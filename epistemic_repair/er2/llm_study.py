"""Auditable 40-cell ER-2 LLM causal-repair study and report generation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Sequence

from epistemic_repair.er2.evaluation import ER2_HYPOTHESES
from epistemic_repair.er2.llm_policy import ER2CausalRepairLLMPolicy
from epistemic_repair.er2.llm_prompts import (
    DIAGNOSIS_CAUSAL_DESCRIPTIONS,
    ER2_REPAIR_PROMPT_VERSION,
    ER2LLMCondition,
    REPAIR_ACTION_DESCRIPTIONS,
    RepairOptionID,
    build_repair_selection_prompt,
    canonical_text_sha256,
    option_permutation,
)
from epistemic_repair.er2.llm_runner import (
    ER2LLMOutcome,
    ER2LLMRepairEpisodeResult,
    ER2LLMRepairEpisodeRunner,
)
from epistemic_repair.er2.policies import FixedRepairPolicy
from epistemic_repair.er2.runner import ER2RepairEpisodeRunner
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient, LLMRateLimitError, LLMRequest, LLMResponse
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.repair.operators import repair_for_failure


ER2_LLM_BENCHMARK_VERSION = "er2_causal_repair_v0"
DEFAULT_ER2_LLM_SEEDS = tuple(range(10))
ARTIFACT_FILENAMES = (
    "episodes.csv",
    "summary.csv",
    "per_hypothesis.csv",
    "wrong_repair_analysis.csv",
    "permutation_audit.csv",
    "wording_audit.json",
    "traces.jsonl",
    "report.md",
    "run_config.json",
)


@dataclass(frozen=True, slots=True)
class ER2LLMStudyConfig:
    """Frozen scientific condition plus provider and pacing controls."""

    provider: str = "gemini"
    model_id: str = "gemini-3.6-flash"
    thinking_level: str = "low"
    seeds: tuple[int, ...] = DEFAULT_ER2_LLM_SEEDS
    timeout_seconds: float = 60.0
    max_output_tokens: int = 1024
    max_retries: int = 1
    min_request_interval_seconds: float = 2.0
    rate_limit_backoff_seconds: float = 30.0
    episode_cooldown_seconds: float = 5.0
    condition: ER2LLMCondition = ER2LLMCondition.CAUSAL_REPAIR_SELECTION

    def __post_init__(self) -> None:
        if not self.seeds or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("seeds must be a non-empty sequence of integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.condition is not ER2LLMCondition.CAUSAL_REPAIR_SELECTION:
            raise ValueError("ER-2 supports exactly CAUSAL_REPAIR_SELECTION")
        for name in (
            "min_request_interval_seconds",
            "rate_limit_backoff_seconds",
            "episode_cooldown_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.llm_config()

    def llm_config(self) -> LLMConfig:
        return LLMConfig(
            provider=self.provider,
            model_id=self.model_id,
            thinking_level=self.thinking_level,
            max_output_tokens=self.max_output_tokens,
            request_timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_decision_calls=1,
        )

    @property
    def episode_count(self) -> int:
        return len(ER2_HYPOTHESES) * len(self.seeds)


class PacedER2LLMClient(LLMClient):
    """Enforce Tier-1 request spacing and bounded rate-limit cooldowns."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        minimum_interval_seconds: float,
        rate_limit_backoff_seconds: float,
        progress: Callable[[str], None],
    ) -> None:
        self.inner = inner
        self.minimum_interval_seconds = minimum_interval_seconds
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self.progress = progress
        self._last_start: float | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        now = monotonic()
        if self._last_start is not None:
            wait = self.minimum_interval_seconds - (now - self._last_start)
            if wait > 0:
                self.progress(f"waiting {wait:.1f}s for provider pacing")
                sleep(wait)
        self._last_start = monotonic()
        try:
            return self.inner.generate(request)
        except LLMRateLimitError:
            if self.rate_limit_backoff_seconds:
                self.progress(
                    f"rate limited; waiting {self.rate_limit_backoff_seconds:.1f}s before retry"
                )
                sleep(self.rate_limit_backoff_seconds)
            raise


def run_er2_llm_study(
    client: LLMClient,
    config: ER2LLMStudyConfig,
    output_directory: Path | str,
    *,
    overwrite: bool = False,
    retry_provider_failures: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run or resume the study, checkpointing every cell and regenerating artifacts."""
    notify = progress or (lambda _message: None)
    output = Path(output_directory)
    checkpoints = output / ".checkpoints"
    _prepare_run(output, checkpoints, config, overwrite=overwrite)
    paced = PacedER2LLMClient(
        client,
        minimum_interval_seconds=config.min_request_interval_seconds,
        rate_limit_backoff_seconds=config.rate_limit_backoff_seconds,
        progress=notify,
    )
    runner = ER2LLMRepairEpisodeRunner(
        ER2CausalRepairLLMPolicy(paced, config.llm_config())
    )
    cells = [(hypothesis, seed) for hypothesis in ER2_HYPOTHESES for seed in config.seeds]
    rows: list[dict[str, Any]] = []
    for index, (hypothesis, seed) in enumerate(cells, start=1):
        checkpoint = checkpoints / f"{hypothesis.value.lower()}_seed{seed}.json"
        prior = _read_json(checkpoint) if checkpoint.is_file() else None
        if prior is not None and not (
            retry_provider_failures
            and prior["outcome"] in {
                ER2LLMOutcome.PROVIDER_FAILURE.value,
                ER2LLMOutcome.RATE_LIMIT_FAILURE.value,
            }
        ):
            row = prior
            notify(f"[{index}/{len(cells)}] {hypothesis.value} seed={seed} checkpoint retained")
        else:
            started = monotonic()
            result = runner.run(
                true_hypothesis=hypothesis,
                diagnosis=hypothesis,
                seed=seed,
            )
            row = episode_record(result, config)
            _write_json(checkpoint, row)
            notify(
                f"[{index}/{len(cells)}] {hypothesis.value} seed={seed} "
                f"complete in {monotonic() - started:.1f}s retries={row['retry_count']} "
                f"provider_failure={row['provider_failure']} "
                f"rate_limit_failure={row['rate_limit_failure']} "
                f"scientific_model_failure={row['scientific_model_failure']}"
            )
        rows.append(row)
        write_study_artifacts(output, rows, config)
        if index < len(cells) and config.episode_cooldown_seconds:
            notify(
                f"cooldown: waiting {config.episode_cooldown_seconds:.1f}s before next episode"
            )
            sleep(config.episode_cooldown_seconds)
    return rows


def write_preflight_artifacts(
    output_directory: Path | str,
    config: ER2LLMStudyConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Write zero-call wording/permutation audits and an example prompt."""
    output = Path(output_directory)
    if output.exists() and not overwrite and any(output.iterdir()):
        raise FileExistsError(f"preflight output already exists at {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    audit = wording_audit(config.seeds)
    _require_prompt_boundary(audit)
    _write_csv(output / "permutation_audit.csv", permutation_audit_rows(config.seeds))
    _write_json(output / "wording_audit.json", audit)
    example = option_permutation(config.seeds[0]).prompt_view(FailureMode.SENSOR_CORRUPTION)
    (output / "example_prompt.txt").write_text(
        build_repair_selection_prompt(example) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        "# ER-2 LLM repair-selection preflight\n\n"
        f"The frozen {config.episode_count}-episode grid passed deterministic wording "
        "and permutation generation. All 40 rendered prompts were scanned and contain "
        "no canonical diagnosis or repair enum labels. No provider was called. See "
        "`wording_audit.json`, `permutation_audit.csv`, and `example_prompt.txt`.\n",
        encoding="utf-8",
    )


def episode_record(
    result: ER2LLMRepairEpisodeResult,
    config: ER2LLMStudyConfig,
) -> dict[str, Any]:
    """Flatten an episode and keep hidden mapping only in evaluation metadata."""
    metrics = asdict(result.metrics) if result.metrics is not None else {}
    attempts = [
        {
            "attempt_number": attempt.attempt_number,
            "status": attempt.status.value,
            "raw_output": attempt.raw_output,
            "provider_request_id": attempt.provider_request_id,
            "error_type": attempt.error_type,
            "error_message": attempt.error_message,
        }
        for attempt in result.policy_result.attempts
    ]
    mapping = {
        option.value: result.permutation.repair_by_option[option].value
        for option in RepairOptionID
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": ER2_LLM_BENCHMARK_VERSION,
        "prompt_version": result.policy_result.prompt_version,
        "provider": config.provider,
        "model": config.model_id,
        "thinking_level": config.thinking_level,
        "condition": result.condition.value,
        "true_hypothesis": result.true_hypothesis.value,
        "supplied_diagnosis": result.supplied_diagnosis.value,
        "seed": result.seed,
        "selected_option": result.selected_option.value if result.selected_option else None,
        "selected_repair": result.selected_repair.value if result.selected_repair else None,
        "correct_repair": result.correct_repair.value,
        "option_permutation": mapping,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "outcome": result.outcome.value,
        "valid_selection": result.valid_selection,
        "repair_selection_correct": result.repair_selection_correct,
        "repair_success": metrics.get("repair_success"),
        "post_repair_accuracy": metrics.get("post_repair_accuracy"),
        "affected_region_accuracy": metrics.get("affected_region_accuracy"),
        "unaffected_region_accuracy": metrics.get("unaffected_region_accuracy"),
        "collateral_damage": metrics.get("collateral_damage"),
        "retry_count": result.policy_result.retry_count,
        "provider_failure": result.outcome is ER2LLMOutcome.PROVIDER_FAILURE,
        "rate_limit_failure": result.outcome is ER2LLMOutcome.RATE_LIMIT_FAILURE,
        "scientific_model_failure": result.outcome is ER2LLMOutcome.SCIENTIFIC_MODEL_FAILURE,
        "prompt": result.policy_result.prompt,
        "attempts": attempts,
    }


def permutation_audit_rows(seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Expose the hidden mapping only in an evaluation audit artifact."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        permutation = option_permutation(seed)
        row: dict[str, Any] = {"seed": seed}
        for option in RepairOptionID:
            row[f"option_{option.value}_repair"] = permutation.repair_by_option[option].value
        for hypothesis in ER2_HYPOTHESES:
            correct = repair_for_failure(hypothesis)
            row[f"correct_option_{hypothesis.value}"] = next(
                option.value
                for option, repair in permutation.repair_by_option.items()
                if repair is correct
            )
        rows.append(row)
    return rows


def wording_audit(seeds: Sequence[int]) -> dict[str, Any]:
    """Prove all generated cells use only the centrally frozen canonical strings."""
    cells = 0
    observed_repairs: dict[str, set[str]] = {
        repair.value: set() for repair in REPAIR_ACTION_DESCRIPTIONS
    }
    observed_diagnoses: dict[str, set[str]] = {
        hypothesis.value: set() for hypothesis in ER2_HYPOTHESES
    }
    deterministic = True
    label_leaks: list[dict[str, Any]] = []
    normalized_prompts: dict[str, set[str]] = {
        hypothesis.value: set() for hypothesis in ER2_HYPOTHESES
    }
    hidden_diagnosis_labels = tuple(hypothesis.value for hypothesis in ER2_HYPOTHESES)
    hidden_repair_labels = tuple(repair.value for repair in REPAIR_ACTION_DESCRIPTIONS)
    for hypothesis in ER2_HYPOTHESES:
        for seed in seeds:
            permutation = option_permutation(seed)
            view = permutation.prompt_view(hypothesis)
            prompt = build_repair_selection_prompt(view)
            cells += 1
            observed_diagnoses[hypothesis.value].add(view.diagnosis_description)
            for option in view.options:
                repair = permutation.repair_by_option[option.option_id]
                observed_repairs[repair.value].add(option.description)
            deterministic &= prompt == build_repair_selection_prompt(
                option_permutation(seed).prompt_view(hypothesis)
            )
            normalized_prompts[hypothesis.value].add(
                _remove_option_assignment_block(prompt)
            )
            for label in (*hidden_diagnosis_labels, *hidden_repair_labels):
                if label in prompt:
                    label_leaks.append({
                        "hypothesis": hypothesis.value,
                        "seed": seed,
                        "leaked_label": label,
                    })
    diagnosis_entries = {
        hypothesis.value: {
            "text": text,
            "sha256": canonical_text_sha256(text),
            "observed_variant_count": len(observed_diagnoses[hypothesis.value]),
        }
        for hypothesis, text in DIAGNOSIS_CAUSAL_DESCRIPTIONS.items()
        if hypothesis in ER2_HYPOTHESES
    }
    repair_entries = {
        repair.value: {
            "text": text,
            "sha256": canonical_text_sha256(text),
            "observed_variant_count": len(observed_repairs[repair.value]),
        }
        for repair, text in REPAIR_ACTION_DESCRIPTIONS.items()
    }
    return {
        "prompt_version": ER2_REPAIR_PROMPT_VERSION,
        "episode_count": cells,
        "canonical_diagnoses": diagnosis_entries,
        "canonical_repairs": repair_entries,
        "all_episode_strings_are_canonical": all(
            entry["observed_variant_count"] == 1
            for entry in (*diagnosis_entries.values(), *repair_entries.values())
        ),
        "same_input_prompt_is_deterministic": deterministic,
        "only_option_assignment_varies_across_seeds": all(
            len(prompts) == 1 for prompts in normalized_prompts.values()
        ),
        "scanned_prompt_count": cells,
        "hidden_diagnosis_labels_scanned": list(hidden_diagnosis_labels),
        "hidden_repair_labels_scanned": list(hidden_repair_labels),
        "prompt_label_leakage_free": not label_leaks,
        "prompt_label_leaks": label_leaks,
    }


def write_study_artifacts(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    config: ER2LLMStudyConfig,
) -> None:
    """Regenerate all public artifacts from completed checkpoint rows."""
    public_rows = [_public_episode_row(row) for row in rows]
    _write_csv(output / "episodes.csv", public_rows)
    summary = [_aggregate_row("ALL", rows)]
    per_hypothesis = [
        _aggregate_row(
            hypothesis.value,
            [row for row in rows if row["true_hypothesis"] == hypothesis.value],
        )
        for hypothesis in ER2_HYPOTHESES
    ]
    wrong = [
        {
            "truth": row["true_hypothesis"],
            "selected_repair": row["selected_repair"],
            "overall_post_repair_accuracy": row["post_repair_accuracy"],
            "pre_repair_affected_accuracy": _pre_repair_affected_accuracy(
                FailureMode(row["true_hypothesis"])
            ),
            "affected_accuracy": row["affected_region_accuracy"],
            "unaffected_accuracy": row["unaffected_region_accuracy"],
            "collateral_damage": row["collateral_damage"],
            "partial_affected_improvement_with_collateral_damage": (
                row["affected_region_accuracy"]
                > _pre_repair_affected_accuracy(FailureMode(row["true_hypothesis"]))
                and row["collateral_damage"] > 0.0
            ),
        }
        for row in rows
        if row["outcome"] == ER2LLMOutcome.VALID_WRONG_REPAIR.value
    ]
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "per_hypothesis.csv", per_hypothesis)
    _write_csv(output / "wrong_repair_analysis.csv", wrong, fieldnames=(
        "truth",
        "selected_repair",
        "overall_post_repair_accuracy",
        "pre_repair_affected_accuracy",
        "affected_accuracy",
        "unaffected_accuracy",
        "collateral_damage",
        "partial_affected_improvement_with_collateral_damage",
    ))
    _write_csv(output / "permutation_audit.csv", permutation_audit_rows(config.seeds))
    audit = wording_audit(config.seeds)
    _require_prompt_boundary(audit)
    _write_json(output / "wording_audit.json", audit)
    with (output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        _build_report(summary[0], per_hypothesis, wrong, len(rows), config.episode_count),
        encoding="utf-8",
    )


def _aggregate_row(group: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_selection"]]
    return {
        "group": group,
        "episodes_completed": len(rows),
        "valid_selections": len(valid),
        "valid_correct_selections": sum(row["repair_selection_correct"] is True for row in valid),
        "valid_wrong_selections": sum(row["repair_selection_correct"] is False for row in valid),
        "scientific_model_failures": sum(row["scientific_model_failure"] for row in rows),
        "provider_failures": sum(row["provider_failure"] for row in rows),
        "rate_limit_failures": sum(row["rate_limit_failure"] for row in rows),
        "repair_selection_accuracy": _mean_or_none(
            [float(row["repair_selection_correct"]) for row in valid]
        ),
        "repair_success_rate": _mean_or_none([float(row["repair_success"]) for row in valid]),
        "post_repair_accuracy": _mean_or_none([row["post_repair_accuracy"] for row in valid]),
        "affected_region_accuracy": _mean_or_none([row["affected_region_accuracy"] for row in valid]),
        "unaffected_region_accuracy": _mean_or_none([row["unaffected_region_accuracy"] for row in valid]),
        "collateral_damage": _mean_or_none([row["collateral_damage"] for row in valid]),
    }


def _build_report(
    overall: Mapping[str, Any],
    per_hypothesis: Sequence[Mapping[str, Any]],
    wrong: Sequence[Mapping[str, Any]],
    completed: int,
    planned: int,
) -> str:
    rows = "\n".join(
        f"| {row['group']} | {row['valid_selections']} | {_fmt(row['repair_selection_accuracy'])} | "
        f"{_fmt(row['repair_success_rate'])} | {_fmt(row['post_repair_accuracy'])} | "
        f"{_fmt(row['affected_region_accuracy'])} | {_fmt(row['unaffected_region_accuracy'])} | "
        f"{_fmt(row['collateral_damage'])} |"
        for row in per_hypothesis
    )
    damaging = sum(row["partial_affected_improvement_with_collateral_damage"] for row in wrong)
    return (
        "# ER-2 LLM causal repair-selection study\n\n"
        "## Technical summary\n\n"
        f"Completed {completed} of {planned} planned episodes. Valid structured choices: "
        f"{overall['valid_selections']}; repair-selection accuracy among valid choices: "
        f"{_fmt(overall['repair_selection_accuracy'])}. Structured-output and provider "
        "failures are reported separately and are not relabeled as wrong repairs.\n\n"
        f"Among valid wrong choices, {damaging} cases partially improved affected behavior "
        "while causing positive collateral damage to unaffected knowledge.\n\n"
        "## Per-hypothesis metrics\n\n"
        "| Hypothesis | Valid | Selection accuracy | Repair success | Overall | Affected | Unaffected | Collateral damage |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{rows}\n\n"
        "## Method and denominator\n\n"
        "The externally supplied diagnosis equals the evaluation truth. The model sees "
        "only its fixed causal description and seed-permuted A/B/C/D consequence text. "
        "Trusted Python translates the option and invokes the unchanged deterministic "
        "ER-2 mutation/evaluator. Behavioral metric means use valid structured choices; "
        "completion and failure counts retain all planned episodes.\n"
    )


def _public_episode_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"prompt", "attempts"}}


def _remove_option_assignment_block(prompt: str) -> str:
    """Normalize only the four seed-permuted option lines for audit comparison."""
    prefix, remainder = prompt.split("Candidate repair actions:\n", 1)
    _option_block, suffix = remainder.split("\n\nSelection rule:\n", 1)
    return prefix + "Candidate repair actions:\n[PERMUTED OPTIONS]\n\nSelection rule:\n" + suffix


def _require_prompt_boundary(audit: Mapping[str, Any]) -> None:
    """Fail artifact generation if any hidden enum label reached provider text."""
    if not audit["prompt_label_leakage_free"]:
        raise RuntimeError(
            "preflight failed: canonical diagnosis or repair enum label leaked into prompt"
        )


def _pre_repair_affected_accuracy(hypothesis: FailureMode) -> float:
    """Obtain the pre-repair affected score through the unchanged ER-2 evaluator."""
    result = ER2RepairEpisodeRunner().run(
        true_hypothesis=hypothesis,
        diagnosis=hypothesis,
        policy=FixedRepairPolicy(repair_for_failure(FailureMode.NO_STRUCTURAL_CHANGE)),
    )
    return result.metrics.affected_region_accuracy


def _prepare_run(
    output: Path,
    checkpoints: Path,
    config: ER2LLMStudyConfig,
    *,
    overwrite: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ARTIFACT_FILENAMES:
            path = output / name
            if path.is_file():
                path.unlink()
        if checkpoints.is_dir():
            for path in checkpoints.glob("*.json"):
                path.unlink()
    expected = _config_record(config)
    config_path = output / "run_config.json"
    if config_path.is_file() and _read_json(config_path) != expected:
        raise ValueError("existing run_config.json does not match requested configuration")
    checkpoints.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, expected)


def _config_record(config: ER2LLMStudyConfig) -> dict[str, Any]:
    return {
        "benchmark_version": ER2_LLM_BENCHMARK_VERSION,
        "prompt_version": ER2_REPAIR_PROMPT_VERSION,
        "provider": config.provider,
        "model_id": config.model_id,
        "thinking_level": config.thinking_level,
        "condition": config.condition.value,
        "seeds": list(config.seeds),
        "timeout_seconds": config.timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "max_retries": config.max_retries,
        "min_request_interval_seconds": config.min_request_interval_seconds,
        "rate_limit_backoff_seconds": config.rate_limit_backoff_seconds,
        "episode_cooldown_seconds": config.episode_cooldown_seconds,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    names = list(fieldnames or (list(rows[0]) if rows else ()))
    if not names:
        raise ValueError("fieldnames are required for an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in names})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
