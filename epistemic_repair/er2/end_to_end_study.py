"""Resumable reporting harness for the ER-1 V2 -> ER-2 end-to-end study."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Sequence

from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, Context, DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.views import ER1V2BenchmarkAgentView
from epistemic_repair.er2.end_to_end_prompts import (
    END_TO_END_REPAIR_PROMPT_VERSION,
    build_end_to_end_repair_prompt,
    end_to_end_repair_view,
)
from epistemic_repair.er2.end_to_end_runner import (
    VALID_END_TO_END_OUTCOMES,
    EndToEndEpisodeResult,
    EndToEndEpisodeRunner,
    EndToEndOutcome,
)
from epistemic_repair.er2.llm_prompts import (
    DIAGNOSIS_CAUSAL_DESCRIPTIONS,
    REPAIR_ACTION_DESCRIPTIONS,
    RepairOptionID,
    canonical_text_sha256,
    option_permutation,
)
from epistemic_repair.er2.llm_study import PacedER2LLMClient, permutation_audit_rows
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.sanitize import sanitize_text
from epistemic_repair.policies.stochastic_views import StochasticAgentExperimentRecord
from epistemic_repair.prompts.binary_er1_v2 import build_er1_v2_full_autonomous_prompt
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


END_TO_END_BENCHMARK_VERSION = "binary_er1_v2_er2_end_to_end_v0"
DEFAULT_END_TO_END_SEEDS = tuple(range(10))
ARTIFACT_FILENAMES = (
    "episodes.csv",
    "summary.csv",
    "per_hypothesis.csv",
    "failure_chain.csv",
    "wrong_repair_analysis.csv",
    "counterfactual_repairs.csv",
    "permutation_audit.csv",
    "traces.jsonl",
    "report.md",
    "run_config.json",
)


@dataclass(frozen=True, slots=True)
class EndToEndStudyConfig:
    """Frozen end-to-end design and provider/call controls."""

    provider: str = "gemini"
    model_id: str = "gemini-3.6-flash"
    thinking_level: str = "low"
    seeds: tuple[int, ...] = DEFAULT_END_TO_END_SEEDS
    experiment_budget: int = 8
    diagnosis_threshold: float = 0.95
    timeout_seconds: float = 60.0
    max_output_tokens: int = 1024
    max_retries: int = 1
    max_decision_calls: int = 9
    min_request_interval_seconds: float = 2.0
    rate_limit_backoff_seconds: float = 30.0
    episode_cooldown_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.seeds or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("seeds must be a non-empty tuple of integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if type(self.experiment_budget) is not int or self.experiment_budget <= 0:
            raise ValueError("experiment_budget must be positive")
        if not 0.0 < self.diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
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
            max_decision_calls=self.max_decision_calls,
        )

    @property
    def episode_count(self) -> int:
        return len(ER1_HYPOTHESES) * len(self.seeds)


def run_end_to_end_study(
    client: LLMClient,
    config: EndToEndStudyConfig,
    output_directory: Path | str,
    *,
    overwrite: bool = False,
    retry_provider_failures: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run/resume the matched study and checkpoint each end-to-end cell."""
    notify = progress or (lambda _message: None)
    output = Path(output_directory)
    checkpoint_dir = output / ".checkpoints"
    _prepare_run(output, checkpoint_dir, config, overwrite=overwrite)
    paced = PacedER2LLMClient(
        client,
        minimum_interval_seconds=config.min_request_interval_seconds,
        rate_limit_backoff_seconds=config.rate_limit_backoff_seconds,
        progress=notify,
    )
    runner = EndToEndEpisodeRunner(
        paced,
        config.llm_config(),
        experiment_budget=config.experiment_budget,
        diagnosis_threshold=config.diagnosis_threshold,
    )
    cells = [(hypothesis, seed) for hypothesis in ER1_HYPOTHESES for seed in config.seeds]
    records: list[dict[str, Any]] = []
    for index, (hypothesis, seed) in enumerate(cells, start=1):
        checkpoint = checkpoint_dir / f"{hypothesis.value.lower()}_seed{seed}.json"
        prior = _read_json(checkpoint) if checkpoint.is_file() else None
        if prior is not None and not (
            retry_provider_failures
            and prior["row"]["outcome"] in {
                EndToEndOutcome.PROVIDER_FAILURE.value,
                EndToEndOutcome.RATE_LIMIT_FAILURE.value,
            }
        ):
            record = prior
            notify(f"[{index}/{len(cells)}] {hypothesis.value} seed={seed} checkpoint retained")
        else:
            started = monotonic()
            episode = runner.run(true_hypothesis=hypothesis, seed=seed)
            record = episode_record(episode, config)
            _write_json(checkpoint, record)
            row = record["row"]
            notify(
                f"[{index}/{len(cells)}] {hypothesis.value} seed={seed} "
                f"complete in {monotonic() - started:.1f}s retries={row['total_retries']} "
                f"provider_failure={row['provider_failure']} "
                f"rate_limit_failure={row['rate_limit_failure']} "
                f"scientific_model_failure={row['scientific_model_failure']}"
            )
        records.append(record)
        write_end_to_end_artifacts(output, records, config)
        if index < len(cells) and config.episode_cooldown_seconds:
            notify(f"cooldown: waiting {config.episode_cooldown_seconds:.1f}s before next episode")
            sleep(config.episode_cooldown_seconds)
    return records


def episode_record(
    episode: EndToEndEpisodeResult,
    config: EndToEndStudyConfig,
) -> dict[str, Any]:
    """Create CSV row, sanitized trace, and counterfactual rows."""
    investigation_attempts = [
        attempt
        for turn in episode.investigation.trace
        for attempt in turn.policy_result.attempts
    ]
    repair_attempts = (
        list(episode.repair_policy_result.attempts)
        if episode.repair_policy_result is not None
        else []
    )
    metrics = asdict(episode.metrics) if episode.metrics is not None else {}
    actions = [
        turn.experiment_record.action.value
        for turn in episode.investigation.trace
        if turn.experiment_record is not None
    ]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": END_TO_END_BENCHMARK_VERSION,
        "investigation_prompt_version": episode.investigation.run_metadata.prompt_version,
        "repair_prompt_version": END_TO_END_REPAIR_PROMPT_VERSION,
        "provider": config.provider,
        "model": config.model_id,
        "thinking_level": config.thinking_level,
        "condition": "FULL_AUTONOMOUS_END_TO_END_REPAIR",
        "true_hypothesis": episode.true_hypothesis.value,
        "seed": episode.seed,
        "model_diagnosis": episode.model_diagnosis.value if episode.model_diagnosis else None,
        "diagnosis_correct": episode.diagnosis_correct,
        "selected_option": episode.selected_option.value if episode.selected_option else None,
        "selected_repair": episode.selected_repair.value if episode.selected_repair else None,
        "correct_repair": repair_for_failure(episode.true_hypothesis).value,
        "repair_selection_correct": episode.repair_selection_correct,
        "repair_consistent_with_model_diagnosis": episode.repair_consistent_with_model_diagnosis,
        "failure_chain_category": episode.outcome.value,
        "outcome": episode.outcome.value,
        "repair_success": metrics.get("repair_success"),
        "post_repair_accuracy": metrics.get("post_repair_accuracy"),
        "affected_region_accuracy": metrics.get("affected_region_accuracy"),
        "unaffected_region_accuracy": metrics.get("unaffected_region_accuracy"),
        "collateral_damage": metrics.get("collateral_damage"),
        "experiments_used": episode.investigation.experiments_used,
        "decision_calls": episode.investigation.decision_calls,
        "action_sequence": ">".join(actions),
        "investigation_retries": episode.investigation.total_retries,
        "repair_retries": max(0, len(repair_attempts) - 1) if repair_attempts else 0,
        "total_retries": episode.investigation.total_retries + (
            max(0, len(repair_attempts) - 1) if repair_attempts else 0
        ),
        "provider_failure": episode.outcome is EndToEndOutcome.PROVIDER_FAILURE,
        "rate_limit_failure": episode.outcome is EndToEndOutcome.RATE_LIMIT_FAILURE,
        "scientific_model_failure": episode.outcome is EndToEndOutcome.SCIENTIFIC_MODEL_FAILURE,
        "option_permutation": {
            option.value: episode.permutation.repair_by_option[option].value
            for option in RepairOptionID
        },
    }
    trace = {
        "truth_evaluation_only": episode.true_hypothesis.value,
        "seed": episode.seed,
        "investigation": [
            {
                "call_number": turn.call_number,
                "prompt": turn.policy_result.prompt,
                "attempts": [_attempt_record(attempt) for attempt in turn.policy_result.attempts],
                "experiment_result": (
                    _json_safe(turn.experiment_record) if turn.experiment_record else None
                ),
            }
            for turn in episode.investigation.trace
        ],
        "model_diagnosis_evaluation_only": row["model_diagnosis"],
        "repair_prompt": (
            episode.repair_policy_result.prompt if episode.repair_policy_result else None
        ),
        "repair_attempts": [_attempt_record(attempt) for attempt in repair_attempts],
        "selected_option": row["selected_option"],
        "selected_repair_evaluation_only": row["selected_repair"],
        "failure_chain_category": row["failure_chain_category"],
    }
    counterfactual = [
        {
            "true_hypothesis": episode.true_hypothesis.value,
            "seed": episode.seed,
            "model_diagnosis": row["model_diagnosis"],
            "repair": item.repair.value,
            "chosen": item.chosen,
            "oracle": item.oracle,
            **asdict(item.metrics),
        }
        for item in episode.counterfactual_repairs
    ]
    return {"row": row, "trace": trace, "counterfactual": counterfactual}


def write_end_to_end_artifacts(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    config: EndToEndStudyConfig,
) -> None:
    """Regenerate every requested artifact from checkpoint records."""
    rows = [record["row"] for record in records]
    _write_csv(output / "episodes.csv", rows)
    summary = [_aggregate("ALL", rows)]
    per_hypothesis = [
        _aggregate(
            hypothesis.value,
            [row for row in rows if row["true_hypothesis"] == hypothesis.value],
        )
        for hypothesis in ER1_HYPOTHESES
    ]
    failure_chain = [
        {
            "category": category.value,
            "count": sum(row["outcome"] == category.value for row in rows),
            "fraction_of_completed_episodes": (
                sum(row["outcome"] == category.value for row in rows) / len(rows)
                if rows else None
            ),
        }
        for category in EndToEndOutcome
    ]
    wrong = [
        {
            "true_hypothesis": row["true_hypothesis"],
            "seed": row["seed"],
            "model_diagnosis": row["model_diagnosis"],
            "selected_repair": row["selected_repair"],
            "repair_consistent_with_model_diagnosis": row["repair_consistent_with_model_diagnosis"],
            "post_repair_accuracy": row["post_repair_accuracy"],
            "affected_region_accuracy": row["affected_region_accuracy"],
            "unaffected_region_accuracy": row["unaffected_region_accuracy"],
            "collateral_damage": row["collateral_damage"],
            "sensor_corruption_world_update_case": (
                row["true_hypothesis"] == FailureMode.SENSOR_CORRUPTION.value
                and row["selected_repair"] == RepairOperator.UPDATE_WORLD_MODEL.value
            ),
        }
        for row in rows
        if row["repair_selection_correct"] is False
    ]
    counterfactual = [item for record in records for item in record["counterfactual"]]
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "per_hypothesis.csv", per_hypothesis)
    _write_csv(output / "failure_chain.csv", failure_chain)
    _write_csv(
        output / "wrong_repair_analysis.csv",
        wrong,
        fieldnames=(
            "true_hypothesis", "seed", "model_diagnosis", "selected_repair",
            "repair_consistent_with_model_diagnosis", "post_repair_accuracy",
            "affected_region_accuracy", "unaffected_region_accuracy",
            "collateral_damage", "sensor_corruption_world_update_case",
        ),
    )
    _write_csv(
        output / "counterfactual_repairs.csv",
        counterfactual,
        fieldnames=(
            "true_hypothesis", "seed", "model_diagnosis", "repair", "chosen",
            "oracle", "pre_repair_accuracy", "post_repair_accuracy",
            "affected_region_accuracy", "unaffected_region_accuracy",
            "pre_repair_unaffected_accuracy", "collateral_damage", "repair_success",
            "affected_case_count", "unaffected_case_count",
        ),
    )
    _write_csv(output / "permutation_audit.csv", permutation_audit_rows(config.seeds))
    with (output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["trace"], sort_keys=True) + "\n")
    (output / "report.md").write_text(
        build_end_to_end_report(summary[0], per_hypothesis, failure_chain, wrong, len(rows), config.episode_count),
        encoding="utf-8",
    )


def build_end_to_end_report(
    overall: Mapping[str, Any],
    per_hypothesis: Sequence[Mapping[str, Any]],
    failure_chain: Sequence[Mapping[str, Any]],
    wrong_repairs: Sequence[Mapping[str, Any]],
    completed: int,
    planned: int,
) -> str:
    """Build the answer-first technical report required for the live artifact."""
    special = sum(row["sensor_corruption_world_update_case"] for row in wrong_repairs)
    table = "\n".join(
        f"| {row['group']} | {row['valid_end_to_end_episodes']} | {_fmt(row['diagnosis_accuracy'])} | "
        f"{_fmt(row['repair_selection_accuracy'])} | {_fmt(row['repair_success_rate'])} | "
        f"{_fmt(row['post_repair_accuracy'])} | {_fmt(row['collateral_damage'])} |"
        for row in per_hypothesis
    )
    chain = "\n".join(
        f"| {row['category']} | {row['count']} | {_fmt(row['fraction_of_completed_episodes'])} |"
        for row in failure_chain
    )
    return (
        "# ER-1 V2 to ER-2 end-to-end repair study\n\n"
        "## Technical summary\n\n"
        f"Completed {completed} of {planned} planned episodes. Among valid end-to-end "
        f"episodes, diagnosis accuracy is {_fmt(overall['diagnosis_accuracy'])}, repair "
        f"selection accuracy is {_fmt(overall['repair_selection_accuracy'])}, and the "
        f"joint correct-diagnosis/correct-repair rate is "
        f"{_fmt(overall['correct_diagnosis_and_correct_repair_rate'])}. Protocol failures "
        "remain outside valid-choice denominators and are reported separately.\n\n"
        f"The strongest wrong-learning pattern—sensor corruption followed by a world-model "
        f"update—occurred in {special} completed episode(s). Exact cases and collateral "
        "outcomes are in `wrong_repair_analysis.csv`.\n\n"
        "## Diagnosis and repair outcomes by hidden hypothesis\n\n"
        "| Hypothesis | Valid | Diagnosis accuracy | Repair accuracy | Repair success | Post-repair accuracy | Collateral damage |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table}\n\n"
        "## Failure chain remains decomposed\n\n"
        "| Category | Count | Fraction of completed episodes |\n"
        "| --- | ---: | ---: |\n"
        f"{chain}\n\n"
        "## Scope, definitions, and experimental design\n\n"
        "The cohort is the matched four-hypothesis by seed grid. Diagnosis comes from the "
        "unchanged ER-1 V2 FULL_AUTONOMOUS interaction. The repair request receives only "
        "its final agent-visible investigation record and frozen neutral A/B/C/D repair "
        "descriptions. Trusted Python translates the option and invokes the unchanged "
        "deterministic ER-2 mutation/evaluator.\n\n"
        "Behavioral repair means use valid repair selections. Diagnosis accuracy uses "
        "episodes with a completed model diagnosis. Joint and conditional repair metrics "
        "use valid end-to-end episodes. Failure counts retain every completed planned cell.\n\n"
        "## Counterfactual checks distinguish selection from benchmark degeneracy\n\n"
        "For every completed investigation with a diagnosis, all four repairs are evaluated "
        "without additional model calls. `counterfactual_repairs.csv` records chosen, oracle, "
        "and alternative outcomes using the same held-out evaluator.\n\n"
        "## Limitations and robustness\n\n"
        "The provider interface is stateless across calls. To honor the strict no-diagnosis-"
        "label boundary, the repair request repeats the complete agent-visible evidence but "
        "does not re-inject the model's diagnosis label or rationale. The analysis therefore "
        "compares the recorded diagnosis with a repair chosen from the same evidence, rather "
        "than relying on hidden conversational memory. The supplied-diagnosis 40/40 control "
        "remains external and is not pooled into these denominators.\n\n"
        "## Recommended next step\n\n"
        "Run the frozen 40-cell Tier-1 command once, inspect protocol failures before any "
        "scientific interpretation, and do not change prompts based on individual outcomes.\n\n"
        "## Further questions\n\n"
        "If repair errors remain after correct diagnoses, compare their option positions and "
        "counterfactual collateral damage. If errors primarily follow wrong diagnoses, the "
        "bottleneck is evidence interpretation rather than repair semantics.\n"
    )


def preflight_prompt_audit(seeds: Sequence[int]) -> dict[str, Any]:
    """Render 40 representative repair prompts and prove the strict boundary."""
    views = _representative_investigation_views()
    prompts = [
        build_end_to_end_repair_prompt(
            end_to_end_repair_view(view, option_permutation(seed))
        )
        for view in views
        for seed in seeds
    ]
    diagnosis_labels = tuple(hypothesis.value for hypothesis in ER1_HYPOTHESES)
    repair_labels = tuple(repair.value for repair in RepairOperator)
    causal_texts = tuple(DIAGNOSIS_CAUSAL_DESCRIPTIONS.values())
    leaks = [
        {"prompt_index": index, "kind": kind, "value": value}
        for index, prompt in enumerate(prompts)
        for kind, values in (
            ("diagnosis_label", diagnosis_labels),
            ("repair_label", repair_labels),
            ("supplied_causal_interpretation", causal_texts),
        )
        for value in values
        if value in prompt
    ]
    return {
        "prompt_version": END_TO_END_REPAIR_PROMPT_VERSION,
        "scanned_prompt_count": len(prompts),
        "boundary_passed": not leaks,
        "leaks": leaks,
        "same_view_and_seed_is_deterministic": all(
            prompt == build_end_to_end_repair_prompt(
                end_to_end_repair_view(views[index // len(seeds)], option_permutation(seeds[index % len(seeds)]))
            )
            for index, prompt in enumerate(prompts)
        ),
        "repair_wording": {
            repair.value: {
                "text": text,
                "sha256": canonical_text_sha256(text),
            }
            for repair, text in REPAIR_ACTION_DESCRIPTIONS.items()
        },
    }


def write_end_to_end_preflight(
    output_directory: Path | str,
    config: EndToEndStudyConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Write zero-provider-call prompt, permutation, and method artifacts."""
    output = Path(output_directory)
    if output.exists() and not overwrite and any(output.iterdir()):
        raise FileExistsError(f"preflight output already exists at {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    audit = preflight_prompt_audit(config.seeds)
    if not audit["boundary_passed"]:
        raise RuntimeError("end-to-end repair prompt boundary preflight failed")
    _write_json(output / "prompt_audit.json", audit)
    _write_csv(output / "permutation_audit.csv", permutation_audit_rows(config.seeds))
    example = build_end_to_end_repair_prompt(
        end_to_end_repair_view(_example_investigation_view(), option_permutation(0))
    )
    (output / "example_repair_prompt.txt").write_text(example + "\n", encoding="utf-8")
    _write_json(output / "example_complete_prompt_flow.json", preflight_example_flow())
    (output / "report.md").write_text(
        "# End-to-end ER-1 V2 to ER-2 preflight\n\n"
        "## Technical summary\n\n"
        "The 40-cell prospective study is configured but has not called a provider. "
        "All representative repair prompts passed the diagnosis-label, repair-label, "
        "supplied-causal-interpretation, deterministic-permutation, and frozen-wording "
        "checks.\n\n"
        "## Experimental boundary\n\n"
        "The unchanged ER-1 V2 FULL_AUTONOMOUS runner produces the investigation and "
        "records its diagnosis. A separate evidence-only repair request receives the final "
        "ER-1 benchmark-agent view and neutral options. Hidden truth and model diagnosis "
        "remain evaluation-only.\n\n"
        "## Planned evidence and metrics\n\n"
        "The live runner will produce the requested episode, summary, per-hypothesis, "
        "failure-chain, wrong-repair, counterfactual, permutation, trace, and report "
        "artifacts. No scientific result exists at preflight.\n\n"
        "## Limitation\n\n"
        "Because provider requests are stateless, the repair request repeats the complete "
        "agent-visible evidence but does not re-inject the diagnosis label. This preserves "
        "the requested boundary while allowing diagnosis/repair consistency to be analyzed "
        "afterward.\n",
        encoding="utf-8",
    )


def preflight_example_flow() -> dict[str, Any]:
    """Build a complete mocked seed-0 flow with real deterministic ER-1 evidence sampling."""
    env = ER1V2BinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION, episode_seed=0)
    trigger = env.trigger_observation()
    first_view = ER1V2BenchmarkAgentView(
        (trigger,), (), Context.B, BENCHMARK_ACTIONS, 8
    )
    experiment_result = env.run_experiment(
        DiagnosticAction.USE_TRUSTED_SENSOR, x=1
    )
    record = StochasticAgentExperimentRecord(
        1,
        DiagnosticAction.USE_TRUSTED_SENSOR,
        experiment_result,
        Context.B,
        Context.B,
    )
    final_view = ER1V2BenchmarkAgentView(
        (trigger,), (record,), Context.B, BENCHMARK_ACTIONS, 7
    )
    repair_prompt = build_end_to_end_repair_prompt(
        end_to_end_repair_view(final_view, option_permutation(0))
    )
    return {
        "status": "mocked preflight; no provider call and no scientific result",
        "truth_evaluation_only": FailureMode.SENSOR_CORRUPTION.value,
        "seed": 0,
        "calls": [
            {
                "call_number": 1,
                "phase": "ER1_V2_FULL_AUTONOMOUS_INVESTIGATION",
                "prompt": build_er1_v2_full_autonomous_prompt(first_view),
                "mock_response": {
                    "decision": "RUN_EXPERIMENT",
                    "action": DiagnosticAction.USE_TRUSTED_SENSOR.value,
                    "diagnosis": None,
                    "beliefs": {hypothesis.value: 0.25 for hypothesis in ER1_HYPOTHESES},
                    "confidence": None,
                    "reason_summary": "A persistent trusted measurement is needed.",
                },
            },
            {
                "call_number": 2,
                "phase": "ER1_V2_FULL_AUTONOMOUS_DIAGNOSIS",
                "prompt": build_er1_v2_full_autonomous_prompt(final_view),
                "mock_response": {
                    "decision": "DIAGNOSE",
                    "action": None,
                    "diagnosis": FailureMode.SENSOR_CORRUPTION.value,
                    "beliefs": {
                        hypothesis.value: (
                            0.85 if hypothesis is FailureMode.SENSOR_CORRUPTION else 0.05
                        )
                        for hypothesis in ER1_HYPOTHESES
                    },
                    "confidence": 0.85,
                    "reason_summary": "The persistent evidence supports the diagnosis.",
                },
            },
            {
                "call_number": 3,
                "phase": "EVIDENCE_ONLY_NEUTRAL_REPAIR_SELECTION",
                "prompt": repair_prompt,
                "mock_response": {
                    "selected_option": RepairOptionID.A.value,
                    "confidence": 0.9,
                    "rationale": "This is the smallest change supported by the record.",
                },
            },
        ],
    }


def _aggregate(group: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnosed = [row for row in rows if row["model_diagnosis"] is not None]
    valid = [row for row in rows if row["outcome"] in {item.value for item in VALID_END_TO_END_OUTCOMES}]
    correct_diagnosis = [row for row in valid if row["diagnosis_correct"]]
    wrong_diagnosis = [row for row in valid if not row["diagnosis_correct"]]
    wrong_repairs = [row for row in valid if not row["repair_selection_correct"]]
    wrong_repair_wrong_diagnosis = [row for row in wrong_repairs if not row["diagnosis_correct"]]
    wrong_repair_correct_diagnosis = [row for row in wrong_repairs if row["diagnosis_correct"]]
    wrong_diag_correct_repair = [row for row in wrong_diagnosis if row["repair_selection_correct"]]
    return {
        "group": group,
        "episodes_completed": len(rows),
        "diagnosed_episodes": len(diagnosed),
        "valid_end_to_end_episodes": len(valid),
        "diagnosis_accuracy": _mean_or_none([float(row["diagnosis_correct"]) for row in diagnosed]),
        "repair_selection_accuracy": _mean_or_none([float(row["repair_selection_correct"]) for row in valid]),
        "repair_success_rate": _mean_or_none([float(row["repair_success"]) for row in valid]),
        "repair_given_correct_diagnosis_accuracy": _mean_or_none([
            float(row["repair_selection_correct"]) for row in correct_diagnosis
        ]),
        "repair_given_wrong_diagnosis_accuracy": _mean_or_none([
            float(row["repair_selection_correct"]) for row in wrong_diagnosis
        ]),
        "correct_diagnosis_and_correct_repair_rate": _mean_or_none([
            float(row["diagnosis_correct"] and row["repair_selection_correct"]) for row in valid
        ]),
        "post_repair_accuracy": _mean_or_none([row["post_repair_accuracy"] for row in valid]),
        "affected_region_accuracy": _mean_or_none([row["affected_region_accuracy"] for row in valid]),
        "unaffected_region_accuracy": _mean_or_none([row["unaffected_region_accuracy"] for row in valid]),
        "collateral_damage": _mean_or_none([row["collateral_damage"] for row in valid]),
        "fraction_wrong_repairs_caused_by_wrong_diagnosis": _fraction(
            len(wrong_repair_wrong_diagnosis), len(wrong_repairs)
        ),
        "fraction_wrong_repairs_from_selection_failure_despite_correct_diagnosis": _fraction(
            len(wrong_repair_correct_diagnosis), len(wrong_repairs)
        ),
        "fraction_wrong_diagnoses_with_correct_repair": _fraction(
            len(wrong_diag_correct_repair), len(wrong_diagnosis)
        ),
        "scientific_model_failures": sum(row["scientific_model_failure"] for row in rows),
        "provider_failures": sum(row["provider_failure"] for row in rows),
        "rate_limit_failures": sum(row["rate_limit_failure"] for row in rows),
    }


def _representative_investigation_views() -> tuple[ER1V2BenchmarkAgentView, ...]:
    trigger = (Observation(1, 0),)
    return (
        ER1V2BenchmarkAgentView(trigger, (), Context.B, BENCHMARK_ACTIONS, 8),
        ER1V2BenchmarkAgentView(trigger, (
            StochasticAgentExperimentRecord(1, DiagnosticAction.REPEAT_TRIAL, RepeatTrialResult(1, 0), Context.B, Context.B),
        ), Context.B, BENCHMARK_ACTIONS, 7),
        ER1V2BenchmarkAgentView(trigger, (
            StochasticAgentExperimentRecord(1, DiagnosticAction.USE_TRUSTED_SENSOR, StochasticTrustedSensorResult(1, 1), Context.B, Context.B),
        ), Context.B, BENCHMARK_ACTIONS, 7),
        ER1V2BenchmarkAgentView(trigger, (
            StochasticAgentExperimentRecord(1, DiagnosticAction.CHANGE_CONTEXT, ChangeContextResult(Context.A, 1, 1), Context.B, Context.A),
        ), Context.A, BENCHMARK_ACTIONS, 7),
    )


def _example_investigation_view() -> ER1V2BenchmarkAgentView:
    return ER1V2BenchmarkAgentView(
        (Observation(1, 0),),
        (
            StochasticAgentExperimentRecord(1, DiagnosticAction.REPEAT_TRIAL, RepeatTrialResult(1, 0), Context.B, Context.B),
            StochasticAgentExperimentRecord(2, DiagnosticAction.USE_TRUSTED_SENSOR, StochasticTrustedSensorResult(1, 1), Context.B, Context.B),
        ),
        Context.B,
        BENCHMARK_ACTIONS,
        6,
    )


def _attempt_record(attempt) -> dict[str, Any]:
    return {
        "attempt_number": attempt.attempt_number,
        "status": attempt.status.value,
        "raw_output": sanitize_text(attempt.raw_output) if attempt.raw_output else None,
        "provider_request_id": sanitize_text(attempt.provider_request_id) if attempt.provider_request_id else None,
        "error_type": attempt.error_type,
        "error_message": sanitize_text(attempt.error_message) if attempt.error_message else None,
    }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _prepare_run(output: Path, checkpoint_dir: Path, config: EndToEndStudyConfig, *, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ARTIFACT_FILENAMES:
            path = output / name
            if path.is_file():
                path.unlink()
        if checkpoint_dir.is_dir():
            for path in checkpoint_dir.glob("*.json"):
                path.unlink()
    expected = _config_record(config)
    config_path = output / "run_config.json"
    if config_path.is_file() and _read_json(config_path) != expected:
        raise ValueError("existing run_config.json does not match requested configuration")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, expected)


def _config_record(config: EndToEndStudyConfig) -> dict[str, Any]:
    return {
        "benchmark_version": END_TO_END_BENCHMARK_VERSION,
        "investigation_condition": "FULL_AUTONOMOUS",
        "repair_prompt_version": END_TO_END_REPAIR_PROMPT_VERSION,
        "provider": config.provider,
        "model_id": config.model_id,
        "thinking_level": config.thinking_level,
        "seeds": list(config.seeds),
        "experiment_budget": config.experiment_budget,
        "diagnosis_threshold": config.diagnosis_threshold,
        "timeout_seconds": config.timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "max_retries": config.max_retries,
        "max_decision_calls": config.max_decision_calls,
        "min_request_interval_seconds": config.min_request_interval_seconds,
        "rate_limit_backoff_seconds": config.rate_limit_backoff_seconds,
        "episode_cooldown_seconds": config.episode_cooldown_seconds,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames or (list(rows[0]) if rows else ()))
    if not names:
        raise ValueError("fieldnames required for empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in names})


def _csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
