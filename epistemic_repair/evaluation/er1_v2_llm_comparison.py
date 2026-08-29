"""Resumable matched-seed comparison harness for ER-1 V2 LLM policies."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
import subprocess
from time import monotonic, sleep
from typing import Any, Callable, Iterable, Mapping, Sequence

from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.llm_policy import (
    ER1V2FullAutonomousLLMPolicy,
    ER1V2PlannerOnlyLLMPolicy,
    ER1V2ThresholdAwareAutonomousLLMPolicy,
)
from epistemic_repair.er1_v2.llm_runner import (
    ER1_V2_BENCHMARK_VERSION,
    ER1V2LLMEpisodeResult,
    ER1V2LLMEpisodeRunner,
)
from epistemic_repair.evaluation.llm_runner import LLMTerminationReason
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import (
    LLMClient,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.sanitize import sanitize_text
from epistemic_repair.llm.schemas import ER1FullAutonomousDecision, LLMCondition
from epistemic_repair.policies.llm import LLMAttemptStatus
from epistemic_repair.prompts.binary_er1_v2 import (
    BINARY_ER1_V2_PROMPT_VERSION,
    BINARY_ER1_V2_THRESHOLD_AWARE_PROMPT_VERSION,
)


DEFAULT_COMPARISON_SEEDS = tuple(range(10))
DEFAULT_COMPARISON_CONDITIONS = (
    LLMCondition.FULL_AUTONOMOUS,
    LLMCondition.PLANNER_ONLY,
)
SUPPORTED_COMPARISON_CONDITIONS = (
    *DEFAULT_COMPARISON_CONDITIONS,
    LLMCondition.THRESHOLD_AWARE_AUTONOMOUS,
)
RATE_LIMIT_SAFETY_MARGIN_SECONDS = 0.5
_GEMINI_RETRY_DELAY_PATTERN = re.compile(
    r"\bPlease retry in ([0-9]+(?:\.[0-9]+)?)s\b"
)
ARTIFACT_FILENAMES = (
    "episodes.csv",
    "traces.jsonl",
    "summary.csv",
    "per_hypothesis.csv",
    "paired_comparison.csv",
    "confusion_full.json",
    "confusion_planner.json",
    "confusion_threshold_aware.json",
    "report.md",
    "run_config.json",
)
EPISODE_FIELDS = (
    "timestamp",
    "benchmark_version",
    "prompt_version",
    "provider",
    "model",
    "thinking_level",
    "condition",
    "true_hypothesis",
    "seed",
    "experiment_budget",
    "diagnosis_threshold",
    "max_decision_calls",
    "final_diagnosis",
    "diagnosis_correct",
    "diagnosed_correctly_within_budget",
    "threshold_qualified_success",
    "premature_diagnosis",
    "normative_probability_of_final_diagnosis",
    "model_reported_final_confidence",
    "experiments_used",
    "decision_calls",
    "total_retries",
    "termination_reason",
    "cumulative_action_regret",
    "oracle_action_agreements",
    "oracle_action_agreement_rate",
    "mean_autonomous_belief_l1_error",
    "provider_failure_flag",
    "model_failure_flag",
    "scientific_model_failure_flag",
    "provider_rate_limit_failure_flag",
    "provider_transport_failure_flag",
    "provider_error_type",
    "provider_error_message",
    "action_sequence",
    "first_action",
    "repeat_trial_count",
    "use_trusted_sensor_count",
    "change_context_count",
)


@dataclass(frozen=True, slots=True)
class ComparisonCell:
    """One condition, truth, and seed in the matched experimental grid."""

    condition: LLMCondition
    hypothesis: FailureMode
    seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.condition.value, self.hypothesis.value, self.seed)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Exact scientific and provider settings for one comparison run."""

    provider: str = "gemini"
    model_id: str = "gemini-3.6-flash"
    thinking_level: str = "low"
    seeds: tuple[int, ...] = DEFAULT_COMPARISON_SEEDS
    experiment_budget: int = 8
    diagnosis_threshold: float = 0.95
    timeout_seconds: float = 60.0
    max_output_tokens: int = 1024
    max_retries: int = 1
    max_decision_calls: int = 9
    min_request_interval_seconds: float = 3.5
    rate_limit_backoff_seconds: float = 10.0
    episode_cooldown_seconds: float = 0.0
    conditions: tuple[LLMCondition, ...] = DEFAULT_COMPARISON_CONDITIONS

    def __post_init__(self) -> None:
        if not self.seeds or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("seeds must be a non-empty sequence of integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be non-empty and unique")
        if any(condition not in SUPPORTED_COMPARISON_CONDITIONS for condition in self.conditions):
            raise ValueError("unsupported comparison condition")
        if type(self.experiment_budget) is not int or self.experiment_budget <= 0:
            raise ValueError("experiment_budget must be a positive integer")
        if not 0.0 < self.diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        if (
            not isinstance(self.min_request_interval_seconds, (int, float))
            or isinstance(self.min_request_interval_seconds, bool)
            or self.min_request_interval_seconds < 0.0
        ):
            raise ValueError("min_request_interval_seconds must be non-negative")
        if (
            not isinstance(self.rate_limit_backoff_seconds, (int, float))
            or isinstance(self.rate_limit_backoff_seconds, bool)
            or self.rate_limit_backoff_seconds < 0.0
        ):
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        if (
            not isinstance(self.episode_cooldown_seconds, (int, float))
            or isinstance(self.episode_cooldown_seconds, bool)
            or self.episode_cooldown_seconds < 0.0
        ):
            raise ValueError("episode_cooldown_seconds must be non-negative")
        # Reuse the provider-neutral validation for all remaining call controls.
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

    def reproducibility_dict(self, *, timestamp: str, git_commit_sha: str | None) -> dict[str, Any]:
        return {
            "git_commit_sha": git_commit_sha,
            "benchmark_version": ER1_V2_BENCHMARK_VERSION,
            "prompt_version": _configured_prompt_version(self.conditions),
            "provider": self.provider,
            "model": self.model_id,
            "thinking_level": self.thinking_level,
            "experiment_budget": self.experiment_budget,
            "diagnosis_threshold": self.diagnosis_threshold,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "max_decision_calls": self.max_decision_calls,
            "min_request_interval_seconds": self.min_request_interval_seconds,
            "rate_limit_backoff_seconds": self.rate_limit_backoff_seconds,
            "episode_cooldown_seconds": self.episode_cooldown_seconds,
            "seeds": list(self.seeds),
            "conditions": [condition.value for condition in self.conditions],
            "hypotheses": [hypothesis.value for hypothesis in ER1_HYPOTHESES],
            "timestamp": timestamp,
            "planned_episode_count": len(comparison_cells(self)),
        }


EpisodeExecutor = Callable[[ComparisonCell], ER1V2LLMEpisodeResult]
ProgressWriter = Callable[[str], None]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def _condition_prompt_version(condition: LLMCondition) -> str:
    if condition is LLMCondition.THRESHOLD_AWARE_AUTONOMOUS:
        return BINARY_ER1_V2_THRESHOLD_AWARE_PROMPT_VERSION
    return BINARY_ER1_V2_PROMPT_VERSION


def _configured_prompt_version(conditions: Sequence[LLMCondition]) -> str:
    versions = tuple(dict.fromkeys(_condition_prompt_version(item) for item in conditions))
    return versions[0] if len(versions) == 1 else "+".join(versions)


def parse_gemini_retry_delay_seconds(message: str) -> float | None:
    """Parse only Gemini's narrowly defined ``Please retry in <n>s`` hint."""
    match = _GEMINI_RETRY_DELAY_PATTERN.search(sanitize_text(message))
    if match is None:
        return None
    value = float(match.group(1))
    return value if value >= 0.0 else None


class PacedLLMClient(LLMClient):
    """Serialize provider attempts behind one monotonic process-wide schedule."""

    def __init__(
        self,
        client: LLMClient,
        *,
        min_request_interval_seconds: float,
        rate_limit_backoff_seconds: float,
        clock: Clock = monotonic,
        sleeper: Sleeper = sleep,
        event_writer: ProgressWriter = print,
    ) -> None:
        if min_request_interval_seconds < 0.0:
            raise ValueError("min_request_interval_seconds must be non-negative")
        if rate_limit_backoff_seconds < 0.0:
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        self._client = client
        self._min_interval = float(min_request_interval_seconds)
        self._fallback_backoff = float(rate_limit_backoff_seconds)
        self._clock = clock
        self._sleeper = sleeper
        self._event_writer = event_writer
        self._last_attempt_started: float | None = None
        self._rate_limit_not_before: float | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        now = self._clock()
        pacing_not_before = (
            now
            if self._last_attempt_started is None
            else self._last_attempt_started + self._min_interval
        )
        rate_limit_not_before = self._rate_limit_not_before or now
        target = max(now, pacing_not_before, rate_limit_not_before)
        wait_seconds = max(0.0, target - now)
        if wait_seconds > 0.0:
            if rate_limit_not_before >= pacing_not_before and rate_limit_not_before > now:
                self._event_writer(
                    f"rate limited; waiting {wait_seconds:.1f}s before retry"
                )
            else:
                self._event_writer(
                    f"waiting {wait_seconds:.1f}s for provider pacing"
                )
            self._sleeper(wait_seconds)

        self._last_attempt_started = self._clock()
        if (
            self._rate_limit_not_before is not None
            and self._last_attempt_started >= self._rate_limit_not_before
        ):
            self._rate_limit_not_before = None
        try:
            return self._client.generate(request)
        except LLMRateLimitError as error:
            requested = parse_gemini_retry_delay_seconds(str(error))
            delay = (
                requested + RATE_LIMIT_SAFETY_MARGIN_SECONDS
                if requested is not None
                else self._fallback_backoff
            )
            self._rate_limit_not_before = self._clock() + delay
            raise


def comparison_cells(config: ComparisonConfig) -> tuple[ComparisonCell, ...]:
    """Return a deterministic grid with each matched pair adjacent."""
    return tuple(
        ComparisonCell(condition, hypothesis, seed)
        for hypothesis in ER1_HYPOTHESES
        for seed in config.seeds
        for condition in config.conditions
    )


def run_er1_v2_llm_comparison(
    client: LLMClient,
    config: ComparisonConfig,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
    progress: ProgressWriter = print,
    episode_executor: EpisodeExecutor | None = None,
    retry_provider_failures: bool = True,
    clock: Clock = monotonic,
    sleeper: Sleeper = sleep,
) -> tuple[dict[str, Any], ...]:
    """Run or resume the comparison, checkpointing after every episode."""
    destination = Path(output_dir)
    if overwrite:
        _clear_known_outputs(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / ".checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    run_config = config.reproducibility_dict(
        timestamp=timestamp,
        git_commit_sha=_git_commit_sha(),
    )
    _prepare_run_config(destination / "run_config.json", run_config)
    completed = _load_checkpoints(checkpoint_dir)
    cells = comparison_cells(config)
    paced_client = PacedLLMClient(
        client,
        min_request_interval_seconds=config.min_request_interval_seconds,
        rate_limit_backoff_seconds=config.rate_limit_backoff_seconds,
        clock=clock,
        sleeper=sleeper,
        event_writer=progress,
    )
    execute = episode_executor or _default_executor(paced_client, config)
    started = clock()

    try:
        for index, cell in enumerate(cells, start=1):
            existing = completed.get(cell.key)
            should_retry = (
                existing is not None
                and retry_provider_failures
                and bool(existing["row"].get("provider_failure_flag"))
            )
            if existing is not None and not should_retry:
                progress(format_progress(index, len(cells), cell, 0.0, completed[cell.key]["row"], skipped=True))
                continue
            if should_retry:
                progress(
                    f"[{index}/{len(cells)}] retrying provider-failed cell "
                    f"{cell.condition.value} {cell.hypothesis.value} seed={cell.seed}"
                )
            episode_started = clock()
            try:
                episode = execute(cell)
                row = episode_row(episode)
                trace = sanitized_episode_trace(episode)
            except Exception as error:  # keep one bad provider cell from aborting the study
                row, trace = failure_record(cell, config, error)
            record = {"row": row, "trace": trace}
            _write_checkpoint(checkpoint_dir, cell, record)
            completed[cell.key] = record
            _write_current_artifacts(
                destination,
                completed,
                config,
                elapsed_seconds=clock() - started,
            )
            progress(format_progress(index, len(cells), cell, clock() - episode_started, row))
            if (
                config.episode_cooldown_seconds > 0.0
                and _has_pending_cell(
                    cells[index:],
                    completed,
                    retry_provider_failures=retry_provider_failures,
                )
            ):
                progress(
                    "cooldown: waiting "
                    f"{config.episode_cooldown_seconds:.1f}s before next episode"
                )
                sleeper(config.episode_cooldown_seconds)
    finally:
        _write_current_artifacts(
            destination,
            completed,
            config,
            elapsed_seconds=clock() - started,
        )
        progress(
            f"completed={len(completed)}/{len(cells)} elapsed={clock() - started:.1f}s"
        )
    return tuple(record["row"] for _, record in _ordered_records(completed))


def _default_executor(client: LLMClient, config: ComparisonConfig) -> EpisodeExecutor:
    llm_config = config.llm_config()

    def execute(cell: ComparisonCell) -> ER1V2LLMEpisodeResult:
        if cell.condition is LLMCondition.FULL_AUTONOMOUS:
            policy = ER1V2FullAutonomousLLMPolicy(client, llm_config)
        elif cell.condition is LLMCondition.PLANNER_ONLY:
            policy = ER1V2PlannerOnlyLLMPolicy(client, llm_config)
        else:
            policy = ER1V2ThresholdAwareAutonomousLLMPolicy(
                client,
                llm_config,
                config.diagnosis_threshold,
            )
        runner = ER1V2LLMEpisodeRunner(
            condition=cell.condition,
            experiment_budget=config.experiment_budget,
            diagnosis_threshold=config.diagnosis_threshold,
            episode_seed=cell.seed,
        )
        return runner.run(ER1V2BinaryMachine(), cell.hypothesis, policy)

    return execute


def _has_pending_cell(
    cells: Sequence[ComparisonCell],
    completed: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    retry_provider_failures: bool,
) -> bool:
    """Return whether a later cell will actually execute in this invocation."""
    for cell in cells:
        existing = completed.get(cell.key)
        if existing is None:
            return True
        if retry_provider_failures and bool(
            existing["row"].get("provider_failure_flag")
        ):
            return True
    return False


def episode_row(episode: ER1V2LLMEpisodeResult) -> dict[str, Any]:
    """Flatten one completed result without raw prompts or provider output."""
    actions = [
        turn.experiment_record.action
        for turn in episode.trace
        if turn.experiment_record is not None
    ]
    belief_errors = [
        turn.autonomous_belief_l1_error
        for turn in episode.trace
        if turn.autonomous_belief_l1_error is not None
    ]
    attempts = [attempt for turn in episode.trace for attempt in turn.policy_result.attempts]
    provider_attempts = [
        attempt
        for attempt in attempts
        if attempt.status in (LLMAttemptStatus.TRANSIENT_ERROR, LLMAttemptStatus.PROVIDER_ERROR)
    ]
    rate_limit_attempts = [
        attempt
        for attempt in provider_attempts
        if attempt.error_type == LLMRateLimitError.__name__
    ]
    provider_error = provider_attempts[-1] if provider_attempts else None
    experiments = episode.experiments_used
    metadata = episode.run_metadata
    return {
        "timestamp": metadata.timestamp_utc,
        "benchmark_version": metadata.benchmark_version,
        "prompt_version": metadata.prompt_version,
        "provider": metadata.provider,
        "model": metadata.model_id,
        "thinking_level": metadata.thinking_level,
        "condition": metadata.condition.value,
        "true_hypothesis": episode.evaluation_metadata.hidden_failure_mode.value,
        "seed": episode.evaluation_metadata.episode_seed,
        "experiment_budget": metadata.experiment_budget,
        "diagnosis_threshold": episode.evaluation_metadata.diagnosis_threshold,
        "max_decision_calls": metadata.max_decision_calls,
        "final_diagnosis": episode.final_diagnosis.value if episode.final_diagnosis else None,
        "diagnosis_correct": episode.diagnosis_correct,
        "diagnosed_correctly_within_budget": episode.diagnosed_correctly_within_budget,
        "threshold_qualified_success": episode.threshold_qualified_success,
        "premature_diagnosis": episode.premature_diagnosis,
        "normative_probability_of_final_diagnosis": episode.normative_probability_of_final_diagnosis,
        "model_reported_final_confidence": episode.final_confidence,
        "experiments_used": experiments,
        "decision_calls": episode.decision_calls,
        "total_retries": episode.total_retries,
        "termination_reason": episode.termination_reason.value,
        "cumulative_action_regret": episode.cumulative_action_regret,
        "oracle_action_agreements": episode.oracle_action_agreements,
        "oracle_action_agreement_rate": episode.oracle_action_agreements / experiments if experiments else None,
        "mean_autonomous_belief_l1_error": _mean(belief_errors),
        "provider_failure_flag": bool(provider_attempts),
        "model_failure_flag": episode.termination_reason is LLMTerminationReason.MODEL_FAILURE,
        "scientific_model_failure_flag": (
            episode.termination_reason is LLMTerminationReason.MODEL_FAILURE
            and not provider_attempts
        ),
        "provider_rate_limit_failure_flag": bool(rate_limit_attempts),
        "provider_transport_failure_flag": bool(provider_attempts),
        "provider_error_type": provider_error.error_type if provider_error else None,
        "provider_error_message": sanitize_text(provider_error.error_message) if provider_error and provider_error.error_message else None,
        "action_sequence": ">".join(action.value for action in actions),
        "first_action": actions[0].value if actions else None,
        "repeat_trial_count": actions.count(DiagnosticAction.REPEAT_TRIAL),
        "use_trusted_sensor_count": actions.count(DiagnosticAction.USE_TRUSTED_SENSOR),
        "change_context_count": actions.count(DiagnosticAction.CHANGE_CONTEXT),
    }


def sanitized_episode_trace(episode: ER1V2LLMEpisodeResult) -> dict[str, Any]:
    """Build an audit trace containing safe decisions but no prompts/raw output."""
    turns: list[dict[str, Any]] = []
    for turn in episode.trace:
        decision = turn.policy_result.decision
        beliefs = None
        if isinstance(decision, ER1FullAutonomousDecision):
            beliefs = _beliefs_dict(decision.beliefs)
        record = turn.experiment_record
        turns.append({
            "condition": episode.run_metadata.condition.value,
            "truth": episode.evaluation_metadata.hidden_failure_mode.value,
            "seed": episode.evaluation_metadata.episode_seed,
            "call_number": turn.call_number,
            "decision": decision.decision.value if decision else None,
            "model_beliefs": beliefs,
            "action": decision.action.value if decision and decision.action else None,
            "diagnosis": decision.diagnosis.value if decision and decision.diagnosis else None,
            "reason_summary": sanitize_text(decision.reason_summary) if decision else None,
            "normative_belief_before": _beliefs_dict(turn.normative_prior),
            "normative_belief_after": _beliefs_dict(turn.normative_posterior) if turn.normative_posterior else None,
            "action_regret": turn.action_regret,
            "oracle_agreement": turn.oracle_action_agreement,
            "experiment_result": _json_safe(record.result) if record else None,
            "retry_count": turn.policy_result.retry_count,
            "attempts": [
                {
                    "attempt_number": attempt.attempt_number,
                    "status": attempt.status.value,
                    "error_type": attempt.error_type,
                    "error_message": sanitize_text(attempt.error_message) if attempt.error_message else None,
                    "provider_request_id": sanitize_text(attempt.provider_request_id) if attempt.provider_request_id else None,
                }
                for attempt in turn.policy_result.attempts
            ],
        })
    return {
        "condition": episode.run_metadata.condition.value,
        "truth": episode.evaluation_metadata.hidden_failure_mode.value,
        "seed": episode.evaluation_metadata.episode_seed,
        "turns": turns,
    }


def failure_record(
    cell: ComparisonCell,
    config: ComparisonConfig,
    error: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Represent an unexpected per-cell exception as a sanitized failed episode."""
    now = datetime.now(timezone.utc).isoformat()
    error_message = sanitize_text(str(error))
    row = {field: None for field in EPISODE_FIELDS}
    is_rate_limit = isinstance(error, LLMRateLimitError)
    row.update({
        "timestamp": now,
        "benchmark_version": ER1_V2_BENCHMARK_VERSION,
        "prompt_version": _condition_prompt_version(cell.condition),
        "provider": config.provider,
        "model": config.model_id,
        "thinking_level": config.thinking_level,
        "condition": cell.condition.value,
        "true_hypothesis": cell.hypothesis.value,
        "seed": cell.seed,
        "experiment_budget": config.experiment_budget,
        "diagnosis_threshold": config.diagnosis_threshold,
        "max_decision_calls": config.max_decision_calls,
        "diagnosis_correct": False,
        "diagnosed_correctly_within_budget": False,
        "threshold_qualified_success": False,
        "premature_diagnosis": False,
        "experiments_used": 0,
        "decision_calls": 0,
        "total_retries": 0,
        "termination_reason": LLMTerminationReason.MODEL_FAILURE.value,
        "cumulative_action_regret": 0.0,
        "oracle_action_agreements": 0,
        "provider_failure_flag": True,
        "model_failure_flag": True,
        "scientific_model_failure_flag": False,
        "provider_rate_limit_failure_flag": is_rate_limit,
        "provider_transport_failure_flag": True,
        "provider_error_type": type(error).__name__,
        "provider_error_message": error_message,
        "action_sequence": "",
        "repeat_trial_count": 0,
        "use_trusted_sensor_count": 0,
        "change_context_count": 0,
    })
    trace = {
        "condition": cell.condition.value,
        "truth": cell.hypothesis.value,
        "seed": cell.seed,
        "turns": [],
        "harness_error": {"type": type(error).__name__, "message": error_message},
    }
    return row, trace


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, by_hypothesis: bool = False) -> list[dict[str, Any]]:
    """Aggregate requested metrics overall or by condition and truth."""
    groups: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["true_hypothesis"]) if by_hypothesis else None)
        groups.setdefault(key, []).append(row)
    summaries = []
    condition_order = {condition.value: i for i, condition in enumerate(SUPPORTED_COMPARISON_CONDITIONS)}
    hypothesis_order = {hypothesis.value: i for i, hypothesis in enumerate(ER1_HYPOTHESES)}
    for (condition, hypothesis), group in sorted(
        groups.items(),
        key=lambda item: (condition_order[item[0][0]], hypothesis_order.get(item[0][1], -1)),
    ):
        experiment_total = sum(int(row["experiments_used"] or 0) for row in group)
        agreement_total = sum(int(row["oracle_action_agreements"] or 0) for row in group)
        belief_values = [float(row["mean_autonomous_belief_l1_error"]) for row in group if row["mean_autonomous_belief_l1_error"] is not None]
        summary = {
            "condition": condition,
            "episode_count": len(group),
            "diagnosis_accuracy": _rate(group, "diagnosis_correct"),
            "diagnosed_correctly_within_budget": _rate(group, "diagnosed_correctly_within_budget"),
            "threshold_qualified_success": _rate(group, "threshold_qualified_success"),
            "premature_diagnosis_rate": _rate(group, "premature_diagnosis"),
            "mean_experiments": _numeric_mean(group, "experiments_used"),
            "mean_decision_calls": _numeric_mean(group, "decision_calls"),
            "mean_regret": _numeric_mean(group, "cumulative_action_regret"),
            "oracle_action_agreement_rate": agreement_total / experiment_total if experiment_total else None,
            "mean_retries": _numeric_mean(group, "total_retries"),
            "provider_failure_rate": _rate(group, "provider_failure_flag"),
            "provider_rate_limit_failure_rate": _rate(
                group, "provider_rate_limit_failure_flag"
            ),
            "provider_transport_failure_rate": _rate(
                group, "provider_transport_failure_flag"
            ),
            "scientific_model_failure_rate": _rate(
                group, "scientific_model_failure_flag"
            ),
            "mean_autonomous_belief_l1_error": _mean(belief_values),
        }
        if by_hypothesis:
            summary = {"condition": condition, "true_hypothesis": hypothesis, **{k: v for k, v in summary.items() if k != "condition"}}
        summaries.append(summary)
    return summaries


def confusion_matrix(rows: Sequence[Mapping[str, Any]], condition: LLMCondition) -> dict[str, Any]:
    """Count four-way diagnoses and expose failures/no-diagnosis separately."""
    labels = [hypothesis.value for hypothesis in ER1_HYPOTHESES]
    columns = [*labels, "NO_DIAGNOSIS"]
    counts = {truth: {diagnosis: 0 for diagnosis in columns} for truth in labels}
    for row in rows:
        if row["condition"] != condition.value:
            continue
        diagnosis = row["final_diagnosis"] if row["final_diagnosis"] in labels else "NO_DIAGNOSIS"
        counts[str(row["true_hypothesis"])][str(diagnosis)] += 1
    row_rates = {
        truth: {
            diagnosis: (count / sum(values.values()) if sum(values.values()) else 0.0)
            for diagnosis, count in values.items()
        }
        for truth, values in counts.items()
    }
    return {"condition": condition.value, "rows": labels, "columns": columns, "counts": counts, "row_rates": row_rates}


def paired_comparison(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare condition results matched on exactly hypothesis and seed."""
    indexed = {
        (str(row["condition"]), str(row["true_hypothesis"]), int(row["seed"])): row
        for row in rows
    }
    output = []
    for hypothesis in ER1_HYPOTHESES:
        seeds = sorted({int(row["seed"]) for row in rows if row["true_hypothesis"] == hypothesis.value})
        for seed in seeds:
            full = indexed.get((LLMCondition.FULL_AUTONOMOUS.value, hypothesis.value, seed))
            planner = indexed.get((LLMCondition.PLANNER_ONLY.value, hypothesis.value, seed))
            if full is None or planner is None:
                continue
            full_correct = bool(full["diagnosis_correct"])
            planner_correct = bool(planner["diagnosis_correct"])
            category = (
                "BOTH_CORRECT" if full_correct and planner_correct else
                "PLANNER_CORRECT_FULL_WRONG" if planner_correct else
                "FULL_CORRECT_PLANNER_WRONG" if full_correct else
                "BOTH_WRONG"
            )
            output.append({
                "true_hypothesis": hypothesis.value,
                "seed": seed,
                "pair_key": f"{hypothesis.value}:{seed}",
                "outcome_category": category,
                "full_final_diagnosis": full["final_diagnosis"],
                "planner_final_diagnosis": planner["final_diagnosis"],
                "full_correct": full_correct,
                "planner_correct": planner_correct,
                "delta_experiments_planner_minus_full": _delta(planner, full, "experiments_used"),
                "delta_regret_planner_minus_full": _delta(planner, full, "cumulative_action_regret"),
                "delta_threshold_success_planner_minus_full": int(bool(planner["threshold_qualified_success"])) - int(bool(full["threshold_qualified_success"])),
                "delta_prematurity_planner_minus_full": int(bool(planner["premature_diagnosis"])) - int(bool(full["premature_diagnosis"])),
                "delta_oracle_agreement_planner_minus_full": _nullable_delta(planner["oracle_action_agreement_rate"], full["oracle_action_agreement_rate"]),
            })
    return output


def format_progress(
    index: int,
    total: int,
    cell: ComparisonCell,
    elapsed_seconds: float,
    row: Mapping[str, Any],
    *,
    skipped: bool = False,
) -> str:
    status = "skipped (already complete)" if skipped else "complete"
    return (
        f"[{index}/{total}] {cell.condition.value} {cell.hypothesis.value} "
        f"seed={cell.seed} {status} in {elapsed_seconds:.1f}s "
        f"retries={row.get('total_retries', 0)} "
        f"provider_failure={bool(row.get('provider_failure_flag'))} "
        f"rate_limit_failure={bool(row.get('provider_rate_limit_failure_flag'))} "
        f"scientific_model_failure={bool(row.get('scientific_model_failure_flag'))}"
    )


def _write_current_artifacts(
    destination: Path,
    records: Mapping[tuple[str, str, int], Mapping[str, Any]],
    config: ComparisonConfig,
    *,
    elapsed_seconds: float | None = None,
) -> None:
    ordered = _ordered_records(records)
    rows = [record["row"] for _, record in ordered]
    _write_csv(destination / "episodes.csv", rows, EPISODE_FIELDS)
    with (destination / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for _, record in ordered:
            handle.write(json.dumps(record["trace"], sort_keys=True) + "\n")
        handle.flush()
    summary = summarize_rows(rows)
    per_hypothesis = summarize_rows(rows, by_hypothesis=True)
    pairs = paired_comparison(rows)
    _write_csv(destination / "summary.csv", summary)
    _write_csv(destination / "per_hypothesis.csv", per_hypothesis)
    _write_csv(destination / "paired_comparison.csv", pairs)
    _write_json(destination / "confusion_full.json", confusion_matrix(rows, LLMCondition.FULL_AUTONOMOUS))
    _write_json(destination / "confusion_planner.json", confusion_matrix(rows, LLMCondition.PLANNER_ONLY))
    _write_json(
        destination / "confusion_threshold_aware.json",
        confusion_matrix(rows, LLMCondition.THRESHOLD_AWARE_AUTONOMOUS),
    )
    (destination / "report.md").write_text(
        _render_report(
            config,
            rows,
            summary,
            per_hypothesis,
            pairs,
            elapsed_seconds=elapsed_seconds,
        ),
        encoding="utf-8",
    )


def _render_report(
    config: ComparisonConfig,
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    per_hypothesis: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float | None = None,
) -> str:
    planned = len(comparison_cells(config))
    provider_failed = sum(bool(row.get("provider_failure_flag")) for row in rows)
    scientific_failed = sum(
        bool(row.get("scientific_model_failure_flag")) for row in rows
    )
    valid = len(rows) - provider_failed - scientific_failed
    remaining = planned - len(rows)
    lines = [
        "# ER-1 V2 LLM matched-condition comparison",
        "",
        f"Completed **{len(rows)} / {planned}** planned episodes. Results are provisional until complete.",
        "",
        f"This is a small {len(config.seeds)}-seed measurement study; differences should not be treated as stable scientific conclusions without replication.",
        "",
        "## Run progress",
        "",
        f"- Completed valid cells: **{valid}**",
        f"- Provider-failed cells: **{provider_failed}**",
        f"- Scientific-model-failed cells: **{scientific_failed}**",
        f"- Remaining unattempted cells: **{remaining}**",
        (
            f"- Elapsed wall-clock time for this invocation: **{elapsed_seconds:.1f} seconds**"
            if elapsed_seconds is not None
            else "- Elapsed wall-clock time: unavailable"
        ),
        "",
        "Scientific conclusions are deferred until enough valid matched pairs exist.",
        "",
        "## Overall results",
        "",
        "| Condition | N | Accuracy | Threshold success | Premature | Mean experiments | Mean regret | Oracle agreement | Rate-limit failure | Scientific model failure | Belief L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['episode_count']} | {_fmt(row['diagnosis_accuracy'])} | "
            f"{_fmt(row['threshold_qualified_success'])} | {_fmt(row['premature_diagnosis_rate'])} | "
            f"{_fmt(row['mean_experiments'])} | {_fmt(row['mean_regret'])} | "
            f"{_fmt(row['oracle_action_agreement_rate'])} | {_fmt(row['provider_rate_limit_failure_rate'])} | "
            f"{_fmt(row['scientific_model_failure_rate'])} | "
            f"{_fmt(row['mean_autonomous_belief_l1_error'])} |"
        )
    lines.extend(["", "## Per-hypothesis results", "", "| Condition | Truth | N | Accuracy | Threshold success | Premature | Mean regret | Provider failure |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in per_hypothesis:
        lines.append(
            f"| {row['condition']} | {row['true_hypothesis']} | {row['episode_count']} | "
            f"{_fmt(row['diagnosis_accuracy'])} | {_fmt(row['threshold_qualified_success'])} | "
            f"{_fmt(row['premature_diagnosis_rate'])} | {_fmt(row['mean_regret'])} | {_fmt(row['provider_failure_rate'])} |"
        )
    categories = {name: sum(pair["outcome_category"] == name for pair in pairs) for name in ("BOTH_CORRECT", "PLANNER_CORRECT_FULL_WRONG", "FULL_CORRECT_PLANNER_WRONG", "BOTH_WRONG")}
    lines.extend([
        "",
        "## Paired comparison",
        "",
        f"Matched pairs: **{len(pairs)}**. " + ", ".join(f"{name}={count}" for name, count in categories.items()) + ".",
        "",
        "Pairs use `(true hypothesis, seed)` as the key. Every delta is planner-only minus full-autonomous.",
        "",
        "## Interpretation guide",
        "",
        "The following cases are hypotheses to inspect, not automatically selected conclusions:",
        "",
        "- **Case A:** If planner-only substantially outperforms full autonomous, belief estimation/updating is likely a major bottleneck.",
        "- **Case B:** If accuracy is similar but planner-only has much lower regret, authoritative beliefs may indirectly improve planning quality.",
        "- **Case C:** If planner-only still diagnoses prematurely, stopping remains a bottleneck even with correct beliefs.",
        "- **Case D:** If both conditions perform similarly poorly, planning/reasoning or model capability is the larger candidate bottleneck.",
        "- **Case E:** If provider failures are appreciable, transport reliability must be separated from scientific model failure.",
        "",
        _supported_interpretation(summary),
        "",
        "Confusion matrices, individual paired deltas, and sanitized turn traces are provided as separate artifacts.",
        "",
    ])
    return "\n".join(lines)


def _supported_interpretation(summary: Sequence[Mapping[str, Any]]) -> str:
    indexed = {row["condition"]: row for row in summary}
    full = indexed.get(LLMCondition.FULL_AUTONOMOUS.value)
    planner = indexed.get(LLMCondition.PLANNER_ONLY.value)
    if not full or not planner or min(int(full["episode_count"]), int(planner["episode_count"])) < 10:
        return "No interpretation case is selected because the currently completed matched sample is too small."
    observations = []
    accuracy_delta = float(planner["diagnosis_accuracy"]) - float(full["diagnosis_accuracy"])
    if accuracy_delta >= 0.15:
        observations.append("Case A is directionally supported by the observed accuracy gap")
    if float(planner["premature_diagnosis_rate"]) >= 0.20:
        observations.append("Case C is directionally supported by planner-only prematurity")
    if max(float(full["provider_failure_rate"]), float(planner["provider_failure_rate"])) >= 0.10:
        observations.append("Case E is directionally supported by the observed provider-failure rate")
    return ("; ".join(observations) + ". Treat this as descriptive, not confirmatory.") if observations else "No single interpretation case is clearly supported by the current descriptive thresholds."


def _prepare_run_config(path: Path, current: Mapping[str, Any]) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        scheduling_fields = (
            "min_request_interval_seconds",
            "rate_limit_backoff_seconds",
            "episode_cooldown_seconds",
        )
        migrated = dict(previous)
        for field in scheduling_fields:
            migrated.setdefault(field, current[field])
        ignored = {"timestamp", "git_commit_sha"}
        if {k: v for k, v in migrated.items() if k not in ignored} != {k: v for k, v in current.items() if k not in ignored}:
            raise ValueError("output directory contains an incompatible run_config.json; use a new directory or --overwrite")
        if migrated != previous:
            _write_json(path, migrated)
        return
    _write_json(path, current)


def _checkpoint_name(cell: ComparisonCell) -> str:
    return f"{cell.condition.value}__{cell.hypothesis.value}__{cell.seed}.json"


def _write_checkpoint(directory: Path, cell: ComparisonCell, record: Mapping[str, Any]) -> None:
    target = directory / _checkpoint_name(cell)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _load_checkpoints(directory: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records = {}
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        row = record["row"]
        row.setdefault("provider_transport_failure_flag", bool(row.get("provider_failure_flag")))
        row.setdefault(
            "provider_rate_limit_failure_flag",
            row.get("provider_error_type") == LLMRateLimitError.__name__,
        )
        row.setdefault(
            "scientific_model_failure_flag",
            bool(row.get("model_failure_flag"))
            and not bool(row.get("provider_failure_flag")),
        )
        key = (str(row["condition"]), str(row["true_hypothesis"]), int(row["seed"]))
        records[key] = record
    return records


def _ordered_records(records: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> list[tuple[tuple[str, str, int], Mapping[str, Any]]]:
    condition_order = {condition.value: i for i, condition in enumerate(SUPPORTED_COMPARISON_CONDITIONS)}
    hypothesis_order = {hypothesis.value: i for i, hypothesis in enumerate(ER1_HYPOTHESES)}
    return sorted(records.items(), key=lambda item: (hypothesis_order[item[0][1]], item[0][2], condition_order[item[0][0]]))


def _clear_known_outputs(destination: Path) -> None:
    if not destination.exists():
        return
    for name in ARTIFACT_FILENAMES:
        path = destination / name
        if path.is_file():
            path.unlink()
    checkpoint_dir = destination / ".checkpoints"
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.iterdir():
            if path.is_file() and path.suffix in (".json", ".tmp"):
                path.unlink()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if names:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        handle.flush()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _beliefs_dict(beliefs: Any) -> dict[str, float]:
    return {hypothesis.value: beliefs.probability(hypothesis) for hypothesis in ER1_HYPOTHESES}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(row.get(field)) for row in rows) / len(rows)


def _numeric_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(float(row[field] or 0.0) for row in rows) / len(rows)


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], field: str) -> float:
    return float(left[field] or 0.0) - float(right[field] or 0.0)


def _nullable_delta(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
