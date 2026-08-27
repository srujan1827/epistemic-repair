"""Oracle-only statistical calibration utilities for stochastic ER-1."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.beliefs.stochastic_likelihoods import StochasticLikelihoodModel
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.evaluation.stochastic_runner import (
    StochasticDiagnosticEpisodeResult,
    StochasticDiagnosticEpisodeRunner,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.policies.stochastic import (
    StochasticOracleInformationGainPolicy,
)


DEFAULT_CALIBRATION_BUDGETS = (1, 2, 3, 5, 8)
DEFAULT_CALIBRATION_THRESHOLDS = (0.80, 0.90, 0.95)
DEFAULT_CALIBRATION_SEEDS = 1000
CALIBRATION_HYPOTHESIS_ORDER = ER1_HYPOTHESES
CALIBRATION_ACTION_ORDER = (
    DiagnosticAction.REPEAT_TRIAL,
    DiagnosticAction.USE_TRUSTED_SENSOR,
    DiagnosticAction.CHANGE_CONTEXT,
)


@dataclass(frozen=True, slots=True)
class BinomialInterval:
    """Point estimate and 95% Wilson interval for a binomial rate."""

    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class OracleCalibrationEpisode:
    """Compact outcome from one seeded oracle episode."""

    hypothesis: FailureMode
    budget: int
    threshold: float
    seed: int
    final_map_diagnosis: FailureMode
    map_correct: bool
    reached_threshold: bool
    success_at_threshold: bool
    experiments_used: int
    final_true_posterior: float
    cumulative_action_regret: float
    action_sequence: tuple[DiagnosticAction, ...]

    @property
    def budget_exhausted(self) -> bool:
        return not self.reached_threshold


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    """Aggregated metrics for one hypothesis × budget × threshold cell."""

    hypothesis: FailureMode
    budget: int
    threshold: float
    episodes: int
    map_accuracy: float
    map_accuracy_ci_lower: float
    map_accuracy_ci_upper: float
    success_at_threshold: float
    success_ci_lower: float
    success_ci_upper: float
    mean_experiments: float
    median_experiments: float
    stddev_experiments: float
    mean_true_posterior: float
    median_true_posterior: float
    p10_true_posterior: float
    p25_true_posterior: float
    p75_true_posterior: float
    p90_true_posterior: float
    threshold_reached_fraction: float
    budget_exhausted_fraction: float
    cumulative_action_regret: float
    mean_episode_action_regret: float
    mean_action_regret: float
    first_repeat_trial_frequency: float
    first_use_trusted_sensor_frequency: float
    first_change_context_frequency: float
    overall_repeat_trial_frequency: float
    overall_use_trusted_sensor_frequency: float
    overall_change_context_frequency: float
    mean_repeat_trials: float
    mean_trusted_sensor_uses: float
    mean_context_changes: float


@dataclass(frozen=True, slots=True)
class OverallCalibrationCell:
    """Aggregate rates across all four equally represented hypotheses."""

    budget: int
    threshold: float
    episodes: int
    map_accuracy: float
    map_accuracy_ci_lower: float
    map_accuracy_ci_upper: float
    success_at_threshold: float
    success_ci_lower: float
    success_ci_upper: float
    mean_experiments: float
    false_structural_diagnosis_rate: float
    false_structural_ci_lower: float
    false_structural_ci_upper: float
    missed_structural_failure_rate: float
    missed_structural_ci_lower: float
    missed_structural_ci_upper: float


@dataclass(frozen=True, slots=True)
class OracleCalibrationStudy:
    """Complete in-memory aggregate of an ER-1 oracle sweep."""

    seed_count: int
    budgets: tuple[int, ...]
    thresholds: tuple[float, ...]
    episodes: tuple[OracleCalibrationEpisode, ...]
    cells: tuple[CalibrationCell, ...]
    overall_cells: tuple[OverallCalibrationCell, ...]
    confusion_matrices: dict[str, object]
    top_action_sequences: dict[str, object]
    hard_cases: dict[str, object]
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class CalibrationArtifacts:
    """Paths written by :func:`write_calibration_artifacts`."""

    calibration_csv: Path
    overall_csv: Path
    confusion_json: Path
    hard_cases_json: Path
    report_markdown: Path


def wilson_interval(successes: int, trials: int) -> BinomialInterval:
    """Return the two-sided 95% Wilson score interval."""
    if type(successes) is not int or type(trials) is not int:
        raise TypeError("successes and trials must be integers")
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    z = 1.959963984540054
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * sqrt(
            estimate * (1.0 - estimate) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return BinomialInterval(estimate, center - radius, center + radius)


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated sample percentile (R-7 convention)."""
    if not values:
        raise ValueError("at least one value is required")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def run_oracle_calibration(
    *,
    seed_count: int = DEFAULT_CALIBRATION_SEEDS,
    budgets: Iterable[int] = DEFAULT_CALIBRATION_BUDGETS,
    thresholds: Iterable[float] = DEFAULT_CALIBRATION_THRESHOLDS,
) -> OracleCalibrationStudy:
    """Run a reproducible oracle-only calibration grid using seeds 0..N-1."""
    budgets_tuple = _validate_budgets(budgets)
    thresholds_tuple = _validate_thresholds(thresholds)
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be a positive integer")

    started = perf_counter()
    episode_records: list[OracleCalibrationEpisode] = []
    likelihood_model = StochasticLikelihoodModel()
    policy = StochasticOracleInformationGainPolicy()

    for budget in budgets_tuple:
        for threshold in thresholds_tuple:
            runner = StochasticDiagnosticEpisodeRunner(
                diagnosis_threshold=threshold,
                max_experiments=budget,
                likelihood_model=likelihood_model,
            )
            environment = StochasticBinaryMachine()
            for hypothesis in CALIBRATION_HYPOTHESIS_ORDER:
                for seed in range(seed_count):
                    result = runner.run(
                        environment,
                        hypothesis,
                        policy,
                        episode_seed=seed,
                    )
                    episode_records.append(
                        _compact_episode(result, hypothesis, budget, threshold, seed)
                    )

    episodes = tuple(episode_records)
    cells = aggregate_calibration_cells(episodes)
    overall = aggregate_overall_cells(episodes)
    confusions = build_confusion_matrices(episodes)
    sequences = build_top_action_sequences(episodes)
    hard_cases = build_hard_cases(episodes)
    return OracleCalibrationStudy(
        seed_count=seed_count,
        budgets=budgets_tuple,
        thresholds=thresholds_tuple,
        episodes=episodes,
        cells=cells,
        overall_cells=overall,
        confusion_matrices=confusions,
        top_action_sequences=sequences,
        hard_cases=hard_cases,
        runtime_seconds=perf_counter() - started,
    )


def aggregate_calibration_cells(
    episodes: Sequence[OracleCalibrationEpisode],
) -> tuple[CalibrationCell, ...]:
    """Aggregate per-hypothesis metrics from compact episode records."""
    grouped: dict[tuple[int, float, FailureMode], list[OracleCalibrationEpisode]] = (
        defaultdict(list)
    )
    for episode in episodes:
        grouped[(episode.budget, episode.threshold, episode.hypothesis)].append(
            episode
        )
    cells = []
    for key in sorted(
        grouped,
        key=lambda item: (
            item[0],
            item[1],
            CALIBRATION_HYPOTHESIS_ORDER.index(item[2]),
        ),
    ):
        cells.append(_aggregate_cell(grouped[key]))
    return tuple(cells)


def aggregate_overall_cells(
    episodes: Sequence[OracleCalibrationEpisode],
) -> tuple[OverallCalibrationCell, ...]:
    """Aggregate rates and structural-error intervals across hypotheses."""
    grouped: dict[tuple[int, float], list[OracleCalibrationEpisode]] = defaultdict(
        list
    )
    for episode in episodes:
        grouped[(episode.budget, episode.threshold)].append(episode)
    rows = []
    for budget, threshold in sorted(grouped):
        group = grouped[(budget, threshold)]
        map_interval = wilson_interval(sum(item.map_correct for item in group), len(group))
        success_interval = wilson_interval(
            sum(item.success_at_threshold for item in group), len(group)
        )
        no_change = [
            item
            for item in group
            if item.hypothesis is FailureMode.NO_STRUCTURAL_CHANGE
        ]
        structural = [
            item
            for item in group
            if item.hypothesis is not FailureMode.NO_STRUCTURAL_CHANGE
        ]
        false_interval = wilson_interval(
            sum(
                item.final_map_diagnosis is not FailureMode.NO_STRUCTURAL_CHANGE
                for item in no_change
            ),
            len(no_change),
        )
        missed_interval = wilson_interval(
            sum(
                item.final_map_diagnosis is FailureMode.NO_STRUCTURAL_CHANGE
                for item in structural
            ),
            len(structural),
        )
        rows.append(
            OverallCalibrationCell(
                budget=budget,
                threshold=threshold,
                episodes=len(group),
                map_accuracy=map_interval.estimate,
                map_accuracy_ci_lower=map_interval.lower,
                map_accuracy_ci_upper=map_interval.upper,
                success_at_threshold=success_interval.estimate,
                success_ci_lower=success_interval.lower,
                success_ci_upper=success_interval.upper,
                mean_experiments=mean(item.experiments_used for item in group),
                false_structural_diagnosis_rate=false_interval.estimate,
                false_structural_ci_lower=false_interval.lower,
                false_structural_ci_upper=false_interval.upper,
                missed_structural_failure_rate=missed_interval.estimate,
                missed_structural_ci_lower=missed_interval.lower,
                missed_structural_ci_upper=missed_interval.upper,
            )
        )
    return tuple(rows)


def build_confusion_matrices(
    episodes: Sequence[OracleCalibrationEpisode],
) -> dict[str, object]:
    """Return counts and row-normalized percentages for every condition."""
    grouped: dict[tuple[int, float], list[OracleCalibrationEpisode]] = defaultdict(
        list
    )
    for episode in episodes:
        grouped[(episode.budget, episode.threshold)].append(episode)
    matrices: dict[str, object] = {
        "hypothesis_order": [item.value for item in CALIBRATION_HYPOTHESIS_ORDER],
        "conditions": {},
    }
    conditions = matrices["conditions"]
    assert isinstance(conditions, dict)
    for budget, threshold in sorted(grouped):
        counts = {
            truth.value: {diagnosis.value: 0 for diagnosis in CALIBRATION_HYPOTHESIS_ORDER}
            for truth in CALIBRATION_HYPOTHESIS_ORDER
        }
        for episode in grouped[(budget, threshold)]:
            counts[episode.hypothesis.value][episode.final_map_diagnosis.value] += 1
        percentages = {}
        for truth in CALIBRATION_HYPOTHESIS_ORDER:
            row = counts[truth.value]
            total = sum(row.values())
            percentages[truth.value] = {
                diagnosis: 100.0 * count / total if total else 0.0
                for diagnosis, count in row.items()
            }
        conditions[_condition_key(budget, threshold)] = {
            "budget": budget,
            "threshold": threshold,
            "counts": counts,
            "row_percentages": percentages,
        }
    return matrices


def build_top_action_sequences(
    episodes: Sequence[OracleCalibrationEpisode],
    *,
    limit: int = 5,
) -> dict[str, object]:
    """Return common action sequences for budget 5, threshold 0.90."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    output: dict[str, object] = {
        "budget": 5,
        "threshold": 0.90,
        "by_hypothesis": {},
    }
    by_hypothesis = output["by_hypothesis"]
    assert isinstance(by_hypothesis, dict)
    for hypothesis in CALIBRATION_HYPOTHESIS_ORDER:
        selected = [
            item
            for item in episodes
            if item.hypothesis is hypothesis
            and item.budget == 5
            and _same_threshold(item.threshold, 0.90)
        ]
        counter = Counter(
            tuple(action.value for action in item.action_sequence) for item in selected
        )
        by_hypothesis[hypothesis.value] = [
            {
                "sequence": list(sequence),
                "count": count,
                "percentage": 100.0 * count / len(selected),
            }
            for sequence, count in sorted(
                counter.items(), key=lambda pair: (-pair[1], pair[0])
            )[:limit]
        ] if selected else []
    return output


def build_hard_cases(
    episodes: Sequence[OracleCalibrationEpisode],
) -> dict[str, object]:
    """Re-run compact representative budget-5/threshold-0.90 traces."""
    output: dict[str, object] = {
        "budget": 5,
        "threshold": 0.90,
        "initial_posterior": _belief_dict(
            StochasticLikelihoodModel().conditioned_initial_beliefs()
        ),
        "by_hypothesis": {},
    }
    by_hypothesis = output["by_hypothesis"]
    assert isinstance(by_hypothesis, dict)
    for hypothesis in CALIBRATION_HYPOTHESIS_ORDER:
        selected = [
            item
            for item in episodes
            if item.hypothesis is hypothesis
            and item.budget == 5
            and _same_threshold(item.threshold, 0.90)
        ]
        quick = sorted(
            (item for item in selected if item.success_at_threshold),
            key=lambda item: (item.experiments_used, item.seed),
        )
        correct_without_threshold = sorted(
            (
                item
                for item in selected
                if item.map_correct and not item.reached_threshold
            ),
            key=lambda item: item.seed,
        )
        incorrect = sorted(
            (item for item in selected if not item.map_correct),
            key=lambda item: item.seed,
        )
        by_hypothesis[hypothesis.value] = {
            "quick_threshold_success": _rerun_hard_case(quick[0]) if quick else None,
            "correct_map_without_threshold": (
                _rerun_hard_case(correct_without_threshold[0])
                if correct_without_threshold
                else None
            ),
            "incorrect_map": _rerun_hard_case(incorrect[0]) if incorrect else None,
        }
    return output


def write_calibration_artifacts(
    study: OracleCalibrationStudy,
    output_dir: Path | str,
) -> CalibrationArtifacts:
    """Write deterministic tabular/JSON outputs and a concise Markdown report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    calibration_csv = directory / "er1_oracle_calibration.csv"
    overall_csv = directory / "er1_oracle_overall.csv"
    confusion_json = directory / "er1_oracle_confusion_matrices.json"
    hard_cases_json = directory / "er1_oracle_hard_cases.json"
    report_markdown = directory / "er1_oracle_calibration.md"

    _write_dataclass_csv(calibration_csv, study.cells)
    _write_dataclass_csv(overall_csv, study.overall_cells)
    _write_json(confusion_json, {
        **study.confusion_matrices,
        "top_action_sequences": study.top_action_sequences,
    })
    _write_json(hard_cases_json, study.hard_cases)
    report_markdown.write_text(build_markdown_report(study), encoding="utf-8")
    return CalibrationArtifacts(
        calibration_csv=calibration_csv,
        overall_csv=overall_csv,
        confusion_json=confusion_json,
        hard_cases_json=hard_cases_json,
        report_markdown=report_markdown,
    )


def build_markdown_report(study: OracleCalibrationStudy) -> str:
    """Render the complete human-readable ER-1 oracle calibration report."""
    target_budget, target_threshold, extended_budget = _report_conditions(study)
    lines = [
        "# ER-1 Oracle Calibration",
        "",
        "## Executive summary",
        "",
        f"This oracle-only sweep contains {len(study.episodes):,} episodes "
        f"({study.seed_count} seeds per hypothesis/condition) and took "
        f"{study.runtime_seconds:.3f} seconds. MAP accuracy and threshold-qualified "
        "success are reported separately.",
        "",
        _recommendation(study),
        "",
        "## Overall results",
        "",
        "| Budget | Threshold | MAP accuracy (95% CI) | Success@threshold (95% CI) | Mean experiments | False structural (95% CI) | Missed structural (95% CI) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in study.overall_cells:
        lines.append(
            f"| {row.budget} | {row.threshold:.2f} | {_ci(row.map_accuracy, row.map_accuracy_ci_lower, row.map_accuracy_ci_upper)} "
            f"| {_ci(row.success_at_threshold, row.success_ci_lower, row.success_ci_upper)} "
            f"| {row.mean_experiments:.3f} "
            f"| {_ci(row.false_structural_diagnosis_rate, row.false_structural_ci_lower, row.false_structural_ci_upper)} "
            f"| {_ci(row.missed_structural_failure_rate, row.missed_structural_ci_lower, row.missed_structural_ci_upper)} |"
        )

    lines.extend(
        _per_hypothesis_section(
            study, budget=target_budget, threshold=target_threshold
        )
    )
    if extended_budget != target_budget:
        lines.extend(
            _per_hypothesis_section(
                study, budget=extended_budget, threshold=target_threshold
            )
        )
    lines.extend(_threshold_sensitivity_section(study, budget=target_budget))
    lines.extend(
        _posterior_section(
            study, budget=target_budget, threshold=target_threshold
        )
    )
    lines.extend(
        _confusion_section(
            study, budget=target_budget, threshold=target_threshold
        )
    )
    if _has_condition(study, 5, 0.90):
        lines.extend(_action_section(study))
        lines.extend(_hard_case_section(study))
    lines.extend([
        "",
        "## Recommendation",
        "",
        _recommendation(study),
        "",
        "No ER-1 probabilities, defaults, prompts, or action semantics were changed by this calibration.",
        "",
    ])
    return "\n".join(lines)


def _compact_episode(
    result: StochasticDiagnosticEpisodeResult,
    hypothesis: FailureMode,
    budget: int,
    threshold: float,
    seed: int,
) -> OracleCalibrationEpisode:
    final_beliefs = result.trace[-1].posterior if result.trace else result.initial_beliefs
    return OracleCalibrationEpisode(
        hypothesis=hypothesis,
        budget=budget,
        threshold=threshold,
        seed=seed,
        final_map_diagnosis=result.predicted_diagnosis,
        map_correct=result.diagnosis_correct,
        reached_threshold=result.reached_threshold,
        success_at_threshold=result.reached_threshold and result.diagnosis_correct,
        experiments_used=result.experiments_used,
        final_true_posterior=final_beliefs.probability(hypothesis),
        cumulative_action_regret=result.cumulative_action_regret,
        action_sequence=tuple(step.chosen_action for step in result.trace),
    )


def _aggregate_cell(group: Sequence[OracleCalibrationEpisode]) -> CalibrationCell:
    first = group[0]
    if any(
        (item.hypothesis, item.budget, item.threshold)
        != (first.hypothesis, first.budget, first.threshold)
        for item in group
    ):
        raise ValueError("calibration cell contains mixed conditions")
    count = len(group)
    map_interval = wilson_interval(sum(item.map_correct for item in group), count)
    success_interval = wilson_interval(
        sum(item.success_at_threshold for item in group), count
    )
    experiments = [item.experiments_used for item in group]
    posteriors = [item.final_true_posterior for item in group]
    all_actions = [action for item in group for action in item.action_sequence]
    total_actions = len(all_actions)
    first_actions = [item.action_sequence[0] for item in group if item.action_sequence]
    total_regret = sum(item.cumulative_action_regret for item in group)

    def first_frequency(action: DiagnosticAction) -> float:
        return first_actions.count(action) / len(first_actions) if first_actions else 0.0

    def overall_frequency(action: DiagnosticAction) -> float:
        return all_actions.count(action) / total_actions if total_actions else 0.0

    def mean_count(action: DiagnosticAction) -> float:
        return mean(item.action_sequence.count(action) for item in group)

    return CalibrationCell(
        hypothesis=first.hypothesis,
        budget=first.budget,
        threshold=first.threshold,
        episodes=count,
        map_accuracy=map_interval.estimate,
        map_accuracy_ci_lower=map_interval.lower,
        map_accuracy_ci_upper=map_interval.upper,
        success_at_threshold=success_interval.estimate,
        success_ci_lower=success_interval.lower,
        success_ci_upper=success_interval.upper,
        mean_experiments=mean(experiments),
        median_experiments=median(experiments),
        stddev_experiments=pstdev(experiments),
        mean_true_posterior=mean(posteriors),
        median_true_posterior=median(posteriors),
        p10_true_posterior=percentile(posteriors, 0.10),
        p25_true_posterior=percentile(posteriors, 0.25),
        p75_true_posterior=percentile(posteriors, 0.75),
        p90_true_posterior=percentile(posteriors, 0.90),
        threshold_reached_fraction=mean(item.reached_threshold for item in group),
        budget_exhausted_fraction=mean(item.budget_exhausted for item in group),
        cumulative_action_regret=total_regret,
        mean_episode_action_regret=total_regret / count,
        mean_action_regret=total_regret / total_actions if total_actions else 0.0,
        first_repeat_trial_frequency=first_frequency(DiagnosticAction.REPEAT_TRIAL),
        first_use_trusted_sensor_frequency=first_frequency(
            DiagnosticAction.USE_TRUSTED_SENSOR
        ),
        first_change_context_frequency=first_frequency(
            DiagnosticAction.CHANGE_CONTEXT
        ),
        overall_repeat_trial_frequency=overall_frequency(
            DiagnosticAction.REPEAT_TRIAL
        ),
        overall_use_trusted_sensor_frequency=overall_frequency(
            DiagnosticAction.USE_TRUSTED_SENSOR
        ),
        overall_change_context_frequency=overall_frequency(
            DiagnosticAction.CHANGE_CONTEXT
        ),
        mean_repeat_trials=mean_count(DiagnosticAction.REPEAT_TRIAL),
        mean_trusted_sensor_uses=mean_count(DiagnosticAction.USE_TRUSTED_SENSOR),
        mean_context_changes=mean_count(DiagnosticAction.CHANGE_CONTEXT),
    )


def _rerun_hard_case(item: OracleCalibrationEpisode) -> dict[str, object]:
    result = StochasticDiagnosticEpisodeRunner(
        diagnosis_threshold=item.threshold,
        max_experiments=item.budget,
    ).run(
        StochasticBinaryMachine(),
        item.hypothesis,
        StochasticOracleInformationGainPolicy(),
        episode_seed=item.seed,
    )
    trace = []
    for step in result.trace:
        trace.append(
            {
                "step": step.step_number,
                "action": step.chosen_action.value,
                "observation": _result_dict(step.experiment_result),
                "posterior": _belief_dict(step.posterior),
            }
        )
    return {
        "seed": item.seed,
        "true_hypothesis": item.hypothesis.value,
        "initial_posterior": _belief_dict(result.initial_beliefs),
        "trace": trace,
        "final_map_diagnosis": result.predicted_diagnosis.value,
        "final_true_posterior": item.final_true_posterior,
        "termination_reason": (
            "THRESHOLD_REACHED" if result.reached_threshold else "BUDGET_EXHAUSTED"
        ),
    }


def _result_dict(result: object) -> dict[str, object]:
    if isinstance(result, RepeatTrialResult):
        return {"primary_o": result.o}
    if isinstance(result, StochasticTrustedSensorResult):
        return {"trusted_t": result.trusted_t}
    if isinstance(result, ChangeContextResult):
        return {"context": result.context.value, "primary_o": result.o}
    raise TypeError("unsupported stochastic result")


def _belief_dict(beliefs: StochasticHypothesisBeliefs) -> dict[str, float]:
    return {
        hypothesis.value: beliefs.probability(hypothesis)
        for hypothesis in CALIBRATION_HYPOTHESIS_ORDER
    }


def _write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        raise ValueError("cannot write an empty calibration table")
    fieldnames = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow(
                {
                    key: value.value if isinstance(value, FailureMode) else value
                    for key, value in values.items()
                }
            )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _per_hypothesis_section(
    study: OracleCalibrationStudy,
    *,
    budget: int,
    threshold: float,
) -> list[str]:
    lines = [
        "",
        f"## Per-hypothesis difficulty: budget {budget}, threshold {threshold:.2f}",
        "",
        "| Hypothesis | MAP accuracy (95% CI) | Success@threshold (95% CI) | Mean experiments | Mean true posterior | Threshold reached |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in _select_cells(study, budget, threshold):
        lines.append(
            f"| `{cell.hypothesis.value}` | "
            f"{_ci(cell.map_accuracy, cell.map_accuracy_ci_lower, cell.map_accuracy_ci_upper)} | "
            f"{_ci(cell.success_at_threshold, cell.success_ci_lower, cell.success_ci_upper)} | "
            f"{cell.mean_experiments:.3f} | "
            f"{cell.mean_true_posterior:.3f} | {cell.threshold_reached_fraction:.3f} |"
        )
    return lines


def _threshold_sensitivity_section(
    study: OracleCalibrationStudy,
    *,
    budget: int,
) -> list[str]:
    lines = [
        "",
        f"## Threshold sensitivity at budget {budget}",
        "",
        "| Threshold | MAP accuracy | Success@threshold | Mean experiments | False structural | Missed structural |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in study.overall_cells:
        if row.budget == budget:
            lines.append(
                f"| {row.threshold:.2f} | {row.map_accuracy:.3f} | "
                f"{row.success_at_threshold:.3f} | {row.mean_experiments:.3f} | "
                f"{row.false_structural_diagnosis_rate:.3f} | "
                f"{row.missed_structural_failure_rate:.3f} |"
            )
    return lines


def _posterior_section(
    study: OracleCalibrationStudy,
    *,
    budget: int,
    threshold: float,
) -> list[str]:
    lines = [
        "",
        f"## Final true-hypothesis posterior: budget {budget}, threshold {threshold:.2f}",
        "",
        "| Hypothesis | Mean | Median | p10 | p25 | p75 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in _select_cells(study, budget, threshold):
        lines.append(
            f"| `{cell.hypothesis.value}` | {cell.mean_true_posterior:.3f} | "
            f"{cell.median_true_posterior:.3f} | {cell.p10_true_posterior:.3f} | "
            f"{cell.p25_true_posterior:.3f} | {cell.p75_true_posterior:.3f} | "
            f"{cell.p90_true_posterior:.3f} |"
        )
    return lines


def _confusion_section(
    study: OracleCalibrationStudy,
    *,
    budget: int,
    threshold: float,
) -> list[str]:
    conditions = study.confusion_matrices["conditions"]
    assert isinstance(conditions, dict)
    matrix = conditions[_condition_key(budget, threshold)]
    assert isinstance(matrix, dict)
    counts = matrix["counts"]
    percentages = matrix["row_percentages"]
    assert isinstance(counts, dict) and isinstance(percentages, dict)
    abbreviations = ["N", "W", "S", "L"]
    lines = [
        "",
        f"## Confusion matrix: budget {budget}, threshold {threshold:.2f}",
        "",
        "Rows are truth; columns are final MAP diagnosis. Cells show count (row %).",
        "",
        "| Truth | N | W | S | L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for truth, label in zip(CALIBRATION_HYPOTHESIS_ORDER, abbreviations):
        values = []
        for diagnosis in CALIBRATION_HYPOTHESIS_ORDER:
            count = counts[truth.value][diagnosis.value]
            percentage_value = percentages[truth.value][diagnosis.value]
            values.append(f"{count} ({percentage_value:.1f}%)")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("N=no change, W=world shift, S=sensor corruption, L=missing latent.")
    return lines


def _action_section(study: OracleCalibrationStudy) -> list[str]:
    lines = [
        "",
        "## Experiment selection: budget 5, threshold 0.90",
        "",
        "| Hypothesis | First trusted | Mean repeat | Mean trusted | Mean context changes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in _select_cells(study, 5, 0.90):
        lines.append(
            f"| `{cell.hypothesis.value}` | {cell.first_use_trusted_sensor_frequency:.3f} | "
            f"{cell.mean_repeat_trials:.3f} | {cell.mean_trusted_sensor_uses:.3f} | "
            f"{cell.mean_context_changes:.3f} |"
        )
    lines.extend(["", "Top five action sequences by hypothesis:", ""])
    by_hypothesis = study.top_action_sequences["by_hypothesis"]
    assert isinstance(by_hypothesis, dict)
    for hypothesis in CALIBRATION_HYPOTHESIS_ORDER:
        lines.append(f"- `{hypothesis.value}`:")
        for item in by_hypothesis[hypothesis.value]:
            sequence = " → ".join(item["sequence"])
            lines.append(
                f"  `{sequence}` — {item['count']} episodes ({item['percentage']:.1f}%)"
            )
    return lines


def _hard_case_section(study: OracleCalibrationStudy) -> list[str]:
    lines = ["", "## Representative hard cases", ""]
    by_hypothesis = study.hard_cases["by_hypothesis"]
    assert isinstance(by_hypothesis, dict)
    for hypothesis in CALIBRATION_HYPOTHESIS_ORDER:
        lines.append(f"### {hypothesis.value}")
        lines.append("")
        categories = by_hypothesis[hypothesis.value]
        for category, case in categories.items():
            if case is None:
                lines.append(f"- `{category}`: no matching seed in the sweep.")
                continue
            steps = "; ".join(
                f"{step['action']} {_compact_observation(step['observation'])} → "
                f"{_compact_posterior(step['posterior'])}"
                for step in case["trace"]
            )
            lines.append(
                f"- `{category}` seed {case['seed']}: {steps}; final "
                f"`{case['final_map_diagnosis']}`, {case['termination_reason']}."
            )
        lines.append("")
    return lines


def _recommendation(study: OracleCalibrationStudy) -> str:
    target_budget, target_threshold, extended_budget = _report_conditions(study)
    target = _select_overall(study, target_budget, target_threshold)
    extended = _select_overall(study, extended_budget, target_threshold)
    extended_hypotheses = _select_cells(
        study, extended_budget, target_threshold
    )
    minimum_extended_map = min(cell.map_accuracy for cell in extended_hypotheses)
    if (
        target.map_accuracy >= 0.90
        and target.success_at_threshold >= 0.80
        and target.false_structural_diagnosis_rate <= 0.10
        and target.missed_structural_failure_rate <= 0.10
    ):
        return f"**A — Current parameters are healthy; freeze ER-1.** Budget {target_budget} at threshold {target_threshold:.2f} provides strong resolution and controlled structural-error rates."
    if (
        extended.map_accuracy >= 0.90
        and extended.success_at_threshold >= 0.75
        and minimum_extended_map >= 0.80
    ):
        return f"**B — The probability model is healthy, but evaluation should use a different budget and/or threshold.** Identifiability is strong at budget {extended_budget}, while budget {target_budget} / threshold {target_threshold:.2f} leaves avoidable unresolved cases."
    return f"**C — One or more hypotheses remain statistically difficult at practical budgets; revisit the generative probabilities before freezing ER-1.** Even budget {extended_budget} / threshold {target_threshold:.2f} does not deliver uniformly strong oracle resolution."


def _compact_observation(observation: Mapping[str, object]) -> str:
    return ",".join(f"{key}={value}" for key, value in observation.items())


def _compact_posterior(posterior: Mapping[str, float]) -> str:
    labels = ("N", "W", "S", "L")
    return "[" + ",".join(
        f"{label}={posterior[hypothesis.value]:.3f}"
        for label, hypothesis in zip(labels, CALIBRATION_HYPOTHESIS_ORDER)
    ) + "]"


def _select_cells(
    study: OracleCalibrationStudy,
    budget: int,
    threshold: float,
) -> tuple[CalibrationCell, ...]:
    selected = tuple(
        cell
        for cell in study.cells
        if cell.budget == budget and _same_threshold(cell.threshold, threshold)
    )
    if len(selected) != len(CALIBRATION_HYPOTHESIS_ORDER):
        raise ValueError(f"calibration grid does not contain budget={budget}, threshold={threshold}")
    return selected


def _select_overall(
    study: OracleCalibrationStudy,
    budget: int,
    threshold: float,
) -> OverallCalibrationCell:
    matches = [
        cell
        for cell in study.overall_cells
        if cell.budget == budget and _same_threshold(cell.threshold, threshold)
    ]
    if len(matches) != 1:
        raise ValueError(f"calibration grid does not contain budget={budget}, threshold={threshold}")
    return matches[0]


def _has_condition(
    study: OracleCalibrationStudy,
    budget: int,
    threshold: float,
) -> bool:
    return any(
        cell.budget == budget and _same_threshold(cell.threshold, threshold)
        for cell in study.overall_cells
    )


def _report_conditions(study: OracleCalibrationStudy) -> tuple[int, float, int]:
    threshold = 0.90 if any(
        _same_threshold(value, 0.90) for value in study.thresholds
    ) else study.thresholds[-1]
    target_budget = 5 if 5 in study.budgets else study.budgets[-1]
    extended_budget = 8 if 8 in study.budgets else study.budgets[-1]
    return target_budget, threshold, extended_budget


def _condition_key(budget: int, threshold: float) -> str:
    return f"budget_{budget}_threshold_{threshold:.2f}"


def _same_threshold(left: float, right: float) -> bool:
    return abs(left - right) < 1e-12


def _ci(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"


def _validate_budgets(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(type(value) is not int or value <= 0 for value in result):
        raise ValueError("budgets must be positive integers")
    if len(set(result)) != len(result):
        raise ValueError("budgets must not contain duplicates")
    return tuple(sorted(result))


def _validate_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not 0.0 < value <= 1.0 for value in result):
        raise ValueError("thresholds must be in (0, 1]")
    if len(set(result)) != len(result):
        raise ValueError("thresholds must not contain duplicates")
    return tuple(sorted(result))
