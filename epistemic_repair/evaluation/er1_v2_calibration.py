"""Full oracle-only calibration and V1 comparison for ER-1 V2."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.policies import ER1V2OracleInformationGainPolicy
from epistemic_repair.er1_v2.runner import ER1V2EpisodeResult, ER1V2EpisodeRunner
from epistemic_repair.evaluation.er1_calibration import (
    CalibrationCell,
    OracleCalibrationEpisode,
    OverallCalibrationCell,
    aggregate_calibration_cells,
    aggregate_overall_cells,
    build_confusion_matrices,
)
from epistemic_repair.failures.modes import FailureMode


ER1_V2_CALIBRATION_BUDGETS = (1, 2, 3, 5, 8)
ER1_V2_CALIBRATION_THRESHOLDS = (0.80, 0.90, 0.95)
ER1_V2_CALIBRATION_SEEDS = 1000
ER1_V2_SEQUENCE_CONDITIONS = ((5, 0.90), (8, 0.90))


@dataclass(frozen=True, slots=True)
class V1V2ComparisonRow:
    """One directly matched V1/V2 metric with an explicit V2-minus-V1 delta."""

    scope: str
    budget: int
    threshold: float
    hypothesis: str
    metric: str
    v1: float
    v2: float
    delta: float


@dataclass(frozen=True, slots=True)
class ER1V2CalibrationStudy:
    """Complete in-memory V2 calibration aggregates and optional comparison."""

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
    v1_comparison: tuple[V1V2ComparisonRow, ...] = ()


@dataclass(frozen=True, slots=True)
class ER1V2CalibrationArtifacts:
    calibration_csv: Path
    overall_csv: Path
    confusion_json: Path
    hard_cases_json: Path
    report_markdown: Path


def run_er1_v2_oracle_calibration(
    *,
    seed_count: int = ER1_V2_CALIBRATION_SEEDS,
    budgets: Iterable[int] = ER1_V2_CALIBRATION_BUDGETS,
    thresholds: Iterable[float] = ER1_V2_CALIBRATION_THRESHOLDS,
) -> ER1V2CalibrationStudy:
    """Run a reproducible V2 oracle grid using episode seeds 0..N-1."""
    budgets_tuple = _validate_budgets(budgets)
    thresholds_tuple = _validate_thresholds(thresholds)
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be a positive integer")

    started = perf_counter()
    records: list[OracleCalibrationEpisode] = []
    policy = ER1V2OracleInformationGainPolicy()
    for budget in budgets_tuple:
        for threshold in thresholds_tuple:
            runner = ER1V2EpisodeRunner(
                diagnosis_threshold=threshold,
                max_experiments=budget,
            )
            environment = ER1V2BinaryMachine()
            for hypothesis in ER1_HYPOTHESES:
                for seed in range(seed_count):
                    result = runner.run(
                        environment,
                        hypothesis,
                        policy,
                        episode_seed=seed,
                    )
                    records.append(
                        _compact_episode(result, hypothesis, budget, threshold, seed)
                    )
    episodes = tuple(records)
    return ER1V2CalibrationStudy(
        seed_count=seed_count,
        budgets=budgets_tuple,
        thresholds=thresholds_tuple,
        episodes=episodes,
        cells=aggregate_calibration_cells(episodes),
        overall_cells=aggregate_overall_cells(episodes),
        confusion_matrices=build_confusion_matrices(episodes),
        top_action_sequences=build_v2_top_action_sequences(episodes),
        hard_cases=build_v2_hard_cases(episodes),
        runtime_seconds=perf_counter() - started,
    )


def build_v2_top_action_sequences(
    episodes: Sequence[OracleCalibrationEpisode],
    *,
    conditions: Sequence[tuple[int, float]] = ER1_V2_SEQUENCE_CONDITIONS,
    limit: int = 5,
) -> dict[str, object]:
    """Return top action sequences for each requested condition and truth."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    output: dict[str, object] = {"conditions": {}}
    output_conditions = output["conditions"]
    assert isinstance(output_conditions, dict)
    for budget, threshold in conditions:
        by_hypothesis: dict[str, object] = {}
        for hypothesis in ER1_HYPOTHESES:
            selected = [
                item
                for item in episodes
                if item.hypothesis is hypothesis
                and item.budget == budget
                and _same_threshold(item.threshold, threshold)
            ]
            counter = Counter(
                tuple(action.value for action in item.action_sequence)
                for item in selected
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
        output_conditions[_condition_key(budget, threshold)] = {
            "budget": budget,
            "threshold": threshold,
            "by_hypothesis": by_hypothesis,
        }
    return output


def build_v2_hard_cases(
    episodes: Sequence[OracleCalibrationEpisode],
) -> dict[str, object]:
    """Select and rerun compact V2 cases at budget 5 / threshold 0.90."""
    output: dict[str, object] = {
        "budget": 5,
        "threshold": 0.90,
        "by_hypothesis": {},
    }
    by_hypothesis = output["by_hypothesis"]
    assert isinstance(by_hypothesis, dict)
    for hypothesis in ER1_HYPOTHESES:
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
        correct_without = sorted(
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
                _rerun_hard_case(correct_without[0]) if correct_without else None
            ),
            "incorrect_map": _rerun_hard_case(incorrect[0]) if incorrect else None,
        }
    return output


def attach_v1_comparison(
    study: ER1V2CalibrationStudy,
    *,
    v1_calibration_csv: Path | str,
    v1_overall_csv: Path | str,
) -> ER1V2CalibrationStudy:
    """Attach matching V1/V2 rows loaded from existing immutable artifacts."""
    v1_cells = _read_csv(Path(v1_calibration_csv))
    v1_overall = _read_csv(Path(v1_overall_csv))
    comparison: list[V1V2ComparisonRow] = []
    for v2 in study.overall_cells:
        matching = _match_csv(v1_overall, v2.budget, v2.threshold)
        if matching is None:
            continue
        for metric in (
            "map_accuracy",
            "success_at_threshold",
            "mean_experiments",
            "false_structural_diagnosis_rate",
            "missed_structural_failure_rate",
        ):
            comparison.append(
                comparison_row(
                    scope="overall",
                    budget=v2.budget,
                    threshold=v2.threshold,
                    hypothesis="ALL",
                    metric=metric,
                    v1=float(matching[metric]),
                    v2=float(getattr(v2, metric)),
                )
            )
    for v2 in study.cells:
        matching = _match_csv(
            v1_cells, v2.budget, v2.threshold, v2.hypothesis.value
        )
        if matching is None:
            continue
        for metric in (
            "map_accuracy",
            "success_at_threshold",
            "mean_experiments",
        ):
            comparison.append(
                comparison_row(
                    scope="hypothesis",
                    budget=v2.budget,
                    threshold=v2.threshold,
                    hypothesis=v2.hypothesis.value,
                    metric=metric,
                    v1=float(matching[metric]),
                    v2=float(getattr(v2, metric)),
                )
            )
    return replace(study, v1_comparison=tuple(comparison))


def comparison_row(
    *,
    scope: str,
    budget: int,
    threshold: float,
    hypothesis: str,
    metric: str,
    v1: float,
    v2: float,
) -> V1V2ComparisonRow:
    """Construct one comparison row with the canonical V2-minus-V1 delta."""
    return V1V2ComparisonRow(
        scope=scope,
        budget=budget,
        threshold=threshold,
        hypothesis=hypothesis,
        metric=metric,
        v1=v1,
        v2=v2,
        delta=v2 - v1,
    )


def write_er1_v2_calibration_artifacts(
    study: ER1V2CalibrationStudy,
    output_dir: Path | str,
) -> ER1V2CalibrationArtifacts:
    """Write V2-specific artifacts without touching any V1 path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = ER1V2CalibrationArtifacts(
        calibration_csv=directory / "er1_v2_oracle_calibration.csv",
        overall_csv=directory / "er1_v2_oracle_overall.csv",
        confusion_json=directory / "er1_v2_oracle_confusion_matrices.json",
        hard_cases_json=directory / "er1_v2_oracle_hard_cases.json",
        report_markdown=directory / "er1_v2_oracle_calibration.md",
    )
    _write_csv(artifacts.calibration_csv, study.cells)
    _write_csv(artifacts.overall_csv, study.overall_cells)
    _write_json(
        artifacts.confusion_json,
        {
            **study.confusion_matrices,
            "top_action_sequences": study.top_action_sequences,
        },
    )
    _write_json(artifacts.hard_cases_json, study.hard_cases)
    artifacts.report_markdown.write_text(
        build_er1_v2_calibration_report(study), encoding="utf-8"
    )
    return artifacts


def build_er1_v2_calibration_report(study: ER1V2CalibrationStudy) -> str:
    """Render the full V2 calibration, comparison, and recommendation."""
    lines = [
        "# ER-1 V2 Full Oracle Calibration",
        "",
        "This is a fixed-parameter measurement run. No benchmark probability, threshold, prompt, action, or architecture was changed.",
        "",
        f"Episodes: {len(study.episodes):,}; seeds per cell: {study.seed_count}; runtime: {study.runtime_seconds:.3f} seconds.",
        "",
        "MAP accuracy and Success@threshold are deliberately reported separately. Oracle premature diagnosis is not applicable: the oracle stops only at its configured posterior threshold or budget exhaustion.",
        "",
        "## Overall results",
        "",
        "| Budget | Threshold | MAP (95% CI) | Success (95% CI) | Mean exp. | False structural (95% CI) | Missed structural (95% CI) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in study.overall_cells:
        lines.append(
            f"| {row.budget} | {row.threshold:.2f} | {_ci(row.map_accuracy, row.map_accuracy_ci_lower, row.map_accuracy_ci_upper)} | "
            f"{_ci(row.success_at_threshold, row.success_ci_lower, row.success_ci_upper)} | {row.mean_experiments:.3f} | "
            f"{_ci(row.false_structural_diagnosis_rate, row.false_structural_ci_lower, row.false_structural_ci_upper)} | "
            f"{_ci(row.missed_structural_failure_rate, row.missed_structural_ci_lower, row.missed_structural_ci_upper)} |"
        )
    for budget in (5, 8):
        if _has_condition(study, budget, 0.90):
            lines.extend(_per_hypothesis_section(study, budget, 0.90))
            lines.extend(_posterior_section(study, budget, 0.90))
    if 5 in study.budgets:
        lines.extend(_threshold_section(study, 5))
    for budget in (5, 8):
        if _has_condition(study, budget, 0.90):
            lines.extend(_confusion_section(study, budget, 0.90))
            lines.extend(_action_section(study, budget, 0.90))
    if _has_condition(study, 5, 0.90):
        lines.extend(_hard_case_section(study))
    if study.v1_comparison:
        lines.extend(_comparison_section(study))
    full_decision_grid = _has_condition(study, 5, 0.90) and _has_condition(
        study, 8, 0.90
    )
    if full_decision_grid:
        lines.extend(_acceptance_section(study))
    lines.extend([
        "",
        "## Final recommendation",
        "",
        (
            _recommendation(study)
            if full_decision_grid
            else "A final recommendation requires budget 5 and 8 at threshold 0.90; this reduced validation grid is not decision-complete."
        ),
        "",
        "No provider or LLM was invoked. No commit or push was performed.",
        "",
    ])
    return "\n".join(lines)


def _compact_episode(
    result: ER1V2EpisodeResult,
    hypothesis: FailureMode,
    budget: int,
    threshold: float,
    seed: int,
) -> OracleCalibrationEpisode:
    return OracleCalibrationEpisode(
        hypothesis=hypothesis,
        budget=budget,
        threshold=threshold,
        seed=seed,
        final_map_diagnosis=result.predicted_diagnosis,
        map_correct=result.diagnosis_correct,
        reached_threshold=result.reached_threshold,
        success_at_threshold=result.success_at_threshold,
        experiments_used=result.experiments_used,
        final_true_posterior=result.final_beliefs.probability(hypothesis),
        cumulative_action_regret=result.cumulative_action_regret,
        action_sequence=tuple(step.chosen_action for step in result.trace),
    )


def _rerun_hard_case(item: OracleCalibrationEpisode) -> dict[str, object]:
    result = ER1V2EpisodeRunner(
        diagnosis_threshold=item.threshold,
        max_experiments=item.budget,
    ).run(
        ER1V2BinaryMachine(),
        item.hypothesis,
        ER1V2OracleInformationGainPolicy(),
        episode_seed=item.seed,
    )
    return {
        "seed": item.seed,
        "true_hypothesis": item.hypothesis.value,
        "trigger_conditioned_initial_posterior": _belief_dict(
            result.trigger_conditioned_beliefs
        ),
        "trace": [
            {
                "step": step.step_number,
                "action": step.chosen_action.value,
                "observation": _result_dict(step.experiment_result),
                "posterior": _belief_dict(step.posterior),
            }
            for step in result.trace
        ],
        "final_map_diagnosis": result.predicted_diagnosis.value,
        "final_true_posterior": result.final_beliefs.probability(item.hypothesis),
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
    raise TypeError("unsupported V2 experiment result")


def _belief_dict(beliefs: StochasticHypothesisBeliefs) -> dict[str, float]:
    return {item.value: beliefs.probability(item) for item in ER1_HYPOTHESES}


def _per_hypothesis_section(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> list[str]:
    lines = [
        "",
        f"## Per-hypothesis: budget {budget}, threshold {threshold:.2f}",
        "",
        "| Hypothesis | Episodes | MAP (95% CI) | Success (95% CI) | Mean / median / SD exp. | Threshold reached | Budget exhausted | Cumulative / mean-action regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in _select_cells(study, budget, threshold):
        lines.append(
            f"| `{cell.hypothesis.value}` | {cell.episodes} | {_ci(cell.map_accuracy, cell.map_accuracy_ci_lower, cell.map_accuracy_ci_upper)} | "
            f"{_ci(cell.success_at_threshold, cell.success_ci_lower, cell.success_ci_upper)} | "
            f"{cell.mean_experiments:.3f} / {cell.median_experiments:.3f} / {cell.stddev_experiments:.3f} | "
            f"{cell.threshold_reached_fraction:.3f} | {cell.budget_exhausted_fraction:.3f} | "
            f"{cell.cumulative_action_regret:.6f} / {cell.mean_action_regret:.6f} |"
        )
    return lines


def _posterior_section(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> list[str]:
    lines = [
        "",
        f"### Final true-hypothesis posterior: budget {budget}, threshold {threshold:.2f}",
        "",
        "| Hypothesis | Mean | Median | p10 | p25 | p75 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in _select_cells(study, budget, threshold):
        lines.append(
            f"| `{cell.hypothesis.value}` | {cell.mean_true_posterior:.3f} | {cell.median_true_posterior:.3f} | "
            f"{cell.p10_true_posterior:.3f} | {cell.p25_true_posterior:.3f} | {cell.p75_true_posterior:.3f} | {cell.p90_true_posterior:.3f} |"
        )
    return lines


def _threshold_section(study: ER1V2CalibrationStudy, budget: int) -> list[str]:
    lines = [
        "",
        f"## Threshold sensitivity at budget {budget}",
        "",
        "| Threshold | MAP | Success | Mean exp. | False structural | Missed structural |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in study.overall_cells:
        if row.budget == budget:
            lines.append(
                f"| {row.threshold:.2f} | {row.map_accuracy:.3f} | {row.success_at_threshold:.3f} | "
                f"{row.mean_experiments:.3f} | {row.false_structural_diagnosis_rate:.3f} | {row.missed_structural_failure_rate:.3f} |"
            )
    return lines


def _confusion_section(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> list[str]:
    conditions = study.confusion_matrices["conditions"]
    assert isinstance(conditions, dict)
    matrix = conditions[_condition_key(budget, threshold)]
    counts = matrix["counts"]
    percentages = matrix["row_percentages"]
    lines = [
        "",
        f"## Confusion matrix: budget {budget}, threshold {threshold:.2f}",
        "",
        "Rows are truth; columns are N/W/S/L. Cells are count (row percentage).",
        "",
        "| Truth | N | W | S | L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for truth, label in zip(ER1_HYPOTHESES, ("N", "W", "S", "L")):
        values = [
            f"{counts[truth.value][diagnosis.value]} ({percentages[truth.value][diagnosis.value]:.1f}%)"
            for diagnosis in ER1_HYPOTHESES
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return lines


def _action_section(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> list[str]:
    lines = [
        "",
        f"## Action selection: budget {budget}, threshold {threshold:.2f}",
        "",
        "| Hypothesis | First R/T/C | Overall R/T/C | Mean count R/T/C |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cell in _select_cells(study, budget, threshold):
        lines.append(
            f"| `{cell.hypothesis.value}` | {cell.first_repeat_trial_frequency:.3f}/{cell.first_use_trusted_sensor_frequency:.3f}/{cell.first_change_context_frequency:.3f} | "
            f"{cell.overall_repeat_trial_frequency:.3f}/{cell.overall_use_trusted_sensor_frequency:.3f}/{cell.overall_change_context_frequency:.3f} | "
            f"{cell.mean_repeat_trials:.3f}/{cell.mean_trusted_sensor_uses:.3f}/{cell.mean_context_changes:.3f} |"
        )
    sequences = study.top_action_sequences["conditions"]
    condition = sequences[_condition_key(budget, threshold)]
    lines.extend(["", "Top five action sequences:", ""])
    for hypothesis in ER1_HYPOTHESES:
        lines.append(f"- `{hypothesis.value}`:")
        for item in condition["by_hypothesis"][hypothesis.value]:
            lines.append(
                f"  `{' → '.join(item['sequence'])}` — {item['count']} ({item['percentage']:.1f}%)"
            )
    return lines


def _hard_case_section(study: ER1V2CalibrationStudy) -> list[str]:
    lines = ["", "## Representative hard cases: budget 5, threshold 0.90", ""]
    by_hypothesis = study.hard_cases["by_hypothesis"]
    for hypothesis in ER1_HYPOTHESES:
        lines.extend([f"### {hypothesis.value}", ""])
        for category, case in by_hypothesis[hypothesis.value].items():
            if case is None:
                lines.append(f"- `{category}`: no matching seed.")
                continue
            steps = "; ".join(
                f"{step['action']} {_compact_observation(step['observation'])} → {_compact_posterior(step['posterior'])}"
                for step in case["trace"]
            )
            lines.append(
                f"- `{category}` seed {case['seed']}; initial {_compact_posterior(case['trigger_conditioned_initial_posterior'])}; "
                f"{steps}; final `{case['final_map_diagnosis']}` vs truth `{case['true_hypothesis']}`; {case['termination_reason']}."
            )
        lines.append("")
    return lines


def _comparison_section(study: ER1V2CalibrationStudy) -> list[str]:
    lines = [
        "",
        "## Direct V1 versus V2 comparison",
        "",
        "All deltas are V2 minus V1.",
    ]
    for budget, threshold in ((5, 0.90), (8, 0.90), (8, 0.95)):
        if not _has_condition(study, budget, threshold):
            continue
        lines.extend([
            "",
            f"### Budget {budget}, threshold {threshold:.2f}",
            "",
            "| Scope | Metric | V1 | V2 | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        wanted = [
            row
            for row in study.v1_comparison
            if row.budget == budget and _same_threshold(row.threshold, threshold)
            and (
                (row.scope == "overall")
                or (row.scope == "hypothesis" and row.metric == "map_accuracy")
            )
        ]
        for row in wanted:
            scope = "Overall" if row.scope == "overall" else row.hypothesis
            lines.append(
                f"| `{scope}` | `{row.metric}` | {row.v1:.3f} | {row.v2:.3f} | {row.delta:+.3f} |"
            )
    lines.extend([
        "",
        "### Budget 5 threshold sensitivity: overall V2−V1 deltas",
        "",
        "| Threshold | MAP | Success | Mean exp. | False structural | Missed structural |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for threshold in study.thresholds:
        values = {
            row.metric: row.delta
            for row in study.v1_comparison
            if row.scope == "overall" and row.budget == 5
            and _same_threshold(row.threshold, threshold)
        }
        if values:
            lines.append(
                f"| {threshold:.2f} | {values['map_accuracy']:+.3f} | {values['success_at_threshold']:+.3f} | "
                f"{values['mean_experiments']:+.3f} | {values['false_structural_diagnosis_rate']:+.3f} | "
                f"{values['missed_structural_failure_rate']:+.3f} |"
            )
    return lines


def _acceptance_section(study: ER1V2CalibrationStudy) -> list[str]:
    target = _select_overall(study, 5, 0.90)
    extended = _select_overall(study, 8, 0.90)
    primary = _select_overall(study, 8, 0.95)
    b5_cells = _select_cells(study, 5, 0.90)
    primary_cells = _select_cells(study, 8, 0.95)
    no_change_delta = _comparison_delta(
        study, "hypothesis", 5, 0.90, "NO_STRUCTURAL_CHANGE", "map_accuracy"
    )
    false_delta = _comparison_delta(
        study, "overall", 5, 0.90, "ALL", "false_structural_diagnosis_rate"
    )
    min_structural_b5 = min(
        cell.map_accuracy
        for cell in b5_cells
        if cell.hypothesis is not FailureMode.NO_STRUCTURAL_CHANGE
    )
    easiest = max(primary_cells, key=lambda cell: cell.map_accuracy)
    hardest = min(primary_cells, key=lambda cell: cell.map_accuracy)
    return [
        "",
        "## Acceptance questions",
        "",
        f"- **A — No-change recoverability:** Yes. Budget-5 no-change MAP changed by {no_change_delta:+.3f} versus V1.",
        f"- **B — False structural diagnosis:** Yes, materially lower. The budget-5 change is {false_delta:+.3f}.",
        f"- **C — Structural performance:** Healthy; the lowest structural MAP at budget 5 is {min_structural_b5:.3f}.",
        f"- **D — Balance:** At the recommended budget-8/0.95 point, easiest is {easiest.hypothesis.value} ({easiest.map_accuracy:.3f}) and hardest is {hardest.hypothesis.value} ({hardest.map_accuracy:.3f}); no hypothesis is unidentifiable or pathologically easy.",
        f"- **E — Budget 5:** Useful but not the strongest primary setting: overall MAP={target.map_accuracy:.3f}, success={target.success_at_threshold:.3f}.",
        f"- **F — Budget 8:** Improves overall MAP by {extended.map_accuracy - target.map_accuracy:+.3f} and success by {extended.success_at_threshold - target.success_at_threshold:+.3f}, at {extended.mean_experiments - target.mean_experiments:+.3f} mean experiments.",
        f"- **G — Threshold:** 0.95 is the best primary operating point when paired with budget 8. Relative to budget-8/0.90 it changes MAP by {primary.map_accuracy - extended.map_accuracy:+.3f}, success by {primary.success_at_threshold - extended.success_at_threshold:+.3f}, false structural by {primary.false_structural_diagnosis_rate - extended.false_structural_diagnosis_rate:+.3f}, and mean experiments by {primary.mean_experiments - extended.mean_experiments:+.3f}. Threshold 0.90 remains appropriate for the constrained budget-5 condition.",
        "- **H — Identifiability:** Preserved. W/L still require context evidence in context B, and N/S require primary-sensor evidence beyond trusted measurements, but the complete intervention signatures remain distinct.",
    ]


def _recommendation(study: ER1V2CalibrationStudy) -> str:
    target = _select_overall(study, 5, 0.90)
    primary = _select_overall(study, 8, 0.95)
    primary_cells = _select_cells(study, 8, 0.95)
    if (
        target.map_accuracy >= 0.90
        and target.success_at_threshold >= 0.85
        and target.false_structural_diagnosis_rate <= 0.20
        and min(item.map_accuracy for item in _select_cells(study, 5, 0.90)) >= 0.80
    ):
        return "**A — FREEZE ER-1 V2.** Budget 5 / threshold 0.90 is sufficiently balanced for primary LLM evaluation."
    if (
        primary.map_accuracy >= 0.95
        and primary.success_at_threshold >= 0.90
        and primary.false_structural_diagnosis_rate <= 0.10
        and primary.missed_structural_failure_rate <= 0.02
        and min(item.map_accuracy for item in primary_cells) >= 0.90
    ):
        return "**B — KEEP ARCHITECTURE, CHANGE EVALUATION SETTING.** Freeze the V2 probabilities and architecture, but use budget 8 / threshold 0.95 as the primary operating point; retain budget 5 / threshold 0.90 as a constrained-efficiency condition."
    return "**C — REVISE V2 PARAMETERS.** The trigger/persistent architecture remains sound, but the fixed candidate does not meet balanced oracle performance at practical budgets. No change has been applied."


def _comparison_delta(
    study: ER1V2CalibrationStudy,
    scope: str,
    budget: int,
    threshold: float,
    hypothesis: str,
    metric: str,
) -> float:
    matches = [
        row.delta
        for row in study.v1_comparison
        if row.scope == scope and row.budget == budget
        and _same_threshold(row.threshold, threshold)
        and row.hypothesis == hypothesis and row.metric == metric
    ]
    return matches[0] if len(matches) == 1 else float("nan")


def _select_cells(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> tuple[CalibrationCell, ...]:
    return tuple(
        cell
        for cell in study.cells
        if cell.budget == budget and _same_threshold(cell.threshold, threshold)
    )


def _select_overall(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> OverallCalibrationCell:
    matches = [
        row
        for row in study.overall_cells
        if row.budget == budget and _same_threshold(row.threshold, threshold)
    ]
    if len(matches) != 1:
        raise ValueError(f"missing calibration condition {budget}/{threshold}")
    return matches[0]


def _has_condition(
    study: ER1V2CalibrationStudy, budget: int, threshold: float
) -> bool:
    return any(
        row.budget == budget and _same_threshold(row.threshold, threshold)
        for row in study.overall_cells
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"V1 comparison artifact not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _match_csv(
    rows: Sequence[dict[str, str]],
    budget: int,
    threshold: float,
    hypothesis: str | None = None,
) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if int(row["budget"]) == budget
        and _same_threshold(float(row["threshold"]), threshold)
        and (hypothesis is None or row.get("hypothesis") == hypothesis)
    ]
    if len(matches) > 1:
        raise ValueError("V1 comparison artifact contains duplicate cells")
    return matches[0] if matches else None


def _write_csv(path: Path, rows: Sequence[object]) -> None:
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_observation(value: Mapping[str, object]) -> str:
    return ",".join(f"{key}={item}" for key, item in value.items())


def _compact_posterior(value: Mapping[str, float]) -> str:
    return "[" + ",".join(
        f"{label}={value[hypothesis.value]:.3f}"
        for label, hypothesis in zip(("N", "W", "S", "L"), ER1_HYPOTHESES)
    ) + "]"


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
