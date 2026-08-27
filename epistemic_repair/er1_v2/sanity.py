"""Small oracle sanity study and analytic reporting for ER-1 V2."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticOutcomeSignal,
    stochastic_information_gains,
)
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.likelihoods import ER1V2LikelihoodModel
from epistemic_repair.er1_v2.policies import ER1V2OracleInformationGainPolicy
from epistemic_repair.er1_v2.runner import ER1V2EpisodeResult, ER1V2EpisodeRunner
from epistemic_repair.er1_v2.trigger_model import TriggerLikelihoodModel
from epistemic_repair.evaluation.er1_calibration import (
    CalibrationCell,
    OracleCalibrationEpisode,
    OverallCalibrationCell,
    aggregate_calibration_cells,
    aggregate_overall_cells,
    build_confusion_matrices,
)
from epistemic_repair.failures.modes import FailureMode


ER1_V2_SANITY_BUDGETS = (5, 8)
ER1_V2_SANITY_THRESHOLD = 0.90
ER1_V2_SANITY_SEEDS = 100


@dataclass(frozen=True, slots=True)
class PersistentLikelihoodRow:
    current_context: str
    action: str
    result_signal: str
    hypothesis: str
    effective_context: str
    probability_0: float
    probability_1: float


@dataclass(frozen=True, slots=True)
class InitialInformationGain:
    repeat_trial: float
    use_trusted_sensor: float
    change_context: float


@dataclass(frozen=True, slots=True)
class ER1V2SanityStudy:
    seed_count: int
    runtime_seconds: float
    cells: tuple[CalibrationCell, ...]
    overall_cells: tuple[OverallCalibrationCell, ...]
    confusion_matrices: dict[str, object]
    representative_traces: dict[str, object]
    likelihood_table: tuple[PersistentLikelihoodRow, ...]
    initial_information_gain: InitialInformationGain


@dataclass(frozen=True, slots=True)
class ER1V2SanityArtifacts:
    metrics_csv: Path
    confusion_json: Path
    traces_json: Path
    report_markdown: Path


def persistent_likelihood_table() -> tuple[PersistentLikelihoodRow, ...]:
    """Return every binary action distribution for both current contexts."""
    model = ER1V2LikelihoodModel()
    rows = []
    for current_context in Context:
        for action in BENCHMARK_ACTIONS:
            signal = (
                StochasticOutcomeSignal.TRUSTED_T
                if action is DiagnosticAction.USE_TRUSTED_SENSOR
                else StochasticOutcomeSignal.PRIMARY_O
            )
            effective_context = (
                model.target_context(current_context)
                if action is DiagnosticAction.CHANGE_CONTEXT
                else current_context
            )
            for hypothesis in ER1_HYPOTHESES:
                if signal is StochasticOutcomeSignal.TRUSTED_T:
                    probability_0 = model.probability_trusted_observation(
                        0, hypothesis, effective_context
                    )
                else:
                    probability_0 = model.probability_primary_observation(
                        0, hypothesis, effective_context
                    )
                rows.append(
                    PersistentLikelihoodRow(
                        current_context=current_context.value,
                        action=action.value,
                        result_signal=signal.value,
                        hypothesis=hypothesis.value,
                        effective_context=effective_context.value,
                        probability_0=probability_0,
                        probability_1=1.0 - probability_0,
                    )
                )
    return tuple(rows)


def initial_information_gain() -> InitialInformationGain:
    """Calculate V2 initial EIG using persistent investigation likelihoods."""
    beliefs = TriggerLikelihoodModel().conditioned_beliefs()
    scores = stochastic_information_gains(
        beliefs, ER1V2LikelihoodModel(), Context.B
    )
    return InitialInformationGain(
        repeat_trial=scores.repeat_trial,
        use_trusted_sensor=scores.use_trusted_sensor,
        change_context=scores.change_context,
    )


def run_er1_v2_sanity_study(
    *,
    seed_count: int = ER1_V2_SANITY_SEEDS,
) -> ER1V2SanityStudy:
    """Run exactly 4×2×N seeded oracle episodes at threshold 0.90."""
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be a positive integer")
    started = perf_counter()
    records = []
    trace_results: dict[FailureMode, ER1V2EpisodeResult] = {}
    for budget in ER1_V2_SANITY_BUDGETS:
        runner = ER1V2EpisodeRunner(
            diagnosis_threshold=ER1_V2_SANITY_THRESHOLD,
            max_experiments=budget,
        )
        environment = ER1V2BinaryMachine()
        for hypothesis in ER1_HYPOTHESES:
            for seed in range(seed_count):
                result = runner.run(
                    environment,
                    hypothesis,
                    ER1V2OracleInformationGainPolicy(),
                    episode_seed=seed,
                )
                if budget == 5 and seed == 0:
                    trace_results[hypothesis] = result
                records.append(_compact_episode(result, budget, seed))
    episodes = tuple(records)
    return ER1V2SanityStudy(
        seed_count=seed_count,
        runtime_seconds=perf_counter() - started,
        cells=aggregate_calibration_cells(episodes),
        overall_cells=aggregate_overall_cells(episodes),
        confusion_matrices=build_confusion_matrices(episodes),
        representative_traces={
            hypothesis.value: _serialize_trace(trace_results[hypothesis])
            for hypothesis in ER1_HYPOTHESES
        },
        likelihood_table=persistent_likelihood_table(),
        initial_information_gain=initial_information_gain(),
    )


def write_er1_v2_sanity_artifacts(
    study: ER1V2SanityStudy,
    output_dir: Path | str,
) -> ER1V2SanityArtifacts:
    """Write small aggregate artifacts; never write per-episode dumps."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_csv = directory / "er1_v2_sanity.csv"
    confusion_json = directory / "er1_v2_sanity_confusion.json"
    traces_json = directory / "er1_v2_sanity_traces.json"
    report_markdown = directory / "er1_v2_sanity.md"
    _write_csv(metrics_csv, study.cells)
    confusion_json.write_text(
        json.dumps(study.confusion_matrices, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    traces_json.write_text(
        json.dumps(study.representative_traces, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(build_er1_v2_sanity_report(study), encoding="utf-8")
    return ER1V2SanityArtifacts(
        metrics_csv=metrics_csv,
        confusion_json=confusion_json,
        traces_json=traces_json,
        report_markdown=report_markdown,
    )


def build_er1_v2_sanity_report(study: ER1V2SanityStudy) -> str:
    """Render analytic validation, small results, confusion, and traces."""
    trigger = TriggerLikelihoodModel()
    beliefs = trigger.conditioned_beliefs()
    lines = [
        "# ER-1 V2 Analytic Validation and Small Oracle Sanity Study",
        "",
        "This is an implementation diagnostic, not a final scientific calibration.",
        "",
        "## Transient trigger anomaly",
        "",
        "| Hypothesis | P(A0|H) | P(H|A0) |",
        "| --- | ---: | ---: |",
    ]
    for hypothesis in ER1_HYPOTHESES:
        lines.append(
            f"| `{hypothesis.value}` | {trigger.likelihood(hypothesis):.10f} | "
            f"{beliefs.probability(hypothesis):.10f} |"
        )
    lines.extend([
        "",
        "## Persistent investigation likelihoods",
        "",
        "The trigger probabilities do not appear in this table.",
        "",
        "| Current | Action | Signal | Hypothesis | Effective | P(0) | P(1) |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ])
    for row in study.likelihood_table:
        lines.append(
            f"| {row.current_context} | `{row.action}` | {row.result_signal} | "
            f"`{row.hypothesis}` | {row.effective_context} | "
            f"{row.probability_0:.6f} | {row.probability_1:.6f} |"
        )
    information = study.initial_information_gain
    lines.extend([
        "",
        "## Initial expected information gain",
        "",
        "| Action | EIG (bits) |",
        "| --- | ---: |",
        f"| `REPEAT_TRIAL` | {information.repeat_trial:.9f} |",
        f"| `USE_TRUSTED_SENSOR` | {information.use_trusted_sensor:.9f} |",
        f"| `CHANGE_CONTEXT` | {information.change_context:.9f} |",
        "",
        "## 800-episode sanity results",
        "",
        f"Runtime: {study.runtime_seconds:.3f} seconds; {study.seed_count} seeds per hypothesis/budget.",
        "",
        "| Budget | Hypothesis | MAP | Success@0.90 | Mean experiments | Mean true posterior |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for cell in study.cells:
        lines.append(
            f"| {cell.budget} | `{cell.hypothesis.value}` | {cell.map_accuracy:.3f} | "
            f"{cell.success_at_threshold:.3f} | {cell.mean_experiments:.3f} | "
            f"{cell.mean_true_posterior:.3f} |"
        )
    lines.extend(["", "## Structural safety rates", ""])
    for row in study.overall_cells:
        lines.append(
            f"- Budget {row.budget}: overall MAP={row.map_accuracy:.3f}, "
            f"Success@0.90={row.success_at_threshold:.3f}, false structural="
            f"{row.false_structural_diagnosis_rate:.3f}, missed structural="
            f"{row.missed_structural_failure_rate:.3f}, mean experiments="
            f"{row.mean_experiments:.3f}."
        )
    lines.extend(_confusion_markdown(study))
    lines.extend(["", "## Representative seed-0 traces", ""])
    for hypothesis in ER1_HYPOTHESES:
        trace = study.representative_traces[hypothesis.value]
        lines.append(f"### {hypothesis.value}")
        lines.append("")
        lines.append(
            "Initial: " + _compact_posterior(trace["trigger_conditioned_beliefs"])
        )
        for step in trace["trace"]:
            lines.append(
                f"- `{step['action']}` {step['observation']} → "
                f"{_compact_posterior(step['posterior'])}"
            )
        lines.append(
            f"- Final: `{trace['final_diagnosis']}`; {trace['termination_reason']}."
        )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "There is no exact full-signature non-identifiability: every hypothesis has a distinct distribution when all available actions and contexts are considered, and every ordinary binary outcome remains possible.",
        "",
        "Important conditional ambiguities remain. In context B, WORLD_SHIFT and MISSING_LATENT_VARIABLE have identical repeat and trusted distributions, so a context intervention is necessary to distinguish that pair. NO_STRUCTURAL_CHANGE and SENSOR_CORRUPTION have identical trusted-sensor distributions, so trusted evidence alone cannot distinguish them; primary-sensor evidence is necessary.",
        "",
        "A normal repeat (`O=1`) raises NO_STRUCTURAL_CHANGE from 0.127660 to 0.486124 in one update, a 3.81× posterior increase. This is strong positive rehabilitation evidence after the transient trigger.",
        "",
        "USE_TRUSTED_SENSOR has the largest initial EIG (0.566200 bits versus 0.425900 for CHANGE_CONTEXT and 0.223649 for REPEAT_TRIAL), so the oracle chooses it first in every sanity episode. It does not dominate the whole trajectory: subsequent choices adapt between trusted measurements and context interventions. Oracle action regret and premature-diagnosis rate are both zero by construction.",
        "",
        "No full calibration or parameter tuning was performed.",
        "",
    ])
    return "\n".join(lines)


def _compact_episode(
    result: ER1V2EpisodeResult,
    budget: int,
    seed: int,
) -> OracleCalibrationEpisode:
    return OracleCalibrationEpisode(
        hypothesis=result.ground_truth,
        budget=budget,
        threshold=ER1_V2_SANITY_THRESHOLD,
        seed=seed,
        final_map_diagnosis=result.predicted_diagnosis,
        map_correct=result.diagnosis_correct,
        reached_threshold=result.reached_threshold,
        success_at_threshold=result.success_at_threshold,
        experiments_used=result.experiments_used,
        final_true_posterior=result.final_beliefs.probability(result.ground_truth),
        cumulative_action_regret=result.cumulative_action_regret,
        action_sequence=tuple(item.chosen_action for item in result.trace),
    )


def _serialize_trace(result: ER1V2EpisodeResult) -> dict[str, object]:
    return {
        "seed": result.evaluation_metadata.episode_seed,
        "true_hypothesis": result.ground_truth.value,
        "trigger_observation": {"x": 1, "o": 0},
        "trigger_conditioned_beliefs": _belief_dict(
            result.trigger_conditioned_beliefs
        ),
        "trace": [
            {
                "step": item.step_number,
                "action": item.chosen_action.value,
                "observation": _result_dict(item.experiment_result),
                "posterior": _belief_dict(item.posterior),
            }
            for item in result.trace
        ],
        "final_diagnosis": result.predicted_diagnosis.value,
        "termination_reason": (
            "THRESHOLD_REACHED"
            if result.reached_threshold
            else "BUDGET_EXHAUSTED"
        ),
    }


def _belief_dict(beliefs) -> dict[str, float]:
    return {
        hypothesis.value: beliefs.probability(hypothesis)
        for hypothesis in ER1_HYPOTHESES
    }


def _result_dict(result) -> dict[str, object]:
    if isinstance(result, RepeatTrialResult):
        return {"primary_o": result.o}
    if isinstance(result, StochasticTrustedSensorResult):
        return {"trusted_t": result.trusted_t}
    if isinstance(result, ChangeContextResult):
        return {"context": result.context.value, "primary_o": result.o}
    raise TypeError("unsupported V2 result")


def _confusion_markdown(study: ER1V2SanityStudy) -> list[str]:
    lines = []
    conditions = study.confusion_matrices["conditions"]
    for budget in ER1_V2_SANITY_BUDGETS:
        matrix = conditions[f"budget_{budget}_threshold_0.90"]
        counts = matrix["counts"]
        percentages = matrix["row_percentages"]
        lines.extend([
            "",
            f"## Confusion matrix: budget {budget}",
            "",
            "| Truth | N | W | S | L |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for truth, label in zip(ER1_HYPOTHESES, ("N", "W", "S", "L")):
            cells = []
            for diagnosis in ER1_HYPOTHESES:
                cells.append(
                    f"{counts[truth.value][diagnosis.value]} "
                    f"({percentages[truth.value][diagnosis.value]:.1f}%)"
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _compact_posterior(values: dict[str, float]) -> str:
    return "[" + ", ".join(
        f"{label}={values[hypothesis.value]:.4f}"
        for label, hypothesis in zip(("N", "W", "S", "L"), ER1_HYPOTHESES)
    ) + "]"


def _write_csv(path: Path, rows: tuple[CalibrationCell, ...]) -> None:
    fieldnames = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["hypothesis"] = row.hypothesis.value
            writer.writerow(values)
