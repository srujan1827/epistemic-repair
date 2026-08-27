"""Experiment-only staged oracle parameter search for a possible ER-1 V2."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable, Sequence

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticLikelihoodModel,
    stochastic_information_gains,
)
from epistemic_repair.diagnostics.actions import Context, DiagnosticAction
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_DEFAULT_CONFIG, ER1_HYPOTHESES
from epistemic_repair.evaluation.er1_calibration import (
    CALIBRATION_HYPOTHESIS_ORDER,
    CalibrationCell,
    OracleCalibrationEpisode,
    aggregate_calibration_cells,
    aggregate_overall_cells,
)
from epistemic_repair.evaluation.stochastic_runner import (
    StochasticDiagnosticEpisodeRunner,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.policies.stochastic import (
    StochasticOracleInformationGainPolicy,
)


V1_NORMAL_PROCESS_ACCURACY = 0.90
V1_SHIFT_PROCESS_ACCURACY = 0.90
V1_HEALTHY_SENSOR_ACCURACY = 0.95
V1_CORRUPTED_SENSOR_INVERSION_ACCURACY = 0.90
V1_LATENT_PROCESS_ACCURACY = 0.90
SENSOR_CORRUPTION_PROCESS_ACCURACY = 0.90
TRUSTED_SENSOR_RELIABILITY = 0.99

STAGE_A_NORMAL_VALUES = (0.75, 0.80, 0.85, 0.90)
STAGE_B_SHIFT_VALUES = (0.85, 0.90)
STAGE_B_LATENT_VALUES = (0.85, 0.90)
STAGE_B_CORRUPTION_VALUES = (0.85, 0.90)
SEARCH_BUDGETS = (5, 8)
SEARCH_THRESHOLD = 0.90
FULL_BUDGETS = (1, 2, 3, 5, 8)
FULL_THRESHOLDS = (0.80, 0.90, 0.95)


@dataclass(frozen=True, slots=True)
class ER1CandidateParameters:
    """Search-only stochastic parameters; never used as active ER-1 config."""

    normal_process_accuracy: float
    shift_process_accuracy: float
    sensor_normal_accuracy: float
    corrupted_sensor_inversion_accuracy: float
    latent_process_accuracy: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite probability")
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")

    @classmethod
    def v1_baseline(cls) -> "ER1CandidateParameters":
        return cls(
            normal_process_accuracy=V1_NORMAL_PROCESS_ACCURACY,
            shift_process_accuracy=V1_SHIFT_PROCESS_ACCURACY,
            sensor_normal_accuracy=V1_HEALTHY_SENSOR_ACCURACY,
            corrupted_sensor_inversion_accuracy=(
                V1_CORRUPTED_SENSOR_INVERSION_ACCURACY
            ),
            latent_process_accuracy=V1_LATENT_PROCESS_ACCURACY,
        )

    @property
    def is_v1_baseline(self) -> bool:
        return self == self.v1_baseline()

    @property
    def candidate_id(self) -> str:
        if self.is_v1_baseline:
            return "v1_baseline"
        return (
            f"n{self.normal_process_accuracy:.2f}_"
            f"w{self.shift_process_accuracy:.2f}_"
            f"h{self.sensor_normal_accuracy:.2f}_"
            f"c{self.corrupted_sensor_inversion_accuracy:.2f}_"
            f"l{self.latent_process_accuracy:.2f}"
        )


@dataclass(frozen=True, slots=True)
class CandidateAnomalyAnalysis:
    """Analytic anomaly likelihood and conditioned prior for one candidate."""

    no_change_likelihood: float
    world_shift_likelihood: float
    sensor_corruption_likelihood: float
    missing_latent_likelihood: float
    no_change_posterior: float
    world_shift_posterior: float
    sensor_corruption_posterior: float
    missing_latent_posterior: float


@dataclass(frozen=True, slots=True)
class CandidateIdentifiability:
    """Analytic preservation checks for the intended ER-1 task structure."""

    context_required_for_world_vs_latent: bool
    no_change_vs_sensor_primary_distinguishable: bool
    trusted_sensor_informative_about_physical_change: bool
    all_binary_outcomes_nonzero: bool
    all_initial_actions_informative: bool
    repeat_trial_positive_information_gain: bool
    notes: str

    @property
    def passes(self) -> bool:
        return all(
            (
                self.context_required_for_world_vs_latent,
                self.no_change_vs_sensor_primary_distinguishable,
                self.trusted_sensor_informative_about_physical_change,
                self.all_binary_outcomes_nonzero,
                self.all_initial_actions_informative,
                self.repeat_trial_positive_information_gain,
            )
        )


@dataclass(frozen=True, slots=True)
class CandidateScorecard:
    """Transparent Phase-1 or Phase-2 candidate comparison metrics."""

    candidate_id: str
    is_v1_baseline: bool
    normal_process_accuracy: float
    shift_process_accuracy: float
    sensor_normal_accuracy: float
    corrupted_sensor_inversion_accuracy: float
    latent_process_accuracy: float
    initial_no_change_posterior: float
    overall_map_5: float
    overall_map_8: float
    no_change_map_5: float
    no_change_map_8: float
    world_shift_map_5: float
    world_shift_map_8: float
    sensor_corruption_map_5: float
    sensor_corruption_map_8: float
    missing_latent_map_5: float
    missing_latent_map_8: float
    worst_structural_map_5: float
    worst_structural_map_8: float
    false_structural_rate_5: float
    false_structural_rate_8: float
    missed_structural_rate_5: float
    missed_structural_rate_8: float
    mean_experiments_5: float
    mean_experiments_8: float
    max_hypothesis_map_3: float | None
    design_targets_met: int
    heuristic_score: float
    identifiability_pass: bool
    identifiability_notes: str


@dataclass(frozen=True, slots=True)
class CandidateFullCell:
    """One full-calibration cell with candidate parameters attached."""

    candidate_id: str
    normal_process_accuracy: float
    shift_process_accuracy: float
    sensor_normal_accuracy: float
    corrupted_sensor_inversion_accuracy: float
    latent_process_accuracy: float
    hypothesis: str
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


@dataclass(frozen=True, slots=True)
class ParameterSearchStudy:
    """Complete staged search results and finalist full calibrations."""

    phase1_seed_count: int
    phase2_seed_count: int
    promising_normal_values: tuple[float, ...]
    stage_a: tuple[CandidateScorecard, ...]
    stage_b: tuple[CandidateScorecard, ...]
    top_three_parameters: tuple[ER1CandidateParameters, ...]
    phase2_parameters: tuple[ER1CandidateParameters, ...]
    finalists: tuple[CandidateScorecard, ...]
    finalist_cells: tuple[CandidateFullCell, ...]
    anomaly_analyses: dict[str, CandidateAnomalyAnalysis]
    identifiability: dict[str, CandidateIdentifiability]
    recommended_candidate_id: str
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class ParameterSearchArtifacts:
    stage_a_csv: Path
    stage_b_csv: Path
    finalists_csv: Path
    phase2_cells_csv: Path
    report_markdown: Path


class CandidateStochasticLikelihoodModel(StochasticLikelihoodModel):
    """Normative likelihood model parameterized only for search experiments."""

    def __init__(self, parameters: ER1CandidateParameters) -> None:
        super().__init__(ER1_DEFAULT_CONFIG)
        self.parameters = parameters

    def probability_y(
        self,
        y: int,
        hypothesis: FailureMode,
        context: Context,
        *,
        x: int = 1,
    ) -> float:
        self._validate_inputs(hypothesis, context, x)
        self._validate_bit(y, "y")
        accuracy, preferred_matches_x = self._physical_spec(hypothesis, context)
        preferred_y = x if preferred_matches_x else 1 - x
        return accuracy if y == preferred_y else 1.0 - accuracy

    def probability_o_given_y(
        self,
        o: int,
        y: int,
        hypothesis: FailureMode,
    ) -> float:
        self._validate_bit(o, "o")
        self._validate_bit(y, "y")
        if hypothesis not in ER1_HYPOTHESES:
            raise ValueError("hypothesis is not part of ER-1")
        matches = (
            1.0 - self.parameters.corrupted_sensor_inversion_accuracy
            if hypothesis is FailureMode.SENSOR_CORRUPTION
            else self.parameters.sensor_normal_accuracy
        )
        return matches if o == y else 1.0 - matches

    def _physical_spec(
        self,
        hypothesis: FailureMode,
        context: Context,
    ) -> tuple[float, bool]:
        if hypothesis is FailureMode.NO_STRUCTURAL_CHANGE:
            return self.parameters.normal_process_accuracy, True
        if hypothesis is FailureMode.WORLD_SHIFT:
            return self.parameters.shift_process_accuracy, False
        if hypothesis is FailureMode.SENSOR_CORRUPTION:
            return SENSOR_CORRUPTION_PROCESS_ACCURACY, True
        return self.parameters.latent_process_accuracy, context is Context.A


class CandidateStochasticBinaryMachine(StochasticBinaryMachine):
    """Seeded search-only environment matching candidate likelihoods exactly."""

    def __init__(self, parameters: ER1CandidateParameters) -> None:
        super().__init__(ER1_DEFAULT_CONFIG)
        self.parameters = parameters

    def _probability_y_one(self, x: int, context: Context) -> float:
        if self._failure_mode is FailureMode.NO_STRUCTURAL_CHANGE:
            accuracy, matches_x = self.parameters.normal_process_accuracy, True
        elif self._failure_mode is FailureMode.WORLD_SHIFT:
            accuracy, matches_x = self.parameters.shift_process_accuracy, False
        elif self._failure_mode is FailureMode.SENSOR_CORRUPTION:
            accuracy, matches_x = SENSOR_CORRUPTION_PROCESS_ACCURACY, True
        else:
            accuracy = self.parameters.latent_process_accuracy
            matches_x = context is Context.A
        preferred_y = x if matches_x else 1 - x
        return accuracy if preferred_y == 1 else 1.0 - accuracy

    def _sample_primary_observation(self, y: int) -> int:
        matches = (
            1.0 - self.parameters.corrupted_sensor_inversion_accuracy
            if self._failure_mode is FailureMode.SENSOR_CORRUPTION
            else self.parameters.sensor_normal_accuracy
        )
        return self._sample_reliable_copy(y, matches)

    def _primary_probability(self, o: int, y: int) -> float:
        matches = (
            1.0 - self.parameters.corrupted_sensor_inversion_accuracy
            if self._failure_mode is FailureMode.SENSOR_CORRUPTION
            else self.parameters.sensor_normal_accuracy
        )
        return matches if o == y else 1.0 - matches


def analyze_initial_anomaly(
    parameters: ER1CandidateParameters,
) -> CandidateAnomalyAnalysis:
    """Calculate P(A0|H) and the equal-prior posterior analytically."""
    model = CandidateStochasticLikelihoodModel(parameters)
    likelihoods = {
        hypothesis: model.initial_anomaly_likelihood(hypothesis)
        for hypothesis in ER1_HYPOTHESES
    }
    denominator = sum(likelihoods.values())
    posteriors = {
        hypothesis: likelihoods[hypothesis] / denominator
        for hypothesis in ER1_HYPOTHESES
    }
    return CandidateAnomalyAnalysis(
        no_change_likelihood=likelihoods[FailureMode.NO_STRUCTURAL_CHANGE],
        world_shift_likelihood=likelihoods[FailureMode.WORLD_SHIFT],
        sensor_corruption_likelihood=likelihoods[FailureMode.SENSOR_CORRUPTION],
        missing_latent_likelihood=likelihoods[
            FailureMode.MISSING_LATENT_VARIABLE
        ],
        no_change_posterior=posteriors[FailureMode.NO_STRUCTURAL_CHANGE],
        world_shift_posterior=posteriors[FailureMode.WORLD_SHIFT],
        sensor_corruption_posterior=posteriors[FailureMode.SENSOR_CORRUPTION],
        missing_latent_posterior=posteriors[
            FailureMode.MISSING_LATENT_VARIABLE
        ],
    )


def check_candidate_identifiability(
    parameters: ER1CandidateParameters,
) -> CandidateIdentifiability:
    """Verify the intended action distinctions under a candidate model."""
    model = CandidateStochasticLikelihoodModel(parameters)
    beliefs = model.conditioned_initial_beliefs()
    information = stochastic_information_gains(beliefs, model, Context.B)
    no_primary = model.probability_primary_observation(
        0, FailureMode.NO_STRUCTURAL_CHANGE, Context.B
    )
    corrupted_primary = model.probability_primary_observation(
        0, FailureMode.SENSOR_CORRUPTION, Context.B
    )
    no_trusted = model.probability_trusted_observation(
        0, FailureMode.NO_STRUCTURAL_CHANGE, Context.B
    )
    shifted_trusted = model.probability_trusted_observation(
        0, FailureMode.WORLD_SHIFT, Context.B
    )
    all_nonzero = all(
        0.0
        < probability
        < 1.0
        for hypothesis in ER1_HYPOTHESES
        for context in Context
        for probability in (
            model.probability_primary_observation(0, hypothesis, context),
            model.probability_primary_observation(1, hypothesis, context),
            model.probability_trusted_observation(0, hypothesis, context),
            model.probability_trusted_observation(1, hypothesis, context),
        )
    )
    context_required = abs(
        parameters.shift_process_accuracy - parameters.latent_process_accuracy
    ) < 1e-12
    notes = []
    if not context_required:
        notes.append(
            "WORLD_SHIFT and MISSING_LATENT_VARIABLE differ statistically in context B before a context intervention"
        )
    if abs(no_primary - corrupted_primary) < 1e-12:
        notes.append("NO_STRUCTURAL_CHANGE and SENSOR_CORRUPTION primary channels coincide")
    if abs(no_trusted - shifted_trusted) < 1e-12:
        notes.append("trusted sensor does not distinguish unchanged from shifted physics")
    gains = tuple(information.for_action(action) for action in CALIBRATION_ACTION_ORDER)
    if any(value <= 1e-12 for value in gains):
        notes.append("an initial benchmark action has zero expected information gain")
    return CandidateIdentifiability(
        context_required_for_world_vs_latent=context_required,
        no_change_vs_sensor_primary_distinguishable=(
            abs(no_primary - corrupted_primary) > 1e-12
        ),
        trusted_sensor_informative_about_physical_change=(
            abs(no_trusted - shifted_trusted) > 1e-12
        ),
        all_binary_outcomes_nonzero=all_nonzero,
        all_initial_actions_informative=all(value > 1e-12 for value in gains),
        repeat_trial_positive_information_gain=(
            information.repeat_trial > 1e-12
        ),
        notes="; ".join(notes) if notes else "all intended distinctions preserved",
    )


CALIBRATION_ACTION_ORDER = (
    DiagnosticAction.REPEAT_TRIAL,
    DiagnosticAction.USE_TRUSTED_SENSOR,
    DiagnosticAction.CHANGE_CONTEXT,
)


def run_candidate_episodes(
    parameters: ER1CandidateParameters,
    *,
    seed_count: int,
    budgets: Iterable[int],
    thresholds: Iterable[float],
) -> tuple[OracleCalibrationEpisode, ...]:
    """Run a candidate through the unchanged stochastic oracle semantics."""
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be a positive integer")
    budgets_tuple = tuple(sorted(budgets))
    thresholds_tuple = tuple(sorted(float(value) for value in thresholds))
    if not budgets_tuple or any(value <= 0 for value in budgets_tuple):
        raise ValueError("budgets must be positive")
    if not thresholds_tuple or any(
        not 0.0 < value <= 1.0 for value in thresholds_tuple
    ):
        raise ValueError("thresholds must be in (0, 1]")

    model = CandidateStochasticLikelihoodModel(parameters)
    environment = CandidateStochasticBinaryMachine(parameters)
    policy = StochasticOracleInformationGainPolicy()
    records = []
    for budget in budgets_tuple:
        for threshold in thresholds_tuple:
            runner = StochasticDiagnosticEpisodeRunner(
                diagnosis_threshold=threshold,
                max_experiments=budget,
                likelihood_model=model,
            )
            for hypothesis in ER1_HYPOTHESES:
                for seed in range(seed_count):
                    result = runner.run(
                        environment,
                        hypothesis,
                        policy,
                        episode_seed=seed,
                    )
                    final_beliefs = (
                        result.trace[-1].posterior
                        if result.trace
                        else result.initial_beliefs
                    )
                    records.append(
                        OracleCalibrationEpisode(
                            hypothesis=hypothesis,
                            budget=budget,
                            threshold=threshold,
                            seed=seed,
                            final_map_diagnosis=result.predicted_diagnosis,
                            map_correct=result.diagnosis_correct,
                            reached_threshold=result.reached_threshold,
                            success_at_threshold=(
                                result.reached_threshold
                                and result.diagnosis_correct
                            ),
                            experiments_used=result.experiments_used,
                            final_true_posterior=final_beliefs.probability(
                                hypothesis
                            ),
                            cumulative_action_regret=(
                                result.cumulative_action_regret
                            ),
                            action_sequence=tuple(
                                step.chosen_action for step in result.trace
                            ),
                        )
                    )
    return tuple(records)


def build_candidate_scorecard(
    parameters: ER1CandidateParameters,
    episodes: Sequence[OracleCalibrationEpisode],
) -> CandidateScorecard:
    """Build the raw design scorecard and transparent heuristic ranking aid."""
    cells = aggregate_calibration_cells(episodes)
    overall = aggregate_overall_cells(episodes)

    def hypothesis_map(hypothesis: FailureMode, budget: int) -> float:
        matches = [
            cell
            for cell in cells
            if cell.hypothesis is hypothesis
            and cell.budget == budget
            and abs(cell.threshold - SEARCH_THRESHOLD) < 1e-12
        ]
        if len(matches) != 1:
            raise ValueError("scorecard requires budgets 5 and 8 at threshold 0.90")
        return matches[0].map_accuracy

    def overall_row(budget: int):
        matches = [
            row
            for row in overall
            if row.budget == budget
            and abs(row.threshold - SEARCH_THRESHOLD) < 1e-12
        ]
        if len(matches) != 1:
            raise ValueError("scorecard requires budgets 5 and 8 at threshold 0.90")
        return matches[0]

    row5 = overall_row(5)
    row8 = overall_row(8)
    no5 = hypothesis_map(FailureMode.NO_STRUCTURAL_CHANGE, 5)
    no8 = hypothesis_map(FailureMode.NO_STRUCTURAL_CHANGE, 8)
    world5 = hypothesis_map(FailureMode.WORLD_SHIFT, 5)
    world8 = hypothesis_map(FailureMode.WORLD_SHIFT, 8)
    sensor5 = hypothesis_map(FailureMode.SENSOR_CORRUPTION, 5)
    sensor8 = hypothesis_map(FailureMode.SENSOR_CORRUPTION, 8)
    latent5 = hypothesis_map(FailureMode.MISSING_LATENT_VARIABLE, 5)
    latent8 = hypothesis_map(FailureMode.MISSING_LATENT_VARIABLE, 8)
    worst5 = min(world5, sensor5, latent5)
    worst8 = min(world8, sensor8, latent8)
    budget3_cells = [
        cell
        for cell in cells
        if cell.budget == 3 and abs(cell.threshold - SEARCH_THRESHOLD) < 1e-12
    ]
    max_map_3 = (
        max(cell.map_accuracy for cell in budget3_cells)
        if budget3_cells
        else None
    )
    target_checks = (
        row5.map_accuracy >= 0.75,
        no5 >= 0.65,
        worst5 >= 0.80,
        row8.map_accuracy >= 0.85,
        0.80 <= no8 <= 0.90,
        worst8 >= 0.88,
        row8.false_structural_diagnosis_rate <= 0.20,
        row8.missed_structural_failure_rate <= 0.10,
    )
    analysis = analyze_initial_anomaly(parameters)
    identity = check_candidate_identifiability(parameters)
    score = (
        2.0 * row5.map_accuracy
        + 2.0 * row8.map_accuracy
        + 2.5 * no5
        + 3.0 * no8
        + 1.5 * worst5
        + 2.0 * worst8
        - 2.0 * row5.false_structural_diagnosis_rate
        - 3.0 * row8.false_structural_diagnosis_rate
        - row5.missed_structural_failure_rate
        - 2.0 * row8.missed_structural_failure_rate
        - (2.0 if not identity.passes else 0.0)
    )
    return CandidateScorecard(
        candidate_id=parameters.candidate_id,
        is_v1_baseline=parameters.is_v1_baseline,
        **asdict(parameters),
        initial_no_change_posterior=analysis.no_change_posterior,
        overall_map_5=row5.map_accuracy,
        overall_map_8=row8.map_accuracy,
        no_change_map_5=no5,
        no_change_map_8=no8,
        world_shift_map_5=world5,
        world_shift_map_8=world8,
        sensor_corruption_map_5=sensor5,
        sensor_corruption_map_8=sensor8,
        missing_latent_map_5=latent5,
        missing_latent_map_8=latent8,
        worst_structural_map_5=worst5,
        worst_structural_map_8=worst8,
        false_structural_rate_5=row5.false_structural_diagnosis_rate,
        false_structural_rate_8=row8.false_structural_diagnosis_rate,
        missed_structural_rate_5=row5.missed_structural_failure_rate,
        missed_structural_rate_8=row8.missed_structural_failure_rate,
        mean_experiments_5=row5.mean_experiments,
        mean_experiments_8=row8.mean_experiments,
        max_hypothesis_map_3=max_map_3,
        design_targets_met=sum(target_checks),
        heuristic_score=score,
        identifiability_pass=identity.passes,
        identifiability_notes=identity.notes,
    )


def rank_scorecards(
    scorecards: Iterable[CandidateScorecard],
) -> tuple[CandidateScorecard, ...]:
    """Rank deterministically using identifiability, targets, then raw metrics."""
    return tuple(
        sorted(
            scorecards,
            key=lambda item: (
                not item.identifiability_pass,
                (
                    item.max_hypothesis_map_3 is not None
                    and item.max_hypothesis_map_3 > 0.99
                ),
                -item.design_targets_met,
                -item.heuristic_score,
                -item.no_change_map_8,
                -item.worst_structural_map_8,
                item.candidate_id,
            ),
        )
    )


def run_parameter_search(
    *,
    phase1_seed_count: int = 250,
    phase2_seed_count: int = 1000,
) -> ParameterSearchStudy:
    """Run Stage A, Stage B Phase 1, then full calibration of finalists."""
    started = perf_counter()
    phase1_cache: dict[ER1CandidateParameters, tuple[OracleCalibrationEpisode, ...]] = {}

    def phase1_score(parameters: ER1CandidateParameters) -> CandidateScorecard:
        if parameters not in phase1_cache:
            phase1_cache[parameters] = run_candidate_episodes(
                parameters,
                seed_count=phase1_seed_count,
                budgets=SEARCH_BUDGETS,
                thresholds=(SEARCH_THRESHOLD,),
            )
        return build_candidate_scorecard(parameters, phase1_cache[parameters])

    stage_a_parameters = tuple(
        ER1CandidateParameters(
            normal_process_accuracy=normal,
            shift_process_accuracy=V1_SHIFT_PROCESS_ACCURACY,
            sensor_normal_accuracy=V1_HEALTHY_SENSOR_ACCURACY,
            corrupted_sensor_inversion_accuracy=(
                V1_CORRUPTED_SENSOR_INVERSION_ACCURACY
            ),
            latent_process_accuracy=V1_LATENT_PROCESS_ACCURACY,
        )
        for normal in STAGE_A_NORMAL_VALUES
    )
    stage_a = tuple(phase1_score(parameters) for parameters in stage_a_parameters)
    promising = tuple(
        item.normal_process_accuracy for item in rank_scorecards(stage_a)[:2]
    )

    stage_b_parameters = tuple(
        ER1CandidateParameters(
            normal_process_accuracy=normal,
            shift_process_accuracy=shift,
            sensor_normal_accuracy=V1_HEALTHY_SENSOR_ACCURACY,
            corrupted_sensor_inversion_accuracy=corruption,
            latent_process_accuracy=latent,
        )
        for normal, shift, latent, corruption in product(
            promising,
            STAGE_B_SHIFT_VALUES,
            STAGE_B_LATENT_VALUES,
            STAGE_B_CORRUPTION_VALUES,
        )
    )
    stage_b = tuple(phase1_score(parameters) for parameters in stage_b_parameters)
    baseline = ER1CandidateParameters.v1_baseline()
    candidate_pool = {item.candidate_id: item for item in (*stage_b, phase1_score(baseline))}
    ranked_pool = rank_scorecards(candidate_pool.values())
    top_three_ids = {item.candidate_id for item in ranked_pool[:3]}
    parameter_by_id = {
        parameters.candidate_id: parameters
        for parameters in (*stage_b_parameters, baseline)
    }
    top_three = tuple(parameter_by_id[item.candidate_id] for item in ranked_pool[:3])
    phase2_parameters = list(top_three)
    if baseline.candidate_id not in top_three_ids:
        phase2_parameters.append(baseline)

    finalist_scorecards = []
    finalist_cells = []
    anomaly_analyses = {}
    identities = {}
    for parameters in phase2_parameters:
        episodes = run_candidate_episodes(
            parameters,
            seed_count=phase2_seed_count,
            budgets=FULL_BUDGETS,
            thresholds=FULL_THRESHOLDS,
        )
        scorecard = build_candidate_scorecard(parameters, episodes)
        finalist_scorecards.append(scorecard)
        anomaly_analyses[parameters.candidate_id] = analyze_initial_anomaly(parameters)
        identities[parameters.candidate_id] = check_candidate_identifiability(parameters)
        for cell in aggregate_calibration_cells(episodes):
            finalist_cells.append(_attach_candidate(parameters, cell))

    ranked_finalists = rank_scorecards(finalist_scorecards)
    recommendation = ranked_finalists[0].candidate_id
    return ParameterSearchStudy(
        phase1_seed_count=phase1_seed_count,
        phase2_seed_count=phase2_seed_count,
        promising_normal_values=promising,
        stage_a=stage_a,
        stage_b=stage_b,
        top_three_parameters=top_three,
        phase2_parameters=tuple(phase2_parameters),
        finalists=tuple(finalist_scorecards),
        finalist_cells=tuple(finalist_cells),
        anomaly_analyses=anomaly_analyses,
        identifiability=identities,
        recommended_candidate_id=recommendation,
        runtime_seconds=perf_counter() - started,
    )


def write_parameter_search_artifacts(
    study: ParameterSearchStudy,
    output_dir: Path | str,
) -> ParameterSearchArtifacts:
    """Write compact deterministic search tables and a comparison report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stage_a_csv = directory / "er1_parameter_search_stage1.csv"
    stage_b_csv = directory / "er1_parameter_search_stage_b.csv"
    finalists_csv = directory / "er1_parameter_search_finalists.csv"
    phase2_cells_csv = directory / "er1_parameter_search_phase2_cells.csv"
    report_markdown = directory / "er1_parameter_search.md"
    _write_csv(stage_a_csv, study.stage_a)
    _write_csv(stage_b_csv, study.stage_b)
    _write_csv(finalists_csv, study.finalists)
    _write_csv(phase2_cells_csv, study.finalist_cells)
    report_markdown.write_text(build_parameter_search_report(study), encoding="utf-8")
    return ParameterSearchArtifacts(
        stage_a_csv=stage_a_csv,
        stage_b_csv=stage_b_csv,
        finalists_csv=finalists_csv,
        phase2_cells_csv=phase2_cells_csv,
        report_markdown=report_markdown,
    )


def build_parameter_search_report(study: ParameterSearchStudy) -> str:
    """Render staged results, finalist analyses, and an unapplied recommendation."""
    lines = [
        "# ER-1 V2 Oracle-Only Parameter Search",
        "",
        "## Executive summary",
        "",
        f"The staged search took {study.runtime_seconds:.3f} seconds. Phase 1 used "
        f"{study.phase1_seed_count} seeds per hypothesis/cell; Phase 2 used "
        f"{study.phase2_seed_count}. Active ER-1 V1 configuration was not modified.",
        "",
        f"Recommended experimental candidate: `{study.recommended_candidate_id}`. This recommendation is not applied.",
        "",
        "## Stage A — ordinary no-change noise",
        "",
        *_scorecard_table(study.stage_a),
        "",
        "Promising normal-process accuracies advanced to Stage B: "
        + ", ".join(f"{value:.2f}" for value in study.promising_normal_values)
        + ".",
        "",
        "## Stage B — modest structural contrast search",
        "",
        *_scorecard_table(rank_scorecards(study.stage_b)),
        "",
        "## Phase 2 finalists and V1 baseline",
        "",
        *_scorecard_table(rank_scorecards(study.finalists)),
        "",
        "## Finalist structural performance",
        "",
        *_structural_table(rank_scorecards(study.finalists)),
        "",
        "## Direct comparison with V1",
        "",
        *_v1_delta_table(rank_scorecards(study.finalists)),
        "",
        "## Initial anomaly analysis",
        "",
        "| Candidate | P(A0|N) | P(A0|W) | P(A0|S) | P(A0|L) | P(N|A0) | P(W|A0) | P(S|A0) | P(L|A0) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for parameters in study.phase2_parameters:
        analysis = study.anomaly_analyses[parameters.candidate_id]
        lines.append(
            f"| `{parameters.candidate_id}` | {analysis.no_change_likelihood:.4f} | "
            f"{analysis.world_shift_likelihood:.4f} | {analysis.sensor_corruption_likelihood:.4f} | "
            f"{analysis.missing_latent_likelihood:.4f} | {analysis.no_change_posterior:.4f} | "
            f"{analysis.world_shift_posterior:.4f} | {analysis.sensor_corruption_posterior:.4f} | "
            f"{analysis.missing_latent_posterior:.4f} |"
        )
    lines.extend([
        "",
        "## Identifiability checks",
        "",
    ])
    for parameters in study.phase2_parameters:
        identity = study.identifiability[parameters.candidate_id]
        lines.append(
            f"- `{parameters.candidate_id}`: {'PASS' if identity.passes else 'CONCERN'} — {identity.notes}."
        )
    recommended = next(
        item
        for item in study.finalists
        if item.candidate_id == study.recommended_candidate_id
    )
    if recommended.is_v1_baseline:
        recommendation_text = (
            "The search recommends retaining the V1 parameter set rather than "
            "promoting an inferior V2 candidate. None of the searched candidates "
            "improved no-change recovery while preserving structural performance."
        )
        rationale = (
            "V1 is the highest-ranked set after preserved identifiability, soft "
            "design targets, raw accuracy, structural-error rates, and the "
            "transparent heuristic are considered together."
        )
    else:
        recommendation_text = (
            f"Recommend `{recommended.candidate_id}` for manual ER-1 V2 review: "
            f"N={recommended.normal_process_accuracy:.2f}, "
            f"W={recommended.shift_process_accuracy:.2f}, healthy sensor="
            f"{recommended.sensor_normal_accuracy:.2f}, corrupted inversion="
            f"{recommended.corrupted_sensor_inversion_accuracy:.2f}, "
            f"latent={recommended.latent_process_accuracy:.2f}."
        )
        rationale = (
            "It is the highest-ranked finalist after prioritizing preserved "
            "identifiability, soft design targets, raw no-change recovery, worst "
            "structural performance, and the transparent heuristic."
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        recommendation_text,
        "",
        rationale,
        "",
        "No candidate parameters were applied to active ER-1 configuration.",
        "",
    ])
    return "\n".join(lines)


def _scorecard_table(scorecards: Sequence[CandidateScorecard]) -> list[str]:
    lines = [
        "| Candidate | P(N|A0) | MAP@5 | MAP@8 | N@5 | N@8 | Worst structural@5 | Worst structural@8 | False@5 | False@8 | Missed@5 | Missed@8 | Mean exp@5 | Mean exp@8 | Max hypothesis@3 | Targets | ID | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for item in scorecards:
        max_map_3 = (
            f"{item.max_hypothesis_map_3:.3f}"
            if item.max_hypothesis_map_3 is not None
            else "—"
        )
        lines.append(
            f"| `{item.candidate_id}` | {item.initial_no_change_posterior:.3f} | "
            f"{item.overall_map_5:.3f} | {item.overall_map_8:.3f} | "
            f"{item.no_change_map_5:.3f} | {item.no_change_map_8:.3f} | "
            f"{item.worst_structural_map_5:.3f} | {item.worst_structural_map_8:.3f} | "
            f"{item.false_structural_rate_5:.3f} | {item.false_structural_rate_8:.3f} | "
            f"{item.missed_structural_rate_5:.3f} | {item.missed_structural_rate_8:.3f} | "
            f"{item.mean_experiments_5:.3f} | {item.mean_experiments_8:.3f} | "
            f"{max_map_3} | "
            f"{item.design_targets_met}/8 | {'PASS' if item.identifiability_pass else 'CONCERN'} | "
            f"{item.heuristic_score:.3f} |"
        )
    return lines


def _structural_table(scorecards: Sequence[CandidateScorecard]) -> list[str]:
    lines = [
        "| Candidate | World@5 | World@8 | Sensor@5 | Sensor@8 | Latent@5 | Latent@8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in scorecards:
        lines.append(
            f"| `{item.candidate_id}` | {item.world_shift_map_5:.3f} | "
            f"{item.world_shift_map_8:.3f} | {item.sensor_corruption_map_5:.3f} | "
            f"{item.sensor_corruption_map_8:.3f} | {item.missing_latent_map_5:.3f} | "
            f"{item.missing_latent_map_8:.3f} |"
        )
    return lines


def _v1_delta_table(scorecards: Sequence[CandidateScorecard]) -> list[str]:
    baseline = next(item for item in scorecards if item.is_v1_baseline)
    lines = [
        "| Candidate | Δ MAP@5 | Δ MAP@8 | Δ N@5 | Δ N@8 | Δ false@5 | Δ false@8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in scorecards:
        lines.append(
            f"| `{item.candidate_id}` | {item.overall_map_5 - baseline.overall_map_5:+.3f} | "
            f"{item.overall_map_8 - baseline.overall_map_8:+.3f} | "
            f"{item.no_change_map_5 - baseline.no_change_map_5:+.3f} | "
            f"{item.no_change_map_8 - baseline.no_change_map_8:+.3f} | "
            f"{item.false_structural_rate_5 - baseline.false_structural_rate_5:+.3f} | "
            f"{item.false_structural_rate_8 - baseline.false_structural_rate_8:+.3f} |"
        )
    return lines


def _attach_candidate(
    parameters: ER1CandidateParameters,
    cell: CalibrationCell,
) -> CandidateFullCell:
    values = asdict(cell)
    values["hypothesis"] = cell.hypothesis.value
    retained = {
        key: values[key]
        for key in CandidateFullCell.__dataclass_fields__
        if key not in {
            "candidate_id",
            "normal_process_accuracy",
            "shift_process_accuracy",
            "sensor_normal_accuracy",
            "corrupted_sensor_inversion_accuracy",
            "latent_process_accuracy",
        }
    }
    return CandidateFullCell(
        candidate_id=parameters.candidate_id,
        **asdict(parameters),
        **retained,
    )


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        raise ValueError("cannot write an empty search table")
    fieldnames = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
