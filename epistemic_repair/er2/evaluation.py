"""Deterministic held-out evaluation for explicit ER-2 repair states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Iterable

from epistemic_repair.diagnostics.actions import Context
from epistemic_repair.er2.state import RepairableAgentState
from epistemic_repair.failures.modes import FailureMode


ER2_HYPOTHESES: tuple[FailureMode, ...] = (
    FailureMode.NO_STRUCTURAL_CHANGE,
    FailureMode.WORLD_SHIFT,
    FailureMode.SENSOR_CORRUPTION,
    FailureMode.MISSING_LATENT_VARIABLE,
)


class PredictionTarget(str, Enum):
    """Observable scoring task used in the held-out repair suite."""

    PHYSICAL_OUTPUT = "PHYSICAL_OUTPUT"
    SENSOR_MAPPING = "SENSOR_MAPPING"
    END_TO_END_OBSERVATION = "END_TO_END_OBSERVATION"


@dataclass(frozen=True, slots=True)
class RepairEvaluationCase:
    """One deterministic held-out prediction with an evaluation-only target."""

    case_id: str
    target: PredictionTarget
    expected: int
    affected: bool
    x: int | None = None
    y: int | None = None
    context: Context | None = None


@dataclass(frozen=True, slots=True)
class PostRepairMetrics:
    """Behavioral repair scores over affected and unaffected held-out cases."""

    pre_repair_accuracy: float
    post_repair_accuracy: float
    affected_region_accuracy: float
    unaffected_region_accuracy: float
    pre_repair_unaffected_accuracy: float
    collateral_damage: float
    repair_success: bool
    affected_case_count: int
    unaffected_case_count: int


def build_evaluation_suite(hypothesis: FailureMode) -> tuple[RepairEvaluationCase, ...]:
    """Build the fixed ten-case suite for one hidden ER-2 hypothesis."""
    _validate_hypothesis(hypothesis)
    cases: list[RepairEvaluationCase] = []
    for context in Context:
        for x in (0, 1):
            true_y = _true_y(hypothesis, x, context)
            true_o = _true_o(hypothesis, true_y)
            healthy_y = x
            healthy_o = x
            physical_id = f"physical_{context.value}_x{x}"
            observation_id = f"end_to_end_{context.value}_x{x}"
            cases.append(RepairEvaluationCase(
                case_id=physical_id,
                target=PredictionTarget.PHYSICAL_OUTPUT,
                expected=true_y,
                affected=_is_affected(
                    hypothesis, physical_id, true_y != healthy_y
                ),
                x=x,
                context=context,
            ))
            cases.append(RepairEvaluationCase(
                case_id=observation_id,
                target=PredictionTarget.END_TO_END_OBSERVATION,
                expected=true_o,
                affected=_is_affected(
                    hypothesis, observation_id, true_o != healthy_o
                ),
                x=x,
                context=context,
            ))
    for y in (0, 1):
        true_o = _true_o(hypothesis, y)
        case_id = f"sensor_mapping_y{y}"
        cases.append(RepairEvaluationCase(
            case_id=case_id,
            target=PredictionTarget.SENSOR_MAPPING,
            expected=true_o,
            affected=_is_affected(hypothesis, case_id, true_o != y),
            y=y,
        ))
    affected_count = sum(case.affected for case in cases)
    if len(cases) != 10 or affected_count == 0 or affected_count == len(cases):
        raise RuntimeError("ER-2 suite must contain nonempty affected and unaffected regions")
    return tuple(cases)


def evaluate_repaired_state(
    hypothesis: FailureMode,
    pre_repair_state: RepairableAgentState,
    post_repair_state: RepairableAgentState,
) -> PostRepairMetrics:
    """Evaluate a repaired state without granting ground truth to the policy."""
    suite = build_evaluation_suite(hypothesis)
    pre_scores = tuple(_case_correct(pre_repair_state, case) for case in suite)
    post_scores = tuple(_case_correct(post_repair_state, case) for case in suite)
    affected_indices = tuple(index for index, case in enumerate(suite) if case.affected)
    unaffected_indices = tuple(index for index, case in enumerate(suite) if not case.affected)
    pre_unaffected = mean(pre_scores[index] for index in unaffected_indices)
    post_unaffected = mean(post_scores[index] for index in unaffected_indices)
    affected_accuracy = mean(post_scores[index] for index in affected_indices)
    collateral_damage = pre_unaffected - post_unaffected
    return PostRepairMetrics(
        pre_repair_accuracy=mean(pre_scores),
        post_repair_accuracy=mean(post_scores),
        affected_region_accuracy=affected_accuracy,
        unaffected_region_accuracy=post_unaffected,
        pre_repair_unaffected_accuracy=pre_unaffected,
        collateral_damage=collateral_damage,
        repair_success=(
            affected_accuracy == 1.0
            and collateral_damage <= 0.0
            and mean(post_scores) >= mean(pre_scores)
        ),
        affected_case_count=len(affected_indices),
        unaffected_case_count=len(unaffected_indices),
    )


def predictions_for_cases(
    state: RepairableAgentState,
    cases: Iterable[RepairEvaluationCase],
) -> tuple[tuple[str, int, int, bool], ...]:
    """Return case ID, prediction, target, and correctness for audit reports."""
    return tuple(
        (case.case_id, _predict_case(state, case), case.expected, _case_correct(state, case))
        for case in cases
    )


def _predict_case(state: RepairableAgentState, case: RepairEvaluationCase) -> int:
    if case.target is PredictionTarget.PHYSICAL_OUTPUT:
        assert case.x is not None and case.context is not None
        return state.predict_physical_output(case.x, case.context)
    if case.target is PredictionTarget.SENSOR_MAPPING:
        assert case.y is not None
        return state.predict_primary_from_physical(case.y)
    assert case.x is not None and case.context is not None
    return state.predict_primary_observation(case.x, case.context)


def _case_correct(state: RepairableAgentState, case: RepairEvaluationCase) -> bool:
    return _predict_case(state, case) == case.expected


def _true_y(hypothesis: FailureMode, x: int, context: Context) -> int:
    if hypothesis is FailureMode.WORLD_SHIFT:
        return 1 - x
    if hypothesis is FailureMode.MISSING_LATENT_VARIABLE and context is Context.B:
        return 1 - x
    return x


def _true_o(hypothesis: FailureMode, y: int) -> int:
    return 1 - y if hypothesis is FailureMode.SENSOR_CORRUPTION else y


def _is_affected(hypothesis: FailureMode, case_id: str, differs_from_healthy: bool) -> bool:
    if hypothesis is FailureMode.NO_STRUCTURAL_CHANGE:
        # A transient anomaly has no persistent changed cases. Two held-out
        # trigger-adjacent probes test whether the agent preserves the healthy rule.
        return case_id in {"physical_A_x1", "end_to_end_A_x1"}
    return differs_from_healthy


def _validate_hypothesis(hypothesis: FailureMode) -> None:
    if not isinstance(hypothesis, FailureMode):
        raise TypeError("hypothesis must be a FailureMode")
    if hypothesis not in ER2_HYPOTHESES:
        raise ValueError("ER-2 supports exactly the four ER-1 V2 hypotheses")
