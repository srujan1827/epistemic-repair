"""Aggregate metrics for stochastic ER-1 LLM investigator episodes."""

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from epistemic_repair.evaluation.er1_llm_runner import ER1LLMEpisodeResult
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.schemas import ER1FullAutonomousDecision, LLMCondition


@dataclass(frozen=True, slots=True)
class ER1LLMConditionSummary:
    condition: LLMCondition
    episode_count: int
    diagnosis_accuracy: float
    success_within_budget: float
    mean_experiments: float
    mean_action_regret: float
    success_at_1: float
    success_at_2: float
    success_at_3: float
    success_at_5: float
    success_at_8: float
    oracle_action_agreement: float
    valid_decision_rate: float
    belief_normalization_validity: float | None
    mean_final_confidence: float | None
    mean_autonomous_belief_l1_error: float | None
    false_structural_diagnosis_rate: float
    missed_structural_failure_rate: float
    premature_diagnosis_rate: float


def summarize_er1_llm_results(
    results: Sequence[ER1LLMEpisodeResult],
) -> ER1LLMConditionSummary:
    """Summarize one ER-1 LLM condition without pooling conditions."""
    if not results:
        raise ValueError("at least one ER-1 LLM result is required")
    conditions = {result.run_metadata.condition for result in results}
    if len(conditions) != 1:
        raise ValueError("FULL_AUTONOMOUS and PLANNER_ONLY cannot be pooled")
    condition = next(iter(conditions))
    turns = [turn for result in results for turn in result.trace]
    action_turns = [turn for turn in turns if turn.experiment_record is not None]
    no_change = [
        result
        for result in results
        if result.evaluation_metadata.hidden_failure_mode
        is FailureMode.NO_STRUCTURAL_CHANGE
    ]
    structural = [
        result
        for result in results
        if result.evaluation_metadata.hidden_failure_mode
        is not FailureMode.NO_STRUCTURAL_CHANGE
    ]
    confidences = [
        result.final_confidence
        for result in results
        if result.final_confidence is not None
    ]
    belief_errors = [
        turn.autonomous_belief_l1_error
        for turn in turns
        if turn.autonomous_belief_l1_error is not None
    ]
    return ER1LLMConditionSummary(
        condition=condition,
        episode_count=len(results),
        diagnosis_accuracy=mean(result.diagnosis_correct for result in results),
        success_within_budget=mean(
            result.success_within_budget for result in results
        ),
        mean_experiments=mean(result.experiments_used for result in results),
        mean_action_regret=mean(
            result.cumulative_action_regret for result in results
        ),
        success_at_1=_success_at(results, 1),
        success_at_2=_success_at(results, 2),
        success_at_3=_success_at(results, 3),
        success_at_5=_success_at(results, 5),
        success_at_8=_success_at(results, 8),
        oracle_action_agreement=(
            mean(turn.oracle_action_agreement is True for turn in action_turns)
            if action_turns
            else 0.0
        ),
        valid_decision_rate=(
            mean(turn.policy_result.succeeded for turn in turns) if turns else 0.0
        ),
        belief_normalization_validity=(
            mean(
                isinstance(turn.policy_result.decision, ER1FullAutonomousDecision)
                for turn in turns
            )
            if condition is LLMCondition.FULL_AUTONOMOUS and turns
            else None
        ),
        mean_final_confidence=mean(confidences) if confidences else None,
        mean_autonomous_belief_l1_error=(
            mean(belief_errors) if belief_errors else None
        ),
        false_structural_diagnosis_rate=(
            mean(
                result.final_diagnosis is not None
                and result.final_diagnosis is not FailureMode.NO_STRUCTURAL_CHANGE
                for result in no_change
            )
            if no_change
            else 0.0
        ),
        missed_structural_failure_rate=(
            mean(
                result.final_diagnosis is FailureMode.NO_STRUCTURAL_CHANGE
                for result in structural
            )
            if structural
            else 0.0
        ),
        premature_diagnosis_rate=mean(
            result.premature_diagnosis for result in results
        ),
    )


def _success_at(results: Sequence[ER1LLMEpisodeResult], budget: int) -> float:
    return mean(
        result.diagnosis_correct
        and result.final_diagnosis is not None
        and result.experiments_used <= budget
        for result in results
    )
