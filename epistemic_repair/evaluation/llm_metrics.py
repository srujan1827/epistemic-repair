"""Metrics for LLM investigator conditions without pooling their results."""

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from epistemic_repair.evaluation.llm_runner import LLMEpisodeResult
from epistemic_repair.llm.schemas import FullAutonomousDecision, LLMCondition


@dataclass(frozen=True, slots=True)
class LLMConditionSummary:
    """Aggregate performance for exactly one LLM experimental condition."""

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
    oracle_action_agreement: float
    valid_decision_rate: float
    belief_normalization_validity: float | None
    mean_final_confidence: float | None
    mean_autonomous_belief_l1_error: float | None


def summarize_llm_results(
    results: Sequence[LLMEpisodeResult],
) -> LLMConditionSummary:
    """Summarize non-empty results from one condition only."""
    if not results:
        raise ValueError("at least one LLM episode result is required")
    conditions = {result.run_metadata.condition for result in results}
    if len(conditions) != 1:
        raise ValueError("FULL_AUTONOMOUS and PLANNER_ONLY results cannot be pooled")
    condition = next(iter(conditions))
    episode_count = len(results)
    action_turns = [
        turn
        for result in results
        for turn in result.trace
        if turn.experiment_record is not None
    ]
    all_turns = [turn for result in results for turn in result.trace]
    valid_turns = sum(turn.policy_result.succeeded for turn in all_turns)
    confidences = [
        result.final_confidence
        for result in results
        if result.final_confidence is not None
    ]
    belief_errors = [
        turn.autonomous_belief_l1_error
        for turn in all_turns
        if turn.autonomous_belief_l1_error is not None
        and isinstance(turn.policy_result.decision, FullAutonomousDecision)
    ]
    return LLMConditionSummary(
        condition=condition,
        episode_count=episode_count,
        diagnosis_accuracy=sum(result.diagnosis_correct for result in results)
        / episode_count,
        success_within_budget=sum(
            result.success_within_budget for result in results
        )
        / episode_count,
        mean_experiments=mean(result.experiments_used for result in results),
        mean_action_regret=mean(
            result.cumulative_action_regret for result in results
        ),
        success_at_1=_success_at(results, 1),
        success_at_2=_success_at(results, 2),
        success_at_3=_success_at(results, 3),
        success_at_5=_success_at(results, 5),
        oracle_action_agreement=(
            sum(turn.oracle_action_agreement is True for turn in action_turns)
            / len(action_turns)
            if action_turns
            else 0.0
        ),
        valid_decision_rate=valid_turns / len(all_turns) if all_turns else 0.0,
        belief_normalization_validity=(
            sum(
                isinstance(turn.policy_result.decision, FullAutonomousDecision)
                for turn in all_turns
            )
            / len(all_turns)
            if condition is LLMCondition.FULL_AUTONOMOUS and all_turns
            else None
        ),
        mean_final_confidence=mean(confidences) if confidences else None,
        mean_autonomous_belief_l1_error=(
            mean(belief_errors) if belief_errors else None
        ),
    )


def _success_at(results: Sequence[LLMEpisodeResult], budget: int) -> float:
    return sum(
        result.diagnosis_correct and result.experiments_used <= budget
        for result in results
    ) / len(results)
