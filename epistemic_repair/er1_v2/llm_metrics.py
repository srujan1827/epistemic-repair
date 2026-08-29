"""Scientifically explicit aggregate metrics for ER-1 V2 LLM episodes."""

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from epistemic_repair.er1_v2.llm_runner import ER1V2LLMEpisodeResult
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.schemas import ER1FullAutonomousDecision, LLMCondition


@dataclass(frozen=True, slots=True)
class ER1V2LLMConditionSummary:
    """Separate classification, bounded completion, and epistemic success."""

    condition: LLMCondition
    episode_count: int
    diagnosis_accuracy: float
    diagnosed_correctly_within_budget: float
    threshold_qualified_success: float
    premature_diagnosis_rate: float
    mean_experiments: float
    mean_action_regret: float
    oracle_action_agreement: float
    false_structural_diagnosis_rate: float
    missed_structural_failure_rate: float
    valid_decision_rate: float
    belief_normalization_validity: float | None
    mean_final_confidence: float | None
    mean_autonomous_belief_l1_error: float | None

    @property
    def success_within_budget(self) -> float:
        """Compatibility alias retaining the historical bounded-success rate."""
        return self.diagnosed_correctly_within_budget


def summarize_er1_v2_llm_results(
    results: Sequence[ER1V2LLMEpisodeResult],
) -> ER1V2LLMConditionSummary:
    """Summarize one V2 condition without conflating its success concepts."""
    if not results:
        raise ValueError("at least one ER-1 V2 LLM result is required")
    conditions = {result.run_metadata.condition for result in results}
    if len(conditions) != 1:
        raise ValueError("different LLM conditions cannot be pooled")
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
    return ER1V2LLMConditionSummary(
        condition=condition,
        episode_count=len(results),
        diagnosis_accuracy=mean(result.diagnosis_correct for result in results),
        diagnosed_correctly_within_budget=mean(
            result.diagnosed_correctly_within_budget for result in results
        ),
        threshold_qualified_success=mean(
            result.threshold_qualified_success for result in results
        ),
        premature_diagnosis_rate=mean(
            result.premature_diagnosis for result in results
        ),
        mean_experiments=mean(result.experiments_used for result in results),
        mean_action_regret=mean(
            result.cumulative_action_regret for result in results
        ),
        oracle_action_agreement=(
            mean(turn.oracle_action_agreement is True for turn in action_turns)
            if action_turns
            else 0.0
        ),
        false_structural_diagnosis_rate=(
            mean(
                result.final_diagnosis is not None
                and result.final_diagnosis
                is not FailureMode.NO_STRUCTURAL_CHANGE
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
        valid_decision_rate=(
            mean(turn.policy_result.succeeded for turn in turns) if turns else 0.0
        ),
        belief_normalization_validity=(
            mean(
                isinstance(
                    turn.policy_result.decision,
                    ER1FullAutonomousDecision,
                )
                for turn in turns
            )
            if condition in (
                LLMCondition.FULL_AUTONOMOUS,
                LLMCondition.THRESHOLD_AWARE_AUTONOMOUS,
            )
            and turns
            else None
        ),
        mean_final_confidence=mean(confidences) if confidences else None,
        mean_autonomous_belief_l1_error=(
            mean(belief_errors) if belief_errors else None
        ),
    )
