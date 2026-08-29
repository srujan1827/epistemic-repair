"""Trusted bridge from an ER-2 LLM option choice to the frozen evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from epistemic_repair.er2.evaluation import ER2_HYPOTHESES, PostRepairMetrics
from epistemic_repair.er2.llm_policy import ER2CausalRepairLLMPolicy, ER2LLMPolicyResult
from epistemic_repair.er2.llm_prompts import (
    ER2LLMCondition,
    RepairOptionID,
    RepairOptionPermutation,
    option_permutation,
)
from epistemic_repair.er2.policies import FixedRepairPolicy
from epistemic_repair.er2.runner import ER2RepairEpisodeRunner
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.policies.llm import LLMAttemptStatus
from epistemic_repair.repair.operators import RepairOperator


class ER2LLMOutcome(str, Enum):
    """Disjoint outcome taxonomy for one model episode."""

    VALID_CORRECT_REPAIR = "VALID_CORRECT_REPAIR"
    VALID_WRONG_REPAIR = "VALID_WRONG_REPAIR"
    SCIENTIFIC_MODEL_FAILURE = "SCIENTIFIC_MODEL_FAILURE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    RATE_LIMIT_FAILURE = "RATE_LIMIT_FAILURE"


@dataclass(frozen=True, slots=True)
class ER2LLMRepairEpisodeResult:
    """Model selection, hidden translation, and deterministic evaluation result."""

    condition: ER2LLMCondition
    true_hypothesis: FailureMode
    supplied_diagnosis: FailureMode
    seed: int
    permutation: RepairOptionPermutation
    selected_option: RepairOptionID | None
    selected_repair: RepairOperator | None
    correct_repair: RepairOperator
    confidence: float | None
    rationale: str | None
    outcome: ER2LLMOutcome
    metrics: PostRepairMetrics | None
    policy_result: ER2LLMPolicyResult

    @property
    def valid_selection(self) -> bool:
        return self.selected_repair is not None

    @property
    def repair_selection_correct(self) -> bool | None:
        if not self.valid_selection:
            return None
        return self.selected_repair is self.correct_repair


class ER2LLMRepairEpisodeRunner:
    """Run one provider decision without giving the provider evaluator access."""

    def __init__(
        self,
        policy: ER2CausalRepairLLMPolicy,
        *,
        deterministic_runner: ER2RepairEpisodeRunner | None = None,
    ) -> None:
        self.policy = policy
        self._deterministic_runner = deterministic_runner or ER2RepairEpisodeRunner()

    def run(
        self,
        *,
        true_hypothesis: FailureMode,
        diagnosis: FailureMode,
        seed: int,
    ) -> ER2LLMRepairEpisodeResult:
        if true_hypothesis not in ER2_HYPOTHESES or diagnosis not in ER2_HYPOTHESES:
            raise ValueError("true_hypothesis and diagnosis must be ER-2 hypotheses")
        permutation = option_permutation(seed)
        policy_result = self.policy.decide(permutation.prompt_view(diagnosis))
        if policy_result.decision is None:
            return ER2LLMRepairEpisodeResult(
                condition=ER2LLMCondition.CAUSAL_REPAIR_SELECTION,
                true_hypothesis=true_hypothesis,
                supplied_diagnosis=diagnosis,
                seed=seed,
                permutation=permutation,
                selected_option=None,
                selected_repair=None,
                correct_repair=_correct_repair(true_hypothesis),
                confidence=None,
                rationale=None,
                outcome=_failure_outcome(policy_result),
                metrics=None,
                policy_result=policy_result,
            )

        decision = policy_result.decision
        selected_repair = permutation.repair_for(decision.selected_option)
        deterministic_result = self._deterministic_runner.run(
            true_hypothesis=true_hypothesis,
            diagnosis=diagnosis,
            policy=FixedRepairPolicy(selected_repair),
        )
        outcome = (
            ER2LLMOutcome.VALID_CORRECT_REPAIR
            if deterministic_result.repair_selection_correct
            else ER2LLMOutcome.VALID_WRONG_REPAIR
        )
        return ER2LLMRepairEpisodeResult(
            condition=ER2LLMCondition.CAUSAL_REPAIR_SELECTION,
            true_hypothesis=true_hypothesis,
            supplied_diagnosis=diagnosis,
            seed=seed,
            permutation=permutation,
            selected_option=decision.selected_option,
            selected_repair=selected_repair,
            correct_repair=deterministic_result.correct_repair,
            confidence=decision.confidence,
            rationale=decision.rationale,
            outcome=outcome,
            metrics=deterministic_result.metrics,
            policy_result=policy_result,
        )


def _failure_outcome(result: ER2LLMPolicyResult) -> ER2LLMOutcome:
    if not result.attempts:
        return ER2LLMOutcome.PROVIDER_FAILURE
    final = result.attempts[-1]
    if (
        final.status is LLMAttemptStatus.INVALID_FORMAT
        or final.error_type == "LLMFormatError"
    ):
        return ER2LLMOutcome.SCIENTIFIC_MODEL_FAILURE
    if final.error_type == "LLMRateLimitError":
        return ER2LLMOutcome.RATE_LIMIT_FAILURE
    return ER2LLMOutcome.PROVIDER_FAILURE


def _correct_repair(hypothesis: FailureMode) -> RepairOperator:
    # Use the same mapping as the frozen deterministic runner, without exposing it.
    from epistemic_repair.repair.operators import repair_for_failure

    return repair_for_failure(hypothesis)
