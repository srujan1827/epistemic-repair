"""Trusted end-to-end ER-1 investigation -> ER-2 repair/evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.llm_policy import ER1V2FullAutonomousLLMPolicy
from epistemic_repair.er1_v2.llm_runner import (
    ER1V2LLMEpisodeResult,
    ER1V2LLMEpisodeRunner,
)
from epistemic_repair.er2.end_to_end_policy import EndToEndRepairLLMPolicy
from epistemic_repair.er2.end_to_end_prompts import end_to_end_repair_view
from epistemic_repair.er2.evaluation import ER2_HYPOTHESES, PostRepairMetrics
from epistemic_repair.er2.llm_policy import ER2LLMPolicyResult
from epistemic_repair.er2.llm_prompts import (
    RepairOptionID,
    RepairOptionPermutation,
    option_permutation,
)
from epistemic_repair.er2.policies import FixedRepairPolicy
from epistemic_repair.er2.runner import ER2RepairEpisodeRunner
from epistemic_repair.evaluation.llm_runner import LLMTerminationReason
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import LLMCondition
from epistemic_repair.policies.llm import LLMAttemptRecord, LLMAttemptStatus
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


class EndToEndOutcome(str, Enum):
    """Mutually exclusive diagnosis/repair chain and protocol outcomes."""

    CORRECT_DIAGNOSIS_CORRECT_REPAIR = "CORRECT_DIAGNOSIS_CORRECT_REPAIR"
    CORRECT_DIAGNOSIS_WRONG_REPAIR = "CORRECT_DIAGNOSIS_WRONG_REPAIR"
    WRONG_DIAGNOSIS_REPAIR_CONSISTENT_WITH_WRONG_DIAGNOSIS = (
        "WRONG_DIAGNOSIS_REPAIR_CONSISTENT_WITH_WRONG_DIAGNOSIS"
    )
    WRONG_DIAGNOSIS_BUT_CORRECT_REPAIR = "WRONG_DIAGNOSIS_BUT_CORRECT_REPAIR"
    WRONG_DIAGNOSIS_OTHER_WRONG_REPAIR = "WRONG_DIAGNOSIS_OTHER_WRONG_REPAIR"
    SCIENTIFIC_MODEL_FAILURE = "SCIENTIFIC_MODEL_FAILURE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    RATE_LIMIT_FAILURE = "RATE_LIMIT_FAILURE"


VALID_END_TO_END_OUTCOMES = frozenset({
    EndToEndOutcome.CORRECT_DIAGNOSIS_CORRECT_REPAIR,
    EndToEndOutcome.CORRECT_DIAGNOSIS_WRONG_REPAIR,
    EndToEndOutcome.WRONG_DIAGNOSIS_REPAIR_CONSISTENT_WITH_WRONG_DIAGNOSIS,
    EndToEndOutcome.WRONG_DIAGNOSIS_BUT_CORRECT_REPAIR,
    EndToEndOutcome.WRONG_DIAGNOSIS_OTHER_WRONG_REPAIR,
})


@dataclass(frozen=True, slots=True)
class CounterfactualRepairResult:
    """Zero-call deterministic outcome for one candidate repair."""

    repair: RepairOperator
    chosen: bool
    oracle: bool
    metrics: PostRepairMetrics


@dataclass(frozen=True, slots=True)
class EndToEndEpisodeResult:
    """Investigation, repair decision, and held-out repair outcome."""

    true_hypothesis: FailureMode
    seed: int
    investigation: ER1V2LLMEpisodeResult
    permutation: RepairOptionPermutation
    repair_policy_result: ER2LLMPolicyResult | None
    selected_option: RepairOptionID | None
    selected_repair: RepairOperator | None
    metrics: PostRepairMetrics | None
    outcome: EndToEndOutcome
    counterfactual_repairs: tuple[CounterfactualRepairResult, ...]

    @property
    def model_diagnosis(self) -> FailureMode | None:
        return self.investigation.final_diagnosis

    @property
    def diagnosis_correct(self) -> bool:
        return self.investigation.diagnosis_correct

    @property
    def repair_selection_correct(self) -> bool | None:
        if self.selected_repair is None:
            return None
        return self.selected_repair is repair_for_failure(self.true_hypothesis)

    @property
    def repair_consistent_with_model_diagnosis(self) -> bool | None:
        if self.selected_repair is None or self.model_diagnosis is None:
            return None
        return self.selected_repair is repair_for_failure(self.model_diagnosis)


class EndToEndEpisodeRunner:
    """Connect existing ER-1 V2 and ER-2 runners without changing either."""

    def __init__(
        self,
        client: LLMClient,
        config: LLMConfig,
        *,
        experiment_budget: int = 8,
        diagnosis_threshold: float = 0.95,
        deterministic_runner: ER2RepairEpisodeRunner | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._experiment_budget = experiment_budget
        self._diagnosis_threshold = diagnosis_threshold
        self._deterministic_runner = deterministic_runner or ER2RepairEpisodeRunner()

    def run(self, *, true_hypothesis: FailureMode, seed: int) -> EndToEndEpisodeResult:
        if true_hypothesis not in ER2_HYPOTHESES:
            raise ValueError("true_hypothesis must be one of the four ER-1 V2 hypotheses")
        investigation_policy = ER1V2FullAutonomousLLMPolicy(
            self._client, self._config
        )
        investigation_runner = ER1V2LLMEpisodeRunner(
            condition=LLMCondition.FULL_AUTONOMOUS,
            experiment_budget=self._experiment_budget,
            diagnosis_threshold=self._diagnosis_threshold,
            episode_seed=seed,
        )
        investigation = investigation_runner.run(
            ER1V2BinaryMachine(), true_hypothesis, investigation_policy
        )
        permutation = option_permutation(seed)
        if investigation.final_diagnosis is None or not investigation.trace:
            outcome = _investigation_failure_outcome(investigation)
            return EndToEndEpisodeResult(
                true_hypothesis,
                seed,
                investigation,
                permutation,
                None,
                None,
                None,
                None,
                outcome,
                (),
            )

        final_agent_view = investigation.trace[-1].agent_view
        repair_policy = EndToEndRepairLLMPolicy(self._client, self._config)
        repair_result = repair_policy.decide(
            end_to_end_repair_view(final_agent_view, permutation)
        )
        if repair_result.decision is None:
            outcome = _attempt_failure_outcome(repair_result.attempts)
            return EndToEndEpisodeResult(
                true_hypothesis,
                seed,
                investigation,
                permutation,
                repair_result,
                None,
                None,
                None,
                outcome,
                counterfactual_repairs(
                    true_hypothesis=true_hypothesis,
                    model_diagnosis=investigation.final_diagnosis,
                    selected_repair=None,
                    deterministic_runner=self._deterministic_runner,
                ),
            )

        decision = repair_result.decision
        selected_repair = permutation.repair_for(decision.selected_option)
        repair_episode = self._deterministic_runner.run(
            true_hypothesis=true_hypothesis,
            diagnosis=investigation.final_diagnosis,
            policy=FixedRepairPolicy(selected_repair),
        )
        outcome = classify_failure_chain(
            true_hypothesis=true_hypothesis,
            model_diagnosis=investigation.final_diagnosis,
            selected_repair=selected_repair,
        )
        return EndToEndEpisodeResult(
            true_hypothesis,
            seed,
            investigation,
            permutation,
            repair_result,
            decision.selected_option,
            selected_repair,
            repair_episode.metrics,
            outcome,
            counterfactual_repairs(
                true_hypothesis=true_hypothesis,
                model_diagnosis=investigation.final_diagnosis,
                selected_repair=selected_repair,
                deterministic_runner=self._deterministic_runner,
            ),
        )


def classify_failure_chain(
    *,
    true_hypothesis: FailureMode,
    model_diagnosis: FailureMode,
    selected_repair: RepairOperator,
) -> EndToEndOutcome:
    """Classify one valid diagnosis/repair pair into exactly one category."""
    diagnosis_correct = model_diagnosis is true_hypothesis
    repair_correct = selected_repair is repair_for_failure(true_hypothesis)
    if diagnosis_correct:
        return (
            EndToEndOutcome.CORRECT_DIAGNOSIS_CORRECT_REPAIR
            if repair_correct
            else EndToEndOutcome.CORRECT_DIAGNOSIS_WRONG_REPAIR
        )
    if repair_correct:
        return EndToEndOutcome.WRONG_DIAGNOSIS_BUT_CORRECT_REPAIR
    if selected_repair is repair_for_failure(model_diagnosis):
        return EndToEndOutcome.WRONG_DIAGNOSIS_REPAIR_CONSISTENT_WITH_WRONG_DIAGNOSIS
    return EndToEndOutcome.WRONG_DIAGNOSIS_OTHER_WRONG_REPAIR


def counterfactual_repairs(
    *,
    true_hypothesis: FailureMode,
    model_diagnosis: FailureMode,
    selected_repair: RepairOperator | None,
    deterministic_runner: ER2RepairEpisodeRunner | None = None,
) -> tuple[CounterfactualRepairResult, ...]:
    """Evaluate all repairs deterministically without an LLM or network call."""
    runner = deterministic_runner or ER2RepairEpisodeRunner()
    oracle = repair_for_failure(true_hypothesis)
    return tuple(
        CounterfactualRepairResult(
            repair=repair,
            chosen=repair is selected_repair,
            oracle=repair is oracle,
            metrics=runner.run(
                true_hypothesis=true_hypothesis,
                diagnosis=model_diagnosis,
                policy=FixedRepairPolicy(repair),
            ).metrics,
        )
        for repair in RepairOperator
    )


def _investigation_failure_outcome(
    investigation: ER1V2LLMEpisodeResult,
) -> EndToEndOutcome:
    attempts = tuple(
        attempt
        for turn in investigation.trace
        for attempt in turn.policy_result.attempts
    )
    if investigation.termination_reason is LLMTerminationReason.MODEL_FAILURE:
        return _attempt_failure_outcome(attempts)
    return EndToEndOutcome.SCIENTIFIC_MODEL_FAILURE


def _attempt_failure_outcome(
    attempts: tuple[LLMAttemptRecord, ...],
) -> EndToEndOutcome:
    if not attempts:
        return EndToEndOutcome.PROVIDER_FAILURE
    final = attempts[-1]
    if final.error_type == "LLMRateLimitError":
        return EndToEndOutcome.RATE_LIMIT_FAILURE
    if (
        final.status is LLMAttemptStatus.INVALID_FORMAT
        or final.error_type == "LLMFormatError"
    ):
        return EndToEndOutcome.SCIENTIFIC_MODEL_FAILURE
    if final.status in {
        LLMAttemptStatus.TRANSIENT_ERROR,
        LLMAttemptStatus.PROVIDER_ERROR,
    }:
        return EndToEndOutcome.PROVIDER_FAILURE
    return EndToEndOutcome.SCIENTIFIC_MODEL_FAILURE
