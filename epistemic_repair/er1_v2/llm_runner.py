"""Trusted ER-1 V2 interaction loop for LLM investigators."""

from dataclasses import dataclass
from datetime import datetime, timezone

from epistemic_repair.beliefs.stochastic_likelihoods import stochastic_information_gains
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, Context, DiagnosticAction
from epistemic_repair.diagnostics.results import StochasticBenchmarkExperimentResult
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.likelihoods import ER1V2LikelihoodModel
from epistemic_repair.er1_v2.llm_policy import (
    ER1V2FullAutonomousLLMPolicy,
    ER1V2PlannerOnlyLLMPolicy,
    ER1V2ThresholdAwareAutonomousLLMPolicy,
)
from epistemic_repair.er1_v2.trigger_model import TriggerLikelihoodModel
from epistemic_repair.er1_v2.views import ER1V2BenchmarkAgentView
from epistemic_repair.evaluation.llm_runner import LLMRunMetadata, LLMTerminationReason
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import (
    DecisionType,
    ER1FullAutonomousDecision,
    ER1LLMDecision,
    ER1PlannerOnlyDecision,
    LLMCondition,
)
from epistemic_repair.policies.llm import LLMPolicyResult
from epistemic_repair.policies.stochastic_views import StochasticAgentExperimentRecord


ER1_V2_BENCHMARK_VERSION = "binary_er1_v2"


@dataclass(frozen=True, slots=True)
class ER1V2LLMEvaluationMetadata:
    """Hidden V2 truth attached only after model interaction."""

    hidden_failure_mode: FailureMode
    episode_seed: int
    diagnosis_threshold: float


@dataclass(frozen=True, slots=True)
class ER1V2DiagnosisEvaluation:
    """Orthogonal correctness, budget, threshold-support, and timing labels."""

    diagnosis_correct: bool
    diagnosed_correctly_within_budget: bool
    threshold_qualified_success: bool
    premature_diagnosis: bool
    normative_probability_of_final_diagnosis: float | None


@dataclass(frozen=True, slots=True)
class ER1V2LLMTurnTrace:
    """One model call, its exact safe view, and evaluation-only analysis."""

    call_number: int
    agent_view: ER1V2BenchmarkAgentView
    policy_result: LLMPolicyResult
    normative_prior: StochasticHypothesisBeliefs
    normative_posterior: StochasticHypothesisBeliefs | None
    experiment_record: StochasticAgentExperimentRecord | None
    action_regret: float | None
    oracle_action_agreement: bool | None
    autonomous_belief_l1_error: float | None


@dataclass(frozen=True, slots=True)
class ER1V2LLMEpisodeResult:
    """Complete V2 LLM episode with privileged truth kept in metadata."""

    run_metadata: LLMRunMetadata
    evaluation_metadata: ER1V2LLMEvaluationMetadata
    initial_observation: Observation
    initial_beliefs: StochasticHypothesisBeliefs
    trace: tuple[ER1V2LLMTurnTrace, ...]
    final_diagnosis: FailureMode | None
    final_confidence: float | None
    termination_reason: LLMTerminationReason
    experiments_used: int
    decision_calls: int
    total_retries: int
    cumulative_action_regret: float
    oracle_action_agreements: int
    success_within_budget: bool
    threshold_qualified_success: bool
    premature_diagnosis: bool
    normative_probability_of_final_diagnosis: float | None

    @property
    def diagnosis_correct(self) -> bool:
        return self.final_diagnosis is self.evaluation_metadata.hidden_failure_mode

    @property
    def diagnosed_correctly_within_budget(self) -> bool:
        """Explicit V2 label for the historical success_within_budget field."""
        return self.success_within_budget


def evaluate_er1_v2_diagnosis(
    *,
    final_diagnosis: FailureMode | None,
    hidden_failure_mode: FailureMode,
    normative_beliefs: StochasticHypothesisBeliefs,
    diagnosis_threshold: float,
    termination_reason: LLMTerminationReason,
) -> ER1V2DiagnosisEvaluation:
    """Classify a V2 diagnosis using support for the chosen hypothesis itself."""
    if not 0.0 < diagnosis_threshold <= 1.0:
        raise ValueError("diagnosis_threshold must be in (0, 1]")
    if final_diagnosis is None:
        return ER1V2DiagnosisEvaluation(
            diagnosis_correct=False,
            diagnosed_correctly_within_budget=False,
            threshold_qualified_success=False,
            premature_diagnosis=False,
            normative_probability_of_final_diagnosis=None,
        )
    support = normative_beliefs.probability(final_diagnosis)
    correct = final_diagnosis is hidden_failure_mode
    diagnosed = termination_reason is LLMTerminationReason.DIAGNOSED
    return ER1V2DiagnosisEvaluation(
        diagnosis_correct=correct,
        diagnosed_correctly_within_budget=correct and diagnosed,
        threshold_qualified_success=correct and support >= diagnosis_threshold,
        premature_diagnosis=support < diagnosis_threshold,
        normative_probability_of_final_diagnosis=support,
    )


ER1V2LLMPolicy = (
    ER1V2FullAutonomousLLMPolicy
    | ER1V2PlannerOnlyLLMPolicy
    | ER1V2ThresholdAwareAutonomousLLMPolicy
)


class ER1V2LLMEpisodeRunner:
    """Condition on the trigger once, then execute only safe persistent actions."""

    def __init__(
        self,
        *,
        condition: LLMCondition,
        experiment_budget: int = 5,
        diagnosis_threshold: float = 0.90,
        episode_seed: int = 0,
        trigger_model: TriggerLikelihoodModel | None = None,
        investigation_model: ER1V2LikelihoodModel | None = None,
    ) -> None:
        if type(experiment_budget) is not int or experiment_budget <= 0:
            raise ValueError("experiment_budget must be a positive integer")
        if not 0.0 < diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        self._condition = condition
        self._experiment_budget = experiment_budget
        self._diagnosis_threshold = diagnosis_threshold
        self._episode_seed = episode_seed
        self._trigger_model = trigger_model or TriggerLikelihoodModel()
        self._investigation_model = investigation_model or ER1V2LikelihoodModel()

    def run(
        self,
        env: ER1V2BinaryMachine,
        hidden_failure_mode: FailureMode,
        policy: ER1V2LLMPolicy,
    ) -> ER1V2LLMEpisodeResult:
        self._validate_policy(policy)
        if hidden_failure_mode not in ER1_HYPOTHESES:
            raise ValueError("hidden_failure_mode must be an ER-1 V2 hypothesis")
        config = policy.config
        env.reset(hidden_failure_mode, episode_seed=self._episode_seed)
        trigger = env.trigger_observation()
        beliefs = self._trigger_model.conditioned_beliefs()
        initial_beliefs = beliefs
        context = Context.B
        history: list[StochasticAgentExperimentRecord] = []
        trace: list[ER1V2LLMTurnTrace] = []
        diagnosis = None
        confidence = None
        termination = LLMTerminationReason.DECISION_CALL_BUDGET_EXHAUSTED

        while len(trace) < config.max_decision_calls:
            view = ER1V2BenchmarkAgentView(
                trigger_history=(trigger,),
                experiment_history=tuple(history),
                current_context=context,
                available_actions=BENCHMARK_ACTIONS,
                steps_remaining=self._experiment_budget - len(history),
            )
            policy_result = self._request(policy, view, beliefs)
            decision = policy_result.decision
            if decision is None:
                trace.append(self._turn(trace, view, policy_result, beliefs))
                termination = LLMTerminationReason.MODEL_FAILURE
                break
            if not isinstance(decision, (ER1FullAutonomousDecision, ER1PlannerOnlyDecision)):
                raise TypeError("ER-1 V2 policy returned a non-ER-1 decision")
            belief_error = self._belief_error(decision, beliefs)
            if decision.decision is DecisionType.DIAGNOSE:
                diagnosis = decision.diagnosis
                confidence = (
                    decision.confidence
                    if isinstance(decision, ER1FullAutonomousDecision)
                    else beliefs.probability(diagnosis)
                )
                trace.append(self._turn(trace, view, policy_result, beliefs, posterior=beliefs, belief_error=belief_error))
                termination = LLMTerminationReason.DIAGNOSED
                break
            if len(history) >= self._experiment_budget:
                trace.append(self._turn(trace, view, policy_result, beliefs, belief_error=belief_error))
                termination = LLMTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
                break

            assert decision.action is not None
            scores = stochastic_information_gains(beliefs, self._investigation_model, context)
            result, next_context = self._run_action(env, decision.action, context)
            outcome = self._investigation_model.outcome_from_result(decision.action, result, context)
            posterior = self._investigation_model.update(beliefs, decision.action, outcome, context)
            regret = scores.best_value() - scores.for_action(decision.action)
            if abs(regret) < 1e-12:
                regret = 0.0
            record = StochasticAgentExperimentRecord(
                step_number=len(history) + 1,
                action=decision.action,
                result=result,
                context_before=context,
                context_after=next_context,
            )
            history.append(record)
            trace.append(self._turn(
                trace, view, policy_result, beliefs,
                posterior=posterior, record=record, regret=regret,
                agreement=regret == 0.0, belief_error=belief_error,
            ))
            beliefs = posterior
            context = next_context

        truth = env.get_ground_truth().failure_mode
        diagnosis_evaluation = evaluate_er1_v2_diagnosis(
            final_diagnosis=diagnosis,
            hidden_failure_mode=truth,
            normative_beliefs=beliefs,
            diagnosis_threshold=self._diagnosis_threshold,
            termination_reason=termination,
        )
        now = datetime.now(timezone.utc).isoformat()
        metadata = LLMRunMetadata(
            provider=config.provider,
            model_id=config.model_id,
            thinking_level=config.thinking_level,
            max_output_tokens=config.max_output_tokens,
            request_timeout_seconds=config.request_timeout_seconds,
            prompt_version=policy.prompt_version,
            timestamp_utc=now,
            benchmark_version=ER1_V2_BENCHMARK_VERSION,
            condition=self._condition,
            experiment_budget=self._experiment_budget,
            max_decision_calls=config.max_decision_calls,
            episode_seed=self._episode_seed,
        )
        return ER1V2LLMEpisodeResult(
            run_metadata=metadata,
            evaluation_metadata=ER1V2LLMEvaluationMetadata(
                truth,
                self._episode_seed,
                self._diagnosis_threshold,
            ),
            initial_observation=trigger,
            initial_beliefs=initial_beliefs,
            trace=tuple(trace),
            final_diagnosis=diagnosis,
            final_confidence=confidence,
            termination_reason=termination,
            experiments_used=len(history),
            decision_calls=len(trace),
            total_retries=sum(turn.policy_result.retry_count for turn in trace),
            cumulative_action_regret=sum(turn.action_regret or 0.0 for turn in trace),
            oracle_action_agreements=sum(turn.oracle_action_agreement is True for turn in trace),
            success_within_budget=(
                diagnosis_evaluation.diagnosed_correctly_within_budget
            ),
            threshold_qualified_success=(
                diagnosis_evaluation.threshold_qualified_success
            ),
            premature_diagnosis=diagnosis_evaluation.premature_diagnosis,
            normative_probability_of_final_diagnosis=(
                diagnosis_evaluation.normative_probability_of_final_diagnosis
            ),
        )

    def _request(self, policy, view, beliefs):
        if isinstance(
            policy,
            (
                ER1V2FullAutonomousLLMPolicy,
                ER1V2ThresholdAwareAutonomousLLMPolicy,
            ),
        ):
            return policy.decide(view)
        return policy.decide(view, beliefs)

    def _run_action(self, env, action, context) -> tuple[StochasticBenchmarkExperimentResult, Context]:
        if action is DiagnosticAction.CHANGE_CONTEXT:
            target = self._investigation_model.target_context(context)
            return env.run_experiment(action, x=1, context=target), target
        return env.run_experiment(action, x=1), context

    def _validate_policy(self, policy) -> None:
        if self._condition is LLMCondition.FULL_AUTONOMOUS and not isinstance(policy, ER1V2FullAutonomousLLMPolicy):
            raise TypeError("FULL_AUTONOMOUS requires ER1V2FullAutonomousLLMPolicy")
        if self._condition is LLMCondition.PLANNER_ONLY and not isinstance(policy, ER1V2PlannerOnlyLLMPolicy):
            raise TypeError("PLANNER_ONLY requires ER1V2PlannerOnlyLLMPolicy")
        if self._condition is LLMCondition.THRESHOLD_AWARE_AUTONOMOUS:
            if not isinstance(policy, ER1V2ThresholdAwareAutonomousLLMPolicy):
                raise TypeError(
                    "THRESHOLD_AWARE_AUTONOMOUS requires "
                    "ER1V2ThresholdAwareAutonomousLLMPolicy"
                )
            if abs(policy.diagnosis_threshold - self._diagnosis_threshold) > 1e-12:
                raise ValueError(
                    "policy and runner diagnosis thresholds must match"
                )

    @staticmethod
    def _belief_error(decision: ER1LLMDecision, beliefs: StochasticHypothesisBeliefs) -> float | None:
        if not isinstance(decision, ER1FullAutonomousDecision):
            return None
        return sum(abs(decision.beliefs.probability(h) - beliefs.probability(h)) for h in ER1_HYPOTHESES)

    @staticmethod
    def _turn(trace, view, policy_result, prior, *, posterior=None, record=None, regret=None, agreement=None, belief_error=None):
        return ER1V2LLMTurnTrace(
            call_number=len(trace) + 1,
            agent_view=view,
            policy_result=policy_result,
            normative_prior=prior,
            normative_posterior=posterior,
            experiment_record=record,
            action_regret=regret,
            oracle_action_agreement=agreement,
            autonomous_belief_l1_error=belief_error,
        )
