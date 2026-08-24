"""Trusted interaction loop for restricted LLM investigator policies."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from epistemic_repair.beliefs.likelihoods import (
    DeterministicLikelihoodModel,
    information_gains,
)
from epistemic_repair.beliefs.state import HYPOTHESES, HypothesisBeliefs
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.diagnostics.results import (
    BenchmarkExperimentResult,
    ChangeContextResult,
    RepeatTrialResult,
    TrustedSensorResult,
)
from epistemic_repair.environments.binary_machine import BinaryMachine, Observation
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import (
    DecisionType,
    FullAutonomousDecision,
    LLMCondition,
    LLMDecision,
)
from epistemic_repair.policies.llm import (
    FullAutonomousLLMPolicy,
    LLMPolicyResult,
    PlannerOnlyLLMPolicy,
)
from epistemic_repair.policies.views import AgentExperimentRecord, BenchmarkAgentView
from epistemic_repair.prompts.binary_v0 import BINARY_V0_PROMPT_VERSION


BENCHMARK_VERSION = "binary_v0"


class LLMTerminationReason(str, Enum):
    """Explicit terminal state for every LLM episode."""

    DIAGNOSED = "DIAGNOSED"
    MODEL_FAILURE = "MODEL_FAILURE"
    EXPERIMENT_BUDGET_EXHAUSTED = "EXPERIMENT_BUDGET_EXHAUSTED"
    DECISION_CALL_BUDGET_EXHAUSTED = "DECISION_CALL_BUDGET_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class LLMRunMetadata:
    """Reproducibility metadata safe to retain without evaluation ground truth."""

    provider: str
    model_id: str
    thinking_level: str
    max_output_tokens: int
    request_timeout_seconds: float | None
    prompt_version: str
    timestamp_utc: str
    benchmark_version: str
    condition: LLMCondition
    experiment_budget: int
    max_decision_calls: int
    episode_seed: int


@dataclass(frozen=True, slots=True)
class LLMEvaluationMetadata:
    """Outer evaluation-only metadata carrying the hidden condition label."""

    hidden_failure_mode: FailureMode


@dataclass(frozen=True, slots=True)
class LLMTurnTrace:
    """Evaluation trace containing safe request data and normative comparisons."""

    call_number: int
    agent_view: BenchmarkAgentView
    policy_result: LLMPolicyResult
    normative_prior: HypothesisBeliefs
    normative_posterior: HypothesisBeliefs | None
    experiment_record: AgentExperimentRecord | None
    action_regret: float | None
    oracle_action_agreement: bool | None
    autonomous_belief_l1_error: float | None


@dataclass(frozen=True, slots=True)
class LLMEpisodeResult:
    """Complete LLM interaction result with truth only at the evaluation boundary."""

    run_metadata: LLMRunMetadata
    evaluation_metadata: LLMEvaluationMetadata
    initial_observation: Observation
    trace: tuple[LLMTurnTrace, ...]
    final_diagnosis: FailureMode | None
    final_confidence: float | None
    termination_reason: LLMTerminationReason
    experiments_used: int
    decision_calls: int
    total_retries: int
    cumulative_action_regret: float
    oracle_action_agreements: int
    success_within_budget: bool

    @property
    def diagnosis_correct(self) -> bool:
        """Return whether a final diagnosis matches evaluation ground truth."""
        return (
            self.final_diagnosis
            is self.evaluation_metadata.hidden_failure_mode
        )


LLMPolicy = FullAutonomousLLMPolicy | PlannerOnlyLLMPolicy


class LLMEpisodeRunner:
    """Execute model decisions while keeping the environment in trusted Python."""

    def __init__(
        self,
        *,
        condition: LLMCondition,
        experiment_budget: int = 2,
        episode_seed: int = 0,
        likelihood_model: DeterministicLikelihoodModel | None = None,
        timestamp_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if type(experiment_budget) is not int or experiment_budget <= 0:
            raise ValueError("experiment_budget must be a positive integer")
        if type(episode_seed) is not int:
            raise ValueError("episode_seed must be an integer")
        self._condition = condition
        self._experiment_budget = experiment_budget
        self._episode_seed = episode_seed
        self._likelihood_model = likelihood_model or DeterministicLikelihoodModel()
        self._timestamp_factory = timestamp_factory or (
            lambda: datetime.now(timezone.utc)
        )

    def run(
        self,
        env: BinaryMachine,
        hidden_failure_mode: FailureMode,
        policy: LLMPolicy,
    ) -> LLMEpisodeResult:
        """Run an LLM episode without ever passing environment or truth to policy."""
        self._validate_policy(policy)
        if hidden_failure_mode not in HYPOTHESES:
            raise ValueError("hidden_failure_mode must be a benchmark hypothesis")

        config = policy.config
        env.reset(hidden_failure_mode)
        initial_observation = env.step(self._likelihood_model.input_x)
        if initial_observation != Observation(x=1, o=0):
            raise RuntimeError("LLM benchmark did not start from X=1 -> O=0")

        beliefs = HypothesisBeliefs.uniform()
        context = Context.B
        history: list[AgentExperimentRecord] = []
        trace: list[LLMTurnTrace] = []
        final_diagnosis: FailureMode | None = None
        final_confidence: float | None = None
        termination = LLMTerminationReason.DECISION_CALL_BUDGET_EXHAUSTED

        while len(trace) < config.max_decision_calls:
            view = BenchmarkAgentView(
                initial_history=(initial_observation,),
                experiment_history=tuple(history),
                current_context=context,
                available_actions=BENCHMARK_ACTIONS,
                steps_remaining=self._experiment_budget - len(history),
            )
            policy_result = self._request_decision(policy, view, beliefs)
            decision = policy_result.decision
            if decision is None:
                trace.append(
                    LLMTurnTrace(
                        call_number=len(trace) + 1,
                        agent_view=view,
                        policy_result=policy_result,
                        normative_prior=beliefs,
                        normative_posterior=None,
                        experiment_record=None,
                        action_regret=None,
                        oracle_action_agreement=None,
                        autonomous_belief_l1_error=None,
                    )
                )
                termination = LLMTerminationReason.MODEL_FAILURE
                break

            belief_error = self._autonomous_belief_error(decision, beliefs)
            if decision.decision is DecisionType.DIAGNOSE:
                final_diagnosis = decision.diagnosis
                if isinstance(decision, FullAutonomousDecision):
                    final_confidence = decision.confidence
                else:
                    final_confidence = beliefs.probability(final_diagnosis)
                trace.append(
                    LLMTurnTrace(
                        call_number=len(trace) + 1,
                        agent_view=view,
                        policy_result=policy_result,
                        normative_prior=beliefs,
                        normative_posterior=beliefs,
                        experiment_record=None,
                        action_regret=None,
                        oracle_action_agreement=None,
                        autonomous_belief_l1_error=belief_error,
                    )
                )
                termination = LLMTerminationReason.DIAGNOSED
                break

            if len(history) >= self._experiment_budget:
                trace.append(
                    LLMTurnTrace(
                        call_number=len(trace) + 1,
                        agent_view=view,
                        policy_result=policy_result,
                        normative_prior=beliefs,
                        normative_posterior=None,
                        experiment_record=None,
                        action_regret=None,
                        oracle_action_agreement=None,
                        autonomous_belief_l1_error=belief_error,
                    )
                )
                termination = LLMTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
                break

            assert decision.action is not None
            scores = information_gains(beliefs, self._likelihood_model, context)
            result, context_after = self._run_action(
                env, decision.action, context
            )
            outcome = self._likelihood_model.outcome_from_result(
                decision.action, result, context
            )
            posterior = self._likelihood_model.update(
                beliefs, decision.action, outcome, context
            )
            chosen_gain = scores.for_action(decision.action)
            regret = scores.best_value() - chosen_gain
            if abs(regret) < 1e-12:
                regret = 0.0
            agreement = abs(chosen_gain - scores.best_value()) < 1e-12
            record = AgentExperimentRecord(
                step_number=len(history) + 1,
                action=decision.action,
                result=result,
                context_before=context,
                context_after=context_after,
            )
            history.append(record)
            trace.append(
                LLMTurnTrace(
                    call_number=len(trace) + 1,
                    agent_view=view,
                    policy_result=policy_result,
                    normative_prior=beliefs,
                    normative_posterior=posterior,
                    experiment_record=record,
                    action_regret=regret,
                    oracle_action_agreement=agreement,
                    autonomous_belief_l1_error=belief_error,
                )
            )
            beliefs = posterior
            context = context_after

        # Ground truth is accessed only after every model interaction is complete.
        ground_truth = env.get_ground_truth().failure_mode
        correct = final_diagnosis is ground_truth
        return LLMEpisodeResult(
            run_metadata=self._metadata(config),
            evaluation_metadata=LLMEvaluationMetadata(ground_truth),
            initial_observation=initial_observation,
            trace=tuple(trace),
            final_diagnosis=final_diagnosis,
            final_confidence=final_confidence,
            termination_reason=termination,
            experiments_used=len(history),
            decision_calls=len(trace),
            total_retries=sum(turn.policy_result.retry_count for turn in trace),
            cumulative_action_regret=sum(
                turn.action_regret or 0.0 for turn in trace
            ),
            oracle_action_agreements=sum(
                turn.oracle_action_agreement is True for turn in trace
            ),
            success_within_budget=(
                correct and termination is LLMTerminationReason.DIAGNOSED
            ),
        )

    def _request_decision(
        self,
        policy: LLMPolicy,
        view: BenchmarkAgentView,
        beliefs: HypothesisBeliefs,
    ) -> LLMPolicyResult:
        if isinstance(policy, FullAutonomousLLMPolicy):
            return policy.decide(view)
        assert isinstance(policy, PlannerOnlyLLMPolicy)
        return policy.decide(view, beliefs)

    def _run_action(
        self,
        env: BinaryMachine,
        action: DiagnosticAction,
        context: Context,
    ) -> tuple[BenchmarkExperimentResult, Context]:
        x = self._likelihood_model.input_x
        if action is DiagnosticAction.REPEAT_TRIAL:
            result = env.run_experiment(action, x=x)
            assert isinstance(result, RepeatTrialResult)
            return result, context
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            result = env.run_experiment(action, x=x)
            assert isinstance(result, TrustedSensorResult)
            return result, context
        target = self._likelihood_model.target_context(context)
        result = env.run_experiment(action, x=x, context=target)
        assert isinstance(result, ChangeContextResult)
        return result, target

    def _validate_policy(self, policy: LLMPolicy) -> None:
        if self._condition is LLMCondition.FULL_AUTONOMOUS and not isinstance(
            policy, FullAutonomousLLMPolicy
        ):
            raise TypeError("FULL_AUTONOMOUS requires FullAutonomousLLMPolicy")
        if self._condition is LLMCondition.PLANNER_ONLY and not isinstance(
            policy, PlannerOnlyLLMPolicy
        ):
            raise TypeError("PLANNER_ONLY requires PlannerOnlyLLMPolicy")

    def _metadata(self, config: LLMConfig) -> LLMRunMetadata:
        timestamp = self._timestamp_factory()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return LLMRunMetadata(
            provider=config.provider,
            model_id=config.model_id,
            thinking_level=config.thinking_level,
            max_output_tokens=config.max_output_tokens,
            request_timeout_seconds=config.request_timeout_seconds,
            prompt_version=BINARY_V0_PROMPT_VERSION,
            timestamp_utc=timestamp.astimezone(timezone.utc).isoformat(),
            benchmark_version=BENCHMARK_VERSION,
            condition=self._condition,
            experiment_budget=self._experiment_budget,
            max_decision_calls=config.max_decision_calls,
            episode_seed=self._episode_seed,
        )

    @staticmethod
    def _autonomous_belief_error(
        decision: LLMDecision,
        normative_beliefs: HypothesisBeliefs,
    ) -> float | None:
        if not isinstance(decision, FullAutonomousDecision):
            return None
        return sum(
            abs(
                decision.beliefs.probability(hypothesis)
                - normative_beliefs.probability(hypothesis)
            )
            for hypothesis in HYPOTHESES
        )

