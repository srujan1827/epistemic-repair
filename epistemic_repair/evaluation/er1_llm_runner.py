"""Trusted ER-1 interaction loop for stochastic LLM investigators."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticLikelihoodModel,
    stochastic_information_gains,
)
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.diagnostics.results import StochasticBenchmarkExperimentResult
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_DEFAULT_CONFIG, ER1_HYPOTHESES
from epistemic_repair.evaluation.llm_runner import (
    LLMRunMetadata,
    LLMTerminationReason,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import (
    DecisionType,
    ER1FullAutonomousDecision,
    ER1LLMDecision,
    ER1PlannerOnlyDecision,
    LLMCondition,
)
from epistemic_repair.policies.llm import (
    ER1FullAutonomousLLMPolicy,
    ER1PlannerOnlyLLMPolicy,
    LLMPolicyResult,
)
from epistemic_repair.policies.stochastic_views import (
    StochasticAgentExperimentRecord,
    StochasticBenchmarkAgentView,
)
from epistemic_repair.prompts.binary_er1 import BINARY_ER1_PROMPT_VERSION


ER1_BENCHMARK_VERSION = "binary_er1"


@dataclass(frozen=True, slots=True)
class ER1LLMEvaluationMetadata:
    """ER-1 truth and seed attached outside the model-visible trace."""

    hidden_failure_mode: FailureMode
    episode_seed: int


@dataclass(frozen=True, slots=True)
class ER1LLMTurnTrace:
    """One ER-1 LLM call with safe input and privileged analysis fields."""

    call_number: int
    agent_view: StochasticBenchmarkAgentView
    policy_result: LLMPolicyResult
    normative_prior: StochasticHypothesisBeliefs
    normative_posterior: StochasticHypothesisBeliefs | None
    experiment_record: StochasticAgentExperimentRecord | None
    action_regret: float | None
    oracle_action_agreement: bool | None
    autonomous_belief_l1_error: float | None


@dataclass(frozen=True, slots=True)
class ER1LLMEpisodeResult:
    """Complete ER-1 LLM episode and stochastic diagnosis evaluation."""

    run_metadata: LLMRunMetadata
    evaluation_metadata: ER1LLMEvaluationMetadata
    initial_observation: Observation
    initial_beliefs: StochasticHypothesisBeliefs
    trace: tuple[ER1LLMTurnTrace, ...]
    final_diagnosis: FailureMode | None
    final_confidence: float | None
    termination_reason: LLMTerminationReason
    experiments_used: int
    decision_calls: int
    total_retries: int
    cumulative_action_regret: float
    oracle_action_agreements: int
    success_within_budget: bool
    premature_diagnosis: bool

    @property
    def diagnosis_correct(self) -> bool:
        return self.final_diagnosis is self.evaluation_metadata.hidden_failure_mode


ER1LLMPolicy = ER1FullAutonomousLLMPolicy | ER1PlannerOnlyLLMPolicy


class ER1LLMEpisodeRunner:
    """Execute stochastic ER-1 LLM decisions without exposing environment or truth."""

    def __init__(
        self,
        *,
        condition: LLMCondition,
        experiment_budget: int = 5,
        diagnosis_threshold: float = ER1_DEFAULT_CONFIG.diagnosis_threshold,
        episode_seed: int = 0,
        likelihood_model: StochasticLikelihoodModel | None = None,
        timestamp_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if type(experiment_budget) is not int or experiment_budget <= 0:
            raise ValueError("experiment_budget must be a positive integer")
        if not 0.0 < diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        if type(episode_seed) is not int:
            raise ValueError("episode_seed must be an integer")
        self._condition = condition
        self._experiment_budget = experiment_budget
        self._diagnosis_threshold = diagnosis_threshold
        self._episode_seed = episode_seed
        self._likelihood_model = likelihood_model or StochasticLikelihoodModel()
        self._timestamp_factory = timestamp_factory or (
            lambda: datetime.now(timezone.utc)
        )

    def run(
        self,
        env: StochasticBinaryMachine,
        hidden_failure_mode: FailureMode,
        policy: ER1LLMPolicy,
    ) -> ER1LLMEpisodeResult:
        """Run one ER-1 LLM episode with a fixed conditioned entry anomaly."""
        self._validate_policy(policy)
        if hidden_failure_mode not in ER1_HYPOTHESES:
            raise ValueError("hidden_failure_mode must be an ER-1 hypothesis")
        config = policy.config
        env.reset(hidden_failure_mode, episode_seed=self._episode_seed)
        initial_observation = env.initial_anomaly(x=self._likelihood_model.input_x)
        beliefs = self._likelihood_model.conditioned_initial_beliefs()
        initial_beliefs = beliefs
        context = Context.B
        history: list[StochasticAgentExperimentRecord] = []
        trace: list[ER1LLMTurnTrace] = []
        final_diagnosis: FailureMode | None = None
        final_confidence: float | None = None
        premature = False
        termination = LLMTerminationReason.DECISION_CALL_BUDGET_EXHAUSTED

        while len(trace) < config.max_decision_calls:
            view = StochasticBenchmarkAgentView(
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
                    self._trace(
                        trace,
                        view,
                        policy_result,
                        beliefs,
                    )
                )
                termination = LLMTerminationReason.MODEL_FAILURE
                break
            if not isinstance(
                decision,
                (ER1FullAutonomousDecision, ER1PlannerOnlyDecision),
            ):
                raise TypeError("ER-1 policy returned a non-ER-1 decision")
            belief_error = self._autonomous_belief_error(decision, beliefs)
            if decision.decision is DecisionType.DIAGNOSE:
                final_diagnosis = decision.diagnosis
                premature = beliefs.confidence() < self._diagnosis_threshold
                if isinstance(decision, ER1FullAutonomousDecision):
                    final_confidence = decision.confidence
                else:
                    assert final_diagnosis is not None
                    final_confidence = beliefs.probability(final_diagnosis)
                trace.append(
                    self._trace(
                        trace,
                        view,
                        policy_result,
                        beliefs,
                        posterior=beliefs,
                        belief_error=belief_error,
                    )
                )
                termination = LLMTerminationReason.DIAGNOSED
                break
            if len(history) >= self._experiment_budget:
                trace.append(
                    self._trace(
                        trace,
                        view,
                        policy_result,
                        beliefs,
                        belief_error=belief_error,
                    )
                )
                termination = LLMTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
                break

            assert decision.action is not None
            scores = stochastic_information_gains(
                beliefs, self._likelihood_model, context
            )
            result, context_after = self._run_action(
                env, decision.action, context
            )
            outcome = self._likelihood_model.outcome_from_result(
                decision.action, result, context
            )
            posterior = self._likelihood_model.update(
                beliefs, decision.action, outcome, context
            )
            regret = scores.best_value() - scores.for_action(decision.action)
            if abs(regret) < 1e-12:
                regret = 0.0
            agreement = abs(regret) < 1e-12
            record = StochasticAgentExperimentRecord(
                step_number=len(history) + 1,
                action=decision.action,
                result=result,
                context_before=context,
                context_after=context_after,
            )
            history.append(record)
            trace.append(
                self._trace(
                    trace,
                    view,
                    policy_result,
                    beliefs,
                    posterior=posterior,
                    experiment_record=record,
                    action_regret=regret,
                    agreement=agreement,
                    belief_error=belief_error,
                )
            )
            beliefs = posterior
            context = context_after

        truth = env.get_ground_truth().failure_mode
        correct = final_diagnosis is truth
        return ER1LLMEpisodeResult(
            run_metadata=self._metadata(config),
            evaluation_metadata=ER1LLMEvaluationMetadata(
                hidden_failure_mode=truth,
                episode_seed=self._episode_seed,
            ),
            initial_observation=initial_observation,
            initial_beliefs=initial_beliefs,
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
            premature_diagnosis=premature,
        )

    @staticmethod
    def _trace(
        trace: list[ER1LLMTurnTrace],
        view: StochasticBenchmarkAgentView,
        policy_result: LLMPolicyResult,
        prior: StochasticHypothesisBeliefs,
        *,
        posterior: StochasticHypothesisBeliefs | None = None,
        experiment_record: StochasticAgentExperimentRecord | None = None,
        action_regret: float | None = None,
        agreement: bool | None = None,
        belief_error: float | None = None,
    ) -> ER1LLMTurnTrace:
        return ER1LLMTurnTrace(
            call_number=len(trace) + 1,
            agent_view=view,
            policy_result=policy_result,
            normative_prior=prior,
            normative_posterior=posterior,
            experiment_record=experiment_record,
            action_regret=action_regret,
            oracle_action_agreement=agreement,
            autonomous_belief_l1_error=belief_error,
        )

    def _request_decision(
        self,
        policy: ER1LLMPolicy,
        view: StochasticBenchmarkAgentView,
        beliefs: StochasticHypothesisBeliefs,
    ) -> LLMPolicyResult:
        if isinstance(policy, ER1FullAutonomousLLMPolicy):
            return policy.decide(view)
        assert isinstance(policy, ER1PlannerOnlyLLMPolicy)
        return policy.decide(view, beliefs)

    def _run_action(
        self,
        env: StochasticBinaryMachine,
        action: DiagnosticAction,
        context: Context,
    ) -> tuple[StochasticBenchmarkExperimentResult, Context]:
        x = self._likelihood_model.input_x
        if action is DiagnosticAction.CHANGE_CONTEXT:
            target = self._likelihood_model.target_context(context)
            return env.run_experiment(action, x=x, context=target), target
        return env.run_experiment(action, x=x), context

    def _validate_policy(self, policy: ER1LLMPolicy) -> None:
        if self._condition is LLMCondition.FULL_AUTONOMOUS and not isinstance(
            policy, ER1FullAutonomousLLMPolicy
        ):
            raise TypeError("FULL_AUTONOMOUS requires ER1FullAutonomousLLMPolicy")
        if self._condition is LLMCondition.PLANNER_ONLY and not isinstance(
            policy, ER1PlannerOnlyLLMPolicy
        ):
            raise TypeError("PLANNER_ONLY requires ER1PlannerOnlyLLMPolicy")

    @staticmethod
    def _autonomous_belief_error(
        decision: ER1LLMDecision,
        beliefs: StochasticHypothesisBeliefs,
    ) -> float | None:
        if not isinstance(decision, ER1FullAutonomousDecision):
            return None
        return sum(
            abs(
                decision.beliefs.probability(hypothesis)
                - beliefs.probability(hypothesis)
            )
            for hypothesis in ER1_HYPOTHESES
        )

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
            prompt_version=BINARY_ER1_PROMPT_VERSION,
            timestamp_utc=timestamp.astimezone(timezone.utc).isoformat(),
            benchmark_version=ER1_BENCHMARK_VERSION,
            condition=self._condition,
            experiment_budget=self._experiment_budget,
            max_decision_calls=config.max_decision_calls,
            episode_seed=self._episode_seed,
        )
