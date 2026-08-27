"""Trusted Bayesian episode runner for ER-1 V2."""

from dataclasses import dataclass

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticActionInformationGains,
    StochasticExperimentOutcome,
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
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.config import ER1_V2_DEFAULT_INVESTIGATION_CONFIG
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.likelihoods import ER1V2LikelihoodModel
from epistemic_repair.er1_v2.policies import (
    ER1V2BenchmarkDiagnosticPolicy,
    ER1V2DiagnosticPolicy,
    ER1V2OracleDiagnosticPolicy,
)
from epistemic_repair.er1_v2.trigger_model import TriggerLikelihoodModel
from epistemic_repair.er1_v2.views import (
    ER1V2BenchmarkAgentView,
    ER1V2OraclePolicyView,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.policies.stochastic_views import StochasticAgentExperimentRecord


@dataclass(frozen=True, slots=True)
class ER1V2TraceStep:
    """One persistent likelihood update after trigger conditioning."""

    step_number: int
    prior: StochasticHypothesisBeliefs
    information_gains: StochasticActionInformationGains
    chosen_action: DiagnosticAction
    experiment_result: StochasticBenchmarkExperimentResult
    outcome: StochasticExperimentOutcome
    posterior: StochasticHypothesisBeliefs
    realized_information_gain: float
    action_regret: float


@dataclass(frozen=True, slots=True)
class ER1V2EvaluationMetadata:
    """Hidden truth attached only after policy interaction."""

    hidden_failure_mode: FailureMode
    episode_seed: int


@dataclass(frozen=True, slots=True)
class ER1V2EpisodeResult:
    """Complete trigger-plus-investigation oracle episode."""

    trigger_observation: Observation
    trigger_conditioned_beliefs: StochasticHypothesisBeliefs
    trace: tuple[ER1V2TraceStep, ...]
    predicted_diagnosis: FailureMode
    evaluation_metadata: ER1V2EvaluationMetadata
    experiments_used: int
    reached_threshold: bool
    success_at_threshold: bool
    cumulative_information_gain: float
    cumulative_action_regret: float

    @property
    def ground_truth(self) -> FailureMode:
        return self.evaluation_metadata.hidden_failure_mode

    @property
    def diagnosis_correct(self) -> bool:
        return self.predicted_diagnosis is self.ground_truth

    @property
    def final_beliefs(self) -> StochasticHypothesisBeliefs:
        return (
            self.trace[-1].posterior
            if self.trace
            else self.trigger_conditioned_beliefs
        )


class ER1V2EpisodeRunner:
    """Condition once on the trigger, then use only persistent likelihoods."""

    def __init__(
        self,
        *,
        diagnosis_threshold: float = (
            ER1_V2_DEFAULT_INVESTIGATION_CONFIG.diagnosis_threshold
        ),
        max_experiments: int = 5,
        trigger_model: TriggerLikelihoodModel | None = None,
        investigation_model: ER1V2LikelihoodModel | None = None,
    ) -> None:
        if not 0.0 < diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        if type(max_experiments) is not int or max_experiments <= 0:
            raise ValueError("max_experiments must be a positive integer")
        self.diagnosis_threshold = diagnosis_threshold
        self.max_experiments = max_experiments
        self.trigger_model = trigger_model or TriggerLikelihoodModel()
        self.investigation_model = investigation_model or ER1V2LikelihoodModel()

    def run(
        self,
        environment: ER1V2BinaryMachine,
        hidden_failure_mode: FailureMode,
        policy: ER1V2DiagnosticPolicy,
        *,
        episode_seed: int = 0,
    ) -> ER1V2EpisodeResult:
        """Run one seeded V2 episode without exposing ground truth to policy."""
        if hidden_failure_mode not in ER1_HYPOTHESES:
            raise ValueError("hidden_failure_mode must be an ER-1 V2 hypothesis")
        environment.reset(hidden_failure_mode, episode_seed=episode_seed)
        trigger_observation = environment.trigger_observation()
        beliefs = self.trigger_model.conditioned_beliefs()
        initial_beliefs = beliefs
        context = Context.B
        trace: list[ER1V2TraceStep] = []
        history: list[StochasticAgentExperimentRecord] = []

        while (
            beliefs.confidence() < self.diagnosis_threshold
            and len(trace) < self.max_experiments
        ):
            scores = stochastic_information_gains(
                beliefs, self.investigation_model, context
            )
            action = self._choose_action(
                policy,
                beliefs,
                trigger_observation,
                history,
                context,
            )
            prior = beliefs
            result, context_after = self._run_action(
                environment, action, context
            )
            outcome = self.investigation_model.outcome_from_result(
                action, result, context
            )
            beliefs = self.investigation_model.update(
                prior, action, outcome, context
            )
            regret = scores.best_value() - scores.for_action(action)
            if abs(regret) < 1e-12:
                regret = 0.0
            history.append(
                StochasticAgentExperimentRecord(
                    step_number=len(trace) + 1,
                    action=action,
                    result=result,
                    context_before=context,
                    context_after=context_after,
                )
            )
            trace.append(
                ER1V2TraceStep(
                    step_number=len(trace) + 1,
                    prior=prior,
                    information_gains=scores,
                    chosen_action=action,
                    experiment_result=result,
                    outcome=outcome,
                    posterior=beliefs,
                    realized_information_gain=prior.entropy() - beliefs.entropy(),
                    action_regret=regret,
                )
            )
            context = context_after

        truth = environment.get_ground_truth().failure_mode
        prediction = beliefs.most_likely()
        reached = beliefs.confidence() >= self.diagnosis_threshold
        return ER1V2EpisodeResult(
            trigger_observation=trigger_observation,
            trigger_conditioned_beliefs=initial_beliefs,
            trace=tuple(trace),
            predicted_diagnosis=prediction,
            evaluation_metadata=ER1V2EvaluationMetadata(truth, episode_seed),
            experiments_used=len(trace),
            reached_threshold=reached,
            success_at_threshold=reached and prediction is truth,
            cumulative_information_gain=sum(
                item.realized_information_gain for item in trace
            ),
            cumulative_action_regret=sum(item.action_regret for item in trace),
        )

    def _choose_action(
        self,
        policy: ER1V2DiagnosticPolicy,
        beliefs: StochasticHypothesisBeliefs,
        trigger_observation: Observation,
        history: list[StochasticAgentExperimentRecord],
        context: Context,
    ) -> DiagnosticAction:
        if isinstance(policy, ER1V2OracleDiagnosticPolicy):
            action = policy.choose_action(
                ER1V2OraclePolicyView(
                    beliefs=beliefs,
                    current_context=context,
                    available_actions=BENCHMARK_ACTIONS,
                    investigation_likelihood_model=self.investigation_model,
                )
            )
        elif isinstance(policy, ER1V2BenchmarkDiagnosticPolicy):
            action = policy.choose_action(
                ER1V2BenchmarkAgentView(
                    trigger_history=(trigger_observation,),
                    experiment_history=tuple(history),
                    current_context=context,
                    available_actions=BENCHMARK_ACTIONS,
                    steps_remaining=self.max_experiments - len(history),
                )
            )
        else:
            raise TypeError("policy must be an ER-1 V2 diagnostic policy")
        if action not in BENCHMARK_ACTIONS:
            raise ValueError("policy selected an action outside ER-1 V2")
        return action

    def _run_action(
        self,
        environment: ER1V2BinaryMachine,
        action: DiagnosticAction,
        context: Context,
    ) -> tuple[StochasticBenchmarkExperimentResult, Context]:
        if action is DiagnosticAction.CHANGE_CONTEXT:
            target = self.investigation_model.target_context(context)
            return environment.run_experiment(action, x=1, context=target), target
        return environment.run_experiment(action, x=1), context
