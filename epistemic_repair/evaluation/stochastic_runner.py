"""Trusted seeded episode loop for stochastic ER-1 diagnostic policies."""

from dataclasses import dataclass

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticActionInformationGains,
    StochasticExperimentOutcome,
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
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.policies.stochastic import (
    StochasticBenchmarkDiagnosticPolicy,
    StochasticDiagnosticPolicy,
    StochasticOracleDiagnosticPolicy,
)
from epistemic_repair.policies.stochastic_views import (
    StochasticAgentExperimentRecord,
    StochasticBenchmarkAgentView,
    StochasticOraclePolicyView,
)


@dataclass(frozen=True, slots=True)
class StochasticDiagnosticTraceStep:
    """Evaluation trace for one ER-1 stochastic belief transition."""

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
class StochasticEvaluationMetadata:
    """ER-1 ground truth and seed kept outside the agent interface."""

    hidden_failure_mode: FailureMode
    episode_seed: int


@dataclass(frozen=True, slots=True)
class StochasticDiagnosticEpisodeResult:
    """Complete ER-1 result with hidden truth only in evaluation metadata."""

    initial_observation: Observation
    initial_beliefs: StochasticHypothesisBeliefs
    trace: tuple[StochasticDiagnosticTraceStep, ...]
    predicted_diagnosis: FailureMode
    evaluation_metadata: StochasticEvaluationMetadata
    experiments_used: int
    reached_threshold: bool
    success_within_budget: bool
    cumulative_information_gain: float
    cumulative_action_regret: float

    @property
    def ground_truth(self) -> FailureMode:
        return self.evaluation_metadata.hidden_failure_mode

    @property
    def diagnosis_correct(self) -> bool:
        return self.predicted_diagnosis is self.ground_truth


class StochasticDiagnosticEpisodeRunner:
    """Run an ER-1 policy until threshold or budget without leaking truth."""

    def __init__(
        self,
        *,
        diagnosis_threshold: float = ER1_DEFAULT_CONFIG.diagnosis_threshold,
        max_experiments: int = 5,
        likelihood_model: StochasticLikelihoodModel | None = None,
    ) -> None:
        if not 0.0 < diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        if type(max_experiments) is not int or max_experiments <= 0:
            raise ValueError("max_experiments must be a positive integer")
        self._diagnosis_threshold = diagnosis_threshold
        self._max_experiments = max_experiments
        self._likelihood_model = likelihood_model or StochasticLikelihoodModel()

    def run(
        self,
        env: StochasticBinaryMachine,
        hidden_failure_mode: FailureMode,
        policy: StochasticDiagnosticPolicy,
        *,
        episode_seed: int = 0,
    ) -> StochasticDiagnosticEpisodeResult:
        """Run one seeded episode and attach truth only after all decisions."""
        if hidden_failure_mode not in ER1_HYPOTHESES:
            raise ValueError("hidden_failure_mode must be an ER-1 hypothesis")
        env.reset(hidden_failure_mode, episode_seed=episode_seed)
        initial_observation = env.initial_anomaly(x=self._likelihood_model.input_x)
        beliefs = self._likelihood_model.conditioned_initial_beliefs()
        initial_beliefs = beliefs
        current_context = self._likelihood_model.initial_context
        trace: list[StochasticDiagnosticTraceStep] = []
        history: list[StochasticAgentExperimentRecord] = []

        while (
            beliefs.confidence() < self._diagnosis_threshold
            and len(trace) < self._max_experiments
        ):
            scores = stochastic_information_gains(
                beliefs, self._likelihood_model, current_context
            )
            action = self._choose_action(
                policy,
                beliefs,
                initial_observation,
                history,
                current_context,
            )
            prior = beliefs
            result, context_after = self._run_action(
                env, action, current_context
            )
            outcome = self._likelihood_model.outcome_from_result(
                action, result, current_context
            )
            beliefs = self._likelihood_model.update(
                prior, action, outcome, current_context
            )
            realized_gain = prior.entropy() - beliefs.entropy()
            regret = scores.best_value() - scores.for_action(action)
            if abs(regret) < 1e-12:
                regret = 0.0
            history.append(
                StochasticAgentExperimentRecord(
                    step_number=len(trace) + 1,
                    action=action,
                    result=result,
                    context_before=current_context,
                    context_after=context_after,
                )
            )
            trace.append(
                StochasticDiagnosticTraceStep(
                    step_number=len(trace) + 1,
                    prior=prior,
                    information_gains=scores,
                    chosen_action=action,
                    experiment_result=result,
                    outcome=outcome,
                    posterior=beliefs,
                    realized_information_gain=realized_gain,
                    action_regret=regret,
                )
            )
            current_context = context_after

        prediction = beliefs.most_likely()
        reached = beliefs.confidence() >= self._diagnosis_threshold
        truth = env.get_ground_truth().failure_mode
        correct = prediction is truth
        return StochasticDiagnosticEpisodeResult(
            initial_observation=initial_observation,
            initial_beliefs=initial_beliefs,
            trace=tuple(trace),
            predicted_diagnosis=prediction,
            evaluation_metadata=StochasticEvaluationMetadata(
                hidden_failure_mode=truth,
                episode_seed=episode_seed,
            ),
            experiments_used=len(trace),
            reached_threshold=reached,
            success_within_budget=correct,
            cumulative_information_gain=sum(
                step.realized_information_gain for step in trace
            ),
            cumulative_action_regret=sum(step.action_regret for step in trace),
        )

    def _choose_action(
        self,
        policy: StochasticDiagnosticPolicy,
        beliefs: StochasticHypothesisBeliefs,
        initial_observation: Observation,
        history: list[StochasticAgentExperimentRecord],
        current_context: Context,
    ) -> DiagnosticAction:
        if isinstance(policy, StochasticOracleDiagnosticPolicy):
            action = policy.choose_action(
                StochasticOraclePolicyView(
                    beliefs=beliefs,
                    current_context=current_context,
                    available_actions=BENCHMARK_ACTIONS,
                    likelihood_model=self._likelihood_model,
                )
            )
        elif isinstance(policy, StochasticBenchmarkDiagnosticPolicy):
            action = policy.choose_action(
                StochasticBenchmarkAgentView(
                    initial_history=(initial_observation,),
                    experiment_history=tuple(history),
                    current_context=current_context,
                    available_actions=BENCHMARK_ACTIONS,
                    steps_remaining=self._max_experiments - len(history),
                )
            )
        else:
            raise TypeError("policy must be an ER-1 diagnostic policy")
        if action not in BENCHMARK_ACTIONS:
            raise ValueError("policy selected an action outside ER-1")
        return action

    def _run_action(
        self,
        env: StochasticBinaryMachine,
        action: DiagnosticAction,
        current_context: Context,
    ) -> tuple[StochasticBenchmarkExperimentResult, Context]:
        x = self._likelihood_model.input_x
        if action is DiagnosticAction.CHANGE_CONTEXT:
            target = self._likelihood_model.target_context(current_context)
            return env.run_experiment(action, x=x, context=target), target
        return env.run_experiment(action, x=x), current_context
