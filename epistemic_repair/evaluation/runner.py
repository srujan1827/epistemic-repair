"""Ground-truth-separated loop for deterministic diagnostic episodes."""

from dataclasses import dataclass

from epistemic_repair.beliefs.likelihoods import (
    ActionInformationGains,
    DeterministicLikelihoodModel,
    ExperimentOutcome,
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
from epistemic_repair.policies.diagnostic import (
    BenchmarkDiagnosticPolicy,
    DiagnosticPolicy,
    OracleDiagnosticPolicy,
)
from epistemic_repair.policies.views import (
    AgentExperimentRecord,
    BenchmarkAgentView,
    OraclePolicyView,
)


@dataclass(frozen=True, slots=True)
class DiagnosticTraceStep:
    """Privileged evaluation trace for one experiment and belief transition."""

    step_number: int
    prior: HypothesisBeliefs
    information_gains: ActionInformationGains
    chosen_action: DiagnosticAction
    experiment_result: BenchmarkExperimentResult
    outcome: ExperimentOutcome
    posterior: HypothesisBeliefs
    realized_information_gain: float
    action_regret: float


@dataclass(frozen=True, slots=True)
class DiagnosticEpisodeResult:
    """Full evaluation result; ground truth exists only at this outer boundary."""

    initial_observation: Observation
    trace: tuple[DiagnosticTraceStep, ...]
    predicted_diagnosis: FailureMode
    ground_truth: FailureMode
    experiments_used: int
    reached_threshold: bool
    success_within_budget: bool
    cumulative_information_gain: float
    cumulative_action_regret: float

    @property
    def diagnosis_correct(self) -> bool:
        """Return whether the predicted diagnosis matches evaluation ground truth."""
        return self.predicted_diagnosis is self.ground_truth


class DiagnosticEpisodeRunner:
    """Run policy-controlled experiments without passing hidden state to policy."""

    def __init__(
        self,
        *,
        diagnosis_threshold: float = 0.999,
        max_experiments: int = 5,
        likelihood_model: DeterministicLikelihoodModel | None = None,
    ) -> None:
        if not 0.0 < diagnosis_threshold <= 1.0:
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        if type(max_experiments) is not int or max_experiments <= 0:
            raise ValueError("max_experiments must be a positive integer")
        self._diagnosis_threshold = diagnosis_threshold
        self._max_experiments = max_experiments
        self._likelihood_model = likelihood_model or DeterministicLikelihoodModel()

    def run(
        self,
        env: BinaryMachine,
        hidden_failure_mode: FailureMode,
        policy: DiagnosticPolicy,
    ) -> DiagnosticEpisodeResult:
        """Run one episode and attach hidden truth only after policy interaction."""
        if hidden_failure_mode not in HYPOTHESES:
            raise ValueError("hidden_failure_mode must be a benchmark hypothesis")

        env.reset(hidden_failure_mode)
        initial_observation = env.step(self._likelihood_model.input_x)
        expected_anomaly = Observation(x=1, o=0)
        if initial_observation != expected_anomaly:
            raise RuntimeError("benchmark episode did not start from X=1 -> O=0")

        beliefs = HypothesisBeliefs.uniform()
        current_context = Context.B
        trace: list[DiagnosticTraceStep] = []
        agent_history: list[AgentExperimentRecord] = []

        while (
            beliefs.confidence() < self._diagnosis_threshold
            and len(trace) < self._max_experiments
        ):
            scores = information_gains(
                beliefs, self._likelihood_model, current_context
            )
            chosen_action = self._choose_policy_action(
                policy=policy,
                beliefs=beliefs,
                initial_observation=initial_observation,
                agent_history=agent_history,
                current_context=current_context,
                steps_remaining=self._max_experiments - len(trace),
            )
            if chosen_action not in BENCHMARK_ACTIONS:
                raise ValueError("policy selected an action outside the benchmark set")

            prior = beliefs
            result = self._run_benchmark_action(env, chosen_action, current_context)
            outcome = self._likelihood_model.outcome_from_result(
                chosen_action, result, current_context
            )
            beliefs = self._likelihood_model.update(
                prior, chosen_action, outcome, current_context
            )

            realized_gain = prior.entropy() - beliefs.entropy()
            regret = scores.best_value() - scores.for_action(chosen_action)
            if abs(regret) < 1e-12:
                regret = 0.0

            context_after = (
                self._likelihood_model.target_context(current_context)
                if chosen_action is DiagnosticAction.CHANGE_CONTEXT
                else current_context
            )
            agent_record = AgentExperimentRecord(
                step_number=len(trace) + 1,
                action=chosen_action,
                result=result,
                context_before=current_context,
                context_after=context_after,
            )
            agent_history.append(agent_record)
            trace.append(
                DiagnosticTraceStep(
                    step_number=len(trace) + 1,
                    prior=prior,
                    information_gains=scores,
                    chosen_action=chosen_action,
                    experiment_result=result,
                    outcome=outcome,
                    posterior=beliefs,
                    realized_information_gain=realized_gain,
                    action_regret=regret,
                )
            )

            if chosen_action is DiagnosticAction.CHANGE_CONTEXT:
                current_context = context_after

        predicted_diagnosis = beliefs.most_likely()
        reached_threshold = beliefs.confidence() >= self._diagnosis_threshold

        # Evaluation-only access occurs after every policy decision is complete.
        ground_truth = env.get_ground_truth().failure_mode
        correct = predicted_diagnosis is ground_truth
        return DiagnosticEpisodeResult(
            initial_observation=initial_observation,
            trace=tuple(trace),
            predicted_diagnosis=predicted_diagnosis,
            ground_truth=ground_truth,
            experiments_used=len(trace),
            reached_threshold=reached_threshold,
            success_within_budget=reached_threshold and correct,
            cumulative_information_gain=sum(
                step.realized_information_gain for step in trace
            ),
            cumulative_action_regret=sum(step.action_regret for step in trace),
        )

    def _choose_policy_action(
        self,
        *,
        policy: DiagnosticPolicy,
        beliefs: HypothesisBeliefs,
        initial_observation: Observation,
        agent_history: list[AgentExperimentRecord],
        current_context: Context,
        steps_remaining: int,
    ) -> DiagnosticAction:
        """Construct only the information view authorized for a policy class."""
        if isinstance(policy, OracleDiagnosticPolicy):
            return policy.choose_action(
                OraclePolicyView(
                    beliefs=beliefs,
                    current_context=current_context,
                    available_actions=BENCHMARK_ACTIONS,
                    likelihood_model=self._likelihood_model,
                )
            )
        if isinstance(policy, BenchmarkDiagnosticPolicy):
            return policy.choose_action(
                BenchmarkAgentView(
                    initial_history=(initial_observation,),
                    experiment_history=tuple(agent_history),
                    current_context=current_context,
                    available_actions=BENCHMARK_ACTIONS,
                    steps_remaining=steps_remaining,
                )
            )
        raise TypeError(
            "policy must inherit BenchmarkDiagnosticPolicy or OracleDiagnosticPolicy"
        )

    def _run_benchmark_action(
        self,
        env: BinaryMachine,
        action: DiagnosticAction,
        current_context: Context,
    ) -> BenchmarkExperimentResult:
        """Bind benchmark action names to their fixed V0 experimental parameters."""
        x = self._likelihood_model.input_x
        if action is DiagnosticAction.REPEAT_TRIAL:
            result = env.run_experiment(action, x=x)
            assert isinstance(result, RepeatTrialResult)
            return result
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            result = env.run_experiment(action, x=x)
            assert isinstance(result, TrustedSensorResult)
            return result

        target_context = self._likelihood_model.target_context(current_context)
        result = env.run_experiment(action, x=x, context=target_context)
        assert isinstance(result, ChangeContextResult)
        return result
