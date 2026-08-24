"""Structurally separate benchmark-agent and privileged-oracle inputs."""

from dataclasses import dataclass

from epistemic_repair.beliefs.likelihoods import DeterministicLikelihoodModel
from epistemic_repair.beliefs.state import HypothesisBeliefs
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
from epistemic_repair.environments.binary_machine import Observation


@dataclass(frozen=True, slots=True)
class AgentExperimentRecord:
    """One safe, agent-visible experiment-history entry."""

    step_number: int
    action: DiagnosticAction
    result: BenchmarkExperimentResult
    context_before: Context
    context_after: Context

    def __post_init__(self) -> None:
        if type(self.step_number) is not int or self.step_number <= 0:
            raise ValueError("step_number must be a positive integer")
        if self.action not in BENCHMARK_ACTIONS:
            raise ValueError("record action must be benchmark-visible")
        if not isinstance(self.context_before, Context) or not isinstance(
            self.context_after, Context
        ):
            raise TypeError("record contexts must be Context values")
        expected_result_type = {
            DiagnosticAction.REPEAT_TRIAL: RepeatTrialResult,
            DiagnosticAction.USE_TRUSTED_SENSOR: TrustedSensorResult,
            DiagnosticAction.CHANGE_CONTEXT: ChangeContextResult,
        }[self.action]
        if not isinstance(self.result, expected_result_type):
            raise TypeError("record result type does not match its action")
        if self.action is DiagnosticAction.CHANGE_CONTEXT:
            assert isinstance(self.result, ChangeContextResult)
            if self.context_before is self.context_after:
                raise ValueError("CHANGE_CONTEXT must change the observable context")
            if self.result.context is not self.context_after:
                raise ValueError("context result must match context_after")
        elif self.context_before is not self.context_after:
            raise ValueError("non-context actions cannot change context")


@dataclass(frozen=True, slots=True)
class BenchmarkAgentView:
    """Complete restricted input for a future non-oracle benchmark agent."""

    initial_history: tuple[Observation, ...]
    experiment_history: tuple[AgentExperimentRecord, ...]
    current_context: Context
    available_actions: tuple[DiagnosticAction, ...]
    steps_remaining: int

    def __post_init__(self) -> None:
        if type(self.steps_remaining) is not int or self.steps_remaining < 0:
            raise ValueError("steps_remaining must be a non-negative integer")
        if self.available_actions != BENCHMARK_ACTIONS:
            raise ValueError("available_actions must equal BENCHMARK_ACTIONS")
        if not isinstance(self.current_context, Context):
            raise TypeError("current_context must be a Context")


@dataclass(frozen=True, slots=True)
class OraclePolicyView:
    """Normative information explicitly available only to the oracle baseline."""

    beliefs: HypothesisBeliefs
    current_context: Context
    available_actions: tuple[DiagnosticAction, ...]
    likelihood_model: DeterministicLikelihoodModel

    def __post_init__(self) -> None:
        if self.available_actions != BENCHMARK_ACTIONS:
            raise ValueError("available_actions must equal BENCHMARK_ACTIONS")
        if not isinstance(self.current_context, Context):
            raise TypeError("current_context must be a Context")
        if not isinstance(self.beliefs, HypothesisBeliefs):
            raise TypeError("beliefs must be HypothesisBeliefs")
        if not isinstance(self.likelihood_model, DeterministicLikelihoodModel):
            raise TypeError("likelihood_model must be DeterministicLikelihoodModel")
