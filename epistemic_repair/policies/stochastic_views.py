"""Strictly separated ER-1 agent and oracle policy views."""

from dataclasses import dataclass

from epistemic_repair.beliefs.stochastic_likelihoods import StochasticLikelihoodModel
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticBenchmarkExperimentResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.environments.binary_machine import Observation


@dataclass(frozen=True, slots=True)
class StochasticAgentExperimentRecord:
    """One ER-1 agent-visible stochastic experiment result."""

    step_number: int
    action: DiagnosticAction
    result: StochasticBenchmarkExperimentResult
    context_before: Context
    context_after: Context

    def __post_init__(self) -> None:
        if type(self.step_number) is not int or self.step_number <= 0:
            raise ValueError("step_number must be a positive integer")
        if self.action not in BENCHMARK_ACTIONS:
            raise ValueError("record action must be benchmark-visible")
        expected_type = {
            DiagnosticAction.REPEAT_TRIAL: RepeatTrialResult,
            DiagnosticAction.USE_TRUSTED_SENSOR: StochasticTrustedSensorResult,
            DiagnosticAction.CHANGE_CONTEXT: ChangeContextResult,
        }[self.action]
        if not isinstance(self.result, expected_type):
            raise TypeError("record result type does not match its action")
        if self.action is DiagnosticAction.CHANGE_CONTEXT:
            assert isinstance(self.result, ChangeContextResult)
            if self.context_before is self.context_after:
                raise ValueError("CHANGE_CONTEXT must change context")
            if self.result.context is not self.context_after:
                raise ValueError("result context must match context_after")
        elif self.context_before is not self.context_after:
            raise ValueError("non-context actions cannot change context")


@dataclass(frozen=True, slots=True)
class StochasticBenchmarkAgentView:
    """Complete non-privileged input for an ER-1 benchmark policy."""

    initial_history: tuple[Observation, ...]
    experiment_history: tuple[StochasticAgentExperimentRecord, ...]
    current_context: Context
    available_actions: tuple[DiagnosticAction, ...]
    steps_remaining: int

    def __post_init__(self) -> None:
        if self.available_actions != BENCHMARK_ACTIONS:
            raise ValueError("available_actions must equal BENCHMARK_ACTIONS")
        if type(self.steps_remaining) is not int or self.steps_remaining < 0:
            raise ValueError("steps_remaining must be a non-negative integer")
        if not isinstance(self.current_context, Context):
            raise TypeError("current_context must be a Context")


@dataclass(frozen=True, slots=True)
class StochasticOraclePolicyView:
    """Normative ER-1 beliefs and likelihoods available only to the oracle."""

    beliefs: StochasticHypothesisBeliefs
    current_context: Context
    available_actions: tuple[DiagnosticAction, ...]
    likelihood_model: StochasticLikelihoodModel

    def __post_init__(self) -> None:
        if self.available_actions != BENCHMARK_ACTIONS:
            raise ValueError("available_actions must equal BENCHMARK_ACTIONS")
        if not isinstance(self.beliefs, StochasticHypothesisBeliefs):
            raise TypeError("beliefs must be StochasticHypothesisBeliefs")
        if not isinstance(self.likelihood_model, StochasticLikelihoodModel):
            raise TypeError("likelihood_model must be StochasticLikelihoodModel")
        if not isinstance(self.current_context, Context):
            raise TypeError("current_context must be a Context")
