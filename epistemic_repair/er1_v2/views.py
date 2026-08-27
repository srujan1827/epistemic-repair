"""Strictly separated ER-1 V2 benchmark-agent and oracle views."""

from dataclasses import dataclass

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.er1_v2.likelihoods import ER1V2LikelihoodModel
from epistemic_repair.policies.stochastic_views import StochasticAgentExperimentRecord


@dataclass(frozen=True, slots=True)
class ER1V2BenchmarkAgentView:
    """Only transient-trigger evidence and safe persistent experiment history."""

    trigger_history: tuple[Observation, ...]
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
        if any(type(item) is not Observation for item in self.trigger_history):
            raise TypeError("trigger_history must contain only Observation values")


@dataclass(frozen=True, slots=True)
class ER1V2OraclePolicyView:
    """Normative persistent likelihoods without trigger model or ground truth."""

    beliefs: StochasticHypothesisBeliefs
    current_context: Context
    available_actions: tuple[DiagnosticAction, ...]
    investigation_likelihood_model: ER1V2LikelihoodModel

    def __post_init__(self) -> None:
        if self.available_actions != BENCHMARK_ACTIONS:
            raise ValueError("available_actions must equal BENCHMARK_ACTIONS")
        if not isinstance(self.beliefs, StochasticHypothesisBeliefs):
            raise TypeError("beliefs must be StochasticHypothesisBeliefs")
        if not isinstance(
            self.investigation_likelihood_model, ER1V2LikelihoodModel
        ):
            raise TypeError("likelihood model must be ER1V2LikelihoodModel")
        if not isinstance(self.current_context, Context):
            raise TypeError("current_context must be a Context")
