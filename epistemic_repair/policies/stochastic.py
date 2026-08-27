"""ER-1 oracle and restricted seeded-random diagnostic policies."""

from abc import ABC, abstractmethod
from random import Random
from typing import TypeAlias

from epistemic_repair.beliefs.stochastic_likelihoods import (
    stochastic_information_gains,
)
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.policies.stochastic_views import (
    StochasticBenchmarkAgentView,
    StochasticOraclePolicyView,
)


class StochasticBenchmarkDiagnosticPolicy(ABC):
    """Base policy restricted to the safe ER-1 agent view."""

    @abstractmethod
    def choose_action(
        self, view: StochasticBenchmarkAgentView
    ) -> DiagnosticAction:
        """Choose one exposed ER-1 action."""


class StochasticOracleDiagnosticPolicy(ABC):
    """Base policy explicitly authorized to use ER-1 normative beliefs."""

    @abstractmethod
    def choose_action(self, view: StochasticOraclePolicyView) -> DiagnosticAction:
        """Choose one action without access to hidden ground truth."""


StochasticDiagnosticPolicy: TypeAlias = (
    StochasticBenchmarkDiagnosticPolicy | StochasticOracleDiagnosticPolicy
)


class StochasticOracleInformationGainPolicy(StochasticOracleDiagnosticPolicy):
    """Choose maximum ER-1 expected information gain with stable tie order."""

    def choose_action(self, view: StochasticOraclePolicyView) -> DiagnosticAction:
        scores = stochastic_information_gains(
            view.beliefs,
            view.likelihood_model,
            view.current_context,
        )
        return max(view.available_actions, key=scores.for_action)


class StochasticRandomDiagnosticPolicy(StochasticBenchmarkDiagnosticPolicy):
    """Choose uniformly from safe ER-1 actions using a private seeded RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = Random(seed)

    def choose_action(
        self, view: StochasticBenchmarkAgentView
    ) -> DiagnosticAction:
        return self._random.choice(view.available_actions)
