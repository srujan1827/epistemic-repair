"""ER-1 V2 oracle and restricted random policies."""

from abc import ABC, abstractmethod
from random import Random
from typing import TypeAlias

from epistemic_repair.beliefs.stochastic_likelihoods import (
    stochastic_information_gains,
)
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.er1_v2.views import (
    ER1V2BenchmarkAgentView,
    ER1V2OraclePolicyView,
)


class ER1V2BenchmarkDiagnosticPolicy(ABC):
    @abstractmethod
    def choose_action(self, view: ER1V2BenchmarkAgentView) -> DiagnosticAction:
        """Choose one safe persistent investigation action."""


class ER1V2OracleDiagnosticPolicy(ABC):
    @abstractmethod
    def choose_action(self, view: ER1V2OraclePolicyView) -> DiagnosticAction:
        """Choose using beliefs and persistent likelihoods, never truth."""


ER1V2DiagnosticPolicy: TypeAlias = (
    ER1V2BenchmarkDiagnosticPolicy | ER1V2OracleDiagnosticPolicy
)


class ER1V2OracleInformationGainPolicy(ER1V2OracleDiagnosticPolicy):
    """Choose maximal persistent expected information gain with stable ties."""

    def choose_action(self, view: ER1V2OraclePolicyView) -> DiagnosticAction:
        scores = stochastic_information_gains(
            view.beliefs,
            view.investigation_likelihood_model,
            view.current_context,
        )
        return max(view.available_actions, key=scores.for_action)


class ER1V2RandomDiagnosticPolicy(ER1V2BenchmarkDiagnosticPolicy):
    """Seeded restricted baseline with no normative model access."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = Random(seed)

    def choose_action(self, view: ER1V2BenchmarkAgentView) -> DiagnosticAction:
        return self._random.choice(view.available_actions)
