"""Privileged oracle and restricted random diagnostic policies."""

from abc import ABC, abstractmethod
from random import Random
from typing import TypeAlias

from epistemic_repair.beliefs.likelihoods import information_gains
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.policies.views import BenchmarkAgentView, OraclePolicyView


class BenchmarkDiagnosticPolicy(ABC):
    """Base class for policies restricted to benchmark-agent information."""

    @abstractmethod
    def choose_action(self, view: BenchmarkAgentView) -> DiagnosticAction:
        """Choose from the actions exposed by the restricted view."""


class OracleDiagnosticPolicy(ABC):
    """Base class for policies explicitly granted normative oracle information."""

    @abstractmethod
    def choose_action(self, view: OraclePolicyView) -> DiagnosticAction:
        """Choose from actions in an oracle-privileged view."""


DiagnosticPolicy: TypeAlias = BenchmarkDiagnosticPolicy | OracleDiagnosticPolicy


class OracleInformationGainPolicy(OracleDiagnosticPolicy):
    """Choose maximum expected information gain with deterministic ties.

    Ties follow the order of ``view.available_actions``. The benchmark order is
    REPEAT_TRIAL, USE_TRUSTED_SENSOR, CHANGE_CONTEXT, so the initial tie between
    the latter two resolves to USE_TRUSTED_SENSOR.
    """

    def choose_action(self, view: OraclePolicyView) -> DiagnosticAction:
        """Return the first action attaining maximum expected information gain."""
        scores = information_gains(
            view.beliefs,
            view.likelihood_model,
            view.current_context,
        )
        return max(view.available_actions, key=scores.for_action)


class RandomDiagnosticPolicy(BenchmarkDiagnosticPolicy):
    """Choose uniformly from the restricted action view using a private RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = Random(seed)

    def choose_action(self, view: BenchmarkAgentView) -> DiagnosticAction:
        """Choose without beliefs, likelihoods, environment, or ground truth."""
        return self._random.choice(view.available_actions)

