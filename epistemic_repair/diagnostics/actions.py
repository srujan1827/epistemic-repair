"""Diagnostic actions and controlled experimental contexts."""

from enum import Enum


class DiagnosticAction(str, Enum):
    """Experiments available to an agent in the deterministic diagnostic layer."""

    REPEAT_TRIAL = "REPEAT_TRIAL"
    USE_TRUSTED_SENSOR = "USE_TRUSTED_SENSOR"
    CHANGE_CONTEXT = "CHANGE_CONTEXT"
    INSPECT_LATENT_VARIABLE = "INSPECT_LATENT_VARIABLE"


class Context(str, Enum):
    """Controlled context values used by the context-change experiment."""

    A = "A"
    B = "B"


# This ordered tuple is the public action space for benchmark policies. The
# order also defines deterministic tie-breaking for the oracle baseline.
BENCHMARK_ACTIONS: tuple[DiagnosticAction, ...] = (
    DiagnosticAction.REPEAT_TRIAL,
    DiagnosticAction.USE_TRUSTED_SENSOR,
    DiagnosticAction.CHANGE_CONTEXT,
)
