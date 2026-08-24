"""Agent-visible result types for diagnostic experiments."""

from dataclasses import dataclass
from typing import TypeAlias

from epistemic_repair.diagnostics.actions import Context


@dataclass(frozen=True, slots=True)
class RepeatTrialResult:
    """Primary-sensor result from repeating an ordinary trial."""

    x: int
    o: int


@dataclass(frozen=True, slots=True)
class TrustedSensorResult:
    """Independent trusted measurement of the true physical output."""

    x: int
    trusted_y: int


@dataclass(frozen=True, slots=True)
class ChangeContextResult:
    """Primary-sensor observation after moving to a controlled context."""

    context: Context
    x: int
    o: int


@dataclass(frozen=True, slots=True)
class LatentInspectionResult:
    """Result of explicitly inspecting the candidate latent variable."""

    available: bool
    value: int | None = None

    def __post_init__(self) -> None:
        """Keep availability and value internally consistent."""
        if self.available and self.value not in (0, 1):
            raise ValueError("an available latent value must be 0 or 1")
        if not self.available and self.value is not None:
            raise ValueError("an unavailable latent variable cannot have a value")


ExperimentResult: TypeAlias = (
    RepeatTrialResult
    | TrustedSensorResult
    | ChangeContextResult
    | LatentInspectionResult
)

BenchmarkExperimentResult: TypeAlias = (
    RepeatTrialResult | TrustedSensorResult | ChangeContextResult
)
