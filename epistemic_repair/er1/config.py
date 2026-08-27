"""Central probability and stopping constants for stochastic ER-1."""

from dataclasses import dataclass
from math import isfinite

from epistemic_repair.failures.modes import FailureMode


ER1_HYPOTHESES: tuple[FailureMode, ...] = (
    FailureMode.NO_STRUCTURAL_CHANGE,
    FailureMode.WORLD_SHIFT,
    FailureMode.SENSOR_CORRUPTION,
    FailureMode.MISSING_LATENT_VARIABLE,
)

ER1_BASE_PRIOR: dict[FailureMode, float] = {
    hypothesis: 1.0 / len(ER1_HYPOTHESES) for hypothesis in ER1_HYPOTHESES
}


@dataclass(frozen=True, slots=True)
class ER1ProbabilityConfig:
    """Generative probabilities and normative diagnosis threshold for ER-1."""

    preferred_physical_probability: float = 0.90
    primary_sensor_reliability: float = 0.95
    corrupted_sensor_inversion_probability: float = 0.90
    trusted_sensor_reliability: float = 0.99
    diagnosis_threshold: float = 0.90

    def __post_init__(self) -> None:
        for name in (
            "preferred_physical_probability",
            "primary_sensor_reliability",
            "corrupted_sensor_inversion_probability",
            "trusted_sensor_reliability",
            "diagnosis_threshold",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite probability")
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")


ER1_DEFAULT_CONFIG = ER1ProbabilityConfig()
