"""Separate trigger and persistent-investigation constants for ER-1 V2."""

from dataclasses import dataclass
from math import isfinite


ER1_V2_SUPPORTED_BUDGETS = (1, 2, 3, 5, 8)


@dataclass(frozen=True, slots=True)
class ER1V2TriggerConfig:
    """Likelihoods of the one-time transient trigger anomaly A0."""

    no_structural_change: float = 0.30
    world_shift: float = 0.70
    sensor_corruption: float = 0.65
    missing_latent_variable: float = 0.70

    def __post_init__(self) -> None:
        _validate_probabilities(self)


@dataclass(frozen=True, slots=True)
class ER1V2InvestigationConfig:
    """Persistent generative dynamics used only after episode triggering."""

    no_change_physical_accuracy: float = 0.95
    world_shift_physical_accuracy: float = 0.90
    sensor_corruption_physical_accuracy: float = 0.95
    healthy_primary_sensor_accuracy: float = 0.95
    corrupted_sensor_inversion_accuracy: float = 0.90
    latent_physical_accuracy: float = 0.90
    trusted_sensor_accuracy: float = 0.99
    diagnosis_threshold: float = 0.90

    def __post_init__(self) -> None:
        _validate_probabilities(self)


def _validate_probabilities(config: object) -> None:
    for name in config.__dataclass_fields__:
        value = getattr(config, name)
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError(f"{name} must be a finite probability")
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")


ER1_V2_DEFAULT_TRIGGER_CONFIG = ER1V2TriggerConfig()
ER1_V2_DEFAULT_INVESTIGATION_CONFIG = ER1V2InvestigationConfig()
