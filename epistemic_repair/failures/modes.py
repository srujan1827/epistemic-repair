"""Supported hidden causes of prediction failure."""

from enum import Enum


class FailureMode(str, Enum):
    """Ground-truth environment condition for an episode."""

    NORMAL = "NORMAL"
    WORLD_SHIFT = "WORLD_SHIFT"
    SENSOR_CORRUPTION = "SENSOR_CORRUPTION"
    MISSING_LATENT_VARIABLE = "MISSING_LATENT_VARIABLE"

