"""Supported hidden episode conditions for ER-0 and ER-1."""

from enum import Enum


class FailureMode(str, Enum):
    """Ground-truth environment condition for an episode."""

    NORMAL = "NORMAL"
    NO_STRUCTURAL_CHANGE = "NO_STRUCTURAL_CHANGE"
    WORLD_SHIFT = "WORLD_SHIFT"
    SENSOR_CORRUPTION = "SENSOR_CORRUPTION"
    MISSING_LATENT_VARIABLE = "MISSING_LATENT_VARIABLE"
