"""Repair operators and their ground-truth mapping."""

from enum import Enum

from epistemic_repair.failures.modes import FailureMode


class RepairOperator(str, Enum):
    """Conceptual repair labels shared by ER-0 and ER-1."""

    NO_REPAIR = "NO_REPAIR"
    UPDATE_WORLD_MODEL = "UPDATE_WORLD_MODEL"
    RECALIBRATE_SENSOR = "RECALIBRATE_SENSOR"
    ADD_LATENT_VARIABLE = "ADD_LATENT_VARIABLE"


_REPAIR_BY_FAILURE: dict[FailureMode, RepairOperator] = {
    FailureMode.NORMAL: RepairOperator.NO_REPAIR,
    FailureMode.NO_STRUCTURAL_CHANGE: RepairOperator.NO_REPAIR,
    FailureMode.WORLD_SHIFT: RepairOperator.UPDATE_WORLD_MODEL,
    FailureMode.SENSOR_CORRUPTION: RepairOperator.RECALIBRATE_SENSOR,
    FailureMode.MISSING_LATENT_VARIABLE: RepairOperator.ADD_LATENT_VARIABLE,
}


def repair_for_failure(failure_mode: FailureMode) -> RepairOperator:
    """Return the correct repair for a ground-truth failure mode."""
    if not isinstance(failure_mode, FailureMode):
        raise TypeError("failure_mode must be a FailureMode")
    return _REPAIR_BY_FAILURE[failure_mode]
