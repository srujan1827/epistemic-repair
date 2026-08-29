"""Explicit repairable predictive state for the minimal ER-2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from epistemic_repair.diagnostics.actions import Context
from epistemic_repair.repair.operators import RepairOperator


class WorldRelation(str, Enum):
    """Persistent physical relationship encoded by the agent."""

    IDENTITY = "Y_EQUALS_X"
    INVERTED = "Y_EQUALS_ONE_MINUS_X"


class SensorCalibration(str, Enum):
    """Agent model of how the primary sensor maps physical Y to observed O."""

    DIRECT = "O_EQUALS_Y"
    INVERTED = "O_EQUALS_ONE_MINUS_Y"


class LatentStructure(str, Enum):
    """Whether the predictive model represents context-dependent physics."""

    ABSENT = "ABSENT"
    CONTEXT_DEPENDENT = "CONTEXT_DEPENDENT"


@dataclass(slots=True)
class RepairableAgentState:
    """Minimal mutable state whose components can be repaired independently."""

    world_relation: WorldRelation = WorldRelation.IDENTITY
    sensor_calibration: SensorCalibration = SensorCalibration.DIRECT
    latent_structure: LatentStructure = LatentStructure.ABSENT

    def clone(self) -> "RepairableAgentState":
        """Return an independent copy for pre/post evaluation."""
        return replace(self)

    def predict_physical_output(self, x: int, context: Context) -> int:
        """Predict true Y from X and context using the current world model."""
        _validate_bit(x, "x")
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        if self.latent_structure is LatentStructure.CONTEXT_DEPENDENT:
            return x if context is Context.A else 1 - x
        return x if self.world_relation is WorldRelation.IDENTITY else 1 - x

    def predict_primary_from_physical(self, y: int) -> int:
        """Predict primary observation O from a supplied physical output Y."""
        _validate_bit(y, "y")
        return y if self.sensor_calibration is SensorCalibration.DIRECT else 1 - y

    def predict_primary_observation(self, x: int, context: Context) -> int:
        """Predict end-to-end O from X and context."""
        y = self.predict_physical_output(x, context)
        return self.predict_primary_from_physical(y)


def apply_repair(state: RepairableAgentState, repair: RepairOperator) -> None:
    """Mutate exactly the state component targeted by ``repair``."""
    if not isinstance(state, RepairableAgentState):
        raise TypeError("state must be a RepairableAgentState")
    if not isinstance(repair, RepairOperator):
        raise TypeError("repair must be a RepairOperator")
    if repair is RepairOperator.NO_REPAIR:
        return
    if repair is RepairOperator.UPDATE_WORLD_MODEL:
        state.world_relation = WorldRelation.INVERTED
        return
    if repair is RepairOperator.RECALIBRATE_SENSOR:
        state.sensor_calibration = SensorCalibration.INVERTED
        return
    if repair is RepairOperator.ADD_LATENT_VARIABLE:
        state.latent_structure = LatentStructure.CONTEXT_DEPENDENT
        return
    raise ValueError(f"unsupported repair: {repair}")


def _validate_bit(value: int, name: str) -> None:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be exactly 0 or 1")
