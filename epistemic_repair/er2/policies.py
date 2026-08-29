"""Restricted repair-selection policies for ER-2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from epistemic_repair.er2.evaluation import ER2_HYPOTHESES
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


@dataclass(frozen=True, slots=True)
class ER2RepairAgentView:
    """Only the externally supplied diagnosis and allowed repair choices."""

    diagnosis: FailureMode
    available_repairs: tuple[RepairOperator, ...] = tuple(RepairOperator)

    def __post_init__(self) -> None:
        if self.diagnosis not in ER2_HYPOTHESES:
            raise ValueError("diagnosis must be an ER-2 hypothesis")
        if self.available_repairs != tuple(RepairOperator):
            raise ValueError("available_repairs must contain exactly the canonical repairs")


class ER2RepairPolicy(Protocol):
    """Policy boundary for choosing a repair from a diagnosis-only view."""

    def choose_repair(self, view: ER2RepairAgentView) -> RepairOperator:
        """Choose one canonical repair without environment access."""
        ...


@dataclass(frozen=True, slots=True)
class FixedRepairPolicy:
    """Baseline that always chooses one repair."""

    repair: RepairOperator

    def choose_repair(self, view: ER2RepairAgentView) -> RepairOperator:
        if not isinstance(view, ER2RepairAgentView):
            raise TypeError("view must be an ER2RepairAgentView")
        return self.repair


class OracleRepairPolicy:
    """Map the supplied diagnosis to its canonical minimal repair."""

    def choose_repair(self, view: ER2RepairAgentView) -> RepairOperator:
        if not isinstance(view, ER2RepairAgentView):
            raise TypeError("view must be an ER2RepairAgentView")
        return repair_for_failure(view.diagnosis)


class RepairBaseline(str, Enum):
    """Deterministic ER-2 baseline names."""

    ALWAYS_NO_REPAIR = "ALWAYS_NO_REPAIR"
    ALWAYS_UPDATE_WORLD_MODEL = "ALWAYS_UPDATE_WORLD_MODEL"
    ALWAYS_RECALIBRATE_SENSOR = "ALWAYS_RECALIBRATE_SENSOR"
    ALWAYS_ADD_LATENT_VARIABLE = "ALWAYS_ADD_LATENT_VARIABLE"
    ORACLE_REPAIR = "ORACLE_REPAIR"


def baseline_policy(baseline: RepairBaseline) -> ER2RepairPolicy:
    """Construct one deterministic baseline policy."""
    if not isinstance(baseline, RepairBaseline):
        raise TypeError("baseline must be a RepairBaseline")
    fixed = {
        RepairBaseline.ALWAYS_NO_REPAIR: RepairOperator.NO_REPAIR,
        RepairBaseline.ALWAYS_UPDATE_WORLD_MODEL: RepairOperator.UPDATE_WORLD_MODEL,
        RepairBaseline.ALWAYS_RECALIBRATE_SENSOR: RepairOperator.RECALIBRATE_SENSOR,
        RepairBaseline.ALWAYS_ADD_LATENT_VARIABLE: RepairOperator.ADD_LATENT_VARIABLE,
    }
    if baseline is RepairBaseline.ORACLE_REPAIR:
        return OracleRepairPolicy()
    return FixedRepairPolicy(fixed[baseline])
