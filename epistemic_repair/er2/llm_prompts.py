"""Frozen wording and seed-controlled option permutations for ER-2 LLM repair selection."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from epistemic_repair.er2.evaluation import ER2_HYPOTHESES
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator


ER2_REPAIR_PROMPT_VERSION = "binary_er2_repair_selection_001"


class ER2LLMCondition(str, Enum):
    """The single frozen ER-2 LLM condition."""

    CAUSAL_REPAIR_SELECTION = "CAUSAL_REPAIR_SELECTION"


class RepairOptionID(str, Enum):
    """Neutral identifiers exposed to the model."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


# These are scientific-method constants. Change only with a new prompt version.
_REPAIR_ACTION_DESCRIPTIONS = {
    RepairOperator.NO_REPAIR: (
        "Leave the current internal predictive state unchanged."
    ),
    RepairOperator.UPDATE_WORLD_MODEL: (
        "Change the assumed physical relationship between input and true output, "
        "while leaving observation-channel interpretation and contextual "
        "representation unchanged."
    ),
    RepairOperator.RECALIBRATE_SENSOR: (
        "Change how readings from the primary observation channel are interpreted, "
        "while leaving the physical relationship and contextual representation "
        "unchanged."
    ),
    RepairOperator.ADD_LATENT_VARIABLE: (
        "Represent behavior as context-dependent rather than using one global "
        "relationship, while leaving sensor interpretation unchanged."
    ),
}
REPAIR_ACTION_DESCRIPTIONS: Mapping[RepairOperator, str] = MappingProxyType(
    _REPAIR_ACTION_DESCRIPTIONS
)

_DIAGNOSIS_CAUSAL_DESCRIPTIONS = {
    FailureMode.NO_STRUCTURAL_CHANGE: (
        "Persistent physical, observation-channel, and contextual mechanisms remain "
        "healthy; the triggering anomaly does not justify structural adaptation."
    ),
    FailureMode.WORLD_SHIFT: (
        "The observation mechanism remains healthy, while the underlying physical "
        "relationship has persistently changed."
    ),
    FailureMode.SENSOR_CORRUPTION: (
        "The underlying physical relationship remains stable, while the primary "
        "observation channel persistently misreports it."
    ),
    FailureMode.MISSING_LATENT_VARIABLE: (
        "One global physical relationship is inadequate because behavior differs "
        "systematically with context; the observation mechanism remains healthy."
    ),
}
DIAGNOSIS_CAUSAL_DESCRIPTIONS: Mapping[FailureMode, str] = MappingProxyType(
    _DIAGNOSIS_CAUSAL_DESCRIPTIONS
)


@dataclass(frozen=True, slots=True)
class RepairOption:
    """One safe option exposed to the LLM, without its hidden repair identity."""

    option_id: RepairOptionID
    description: str


@dataclass(frozen=True, slots=True)
class ER2LLMPromptView:
    """The complete provider-safe ER-2 prompt input."""

    diagnosis_description: str
    options: tuple[RepairOption, ...]

    def __post_init__(self) -> None:
        if self.diagnosis_description not in DIAGNOSIS_CAUSAL_DESCRIPTIONS.values():
            raise ValueError("diagnosis_description must use the frozen canonical text")
        if tuple(option.option_id for option in self.options) != tuple(RepairOptionID):
            raise ValueError("options must be ordered A, B, C, D")
        expected = set(REPAIR_ACTION_DESCRIPTIONS.values())
        actual = {option.description for option in self.options}
        if actual != expected or len(actual) != len(self.options):
            raise ValueError("options must use each frozen repair description once")


@dataclass(frozen=True, slots=True)
class RepairOptionPermutation:
    """Evaluation-only option-to-repair mapping hidden from the prompt builder."""

    seed: int
    repair_by_option: Mapping[RepairOptionID, RepairOperator]

    def __post_init__(self) -> None:
        if set(self.repair_by_option) != set(RepairOptionID):
            raise ValueError("repair_by_option must define A, B, C, and D")
        if set(self.repair_by_option.values()) != set(RepairOperator):
            raise ValueError("repair_by_option must contain each repair exactly once")

    def prompt_view(self, diagnosis: FailureMode) -> ER2LLMPromptView:
        """Drop the hidden mapping and construct the provider-safe view."""
        if diagnosis not in ER2_HYPOTHESES:
            raise ValueError("diagnosis must be an ER-2 hypothesis")
        return ER2LLMPromptView(
            diagnosis_description=DIAGNOSIS_CAUSAL_DESCRIPTIONS[diagnosis],
            options=tuple(
                RepairOption(
                    option_id=option_id,
                    description=REPAIR_ACTION_DESCRIPTIONS[self.repair_by_option[option_id]],
                )
                for option_id in RepairOptionID
            ),
        )

    def repair_for(self, option_id: RepairOptionID) -> RepairOperator:
        """Translate an LLM option through the evaluation-only mapping."""
        return self.repair_by_option[option_id]


def option_permutation(seed: int) -> RepairOptionPermutation:
    """Create the deterministic A/B/C/D permutation controlled only by episode seed."""
    repairs = list(RepairOperator)
    random.Random(seed).shuffle(repairs)
    return RepairOptionPermutation(
        seed=seed,
        repair_by_option=MappingProxyType(dict(zip(RepairOptionID, repairs, strict=True))),
    )


def build_repair_selection_prompt(view: ER2LLMPromptView) -> str:
    """Render the single versioned prompt from a structurally safe view."""
    if not isinstance(view, ER2LLMPromptView):
        raise TypeError("view must be an ER2LLMPromptView")
    option_lines = "\n".join(
        f"{option.option_id.value}: {option.description}" for option in view.options
    )
    return (
        "ER-2 causal repair selection\n"
        f"Prompt version: {ER2_REPAIR_PROMPT_VERSION}\n\n"
        "Task:\n"
        "An external investigation has supplied a diagnosis and causal interpretation. "
        "Choose exactly one candidate repair action. You do not directly modify the "
        "system; a trusted evaluator will apply the selected option and test its "
        "consequences on held-out cases.\n\n"
        "Causal finding from the external investigation:\n"
        f"{view.diagnosis_description}\n\n"
        "Candidate repair actions:\n"
        f"{option_lines}\n\n"
        "Selection rule:\n"
        "Choose the action whose stated consequences are best justified by the causal "
        "interpretation, while preserving components described as healthy.\n\n"
        "Output:\n"
        "Return only one JSON object with selected_option (A, B, C, or D), confidence "
        "from 0 to 1, and a short rationale. The rationale may identify which internal "
        "component should or should not change."
    )


def canonical_text_sha256(text: str) -> str:
    """Return the UTF-8 SHA-256 used by the wording audit."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
