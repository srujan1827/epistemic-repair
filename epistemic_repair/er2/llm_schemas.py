"""Strict structured-output schema and parser for ER-2 repair selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from epistemic_repair.er2.llm_prompts import RepairOptionID


class ER2StructuredResponseError(ValueError):
    """The provider response did not satisfy the frozen ER-2 contract."""


@dataclass(frozen=True, slots=True)
class ER2RepairSelectionDecision:
    """One validated neutral repair selection from the LLM."""

    selected_option: RepairOptionID
    confidence: float
    rationale: str


def er2_repair_selection_json_schema() -> dict[str, Any]:
    """Return the provider-neutral strict JSON schema."""
    return {
        "type": "object",
        "properties": {
            "selected_option": {
                "type": "string",
                "enum": [option.value for option in RepairOptionID],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
        },
        "required": ["selected_option", "confidence", "rationale"],
        "additionalProperties": False,
    }


def parse_er2_repair_selection_response(text: str) -> ER2RepairSelectionDecision:
    """Parse exactly one raw JSON object without prose stripping or extraction."""
    if not isinstance(text, str) or not text.strip():
        raise ER2StructuredResponseError("response must be non-empty JSON text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ER2StructuredResponseError("response must be exactly one JSON object") from error
    if not isinstance(payload, dict):
        raise ER2StructuredResponseError("response must be a JSON object")
    required = {"selected_option", "confidence", "rationale"}
    if set(payload) != required:
        raise ER2StructuredResponseError(
            "response must contain exactly selected_option, confidence, and rationale"
        )
    try:
        option = RepairOptionID(payload["selected_option"])
    except (TypeError, ValueError) as error:
        raise ER2StructuredResponseError("selected_option must be A, B, C, or D") from error
    confidence = payload["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ER2StructuredResponseError("confidence must be a finite number from 0 to 1")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ER2StructuredResponseError("rationale must be a non-empty string")
    if len(rationale) > 600:
        raise ER2StructuredResponseError("rationale must contain at most 600 characters")
    return ER2RepairSelectionDecision(
        selected_option=option,
        confidence=float(confidence),
        rationale=rationale.strip(),
    )
