"""Strict structured decisions for the two LLM experimental conditions."""

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from typing import Any, Mapping, TypeAlias

from epistemic_repair.beliefs.state import HypothesisBeliefs
from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, DiagnosticAction
from epistemic_repair.failures.modes import FailureMode


class LLMCondition(str, Enum):
    """Distinct LLM investigator conditions that must not be pooled."""

    FULL_AUTONOMOUS = "FULL_AUTONOMOUS"
    PLANNER_ONLY = "PLANNER_ONLY"


class DecisionType(str, Enum):
    """Whether the model requests evidence or ends with a diagnosis."""

    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    DIAGNOSE = "DIAGNOSE"


class StructuredResponseError(ValueError):
    """A provider response that does not satisfy the experiment schema."""


@dataclass(frozen=True, slots=True)
class FullAutonomousDecision:
    """Validated decision including the autonomous model's stated beliefs."""

    decision: DecisionType
    beliefs: HypothesisBeliefs
    reason_summary: str
    action: DiagnosticAction | None = None
    diagnosis: FailureMode | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class PlannerOnlyDecision:
    """Validated planner decision; normative beliefs remain authoritative."""

    decision: DecisionType
    reason_summary: str
    action: DiagnosticAction | None = None
    diagnosis: FailureMode | None = None


LLMDecision: TypeAlias = FullAutonomousDecision | PlannerOnlyDecision


_HYPOTHESIS_NAMES = tuple(hypothesis.value for hypothesis in (
    FailureMode.WORLD_SHIFT,
    FailureMode.SENSOR_CORRUPTION,
    FailureMode.MISSING_LATENT_VARIABLE,
))
_ACTION_NAMES = tuple(action.value for action in BENCHMARK_ACTIONS)


def full_autonomous_json_schema() -> dict[str, Any]:
    """Return provider-neutral JSON Schema for autonomous decisions."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": [item.value for item in DecisionType]},
            "action": {"type": ["string", "null"], "enum": [*_ACTION_NAMES, None]},
            "diagnosis": {
                "type": ["string", "null"],
                "enum": [*_HYPOTHESIS_NAMES, None],
            },
            "beliefs": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    for name in _HYPOTHESIS_NAMES
                },
                "required": list(_HYPOTHESIS_NAMES),
            },
            "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
            "reason_summary": {"type": "string", "maxLength": 600},
        },
        "required": [
            "decision",
            "action",
            "diagnosis",
            "beliefs",
            "confidence",
            "reason_summary",
        ],
    }


def planner_only_json_schema() -> dict[str, Any]:
    """Return provider-neutral JSON Schema for planner-only decisions."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": [item.value for item in DecisionType]},
            "action": {"type": ["string", "null"], "enum": [*_ACTION_NAMES, None]},
            "diagnosis": {
                "type": ["string", "null"],
                "enum": [*_HYPOTHESIS_NAMES, None],
            },
            "reason_summary": {"type": "string", "maxLength": 600},
        },
        "required": ["decision", "action", "diagnosis", "reason_summary"],
    }


def parse_full_autonomous_response(text: str) -> FullAutonomousDecision:
    """Parse and validate a full-autonomous response without guessing intent."""
    data = _parse_object(text)
    _reject_extra_fields(
        data,
        {"decision", "action", "diagnosis", "beliefs", "confidence", "reason_summary"},
    )
    decision = _parse_decision_type(data)
    beliefs_data = data.get("beliefs")
    if not isinstance(beliefs_data, Mapping):
        raise StructuredResponseError("beliefs must be an object")
    if set(beliefs_data) != set(_HYPOTHESIS_NAMES):
        raise StructuredResponseError("beliefs must contain exactly three hypotheses")
    try:
        beliefs = HypothesisBeliefs.from_weights(
            {
                FailureMode.WORLD_SHIFT: _strict_probability(
                    beliefs_data[FailureMode.WORLD_SHIFT.value], "WORLD_SHIFT"
                ),
                FailureMode.SENSOR_CORRUPTION: _strict_probability(
                    beliefs_data[FailureMode.SENSOR_CORRUPTION.value],
                    "SENSOR_CORRUPTION",
                ),
                FailureMode.MISSING_LATENT_VARIABLE: _strict_probability(
                    beliefs_data[FailureMode.MISSING_LATENT_VARIABLE.value],
                    "MISSING_LATENT_VARIABLE",
                ),
            }
        )
    except ValueError as error:
        raise StructuredResponseError(str(error)) from error
    supplied_total = sum(
        _strict_probability(beliefs_data[name], name) for name in _HYPOTHESIS_NAMES
    )
    if abs(supplied_total - 1.0) > 1e-6:
        raise StructuredResponseError("belief probabilities must sum approximately to 1")

    confidence = data.get("confidence")
    if confidence is not None:
        confidence = _strict_probability(confidence, "confidence")
    action, diagnosis = _validate_decision_fields(data, decision)
    if decision is DecisionType.DIAGNOSE and confidence is None:
        raise StructuredResponseError("DIAGNOSE requires confidence")
    return FullAutonomousDecision(
        decision=decision,
        beliefs=beliefs,
        reason_summary=_parse_reason(data),
        action=action,
        diagnosis=diagnosis,
        confidence=confidence,
    )


def parse_planner_only_response(text: str) -> PlannerOnlyDecision:
    """Parse and validate a planner-only response without guessing intent."""
    data = _parse_object(text)
    _reject_extra_fields(data, {"decision", "action", "diagnosis", "reason_summary"})
    decision = _parse_decision_type(data)
    action, diagnosis = _validate_decision_fields(data, decision)
    return PlannerOnlyDecision(
        decision=decision,
        reason_summary=_parse_reason(data),
        action=action,
        diagnosis=diagnosis,
    )


def _parse_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise StructuredResponseError("provider response is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise StructuredResponseError("provider response is not valid JSON") from error
    if not isinstance(data, dict):
        raise StructuredResponseError("structured response must be a JSON object")
    return data


def _reject_extra_fields(data: Mapping[str, Any], allowed: set[str]) -> None:
    extras = set(data) - allowed
    if extras:
        raise StructuredResponseError(f"unsupported response fields: {sorted(extras)}")


def _parse_decision_type(data: Mapping[str, Any]) -> DecisionType:
    try:
        return DecisionType(data.get("decision"))
    except (TypeError, ValueError) as error:
        raise StructuredResponseError("decision must be RUN_EXPERIMENT or DIAGNOSE") from error


def _validate_decision_fields(
    data: Mapping[str, Any], decision: DecisionType
) -> tuple[DiagnosticAction | None, FailureMode | None]:
    action_value = data.get("action")
    diagnosis_value = data.get("diagnosis")
    if decision is DecisionType.RUN_EXPERIMENT:
        if diagnosis_value is not None:
            raise StructuredResponseError("RUN_EXPERIMENT cannot include diagnosis")
        try:
            action = DiagnosticAction(action_value)
        except (TypeError, ValueError) as error:
            raise StructuredResponseError("unsupported or missing benchmark action") from error
        if action not in BENCHMARK_ACTIONS:
            raise StructuredResponseError("unsupported benchmark action")
        return action, None

    if action_value is not None:
        raise StructuredResponseError("DIAGNOSE cannot include action")
    try:
        diagnosis = FailureMode(diagnosis_value)
    except (TypeError, ValueError) as error:
        raise StructuredResponseError("unsupported or missing diagnosis") from error
    if diagnosis not in (
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        raise StructuredResponseError("diagnosis is not a benchmark hypothesis")
    return None, diagnosis


def _strict_probability(value: object, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise StructuredResponseError(f"{name} must be a finite number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise StructuredResponseError(f"{name} must be in [0, 1]")
    return number


def _parse_reason(data: Mapping[str, Any]) -> str:
    reason = data.get("reason_summary")
    if not isinstance(reason, str) or not reason.strip():
        raise StructuredResponseError("reason_summary must be a non-empty string")
    reason = reason.strip()
    if len(reason) > 600:
        raise StructuredResponseError("reason_summary exceeds 600 characters")
    return reason
