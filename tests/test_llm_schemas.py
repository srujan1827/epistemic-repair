"""Strict parsing tests for provider-independent LLM decisions."""

import json

import pytest

from epistemic_repair import (
    DecisionType,
    DiagnosticAction,
    FailureMode,
    StructuredResponseError,
    parse_full_autonomous_response,
    parse_planner_only_response,
)
from epistemic_repair.llm.schemas import (
    full_autonomous_json_schema,
    planner_only_json_schema,
)


def _full_payload(**overrides: object) -> str:
    payload = {
        "decision": "RUN_EXPERIMENT",
        "action": "USE_TRUSTED_SENSOR",
        "diagnosis": None,
        "beliefs": {
            "WORLD_SHIFT": 1 / 3,
            "SENSOR_CORRUPTION": 1 / 3,
            "MISSING_LATENT_VARIABLE": 1 / 3,
        },
        "confidence": None,
        "reason_summary": "A trusted measurement will provide useful evidence.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_full_autonomous_run_experiment_parses() -> None:
    decision = parse_full_autonomous_response(_full_payload())

    assert decision.decision is DecisionType.RUN_EXPERIMENT
    assert decision.action is DiagnosticAction.USE_TRUSTED_SENSOR
    assert decision.diagnosis is None


def test_provider_schemas_require_complete_structural_envelopes() -> None:
    full_schema = full_autonomous_json_schema()
    planner_schema = planner_only_json_schema()

    assert full_schema["required"] == [
        "decision",
        "action",
        "diagnosis",
        "beliefs",
        "confidence",
        "reason_summary",
    ]
    assert planner_schema["required"] == [
        "decision",
        "action",
        "diagnosis",
        "reason_summary",
    ]
    assert full_schema["properties"]["action"]["type"] == ["string", "null"]
    assert full_schema["properties"]["diagnosis"]["type"] == ["string", "null"]
    assert full_schema["properties"]["confidence"]["type"] == ["number", "null"]
    assert planner_schema["properties"]["action"]["type"] == ["string", "null"]
    assert planner_schema["properties"]["diagnosis"]["type"] == ["string", "null"]


def test_valid_full_autonomous_diagnosis_parses() -> None:
    decision = parse_full_autonomous_response(
        _full_payload(
            decision="DIAGNOSE",
            action=None,
            diagnosis="SENSOR_CORRUPTION",
            beliefs={
                "WORLD_SHIFT": 0.0,
                "SENSOR_CORRUPTION": 1.0,
                "MISSING_LATENT_VARIABLE": 0.0,
            },
            confidence=0.99,
        )
    )

    assert decision.decision is DecisionType.DIAGNOSE
    assert decision.diagnosis is FailureMode.SENSOR_CORRUPTION
    assert decision.confidence == 0.99


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "decision": "RUN_EXPERIMENT",
                "action": "CHANGE_CONTEXT",
                "diagnosis": None,
                "reason_summary": "Test context sensitivity.",
            },
            DiagnosticAction.CHANGE_CONTEXT,
        ),
        (
            {
                "decision": "DIAGNOSE",
                "action": None,
                "diagnosis": "WORLD_SHIFT",
                "reason_summary": "The evidence is decisive.",
            },
            FailureMode.WORLD_SHIFT,
        ),
    ],
)
def test_valid_planner_only_decisions_parse(
    payload: dict[str, object], expected: object
) -> None:
    decision = parse_planner_only_response(json.dumps(payload))
    assert decision.action is expected or decision.diagnosis is expected


@pytest.mark.parametrize(
    "action",
    ("INSPECT_LATENT_VARIABLE", "RUN_PYTHON", "DELETE_FILES", "UNKNOWN"),
)
def test_unsupported_or_tool_like_actions_are_rejected(action: str) -> None:
    with pytest.raises(StructuredResponseError, match="action"):
        parse_full_autonomous_response(_full_payload(action=action))


def test_invalid_diagnosis_label_is_rejected() -> None:
    with pytest.raises(StructuredResponseError, match="diagnosis"):
        parse_planner_only_response(
            json.dumps(
                {
                    "decision": "DIAGNOSE",
                    "action": None,
                    "diagnosis": "NORMAL",
                    "reason_summary": "Done.",
                }
            )
        )


@pytest.mark.parametrize(
    "beliefs",
    [
        {
            "WORLD_SHIFT": -0.1,
            "SENSOR_CORRUPTION": 0.5,
            "MISSING_LATENT_VARIABLE": 0.6,
        },
        {
            "WORLD_SHIFT": 0.2,
            "SENSOR_CORRUPTION": 0.2,
            "MISSING_LATENT_VARIABLE": 0.2,
        },
        {
            "WORLD_SHIFT": True,
            "SENSOR_CORRUPTION": 0.0,
            "MISSING_LATENT_VARIABLE": 0.0,
        },
        {"WORLD_SHIFT": 0.5, "SENSOR_CORRUPTION": 0.5},
    ],
)
def test_invalid_belief_distributions_are_rejected(
    beliefs: dict[str, object]
) -> None:
    with pytest.raises(StructuredResponseError):
        parse_full_autonomous_response(_full_payload(beliefs=beliefs))


@pytest.mark.parametrize("text", ("", "not json", "[]", "null"))
def test_empty_or_malformed_responses_are_rejected(text: str) -> None:
    with pytest.raises(StructuredResponseError):
        parse_planner_only_response(text)


def test_truncated_json_is_rejected_without_repair() -> None:
    with pytest.raises(StructuredResponseError, match="not valid JSON"):
        parse_full_autonomous_response('{"decision":"RUN_EXPERIMENT"')


def test_run_experiment_with_missing_action_is_rejected() -> None:
    payload = json.loads(_full_payload())
    payload.pop("action")

    with pytest.raises(StructuredResponseError, match="action"):
        parse_full_autonomous_response(json.dumps(payload))


def test_diagnose_with_missing_diagnosis_is_rejected() -> None:
    payload = json.loads(
        _full_payload(decision="DIAGNOSE", action=None, confidence=0.9)
    )
    payload.pop("diagnosis")

    with pytest.raises(StructuredResponseError, match="diagnosis"):
        parse_full_autonomous_response(json.dumps(payload))


@pytest.mark.parametrize("omit_confidence", (False, True))
def test_diagnose_with_null_or_missing_confidence_is_rejected(
    omit_confidence: bool,
) -> None:
    payload = json.loads(
        _full_payload(
            decision="DIAGNOSE",
            action=None,
            diagnosis="WORLD_SHIFT",
            confidence=None,
        )
    )
    if omit_confidence:
        payload.pop("confidence")

    with pytest.raises(StructuredResponseError, match="confidence"):
        parse_full_autonomous_response(json.dumps(payload))


@pytest.mark.parametrize(
    "overrides",
    [
        {"diagnosis": "WORLD_SHIFT"},
        {"decision": "DIAGNOSE", "action": "REPEAT_TRIAL", "diagnosis": "WORLD_SHIFT", "confidence": 1.0},
        {"decision": "DIAGNOSE", "action": None, "diagnosis": "WORLD_SHIFT"},
    ],
)
def test_contradictory_or_incomplete_decision_fields_are_rejected(
    overrides: dict[str, object]
) -> None:
    with pytest.raises(StructuredResponseError):
        parse_full_autonomous_response(_full_payload(**overrides))


def test_arbitrary_code_field_is_rejected() -> None:
    with pytest.raises(StructuredResponseError, match="unsupported response fields"):
        parse_planner_only_response(
            json.dumps(
                {
                    "decision": "RUN_EXPERIMENT",
                    "action": "REPEAT_TRIAL",
                    "diagnosis": None,
                    "reason_summary": "Try again.",
                    "python": "import os; os.system('whoami')",
                }
            )
        )
