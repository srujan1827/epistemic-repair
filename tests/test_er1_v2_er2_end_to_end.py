"""Mock-only tests for the prospective ER-1 V2 -> ER-2 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er2.end_to_end_prompts import (
    build_end_to_end_repair_prompt,
    end_to_end_repair_view,
    format_investigation_evidence,
)
from epistemic_repair.er2.end_to_end_runner import (
    VALID_END_TO_END_OUTCOMES,
    EndToEndEpisodeRunner,
    EndToEndOutcome,
    classify_failure_chain,
    counterfactual_repairs,
)
from epistemic_repair.er2.end_to_end_study import (
    ARTIFACT_FILENAMES,
    EndToEndStudyConfig,
    episode_record,
    preflight_prompt_audit,
    preflight_example_flow,
    run_end_to_end_study,
    write_end_to_end_preflight,
)
from epistemic_repair.er2.llm_prompts import (
    DIAGNOSIS_CAUSAL_DESCRIPTIONS,
    REPAIR_ACTION_DESCRIPTIONS,
    RepairOptionID,
    canonical_text_sha256,
    option_permutation,
)
from epistemic_repair.er2.policies import FixedRepairPolicy
from epistemic_repair.er2.runner import ER2RepairEpisodeRunner
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import (
    LLMClient,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


FROZEN_REPAIR_HASHES = {
    RepairOperator.NO_REPAIR: "717e27bcafed789e957242b6613510b1560a31917f864e1ae694102ffc8024a4",
    RepairOperator.UPDATE_WORLD_MODEL: "d92b95f1aa5192c891336ae2dbec50f5c941416f7ff785e4711816b1c429c722",
    RepairOperator.RECALIBRATE_SENSOR: "7d327ea6774fbfc160e50838b719c86245fa057d0e4c809aaba57b135a6e1894",
    RepairOperator.ADD_LATENT_VARIABLE: "cdbee2a6bce1f3713a99447633d06f9c9c98c7100c7233bbc819011cd56fcb32",
}


class QueueClient(LLMClient):
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(item, f"mock-{len(self.requests)}")


def _diagnosis_response(
    diagnosis: FailureMode,
    *,
    decision: str = "DIAGNOSE",
    action: str | None = None,
) -> str:
    return json.dumps({
        "decision": decision,
        "action": action,
        "diagnosis": diagnosis.value if decision == "DIAGNOSE" else None,
        "beliefs": {
            hypothesis.value: (0.85 if hypothesis is diagnosis else 0.05)
            for hypothesis in ER1_HYPOTHESES
        },
        "confidence": 0.85 if decision == "DIAGNOSE" else None,
        "reason_summary": "The persistent observations support this conclusion.",
    })


def _experiment_response(action: str) -> str:
    return json.dumps({
        "decision": "RUN_EXPERIMENT",
        "action": action,
        "diagnosis": None,
        "beliefs": {hypothesis.value: 0.25 for hypothesis in ER1_HYPOTHESES},
        "confidence": None,
        "reason_summary": "A persistent measurement is needed.",
    })


def _repair_response(option: str) -> str:
    return json.dumps({
        "selected_option": option,
        "confidence": 0.9,
        "rationale": "This is the smallest change supported by the experiment record.",
    })


def _config(*, retries: int = 0) -> LLMConfig:
    return LLMConfig(
        provider="mock",
        model_id="mock-model",
        thinking_level="low",
        max_retries=retries,
        max_decision_calls=9,
    )


def _sensor_episode(client: LLMClient):
    return EndToEndEpisodeRunner(
        client,
        _config(),
        experiment_budget=8,
        diagnosis_threshold=0.95,
    ).run(true_hypothesis=FailureMode.SENSOR_CORRUPTION, seed=0)


def test_repair_prompt_contains_evidence_but_no_supplied_diagnosis_or_causal_truth() -> None:
    client = QueueClient([
        _experiment_response("USE_TRUSTED_SENSOR"),
        _diagnosis_response(FailureMode.SENSOR_CORRUPTION),
        _repair_response("A"),
    ])
    episode = _sensor_episode(client)
    repair_prompt = client.requests[-1].prompt
    assert format_investigation_evidence(
        episode.investigation.trace[-1].agent_view
    ) in repair_prompt
    for hypothesis in ER1_HYPOTHESES:
        assert hypothesis.value not in repair_prompt
    for text in DIAGNOSIS_CAUSAL_DESCRIPTIONS.values():
        assert text not in repair_prompt
    assert "Authoritative current probabilities" not in repair_prompt
    assert "oracle" not in repair_prompt.lower()
    assert "normative" not in repair_prompt.lower()
    assert "held-out" not in repair_prompt.lower()
    assert "evaluator" not in repair_prompt.lower()


def test_repair_prompt_uses_control_wording_without_enum_names() -> None:
    client = QueueClient([
        _diagnosis_response(FailureMode.SENSOR_CORRUPTION),
        _repair_response("A"),
    ])
    episode = _sensor_episode(client)
    prompt = client.requests[-1].prompt
    for repair, description in REPAIR_ACTION_DESCRIPTIONS.items():
        assert prompt.count(description) == 1
        assert repair.value not in prompt
        assert canonical_text_sha256(description) == FROZEN_REPAIR_HASHES[repair]
    assert episode.repair_policy_result is not None
    assert not hasattr(
        end_to_end_repair_view(
            episode.investigation.trace[-1].agent_view,
            episode.permutation,
        ),
        "diagnosis",
    )


def test_selection_maps_through_hidden_permutation_and_unchanged_evaluator() -> None:
    client = QueueClient([
        _diagnosis_response(FailureMode.SENSOR_CORRUPTION),
        _repair_response("A"),
    ])
    episode = _sensor_episode(client)
    assert episode.selected_repair is RepairOperator.RECALIBRATE_SENSOR
    assert episode.outcome is EndToEndOutcome.CORRECT_DIAGNOSIS_CORRECT_REPAIR
    expected = ER2RepairEpisodeRunner().run(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        policy=FixedRepairPolicy(RepairOperator.RECALIBRATE_SENSOR),
    )
    assert episode.metrics == expected.metrics
    record = episode_record(episode, EndToEndStudyConfig(provider="mock", model_id="mock"))
    assert record["row"]["true_hypothesis"] == "SENSOR_CORRUPTION"
    assert "SENSOR_CORRUPTION" not in record["trace"]["repair_prompt"]


def test_option_permutation_is_the_existing_deterministic_mapping() -> None:
    assert dict(option_permutation(0).repair_by_option) == {
        RepairOptionID.A: RepairOperator.RECALIBRATE_SENSOR,
        RepairOptionID.B: RepairOperator.NO_REPAIR,
        RepairOptionID.C: RepairOperator.UPDATE_WORLD_MODEL,
        RepairOptionID.D: RepairOperator.ADD_LATENT_VARIABLE,
    }
    assert dict(option_permutation(7).repair_by_option) == dict(
        option_permutation(7).repair_by_option
    )


def test_failure_decomposition_is_mutually_exclusive_and_complete() -> None:
    observed = set()
    truth = FailureMode.SENSOR_CORRUPTION
    for diagnosis in ER1_HYPOTHESES:
        for repair in RepairOperator:
            category = classify_failure_chain(
                true_hypothesis=truth,
                model_diagnosis=diagnosis,
                selected_repair=repair,
            )
            assert category in VALID_END_TO_END_OUTCOMES
            observed.add(category)
    assert observed == VALID_END_TO_END_OUTCOMES
    assert len({category.value for category in EndToEndOutcome}) == len(EndToEndOutcome)


def test_repair_consistency_uses_model_diagnosis_not_truth() -> None:
    category = classify_failure_chain(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        model_diagnosis=FailureMode.WORLD_SHIFT,
        selected_repair=RepairOperator.UPDATE_WORLD_MODEL,
    )
    assert category is EndToEndOutcome.WRONG_DIAGNOSIS_REPAIR_CONSISTENT_WITH_WRONG_DIAGNOSIS
    rescued = classify_failure_chain(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        model_diagnosis=FailureMode.WORLD_SHIFT,
        selected_repair=RepairOperator.RECALIBRATE_SENSOR,
    )
    assert rescued is EndToEndOutcome.WRONG_DIAGNOSIS_BUT_CORRECT_REPAIR


class CountingDeterministicRunner(ER2RepairEpisodeRunner):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        return super().run(**kwargs)


def test_counterfactual_evaluation_uses_no_llm_or_network_calls() -> None:
    runner = CountingDeterministicRunner()
    rows = counterfactual_repairs(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        model_diagnosis=FailureMode.WORLD_SHIFT,
        selected_repair=RepairOperator.UPDATE_WORLD_MODEL,
        deterministic_runner=runner,
    )
    assert runner.calls == 4
    assert len(rows) == 4
    assert sum(row.chosen for row in rows) == 1
    assert sum(row.oracle for row in rows) == 1
    wrong_world = next(row for row in rows if row.repair is RepairOperator.UPDATE_WORLD_MODEL)
    assert wrong_world.metrics.collateral_damage > 0


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        ("not json", EndToEndOutcome.SCIENTIFIC_MODEL_FAILURE),
        (LLMProviderError("unavailable"), EndToEndOutcome.PROVIDER_FAILURE),
        (LLMRateLimitError("429"), EndToEndOutcome.RATE_LIMIT_FAILURE),
    ],
)
def test_investigation_protocol_failures_remain_separate(
    response: str | Exception,
    outcome: EndToEndOutcome,
) -> None:
    episode = _sensor_episode(QueueClient([response]))
    assert episode.outcome is outcome
    assert episode.selected_repair is None


def test_repair_structured_failure_is_not_counted_as_wrong_repair() -> None:
    episode = _sensor_episode(QueueClient([
        _diagnosis_response(FailureMode.SENSOR_CORRUPTION),
        "not json",
    ]))
    assert episode.outcome is EndToEndOutcome.SCIENTIFIC_MODEL_FAILURE
    assert episode.repair_selection_correct is None
    assert len(episode.counterfactual_repairs) == 4


def test_preflight_scans_40_prompts_and_writes_no_call_artifacts(tmp_path: Path) -> None:
    audit = preflight_prompt_audit(tuple(range(10)))
    assert audit["scanned_prompt_count"] == 40
    assert audit["boundary_passed"] is True
    assert audit["leaks"] == []
    assert audit["same_view_and_seed_is_deterministic"] is True
    output = tmp_path / "preflight"
    write_end_to_end_preflight(output, EndToEndStudyConfig(), overwrite=False)
    assert (output / "prompt_audit.json").is_file()
    assert (output / "example_repair_prompt.txt").is_file()
    assert (output / "example_complete_prompt_flow.json").is_file()
    assert (output / "report.md").is_file()
    flow = preflight_example_flow()
    assert len(flow["calls"]) == 3
    assert "SENSOR_CORRUPTION" not in flow["calls"][-1]["prompt"]


class GridClient(LLMClient):
    """Mock one diagnosis and one correct repair for each four-cell seed-0 grid."""

    def __init__(self) -> None:
        self.cell = 0
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        hypothesis = ER1_HYPOTHESES[self.cell]
        if "selected_option" in request.response_schema.get("properties", {}):
            permutation = option_permutation(0)
            correct = repair_for_failure(hypothesis)
            option = next(
                option.value
                for option, repair in permutation.repair_by_option.items()
                if repair is correct
            )
            self.cell += 1
            return LLMResponse(_repair_response(option), "mock-repair")
        return LLMResponse(_diagnosis_response(hypothesis), "mock-diagnosis")


def test_mocked_study_generates_requested_artifacts(tmp_path: Path) -> None:
    config = EndToEndStudyConfig(
        provider="mock",
        model_id="mock",
        seeds=(0,),
        max_retries=0,
        min_request_interval_seconds=0,
        rate_limit_backoff_seconds=0,
        episode_cooldown_seconds=0,
    )
    client = GridClient()
    output = tmp_path / "study"
    records = run_end_to_end_study(client, config, output)
    assert len(records) == 4
    assert len(client.requests) == 8
    for filename in ARTIFACT_FILENAMES:
        assert (output / filename).is_file()
    assert all(record["row"]["repair_selection_correct"] for record in records)
    assert sum(len(record["counterfactual"]) for record in records) == 16
