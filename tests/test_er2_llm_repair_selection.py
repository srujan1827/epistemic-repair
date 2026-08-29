"""Regression tests for the small provider-neutral ER-2 LLM study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from epistemic_repair.er2.evaluation import ER2_HYPOTHESES
from epistemic_repair.er2.llm_policy import ER2CausalRepairLLMPolicy
from epistemic_repair.er2.llm_prompts import (
    DIAGNOSIS_CAUSAL_DESCRIPTIONS,
    REPAIR_ACTION_DESCRIPTIONS,
    RepairOptionID,
    build_repair_selection_prompt,
    option_permutation,
)
from epistemic_repair.er2.llm_runner import (
    ER2LLMOutcome,
    ER2LLMRepairEpisodeRunner,
)
from epistemic_repair.er2.llm_schemas import (
    ER2StructuredResponseError,
    parse_er2_repair_selection_response,
)
from epistemic_repair.er2.llm_study import (
    ARTIFACT_FILENAMES,
    ER2LLMStudyConfig,
    episode_record,
    permutation_audit_rows,
    run_er2_llm_study,
    wording_audit,
    write_preflight_artifacts,
)
from epistemic_repair.er2.policies import FixedRepairPolicy
from epistemic_repair.er2.runner import ER2RepairEpisodeRunner
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import (
    LLMClient,
    LLMFormatError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


class StaticClient(LLMClient):
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(self.text, "mock-request")


class RaisingClient(LLMClient):
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise self.error


def _response(option: str, confidence: float = 0.91) -> str:
    return json.dumps({
        "selected_option": option,
        "confidence": confidence,
        "rationale": "This changes the implicated component and preserves healthy ones.",
    })


def _runner(client: LLMClient, *, retries: int = 0) -> ER2LLMRepairEpisodeRunner:
    config = LLMConfig(max_retries=retries, max_decision_calls=1)
    return ER2LLMRepairEpisodeRunner(ER2CausalRepairLLMPolicy(client, config))


def test_canonical_repair_names_are_absent_from_descriptions_and_prompt() -> None:
    text = "\n".join(REPAIR_ACTION_DESCRIPTIONS.values())
    for repair in RepairOperator:
        assert repair.value not in text
    for hypothesis in ER2_HYPOTHESES:
        for seed in range(10):
            prompt = build_repair_selection_prompt(
                option_permutation(seed).prompt_view(hypothesis)
            )
            for repair in RepairOperator:
                assert repair.value not in prompt


@pytest.mark.parametrize(
    "hidden_label",
    (
        "SENSOR_CORRUPTION",
        "WORLD_SHIFT",
        "NO_STRUCTURAL_CHANGE",
        "MISSING_LATENT_VARIABLE",
    ),
)
def test_canonical_diagnosis_label_is_absent_from_all_prompts(
    hidden_label: str,
) -> None:
    for hypothesis in ER2_HYPOTHESES:
        for seed in range(10):
            prompt = build_repair_selection_prompt(
                option_permutation(seed).prompt_view(hypothesis)
            )
            assert hidden_label not in prompt


def test_option_randomization_is_deterministic_and_changes_correct_positions() -> None:
    assert dict(option_permutation(7).repair_by_option) == dict(
        option_permutation(7).repair_by_option
    )
    for hypothesis in ER2_HYPOTHESES:
        correct = repair_for_failure(hypothesis)
        positions = {
            option
            for seed in range(10)
            for option, repair in option_permutation(seed).repair_by_option.items()
            if repair is correct
        }
        assert len(positions) > 1


def test_only_letters_and_descriptions_cross_the_prompt_boundary() -> None:
    permutation = option_permutation(3)
    view = permutation.prompt_view(FailureMode.WORLD_SHIFT)
    assert not hasattr(view, "repair_by_option")
    assert not hasattr(view, "diagnosis")
    assert not any(hasattr(option, "repair") for option in view.options)
    prompt = build_repair_selection_prompt(view)
    assert "option_permutation" not in prompt
    assert "correct repair" not in prompt.lower()


def test_canonical_wording_is_byte_identical_across_all_40_cells() -> None:
    for hypothesis in ER2_HYPOTHESES:
        diagnosis_texts = {
            option_permutation(seed).prompt_view(hypothesis).diagnosis_description
            for seed in range(10)
        }
        assert diagnosis_texts == {DIAGNOSIS_CAUSAL_DESCRIPTIONS[hypothesis]}
    for repair, canonical in REPAIR_ACTION_DESCRIPTIONS.items():
        observed = set()
        for hypothesis in ER2_HYPOTHESES:
            for seed in range(10):
                permutation = option_permutation(seed)
                view = permutation.prompt_view(hypothesis)
                observed.add(next(
                    option.description
                    for option in view.options
                    if permutation.repair_by_option[option.option_id] is repair
                ))
        assert observed == {canonical}


def test_same_hypothesis_and_permutation_produce_exact_same_prompt() -> None:
    first = build_repair_selection_prompt(
        option_permutation(5).prompt_view(FailureMode.MISSING_LATENT_VARIABLE)
    )
    second = build_repair_selection_prompt(
        option_permutation(5).prompt_view(FailureMode.MISSING_LATENT_VARIABLE)
    )
    assert first == second


def test_selected_letter_maps_through_hidden_permutation_and_reuses_evaluator() -> None:
    result = _runner(StaticClient(_response("A"))).run(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        seed=0,
    )
    assert result.selected_repair is RepairOperator.RECALIBRATE_SENSOR
    assert result.outcome is ER2LLMOutcome.VALID_CORRECT_REPAIR
    expected = ER2RepairEpisodeRunner().run(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        policy=FixedRepairPolicy(RepairOperator.RECALIBRATE_SENSOR),
    )
    assert result.metrics == expected.metrics
    record = episode_record(result, ER2LLMStudyConfig())
    assert record["true_hypothesis"] == "SENSOR_CORRUPTION"
    assert record["supplied_diagnosis"] == "SENSOR_CORRUPTION"
    assert "SENSOR_CORRUPTION" not in record["prompt"]


def test_valid_wrong_repair_is_distinct_from_malformed_output() -> None:
    wrong = _runner(StaticClient(_response("C"))).run(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        seed=0,
    )
    malformed = _runner(StaticClient("not json")).run(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        seed=0,
    )
    assert wrong.outcome is ER2LLMOutcome.VALID_WRONG_REPAIR
    assert wrong.metrics is not None
    assert malformed.outcome is ER2LLMOutcome.SCIENTIFIC_MODEL_FAILURE
    assert malformed.metrics is None

    empty_provider_output = _runner(RaisingClient(LLMFormatError("empty output"))).run(
        true_hypothesis=FailureMode.SENSOR_CORRUPTION,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        seed=0,
    )
    assert empty_provider_output.outcome is ER2LLMOutcome.SCIENTIFIC_MODEL_FAILURE


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (LLMProviderError("unavailable"), ER2LLMOutcome.PROVIDER_FAILURE),
        (LLMRateLimitError("429"), ER2LLMOutcome.RATE_LIMIT_FAILURE),
    ],
)
def test_provider_failure_taxonomy(error: Exception, outcome: ER2LLMOutcome) -> None:
    result = _runner(RaisingClient(error)).run(
        true_hypothesis=FailureMode.WORLD_SHIFT,
        diagnosis=FailureMode.WORLD_SHIFT,
        seed=0,
    )
    assert result.outcome is outcome
    assert result.metrics is None


def test_strict_parser_rejects_prose_prefix_and_malformed_json() -> None:
    with pytest.raises(ER2StructuredResponseError):
        parse_er2_repair_selection_response("Here is the JSON: " + _response("A"))
    with pytest.raises(ER2StructuredResponseError):
        parse_er2_repair_selection_response('{"selected_option":"A"')


def test_wording_and_permutation_audits_cover_the_frozen_grid(tmp_path: Path) -> None:
    audit = wording_audit(tuple(range(10)))
    assert audit["episode_count"] == 40
    assert audit["all_episode_strings_are_canonical"] is True
    assert audit["same_input_prompt_is_deterministic"] is True
    assert audit["only_option_assignment_varies_across_seeds"] is True
    assert audit["scanned_prompt_count"] == 40
    assert audit["prompt_label_leakage_free"] is True
    assert audit["prompt_label_leaks"] == []
    rows = permutation_audit_rows(tuple(range(10)))
    assert len(rows) == 10
    assert len({row["correct_option_SENSOR_CORRUPTION"] for row in rows}) > 1

    output = tmp_path / "preflight"
    write_preflight_artifacts(output, ER2LLMStudyConfig(), overwrite=False)
    written = json.loads((output / "wording_audit.json").read_text(encoding="utf-8"))
    assert written["episode_count"] == 40
    assert written["prompt_label_leakage_free"] is True
    assert (output / "example_prompt.txt").is_file()


def test_mocked_40_cell_study_generates_all_requested_artifacts(tmp_path: Path) -> None:
    config = ER2LLMStudyConfig(
        max_retries=0,
        min_request_interval_seconds=0,
        rate_limit_backoff_seconds=0,
        episode_cooldown_seconds=0,
    )
    client = StaticClient(_response("A"))
    output = tmp_path / "study"
    rows = run_er2_llm_study(client, config, output)
    assert len(rows) == 40
    assert len(client.requests) == 40
    for filename in ARTIFACT_FILENAMES:
        assert (output / filename).is_file()
    with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["valid_selections"] == "40"
    assert summary["scientific_model_failures"] == "0"
    with (output / "wrong_repair_analysis.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        wrong_rows = list(csv.DictReader(handle))
    assert wrong_rows
    assert any(
        row["partial_affected_improvement_with_collateral_damage"] == "True"
        for row in wrong_rows
    )


def test_policy_request_schema_cannot_issue_tools_or_arbitrary_code() -> None:
    client = StaticClient(_response("B"))
    _runner(client).run(
        true_hypothesis=FailureMode.NO_STRUCTURAL_CHANGE,
        diagnosis=FailureMode.NO_STRUCTURAL_CHANGE,
        seed=0,
    )
    schema = client.requests[0].response_schema
    assert schema["properties"]["selected_option"]["enum"] == ["A", "B", "C", "D"]
    assert set(schema["properties"]) == {"selected_option", "confidence", "rationale"}
