"""Tests for zero-call ER-1 V2 three-condition artifact analysis."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

import epistemic_repair.evaluation.er1_v2_three_condition_analysis as analysis_module
from epistemic_repair.evaluation.er1_v2_three_condition_analysis import (
    CONDITIONS,
    HYPOTHESES,
    IncompatibleAnalysisInput,
    combine_three_condition_results,
    stopping_calibration,
)


def _make_run(
    directory: Path,
    conditions: tuple[str, ...],
    *,
    seeds: tuple[int, ...] = (0,),
    threshold: float = 0.95,
) -> None:
    directory.mkdir()
    threshold_aware = conditions == ("THRESHOLD_AWARE_AUTONOMOUS",)
    prompt_version = (
        "binary_er1_v2_threshold_aware_001"
        if threshold_aware
        else "binary_er1_v2_001"
    )
    config = {
        "benchmark_version": "binary_er1_v2",
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "thinking_level": "low",
        "experiment_budget": 8,
        "diagnosis_threshold": threshold,
        "max_decision_calls": 9,
        "seeds": list(seeds),
        "hypotheses": list(HYPOTHESES),
        "conditions": list(conditions),
        "prompt_version": prompt_version,
        "planned_episode_count": len(conditions) * len(HYPOTHESES) * len(seeds),
    }
    (directory / "run_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    rows = []
    for condition in conditions:
        for hypothesis in HYPOTHESES:
            for seed in seeds:
                autonomous = condition != "PLANNER_ONLY"
                rows.append({
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "benchmark_version": "binary_er1_v2",
                    "prompt_version": prompt_version,
                    "provider": "gemini",
                    "model": "gemini-3.6-flash",
                    "thinking_level": "low",
                    "condition": condition,
                    "true_hypothesis": hypothesis,
                    "seed": seed,
                    "experiment_budget": 8,
                    "diagnosis_threshold": threshold,
                    "max_decision_calls": 9,
                    "final_diagnosis": hypothesis,
                    "diagnosis_correct": True,
                    "diagnosed_correctly_within_budget": True,
                    "threshold_qualified_success": condition == "THRESHOLD_AWARE_AUTONOMOUS",
                    "premature_diagnosis": condition != "THRESHOLD_AWARE_AUTONOMOUS",
                    "normative_probability_of_final_diagnosis": 0.90 if condition != "THRESHOLD_AWARE_AUTONOMOUS" else 0.96,
                    "model_reported_final_confidence": 0.95 if autonomous else 0.90,
                    "experiments_used": 2 if condition != "THRESHOLD_AWARE_AUTONOMOUS" else 3,
                    "decision_calls": 3 if condition != "THRESHOLD_AWARE_AUTONOMOUS" else 4,
                    "total_retries": 0,
                    "termination_reason": "DIAGNOSED",
                    "cumulative_action_regret": {
                        "FULL_AUTONOMOUS": 0.3,
                        "PLANNER_ONLY": 0.1,
                        "THRESHOLD_AWARE_AUTONOMOUS": 0.2,
                    }[condition],
                    "oracle_action_agreements": 1,
                    "oracle_action_agreement_rate": 0.5,
                    "mean_autonomous_belief_l1_error": 0.1 if autonomous else "",
                    "provider_failure_flag": False,
                    "scientific_model_failure_flag": False,
                    "action_sequence": "USE_TRUSTED_SENSOR>REPEAT_TRIAL",
                })
    _write_rows(directory / "episodes.csv", rows)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_inputs(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    _make_run(first, ("FULL_AUTONOMOUS", "PLANNER_ONLY"))
    _make_run(second, ("THRESHOLD_AWARE_AUTONOMOUS",))
    return first, second


def test_separate_condition_runs_merge_into_matched_triplets(tmp_path) -> None:
    first, second = _valid_inputs(tmp_path)
    output = tmp_path / "combined"

    result = combine_three_condition_results((first, second), output)

    assert len(result["rows"]) == 12
    assert len(result["matched"]) == 4
    assert {row["condition"] for row in result["summary"]} == set(CONDITIONS)
    assert result["prompt_versions"] == {
        "FULL_AUTONOMOUS": "binary_er1_v2_001",
        "PLANNER_ONLY": "binary_er1_v2_001",
        "THRESHOLD_AWARE_AUTONOMOUS": "binary_er1_v2_threshold_aware_001",
    }
    assert {path.name for path in output.iterdir()} == {
        "episodes_combined.csv",
        "summary_three_conditions.csv",
        "per_hypothesis_three_conditions.csv",
        "matched_three_condition.csv",
        "report.md",
    }


def test_duplicate_condition_across_inputs_is_rejected(tmp_path) -> None:
    first, second = _valid_inputs(tmp_path)
    duplicate = tmp_path / "duplicate"
    _make_run(duplicate, ("FULL_AUTONOMOUS",))

    with pytest.raises(IncompatibleAnalysisInput, match="duplicate conditions"):
        combine_three_condition_results((first, second, duplicate), tmp_path / "out")


def test_duplicate_episode_inside_run_is_rejected(tmp_path) -> None:
    first, second = _valid_inputs(tmp_path)
    with (second / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(dict(rows[0]))
    _write_rows(second / "episodes.csv", rows)
    config = json.loads((second / "run_config.json").read_text(encoding="utf-8"))
    config["planned_episode_count"] += 1
    (second / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(IncompatibleAnalysisInput, match="duplicate episode key"):
        combine_three_condition_results((first, second), tmp_path / "out")


def test_incompatible_configuration_is_rejected(tmp_path) -> None:
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    _make_run(first, ("FULL_AUTONOMOUS", "PLANNER_ONLY"), threshold=0.95)
    _make_run(second, ("THRESHOLD_AWARE_AUTONOMOUS",), threshold=0.90)

    with pytest.raises(IncompatibleAnalysisInput, match="diagnosis_threshold"):
        combine_three_condition_results((first, second), tmp_path / "out")


def test_matched_triplet_has_side_by_side_values_and_numeric_deltas(tmp_path) -> None:
    first, second = _valid_inputs(tmp_path)
    result = combine_three_condition_results((first, second), tmp_path / "out")
    row = result["matched"][0]

    assert row["pair_key"] == "NO_STRUCTURAL_CHANGE:0"
    assert row["full_final_diagnosis"] == "NO_STRUCTURAL_CHANGE"
    assert row["planner_final_self_reported_confidence"] is None
    assert row["threshold_aware_experiments_used"] == 3
    assert row["planner_minus_full_cumulative_action_regret"] == pytest.approx(-0.2)
    assert row["threshold_aware_minus_full_experiments_used"] == 1.0
    assert "threshold_aware_minus_full_final_diagnosis" not in row


def test_confidence_calculation_excludes_missing_values_and_planner() -> None:
    rows = []
    for condition in CONDITIONS:
        for index, hypothesis in enumerate(HYPOTHESES):
            rows.append({
                "condition": condition,
                "true_hypothesis": hypothesis,
                "model_reported_final_confidence": (
                    "" if condition == "PLANNER_ONLY" or index == 0 else "0.95"
                ),
                "normative_probability_of_final_diagnosis": "0.90",
            })
    result = stopping_calibration(rows, 0.95)
    overall = {
        row["condition"]: row
        for row in result
        if row["true_hypothesis"] == "ALL"
    }

    assert set(overall) == {
        "FULL_AUTONOMOUS",
        "THRESHOLD_AWARE_AUTONOMOUS",
    }
    assert overall["FULL_AUTONOMOUS"]["comparable_confidence_count"] == 3
    assert overall["FULL_AUTONOMOUS"]["missing_confidence_count"] == 1
    assert overall["FULL_AUTONOMOUS"]["mean_signed_gap"] == pytest.approx(0.05)
    assert overall["FULL_AUTONOMOUS"]["mean_absolute_gap"] == pytest.approx(0.05)
    assert overall["FULL_AUTONOMOUS"]["overconfidence_frequency"] == 1.0
    assert overall["FULL_AUTONOMOUS"]["self_ge_threshold_normative_lt_count"] == 3


def test_analysis_module_has_no_provider_or_network_dependency() -> None:
    source = inspect.getsource(analysis_module)
    assert "create_llm_client" not in source
    assert "GeminiLLMClient" not in source
    assert "requests." not in source
    assert "urllib" not in source
