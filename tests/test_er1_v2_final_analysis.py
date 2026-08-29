"""Tests for the zero-API-call seeds 0..9 ER-1 V2 final analysis."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

import epistemic_repair.evaluation.er1_v2_final_analysis as final_module
from epistemic_repair.evaluation.er1_v2_final_analysis import (
    build_calibration,
    classify_episode,
    combine_final_analysis,
    wilson_interval,
)
from epistemic_repair.evaluation.er1_v2_three_condition_analysis import (
    CONDITIONS,
    HYPOTHESES,
    IncompatibleAnalysisInput,
)


def _row(condition: str, hypothesis: str, seed: int) -> dict[str, object]:
    autonomous = condition != "PLANNER_ONLY"
    threshold_aware = condition == "THRESHOLD_AWARE_AUTONOMOUS"
    return {
        "timestamp": "2026-08-29T00:00:00+00:00",
        "benchmark_version": "binary_er1_v2",
        "prompt_version": (
            "binary_er1_v2_threshold_aware_001"
            if threshold_aware else "binary_er1_v2_001"
        ),
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "thinking_level": "low",
        "condition": condition,
        "true_hypothesis": hypothesis,
        "seed": seed,
        "experiment_budget": 8,
        "diagnosis_threshold": 0.95,
        "max_decision_calls": 9,
        "final_diagnosis": hypothesis,
        "diagnosis_correct": True,
        "diagnosed_correctly_within_budget": True,
        "threshold_qualified_success": threshold_aware,
        "premature_diagnosis": not threshold_aware,
        "normative_probability_of_final_diagnosis": 0.96 if threshold_aware else 0.80,
        "model_reported_final_confidence": 0.95 if autonomous else 0.80,
        "experiments_used": 3 if threshold_aware else 2,
        "decision_calls": 4 if threshold_aware else 3,
        "total_retries": 0,
        "termination_reason": "DIAGNOSED",
        "cumulative_action_regret": {
            "FULL_AUTONOMOUS": 0.30,
            "PLANNER_ONLY": 0.10,
            "THRESHOLD_AWARE_AUTONOMOUS": 0.20,
        }[condition],
        "oracle_action_agreements": 1,
        "oracle_action_agreement_rate": 0.5,
        "mean_autonomous_belief_l1_error": 0.1 if autonomous else "",
        "provider_failure_flag": False,
        "model_failure_flag": False,
        "scientific_model_failure_flag": False,
        "provider_rate_limit_failure_flag": False,
        "provider_transport_failure_flag": False,
        "provider_error_type": "",
        "provider_error_message": "",
        "action_sequence": "USE_TRUSTED_SENSOR>CHANGE_CONTEXT",
        "first_action": "USE_TRUSTED_SENSOR",
        "repeat_trial_count": 0,
        "use_trusted_sensor_count": 1,
        "change_context_count": 1,
    }


def _write_source(
    directory: Path,
    seeds: tuple[int, ...],
    *,
    combined: bool,
) -> list[dict[str, object]]:
    directory.mkdir()
    rows = [
        _row(condition, hypothesis, seed)
        for condition in CONDITIONS
        for hypothesis in HYPOTHESES
        for seed in seeds
    ]
    filename = "episodes_combined.csv" if combined else "episodes.csv"
    _write_rows(directory / filename, rows)
    if not combined:
        config = {
            "benchmark_version": "binary_er1_v2",
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "thinking_level": "low",
            "experiment_budget": 8,
            "diagnosis_threshold": 0.95,
            "max_decision_calls": 9,
            "conditions": list(CONDITIONS),
            "hypotheses": list(HYPOTHESES),
            "seeds": list(seeds),
            "planned_episode_count": len(rows),
        }
        (directory / "run_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    pilot = tmp_path / "pilot"
    replication = tmp_path / "replication"
    _write_source(pilot, (0, 1), combined=True)
    _write_source(replication, tuple(range(2, 10)), combined=False)
    return pilot, replication


def test_seed0_9_merge_builds_exact_grid_and_artifacts(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    output = tmp_path / "final"

    result = combine_final_analysis((pilot, replication), output)

    assert len(result["rows"]) == 120
    assert len(result["matched"]) == 40
    assert {int(row["seed"]) for row in result["rows"]} == set(range(10))
    assert {path.name for path in output.iterdir()} == set(final_module.FINAL_OUTPUT_FILES)
    combined = [row for row in result["summary"] if row["analysis_split"] == "COMBINED"]
    assert all(row["planned_cells"] == 40 for row in combined)


def test_duplicate_key_across_sources_is_rejected(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    with (replication / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["seed"] = "1"
    _write_rows(replication / "episodes.csv", rows)
    config = json.loads((replication / "run_config.json").read_text(encoding="utf-8"))
    config["seeds"] = [1, *range(3, 10)]
    (replication / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(IncompatibleAnalysisInput):
        combine_final_analysis((pilot, replication), tmp_path / "out")


def test_incompatible_model_is_rejected(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    with (replication / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["model"] = "different-model"
    _write_rows(replication / "episodes.csv", rows)
    config = json.loads((replication / "run_config.json").read_text(encoding="utf-8"))
    config["model"] = "different-model"
    (replication / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(IncompatibleAnalysisInput, match="model"):
        combine_final_analysis((pilot, replication), tmp_path / "out")


def test_scientific_failure_is_not_a_diagnostic_error(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    with (replication / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failed = rows[0]
    failed.update({
        "final_diagnosis": "",
        "diagnosis_correct": "False",
        "threshold_qualified_success": "False",
        "premature_diagnosis": "False",
        "termination_reason": "MODEL_FAILURE",
        "scientific_model_failure_flag": "True",
    })
    _write_rows(replication / "episodes.csv", rows)

    result = combine_final_analysis((pilot, replication), tmp_path / "out")
    full = next(
        row for row in result["summary"]
        if row["analysis_split"] == "COMBINED" and row["condition"] == "FULL_AUTONOMOUS"
    )

    assert full["planned_cells"] == 40
    assert full["valid_episodes"] == 39
    assert full["scientific_model_failures"] == 1
    assert full["diagnostic_errors"] == 0
    assert full["planned_cell_accuracy"] == pytest.approx(39 / 40)
    assert full["valid_episode_accuracy"] == 1.0


def test_recovered_provider_attempt_does_not_invalidate_completed_episode() -> None:
    row = _row("FULL_AUTONOMOUS", "WORLD_SHIFT", 0)
    row["provider_failure_flag"] = True
    row["provider_transport_failure_flag"] = True

    assert classify_episode(row) == ("VALID_EPISODE", "VALID_EPISODE")


def test_planned_and_valid_accuracy_use_different_denominators(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    with (replication / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scientific = rows[0]
    scientific.update({
        "final_diagnosis": "",
        "diagnosis_correct": "False",
        "termination_reason": "MODEL_FAILURE",
        "scientific_model_failure_flag": "True",
    })
    diagnostic = rows[1]
    diagnostic.update({
        "final_diagnosis": "WORLD_SHIFT",
        "diagnosis_correct": "False",
    })
    _write_rows(replication / "episodes.csv", rows)

    result = combine_final_analysis((pilot, replication), tmp_path / "out")
    full = next(
        row for row in result["summary"]
        if row["analysis_split"] == "COMBINED" and row["condition"] == "FULL_AUTONOMOUS"
    )
    assert full["planned_cell_accuracy"] == pytest.approx(38 / 40)
    assert full["valid_episode_accuracy"] == pytest.approx(38 / 39)
    assert full["diagnostic_errors"] == 1


def test_matched_triplet_marks_protocol_failure_as_not_comparable(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    with (replication / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = next(
        row for row in rows
        if row["condition"] == "THRESHOLD_AWARE_AUTONOMOUS"
        and row["true_hypothesis"] == "WORLD_SHIFT"
        and row["seed"] == "2"
    )
    target.update({
        "final_diagnosis": "",
        "diagnosis_correct": "False",
        "termination_reason": "MODEL_FAILURE",
        "scientific_model_failure_flag": "True",
    })
    _write_rows(replication / "episodes.csv", rows)

    result = combine_final_analysis((pilot, replication), tmp_path / "out")
    match = next(row for row in result["matched"] if row["match_key"] == "WORLD_SHIFT:2")

    assert match["threshold_aware_protocol_status"] == "SCIENTIFIC_MODEL_FAILURE"
    assert match["threshold_aware_vs_full_protocol_comparison"] == "LEFT_PROTOCOL_FAILURE"
    assert match["threshold_aware_vs_full_diagnostic_outcome"] == "NOT_COMPARABLE_PROTOCOL_FAILURE"
    assert match["threshold_aware_vs_full_diagnosis_correct_delta"] is None


def test_calibration_uses_only_valid_comparable_autonomous_rows() -> None:
    rows = []
    for condition in CONDITIONS:
        for hypothesis in HYPOTHESES:
            row = _row(condition, hypothesis, 0)
            row["protocol_status"] = "VALID_EPISODE"
            rows.append(row)
    rows[0]["model_reported_final_confidence"] = ""
    failed = rows[-1]
    failed["protocol_status"] = "SCIENTIFIC_MODEL_FAILURE"

    calibration = build_calibration(rows, 0.95)
    full = next(
        row for row in calibration
        if row["condition"] == "FULL_AUTONOMOUS" and row["true_hypothesis"] == "ALL"
    )
    aware = next(
        row for row in calibration
        if row["condition"] == "THRESHOLD_AWARE_AUTONOMOUS" and row["true_hypothesis"] == "ALL"
    )
    assert full["valid_episodes"] == 4
    assert full["comparable_confidence_count"] == 3
    assert full["mean_signed_gap"] == pytest.approx(0.15)
    assert aware["valid_episodes"] == 3
    assert aware["comparable_confidence_count"] == 3
    assert aware["both_ge_threshold_count"] == 3


def test_wilson_interval_known_value_and_empty_denominator() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    assert wilson_interval(0, 0) == (None, None)


def test_pilot_replication_and_combined_are_separate_summary_rows(tmp_path: Path) -> None:
    pilot, replication = _sources(tmp_path)
    result = combine_final_analysis((pilot, replication), tmp_path / "out")

    assert len(result["summary"]) == 9
    counts = {
        (row["analysis_split"], row["condition"]): row["planned_cells"]
        for row in result["summary"]
    }
    assert counts[("PILOT", "FULL_AUTONOMOUS")] == 8
    assert counts[("REPLICATION", "FULL_AUTONOMOUS")] == 32
    assert counts[("COMBINED", "FULL_AUTONOMOUS")] == 40


def test_final_analysis_has_no_provider_or_network_path() -> None:
    source = inspect.getsource(final_module)
    assert "GeminiLLMClient" not in source
    assert "create_llm_client" not in source
    assert "requests." not in source
    assert "urllib" not in source
