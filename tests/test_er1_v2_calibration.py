"""Targeted, no-network tests for the ER-1 V2 calibration layer."""

import ast
import csv
from pathlib import Path

import pytest

from epistemic_repair.evaluation.er1_v2_calibration import (
    ER1_V2_CALIBRATION_BUDGETS,
    ER1_V2_CALIBRATION_SEEDS,
    ER1_V2_CALIBRATION_THRESHOLDS,
    attach_v1_comparison,
    comparison_row,
    run_er1_v2_oracle_calibration,
    write_er1_v2_calibration_artifacts,
)
from epistemic_repair.failures.modes import FailureMode
from scripts.calibrate_er1_v2_oracle import parse_args


def test_default_grid_accounts_for_exactly_60000_episodes() -> None:
    assert ER1_V2_CALIBRATION_BUDGETS == (1, 2, 3, 5, 8)
    assert ER1_V2_CALIBRATION_THRESHOLDS == (0.80, 0.90, 0.95)
    assert ER1_V2_CALIBRATION_SEEDS == 1000
    assert 4 * 5 * 3 * 1000 == 60_000
    args = parse_args([])
    assert tuple(args.budgets) == ER1_V2_CALIBRATION_BUDGETS
    assert tuple(args.thresholds) == ER1_V2_CALIBRATION_THRESHOLDS


def test_small_v2_grid_accounting_and_reproducibility() -> None:
    first = run_er1_v2_oracle_calibration(
        seed_count=3, budgets=(1, 2), thresholds=(0.80, 0.90)
    )
    second = run_er1_v2_oracle_calibration(
        seed_count=3, budgets=(1, 2), thresholds=(0.80, 0.90)
    )
    assert len(first.episodes) == 4 * 2 * 2 * 3
    assert len(first.cells) == 4 * 2 * 2
    assert len(first.overall_cells) == 2 * 2
    assert first.episodes == second.episodes
    assert first.cells == second.cells
    assert first.confusion_matrices == second.confusion_matrices


def test_v2_aggregation_confidence_and_structural_rates_are_consistent() -> None:
    study = run_er1_v2_oracle_calibration(
        seed_count=12, budgets=(2,), thresholds=(0.90,)
    )
    overall = study.overall_cells[0]
    assert overall.episodes == 48
    assert overall.map_accuracy_ci_lower <= overall.map_accuracy <= overall.map_accuracy_ci_upper
    assert overall.success_ci_lower - 1e-15 <= overall.success_at_threshold <= overall.success_ci_upper + 1e-15

    no_change = [
        item for item in study.episodes
        if item.hypothesis is FailureMode.NO_STRUCTURAL_CHANGE
    ]
    structural = [
        item for item in study.episodes
        if item.hypothesis is not FailureMode.NO_STRUCTURAL_CHANGE
    ]
    expected_false = sum(
        item.final_map_diagnosis is not FailureMode.NO_STRUCTURAL_CHANGE
        for item in no_change
    ) / len(no_change)
    expected_missed = sum(
        item.final_map_diagnosis is FailureMode.NO_STRUCTURAL_CHANGE
        for item in structural
    ) / len(structural)
    assert overall.false_structural_diagnosis_rate == expected_false
    assert overall.missed_structural_failure_rate == expected_missed


def test_v2_confusion_counts_account_for_every_episode() -> None:
    study = run_er1_v2_oracle_calibration(
        seed_count=7, budgets=(3,), thresholds=(0.90,)
    )
    condition = study.confusion_matrices["conditions"]["budget_3_threshold_0.90"]
    counts = condition["counts"]
    percentages = condition["row_percentages"]
    assert sum(sum(row.values()) for row in counts.values()) == 28
    for hypothesis in FailureMode:
        if hypothesis.value in percentages:
            assert sum(percentages[hypothesis.value].values()) == pytest.approx(100.0)


def test_action_sequences_are_reported_for_budget_5_and_8() -> None:
    study = run_er1_v2_oracle_calibration(
        seed_count=3, budgets=(5, 8), thresholds=(0.90,)
    )
    conditions = study.top_action_sequences["conditions"]
    assert set(conditions) == {
        "budget_5_threshold_0.90",
        "budget_8_threshold_0.90",
    }
    for condition in conditions.values():
        assert set(condition["by_hypothesis"]) == {
            hypothesis.value
            for hypothesis in (
                FailureMode.NO_STRUCTURAL_CHANGE,
                FailureMode.WORLD_SHIFT,
                FailureMode.SENSOR_CORRUPTION,
                FailureMode.MISSING_LATENT_VARIABLE,
            )
        }


def test_comparison_delta_is_always_v2_minus_v1() -> None:
    row = comparison_row(
        scope="overall",
        budget=5,
        threshold=0.90,
        hypothesis="ALL",
        metric="map_accuracy",
        v1=0.80,
        v2=0.91,
    )
    assert row.delta == pytest.approx(0.11)


def test_v1_comparison_loads_matching_cells(tmp_path: Path) -> None:
    study = run_er1_v2_oracle_calibration(
        seed_count=2, budgets=(5,), thresholds=(0.90,)
    )
    calibration = tmp_path / "v1_calibration.csv"
    overall = tmp_path / "v1_overall.csv"
    _write_v1_fixture(calibration, overall)
    compared = attach_v1_comparison(
        study,
        v1_calibration_csv=calibration,
        v1_overall_csv=overall,
    )
    assert len(compared.v1_comparison) == 5 + 4 * 3
    overall_map = next(
        row for row in compared.v1_comparison
        if row.scope == "overall" and row.metric == "map_accuracy"
    )
    assert overall_map.delta == pytest.approx(overall_map.v2 - 0.5)


def test_v2_artifacts_are_separate_and_csv_is_deterministic(tmp_path: Path) -> None:
    first = run_er1_v2_oracle_calibration(
        seed_count=2, budgets=(1,), thresholds=(0.80,)
    )
    second = run_er1_v2_oracle_calibration(
        seed_count=2, budgets=(1,), thresholds=(0.80,)
    )
    v1_marker = tmp_path / "er1_oracle_calibration.csv"
    v1_marker.write_text("do not overwrite\n", encoding="utf-8")
    first_paths = write_er1_v2_calibration_artifacts(first, tmp_path / "first")
    second_paths = write_er1_v2_calibration_artifacts(second, tmp_path / "second")
    assert first_paths.calibration_csv.name == "er1_v2_oracle_calibration.csv"
    assert first_paths.overall_csv.name == "er1_v2_oracle_overall.csv"
    assert first_paths.calibration_csv.read_bytes() == second_paths.calibration_csv.read_bytes()
    assert first_paths.overall_csv.read_bytes() == second_paths.overall_csv.read_bytes()
    assert v1_marker.read_text(encoding="utf-8") == "do not overwrite\n"


def test_v2_calibration_path_has_no_llm_or_provider_dependency() -> None:
    project = Path(__file__).parents[1]
    sources = (
        project / "scripts" / "calibrate_er1_v2_oracle.py",
        project / "epistemic_repair" / "evaluation" / "er1_v2_calibration.py",
    )
    imported_modules: list[str] = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
    assert not any(
        token in module.lower()
        for module in imported_modules
        for token in ("llm", "gemini", "provider")
    )


def _write_v1_fixture(calibration: Path, overall: Path) -> None:
    with overall.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "budget", "threshold", "map_accuracy", "success_at_threshold",
            "mean_experiments", "false_structural_diagnosis_rate",
            "missed_structural_failure_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "budget": 5, "threshold": 0.9, "map_accuracy": 0.5,
            "success_at_threshold": 0.4, "mean_experiments": 4.0,
            "false_structural_diagnosis_rate": 0.5,
            "missed_structural_failure_rate": 0.1,
        })
    with calibration.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "hypothesis", "budget", "threshold", "map_accuracy",
            "success_at_threshold", "mean_experiments",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for hypothesis in (
            FailureMode.NO_STRUCTURAL_CHANGE,
            FailureMode.WORLD_SHIFT,
            FailureMode.SENSOR_CORRUPTION,
            FailureMode.MISSING_LATENT_VARIABLE,
        ):
            writer.writerow({
                "hypothesis": hypothesis.value,
                "budget": 5,
                "threshold": 0.9,
                "map_accuracy": 0.5,
                "success_at_threshold": 0.4,
                "mean_experiments": 4.0,
            })
