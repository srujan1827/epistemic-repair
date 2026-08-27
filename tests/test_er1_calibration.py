"""Focused statistical and reproducibility tests for ER-1 calibration."""

import ast
from pathlib import Path

import pytest

from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.evaluation.er1_calibration import (
    OracleCalibrationEpisode,
    aggregate_calibration_cells,
    aggregate_overall_cells,
    build_confusion_matrices,
    percentile,
    run_oracle_calibration,
    wilson_interval,
    write_calibration_artifacts,
)
from epistemic_repair.failures.modes import FailureMode


def test_small_calibration_grid_has_expected_size() -> None:
    study = run_oracle_calibration(
        seed_count=3,
        budgets=(1, 2),
        thresholds=(0.80, 0.90),
    )
    assert len(study.episodes) == 4 * 2 * 2 * 3
    assert len(study.cells) == 4 * 2 * 2
    assert len(study.overall_cells) == 2 * 2


def test_calibration_is_reproducible_for_same_seed_grid() -> None:
    first = run_oracle_calibration(seed_count=4, budgets=(2,), thresholds=(0.90,))
    second = run_oracle_calibration(seed_count=4, budgets=(2,), thresholds=(0.90,))
    assert first.episodes == second.episodes
    assert first.cells == second.cells
    assert first.confusion_matrices == second.confusion_matrices
    assert first.hard_cases == second.hard_cases


def test_map_accuracy_and_threshold_success_are_distinct() -> None:
    episodes = (
        _episode(
            FailureMode.WORLD_SHIFT,
            FailureMode.WORLD_SHIFT,
            reached_threshold=False,
        ),
        _episode(
            FailureMode.WORLD_SHIFT,
            FailureMode.WORLD_SHIFT,
            reached_threshold=True,
            seed=1,
        ),
    )
    cell = aggregate_calibration_cells(episodes)[0]
    assert cell.map_accuracy == 1.0
    assert cell.success_at_threshold == 0.5
    assert cell.threshold_reached_fraction == 0.5
    assert cell.budget_exhausted_fraction == 0.5


def test_aggregation_and_posterior_statistics_are_correct() -> None:
    episodes = (
        _episode(
            FailureMode.SENSOR_CORRUPTION,
            FailureMode.SENSOR_CORRUPTION,
            reached_threshold=False,
            true_posterior=0.6,
            experiments=3,
        ),
        _episode(
            FailureMode.SENSOR_CORRUPTION,
            FailureMode.WORLD_SHIFT,
            reached_threshold=False,
            true_posterior=0.2,
            experiments=5,
            seed=1,
        ),
    )
    cell = aggregate_calibration_cells(episodes)[0]
    assert cell.map_accuracy == 0.5
    assert cell.mean_experiments == 4.0
    assert cell.median_experiments == 4.0
    assert cell.stddev_experiments == 1.0
    assert cell.mean_true_posterior == pytest.approx(0.4)
    assert cell.median_true_posterior == pytest.approx(0.4)
    assert cell.p10_true_posterior == pytest.approx(0.24)
    assert percentile((0.2, 0.6), 0.90) == pytest.approx(0.56)


def test_wilson_interval_handles_boundary_rates() -> None:
    none = wilson_interval(0, 100)
    all_success = wilson_interval(100, 100)
    half = wilson_interval(50, 100)
    assert none.estimate == 0.0
    assert none.lower == pytest.approx(0.0, abs=1e-15)
    assert 0.0 < none.upper < 0.05
    assert all_success.upper == pytest.approx(1.0)
    assert 0.4 < half.lower < half.estimate < half.upper < 0.6


def test_confusion_matrix_counts_and_percentages() -> None:
    episodes = (
        _episode(FailureMode.WORLD_SHIFT, FailureMode.WORLD_SHIFT),
        _episode(
            FailureMode.WORLD_SHIFT,
            FailureMode.SENSOR_CORRUPTION,
            seed=1,
        ),
    )
    matrices = build_confusion_matrices(episodes)
    condition = matrices["conditions"]["budget_5_threshold_0.90"]
    counts = condition["counts"][FailureMode.WORLD_SHIFT.value]
    percentages = condition["row_percentages"][FailureMode.WORLD_SHIFT.value]
    assert counts[FailureMode.WORLD_SHIFT.value] == 1
    assert counts[FailureMode.SENSOR_CORRUPTION.value] == 1
    assert percentages[FailureMode.WORLD_SHIFT.value] == 50.0
    assert percentages[FailureMode.SENSOR_CORRUPTION.value] == 50.0


def test_false_structural_and_missed_failure_rates() -> None:
    episodes = (
        _episode(
            FailureMode.NO_STRUCTURAL_CHANGE,
            FailureMode.WORLD_SHIFT,
        ),
        _episode(FailureMode.WORLD_SHIFT, FailureMode.WORLD_SHIFT),
        _episode(
            FailureMode.SENSOR_CORRUPTION,
            FailureMode.NO_STRUCTURAL_CHANGE,
        ),
        _episode(
            FailureMode.MISSING_LATENT_VARIABLE,
            FailureMode.MISSING_LATENT_VARIABLE,
        ),
    )
    overall = aggregate_overall_cells(episodes)[0]
    assert overall.map_accuracy == 0.5
    assert overall.false_structural_diagnosis_rate == 1.0
    assert overall.missed_structural_failure_rate == pytest.approx(1 / 3)


def test_csv_outputs_are_deterministic(tmp_path: Path) -> None:
    first = run_oracle_calibration(seed_count=2, budgets=(1,), thresholds=(0.80,))
    second = run_oracle_calibration(seed_count=2, budgets=(1,), thresholds=(0.80,))
    first_paths = write_calibration_artifacts(first, tmp_path / "first")
    second_paths = write_calibration_artifacts(second, tmp_path / "second")
    assert (
        first_paths.calibration_csv.read_bytes()
        == second_paths.calibration_csv.read_bytes()
    )
    assert first_paths.overall_csv.read_bytes() == second_paths.overall_csv.read_bytes()


def test_calibration_script_has_no_llm_or_provider_imports() -> None:
    imported_modules = []
    project = Path(__file__).parents[1]
    sources = (
        project / "scripts" / "calibrate_er1_oracle.py",
        project / "epistemic_repair" / "evaluation" / "er1_calibration.py",
    )
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


def _episode(
    truth: FailureMode,
    diagnosis: FailureMode,
    *,
    reached_threshold: bool = False,
    true_posterior: float = 0.5,
    experiments: int = 2,
    seed: int = 0,
) -> OracleCalibrationEpisode:
    correct = truth is diagnosis
    return OracleCalibrationEpisode(
        hypothesis=truth,
        budget=5,
        threshold=0.90,
        seed=seed,
        final_map_diagnosis=diagnosis,
        map_correct=correct,
        reached_threshold=reached_threshold,
        success_at_threshold=reached_threshold and correct,
        experiments_used=experiments,
        final_true_posterior=true_posterior,
        cumulative_action_regret=0.0,
        action_sequence=(
            DiagnosticAction.USE_TRUSTED_SENSOR,
            DiagnosticAction.CHANGE_CONTEXT,
        ),
    )
