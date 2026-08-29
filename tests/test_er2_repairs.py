"""Tests for the minimal deterministic ER-2 repair benchmark."""

from __future__ import annotations

import inspect
from dataclasses import asdict

import pytest

import epistemic_repair.er2.evaluation as evaluation_module
import epistemic_repair.er2.report as report_module
import epistemic_repair.er2.runner as runner_module
from epistemic_repair.er2 import (
    ER2_HYPOTHESES,
    ER2RepairAgentView,
    ER2RepairEpisodeRunner,
    FixedRepairPolicy,
    LatentStructure,
    OracleRepairPolicy,
    RepairableAgentState,
    SensorCalibration,
    WorldRelation,
    apply_repair,
    build_evaluation_suite,
    build_wrong_repair_matrix,
    evaluate_baselines,
    evaluate_repaired_state,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


@pytest.mark.parametrize(
    ("repair", "changed_field", "expected_value"),
    (
        (
            RepairOperator.UPDATE_WORLD_MODEL,
            "world_relation",
            WorldRelation.INVERTED,
        ),
        (
            RepairOperator.RECALIBRATE_SENSOR,
            "sensor_calibration",
            SensorCalibration.INVERTED,
        ),
        (
            RepairOperator.ADD_LATENT_VARIABLE,
            "latent_structure",
            LatentStructure.CONTEXT_DEPENDENT,
        ),
    ),
)
def test_each_repair_mutates_only_its_intended_component(
    repair: RepairOperator,
    changed_field: str,
    expected_value: object,
) -> None:
    state = RepairableAgentState()
    before = asdict(state)

    apply_repair(state, repair)

    after = asdict(state)
    assert after[changed_field] is expected_value
    assert {
        field: value for field, value in after.items() if field != changed_field
    } == {
        field: value for field, value in before.items() if field != changed_field
    }


def test_no_repair_is_exact_identity() -> None:
    state = RepairableAgentState()
    before = asdict(state)

    apply_repair(state, RepairOperator.NO_REPAIR)

    assert asdict(state) == before


def test_oracle_diagnosis_to_repair_mapping() -> None:
    policy = OracleRepairPolicy()
    expected = {
        FailureMode.NO_STRUCTURAL_CHANGE: RepairOperator.NO_REPAIR,
        FailureMode.WORLD_SHIFT: RepairOperator.UPDATE_WORLD_MODEL,
        FailureMode.SENSOR_CORRUPTION: RepairOperator.RECALIBRATE_SENSOR,
        FailureMode.MISSING_LATENT_VARIABLE: RepairOperator.ADD_LATENT_VARIABLE,
    }
    for hypothesis, repair in expected.items():
        view = ER2RepairAgentView(diagnosis=hypothesis)
        assert policy.choose_repair(view) is repair
        assert repair_for_failure(hypothesis) is repair


def test_held_out_suite_and_evaluation_are_deterministic() -> None:
    first_suite = build_evaluation_suite(FailureMode.MISSING_LATENT_VARIABLE)
    second_suite = build_evaluation_suite(FailureMode.MISSING_LATENT_VARIABLE)
    before = RepairableAgentState()
    after = before.clone()
    apply_repair(after, RepairOperator.ADD_LATENT_VARIABLE)

    first_metrics = evaluate_repaired_state(
        FailureMode.MISSING_LATENT_VARIABLE, before, after
    )
    second_metrics = evaluate_repaired_state(
        FailureMode.MISSING_LATENT_VARIABLE, before, after
    )

    assert first_suite == second_suite
    assert len(first_suite) == 10
    assert first_metrics == second_metrics
    assert first_metrics.post_repair_accuracy == 1.0
    assert first_metrics.affected_region_accuracy == 1.0
    assert first_metrics.unaffected_region_accuracy == 1.0


def test_wrong_repair_matrix_covers_all_sixteen_combinations() -> None:
    matrix = build_wrong_repair_matrix()
    keys = {
        (row["true_hypothesis"], row["applied_repair"])
        for row in matrix
    }

    assert len(matrix) == 16
    assert len(keys) == 16
    assert keys == {
        (hypothesis.value, repair.value)
        for hypothesis in ER2_HYPOTHESES
        for repair in RepairOperator
    }
    assert sum(row["repair_success"] for row in matrix) == 4


def test_collateral_damage_is_signed_unaffected_accuracy_difference() -> None:
    before = RepairableAgentState()
    after = before.clone()
    apply_repair(after, RepairOperator.UPDATE_WORLD_MODEL)

    metrics = evaluate_repaired_state(
        FailureMode.SENSOR_CORRUPTION, before, after
    )

    assert metrics.pre_repair_unaffected_accuracy == 1.0
    assert metrics.unaffected_region_accuracy == 0.0
    assert metrics.collateral_damage == pytest.approx(
        metrics.pre_repair_unaffected_accuracy
        - metrics.unaffected_region_accuracy
    )
    assert metrics.collateral_damage == 1.0
    assert metrics.affected_region_accuracy == pytest.approx(2 / 3)


def test_missing_latent_global_update_fixes_affected_but_damages_context_a() -> None:
    before = RepairableAgentState()
    after = before.clone()
    apply_repair(after, RepairOperator.UPDATE_WORLD_MODEL)

    metrics = evaluate_repaired_state(
        FailureMode.MISSING_LATENT_VARIABLE, before, after
    )

    assert metrics.affected_region_accuracy == 1.0
    assert metrics.unaffected_region_accuracy == pytest.approx(1 / 3)
    assert metrics.collateral_damage == pytest.approx(2 / 3)
    assert not metrics.repair_success


def test_episode_policy_receives_diagnosis_only_not_true_hypothesis() -> None:
    captured: list[ER2RepairAgentView] = []

    class SpyPolicy:
        def choose_repair(self, view: ER2RepairAgentView) -> RepairOperator:
            captured.append(view)
            return RepairOperator.NO_REPAIR

    result = ER2RepairEpisodeRunner().run(
        true_hypothesis=FailureMode.WORLD_SHIFT,
        diagnosis=FailureMode.SENSOR_CORRUPTION,
        policy=SpyPolicy(),
    )

    assert captured == [
        ER2RepairAgentView(diagnosis=FailureMode.SENSOR_CORRUPTION)
    ]
    assert set(ER2RepairAgentView.__dataclass_fields__) == {
        "diagnosis",
        "available_repairs",
    }
    assert result.true_hypothesis is FailureMode.WORLD_SHIFT
    assert result.supplied_diagnosis is FailureMode.SENSOR_CORRUPTION


def test_oracle_uses_same_episode_evaluator_and_succeeds_on_all_hypotheses() -> None:
    runner = ER2RepairEpisodeRunner()
    results = [
        runner.run(
            true_hypothesis=hypothesis,
            diagnosis=hypothesis,
            policy=OracleRepairPolicy(),
        )
        for hypothesis in ER2_HYPOTHESES
    ]

    assert all(result.repair_selection_correct for result in results)
    assert all(result.metrics.repair_success for result in results)
    assert all(result.metrics.post_repair_accuracy == 1.0 for result in results)
    assert all(result.metrics.collateral_damage == 0.0 for result in results)


def test_baselines_include_four_fixed_policies_and_oracle() -> None:
    baselines = evaluate_baselines()

    assert len(baselines) == 5
    assert {row["baseline"] for row in baselines} == {
        "ALWAYS_NO_REPAIR",
        "ALWAYS_UPDATE_WORLD_MODEL",
        "ALWAYS_RECALIBRATE_SENSOR",
        "ALWAYS_ADD_LATENT_VARIABLE",
        "ORACLE_REPAIR",
    }
    oracle = next(row for row in baselines if row["baseline"] == "ORACLE_REPAIR")
    assert oracle["repair_selection_accuracy"] == 1.0
    assert oracle["repair_success_rate"] == 1.0


def test_er2_modules_have_no_llm_provider_or_network_dependency() -> None:
    source = "\n".join(
        inspect.getsource(module)
        for module in (evaluation_module, runner_module, report_module)
    )
    assert "GeminiLLMClient" not in source
    assert "create_llm_client" not in source
    assert "requests." not in source
    assert "urllib" not in source
