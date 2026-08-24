"""Tests for active diagnostic experiments and information boundaries."""

from dataclasses import fields

import pytest

from epistemic_repair import (
    BinaryMachine,
    ChangeContextResult,
    Context,
    DiagnosticAction,
    FailureMode,
    LatentInspectionResult,
    Observation,
    RepeatTrialResult,
    TrustedSensorResult,
)


FAILURE_MODES = (
    FailureMode.WORLD_SHIFT,
    FailureMode.SENSOR_CORRUPTION,
    FailureMode.MISSING_LATENT_VARIABLE,
)


def test_repeat_trial_reproduces_primary_sensor_behavior() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION)

    result = env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1)

    assert result == RepeatTrialResult(x=1, o=0)


@pytest.mark.parametrize(
    ("mode", "expected_y"),
    [
        (FailureMode.WORLD_SHIFT, 0),
        (FailureMode.SENSOR_CORRUPTION, 1),
        (FailureMode.MISSING_LATENT_VARIABLE, 0),
    ],
)
def test_trusted_sensor_separates_sensor_corruption(
    mode: FailureMode, expected_y: int
) -> None:
    env = BinaryMachine()
    env.reset(mode)

    result = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)

    assert result == TrustedSensorResult(x=1, trusted_y=expected_y)


def test_change_context_does_not_alter_world_shift() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.WORLD_SHIFT)

    in_a = env.run_experiment(
        DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A
    )
    in_b = env.run_experiment(
        DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.B
    )

    assert in_a == ChangeContextResult(context=Context.A, x=1, o=0)
    assert in_b == ChangeContextResult(context=Context.B, x=1, o=0)
    assert env.get_ground_truth().z is None


def test_change_context_does_not_repair_sensor_corruption() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION)

    result = env.run_experiment(
        DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A
    )
    truth = env.get_ground_truth()

    assert result == ChangeContextResult(context=Context.A, x=1, o=0)
    assert truth.y == 1
    assert truth.o == 0
    assert truth.z is None


def test_change_context_changes_latent_and_output_only_in_latent_mode() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE)

    in_a = env.run_experiment(
        DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A
    )
    truth_a = env.get_ground_truth()
    assert in_a == ChangeContextResult(context=Context.A, x=1, o=1)
    assert (truth_a.z, truth_a.y) == (0, 1)

    in_b = env.run_experiment(
        DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.B
    )
    truth_b = env.get_ground_truth()
    assert in_b == ChangeContextResult(context=Context.B, x=1, o=0)
    assert (truth_b.z, truth_b.y) == (1, 0)


@pytest.mark.parametrize(
    "mode", [FailureMode.NORMAL, FailureMode.WORLD_SHIFT, FailureMode.SENSOR_CORRUPTION]
)
def test_latent_inspection_reports_unavailable_outside_latent_mode(
    mode: FailureMode,
) -> None:
    env = BinaryMachine()
    env.reset(mode)

    result = env.run_experiment(DiagnosticAction.INSPECT_LATENT_VARIABLE)

    assert result == LatentInspectionResult(available=False)


def test_latent_inspection_exposes_meaningful_z_in_latent_mode() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE)
    env.run_experiment(DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A)

    result = env.run_experiment(DiagnosticAction.INSPECT_LATENT_VARIABLE)

    assert result == LatentInspectionResult(available=True, value=0)


def test_diagnostic_results_never_contain_failure_labels() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE)
    results = [
        env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1),
        env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1),
        env.run_experiment(
            DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A
        ),
        env.run_experiment(DiagnosticAction.INSPECT_LATENT_VARIABLE),
    ]

    for result in results:
        assert "failure_mode" not in {field.name for field in fields(result)}
        assert not hasattr(result, "correct_repair")


def test_reset_clears_context_and_diagnostic_transition_state() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE)
    env.run_experiment(DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A)

    env.reset(FailureMode.MISSING_LATENT_VARIABLE)
    truth = env.get_ground_truth()

    assert truth.context is Context.B
    assert truth.z == 1
    assert truth.x is None
    assert truth.y is None
    assert truth.o is None
    assert env.step(1) == Observation(x=1, o=0)


def test_all_failures_are_identical_before_diagnostic_experiments() -> None:
    env = BinaryMachine()
    initial_observations = []

    for mode in FAILURE_MODES:
        env.reset(mode)
        initial_observations.append(env.step(1))

    assert initial_observations == [Observation(x=1, o=0)] * 3


def test_actions_require_only_their_meaningful_arguments() -> None:
    env = BinaryMachine()

    with pytest.raises(ValueError, match="x is required"):
        env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR)
    with pytest.raises(ValueError, match="context is required"):
        env.run_experiment(DiagnosticAction.CHANGE_CONTEXT, x=1)
    with pytest.raises(ValueError, match="not valid"):
        env.run_experiment(DiagnosticAction.INSPECT_LATENT_VARIABLE, x=1)


def test_environment_rejects_invalid_context_and_sensor_arguments() -> None:
    env = BinaryMachine()

    with pytest.raises(TypeError, match="Context"):
        env.run_experiment(
            DiagnosticAction.CHANGE_CONTEXT,
            x=1,
            context="A",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="not valid"):
        env.run_experiment(
            DiagnosticAction.USE_TRUSTED_SENSOR,
            x=1,
            context=Context.A,
        )
