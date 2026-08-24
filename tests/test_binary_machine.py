"""Tests for deterministic behavior, ambiguity, and episode isolation."""

import pytest

from epistemic_repair import (
    BinaryMachine,
    FailureMode,
    Observation,
    RepairOperator,
    repair_for_failure,
)


def test_normal_world_for_both_inputs() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.NORMAL)

    assert env.step(0) == Observation(x=0, o=0)
    assert env.get_ground_truth().y == 0
    assert env.step(1) == Observation(x=1, o=1)
    assert env.get_ground_truth().y == 1


def test_world_shift_inverts_physical_output() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.WORLD_SHIFT)

    assert env.step(1) == Observation(x=1, o=0)
    truth = env.get_ground_truth()
    assert truth.y == 0
    assert truth.o == 0


def test_sensor_corruption_preserves_world_and_inverts_observation() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION)

    assert env.step(1) == Observation(x=1, o=0)
    truth = env.get_ground_truth()
    assert truth.y == 1
    assert truth.o == 0


def test_missing_latent_variable_with_z_one_inverts_output() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE, latent_z=1)

    assert env.step(1) == Observation(x=1, o=0)
    truth = env.get_ground_truth()
    assert truth.y == 0
    assert truth.o == 0
    assert truth.z == 1


def test_missing_latent_variable_with_z_zero_looks_normal() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE, latent_z=0)

    assert env.step(1) == Observation(x=1, o=1)
    truth = env.get_ground_truth()
    assert truth.y == 1
    assert truth.z == 0


def test_initial_anomaly_is_observationally_identical_across_failures() -> None:
    env = BinaryMachine()
    observations = []

    for mode in (
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        env.reset(mode)
        observations.append(env.step(1))

    assert observations == [Observation(x=1, o=0)] * 3


def test_identical_anomalies_have_distinct_internal_causes() -> None:
    env = BinaryMachine()
    truths = []

    for mode in (
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        env.reset(mode)
        env.step(1)
        truths.append(env.get_ground_truth())

    assert {truth.failure_mode for truth in truths} == {
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    }
    assert [(truth.y, truth.z) for truth in truths] == [
        (0, None),
        (1, None),
        (0, 1),
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (FailureMode.NORMAL, RepairOperator.NO_REPAIR),
        (FailureMode.WORLD_SHIFT, RepairOperator.UPDATE_WORLD_MODEL),
        (FailureMode.SENSOR_CORRUPTION, RepairOperator.RECALIBRATE_SENSOR),
        (
            FailureMode.MISSING_LATENT_VARIABLE,
            RepairOperator.ADD_LATENT_VARIABLE,
        ),
    ],
)
def test_failure_modes_map_to_intended_repairs(
    mode: FailureMode, expected: RepairOperator
) -> None:
    assert repair_for_failure(mode) is expected


def test_reset_clears_previous_episode_state() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE, latent_z=1)
    env.step(1)

    env.reset(FailureMode.NORMAL)
    truth_after_reset = env.get_ground_truth()
    assert truth_after_reset.x is None
    assert truth_after_reset.y is None
    assert truth_after_reset.o is None
    assert truth_after_reset.z is None
    assert truth_after_reset.failure_mode is FailureMode.NORMAL
    assert truth_after_reset.correct_repair is RepairOperator.NO_REPAIR

    assert env.step(1) == Observation(x=1, o=1)


@pytest.mark.parametrize("bad_x", [-1, 2, True, 0.0, "1"])
def test_step_rejects_values_outside_integer_bit_domain(bad_x: object) -> None:
    env = BinaryMachine()

    with pytest.raises(ValueError):
        env.step(bad_x)  # type: ignore[arg-type]


def test_latent_z_is_rejected_for_unrelated_failure_modes() -> None:
    env = BinaryMachine()

    with pytest.raises(ValueError, match="only valid"):
        env.reset(FailureMode.WORLD_SHIFT, latent_z=1)

