"""ER-1 V2 trigger/persistent separation and oracle regression tests."""

from dataclasses import asdict

import pytest

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticExperimentOutcome,
    StochasticOutcomeSignal,
    stochastic_information_gains,
)
from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, Context, DiagnosticAction
from epistemic_repair.er1.config import ER1_DEFAULT_CONFIG, ER1_HYPOTHESES
from epistemic_repair.er1_v2.config import (
    ER1_V2_SUPPORTED_BUDGETS,
    ER1V2InvestigationConfig,
    ER1V2TriggerConfig,
)
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.likelihoods import ER1V2LikelihoodModel
from epistemic_repair.er1_v2.policies import ER1V2OracleInformationGainPolicy
from epistemic_repair.er1_v2.runner import ER1V2EpisodeRunner
from epistemic_repair.er1_v2.trigger_model import TriggerLikelihoodModel
from epistemic_repair.er1_v2.views import ER1V2OraclePolicyView
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


EXPECTED_TRIGGER = {
    FailureMode.NO_STRUCTURAL_CHANGE: 0.30,
    FailureMode.WORLD_SHIFT: 0.70,
    FailureMode.SENSOR_CORRUPTION: 0.65,
    FailureMode.MISSING_LATENT_VARIABLE: 0.70,
}
EXPECTED_POSTERIOR = {
    FailureMode.NO_STRUCTURAL_CHANGE: 0.30 / 2.35,
    FailureMode.WORLD_SHIFT: 0.70 / 2.35,
    FailureMode.SENSOR_CORRUPTION: 0.65 / 2.35,
    FailureMode.MISSING_LATENT_VARIABLE: 0.70 / 2.35,
}


def test_trigger_likelihoods_and_exact_conditioned_posterior() -> None:
    model = TriggerLikelihoodModel()
    beliefs = model.conditioned_beliefs()
    for hypothesis in ER1_HYPOTHESES:
        assert model.likelihood(hypothesis) == EXPECTED_TRIGGER[hypothesis]
        assert beliefs.probability(hypothesis) == pytest.approx(
            EXPECTED_POSTERIOR[hypothesis], abs=1e-12
        )
    assert sum(beliefs.probability(h) for h in ER1_HYPOTHESES) == pytest.approx(1.0)


@pytest.mark.parametrize("context", tuple(Context))
@pytest.mark.parametrize("action", BENCHMARK_ACTIONS)
@pytest.mark.parametrize("hypothesis", ER1_HYPOTHESES)
def test_every_persistent_action_distribution_sums_to_one(
    context: Context,
    action: DiagnosticAction,
    hypothesis: FailureMode,
) -> None:
    model = ER1V2LikelihoodModel()
    signal = (
        StochasticOutcomeSignal.TRUSTED_T
        if action is DiagnosticAction.USE_TRUSTED_SENSOR
        else StochasticOutcomeSignal.PRIMARY_O
    )
    total = sum(
        model.likelihood(
            StochasticExperimentOutcome(signal, value), hypothesis, action, context
        )
        for value in (0, 1)
    )
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("hypothesis", "context", "primary_one", "trusted_one"),
    (
        (FailureMode.NO_STRUCTURAL_CHANGE, Context.B, 0.905, 0.941),
        (FailureMode.WORLD_SHIFT, Context.B, 0.14, 0.108),
        (FailureMode.SENSOR_CORRUPTION, Context.B, 0.14, 0.941),
        (FailureMode.MISSING_LATENT_VARIABLE, Context.B, 0.14, 0.108),
        (FailureMode.MISSING_LATENT_VARIABLE, Context.A, 0.86, 0.892),
    ),
)
def test_persistent_analytic_probabilities(
    hypothesis: FailureMode,
    context: Context,
    primary_one: float,
    trusted_one: float,
) -> None:
    model = ER1V2LikelihoodModel()
    assert model.probability_primary_observation(1, hypothesis, context) == pytest.approx(primary_one)
    assert model.probability_trusted_observation(1, hypothesis, context) == pytest.approx(trusted_one)


def test_v2_default_persistent_parameters_are_explicit() -> None:
    config = ER1V2InvestigationConfig()
    assert config.no_change_physical_accuracy == 0.95
    assert config.world_shift_physical_accuracy == 0.90
    assert config.sensor_corruption_physical_accuracy == 0.95
    assert config.corrupted_sensor_inversion_accuracy == 0.90
    assert config.latent_physical_accuracy == 0.90
    assert config.healthy_primary_sensor_accuracy == 0.95
    assert config.trusted_sensor_accuracy == 0.99
    assert ER1_V2_SUPPORTED_BUDGETS == (1, 2, 3, 5, 8)


def test_trigger_config_cannot_change_investigation_likelihoods() -> None:
    before = ER1V2LikelihoodModel().probability_primary_observation(
        1, FailureMode.NO_STRUCTURAL_CHANGE, Context.B
    )
    TriggerLikelihoodModel(ER1V2TriggerConfig(no_structural_change=0.8))
    after = ER1V2LikelihoodModel().probability_primary_observation(
        1, FailureMode.NO_STRUCTURAL_CHANGE, Context.B
    )
    assert before == after


def test_investigation_config_cannot_change_trigger_posterior() -> None:
    before = TriggerLikelihoodModel().conditioned_beliefs()
    ER1V2LikelihoodModel(ER1V2InvestigationConfig(no_change_physical_accuracy=0.7))
    after = TriggerLikelihoodModel().conditioned_beliefs()
    assert before == after


def test_investigation_model_rejects_trigger_conditioning() -> None:
    model = ER1V2LikelihoodModel()
    with pytest.raises(RuntimeError, match="TriggerLikelihoodModel"):
        model.initial_anomaly_likelihood(FailureMode.NO_STRUCTURAL_CHANGE)
    with pytest.raises(RuntimeError, match="TriggerLikelihoodModel"):
        model.conditioned_initial_beliefs()


def test_runner_applies_trigger_exactly_once_and_updates_persistently() -> None:
    class CountingTrigger(TriggerLikelihoodModel):
        calls = 0

        def conditioned_beliefs(self, base_prior=None):
            self.calls += 1
            return super().conditioned_beliefs() if base_prior is None else super().conditioned_beliefs(base_prior)

    trigger = CountingTrigger()
    model = ER1V2LikelihoodModel()
    result = ER1V2EpisodeRunner(
        max_experiments=1,
        trigger_model=trigger,
        investigation_model=model,
    ).run(
        ER1V2BinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        ER1V2OracleInformationGainPolicy(),
        episode_seed=0,
    )
    assert trigger.calls == 1
    step = result.trace[0]
    expected = model.update(
        result.trigger_conditioned_beliefs,
        step.chosen_action,
        step.outcome,
        Context.B,
    )
    assert step.posterior == expected


def _trajectory(env: ER1V2BinaryMachine, seed: int) -> tuple[object, ...]:
    env.reset(FailureMode.MISSING_LATENT_VARIABLE, episode_seed=seed)
    trigger = env.trigger_observation()
    return (
        trigger,
        env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1),
        env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1),
        env.run_experiment(DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A),
    )


def test_seeded_trajectory_reset_and_environment_independence() -> None:
    first = ER1V2BinaryMachine()
    second = ER1V2BinaryMachine()
    expected = _trajectory(first, 19)
    assert _trajectory(first, 19) == expected
    assert _trajectory(second, 19) == expected


def test_constructed_trigger_consumes_no_investigation_rng_draw() -> None:
    with_trigger = ER1V2BinaryMachine()
    without_trigger = ER1V2BinaryMachine()
    with_trigger.reset(FailureMode.WORLD_SHIFT, episode_seed=17)
    without_trigger.reset(FailureMode.WORLD_SHIFT, episode_seed=17)
    with_trigger.trigger_observation()
    assert with_trigger.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1) == (
        without_trigger.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1)
    )


def test_trigger_has_no_hidden_y_and_trusted_result_exposes_t_only() -> None:
    env = ER1V2BinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION, episode_seed=3)
    trigger = env.trigger_observation()
    assert env.get_ground_truth().y is None
    trusted = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)
    assert set(asdict(trusted)) == {"x", "trusted_t"}
    assert "y" not in asdict(trusted)
    assert trigger.x == 1 and trigger.o == 0


def test_initial_oracle_eig_and_zero_regret() -> None:
    beliefs = TriggerLikelihoodModel().conditioned_beliefs()
    model = ER1V2LikelihoodModel()
    scores = stochastic_information_gains(beliefs, model, Context.B)
    assert scores.repeat_trial == pytest.approx(0.22364893609980618)
    assert scores.use_trusted_sensor == pytest.approx(0.5662003117800551)
    assert scores.change_context == pytest.approx(0.425899648818598)
    result = ER1V2EpisodeRunner(max_experiments=5).run(
        ER1V2BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        ER1V2OracleInformationGainPolicy(),
        episode_seed=9,
    )
    assert all(step.action_regret == 0.0 for step in result.trace)


def test_oracle_view_contains_no_truth_trigger_model_or_repair() -> None:
    view = ER1V2OraclePolicyView(
        beliefs=TriggerLikelihoodModel().conditioned_beliefs(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        investigation_likelihood_model=ER1V2LikelihoodModel(),
    )
    assert set(asdict(view)) == {
        "beliefs", "current_context", "available_actions", "investigation_likelihood_model"
    }
    assert not hasattr(view, "failure_mode")
    assert not hasattr(view, "trigger_model")
    assert not hasattr(view, "correct_repair")


def test_repair_mapping_includes_no_change() -> None:
    assert repair_for_failure(FailureMode.NO_STRUCTURAL_CHANGE) is RepairOperator.NO_REPAIR


def test_er1_v1_active_constants_remain_candidate_001() -> None:
    assert ER1_DEFAULT_CONFIG.preferred_physical_probability == 0.90
    assert ER1_DEFAULT_CONFIG.primary_sensor_reliability == 0.95
    assert ER1_DEFAULT_CONFIG.corrupted_sensor_inversion_probability == 0.90
    assert ER1_DEFAULT_CONFIG.trusted_sensor_reliability == 0.99
    assert ER1_DEFAULT_CONFIG.diagnosis_threshold == 0.90
