"""Analytic, sampling, Bayesian, oracle, and metric tests for ER-1."""

from dataclasses import replace
import inspect
from math import isclose

import pytest

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticExperimentOutcome,
    StochasticLikelihoodModel,
    StochasticOutcomeSignal,
    stochastic_expected_information_gain,
    stochastic_information_gains,
)
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import Context, DiagnosticAction
from epistemic_repair.diagnostics.results import StochasticTrustedSensorResult
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.environments.binary_machine import BinaryMachine
from epistemic_repair.er1.config import (
    ER1_DEFAULT_CONFIG,
    ER1_HYPOTHESES,
    ER1ProbabilityConfig,
)
from epistemic_repair.evaluation.stochastic_metrics import (
    summarize_stochastic_results,
)
from epistemic_repair.evaluation.stochastic_budgets import (
    evaluate_stochastic_policy_budgets,
)
from epistemic_repair.evaluation.stochastic_runner import (
    StochasticDiagnosticEpisodeRunner,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.policies.stochastic import (
    StochasticOracleInformationGainPolicy,
    StochasticRandomDiagnosticPolicy,
)
from epistemic_repair.policies.stochastic_views import StochasticOraclePolicyView
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


@pytest.fixture
def model() -> StochasticLikelihoodModel:
    return StochasticLikelihoodModel()


def test_er1_probability_constants_match_specification() -> None:
    config = ER1_DEFAULT_CONFIG
    assert config.preferred_physical_probability == 0.90
    assert config.primary_sensor_reliability == 0.95
    assert config.corrupted_sensor_inversion_probability == 0.90
    assert config.trusted_sensor_reliability == 0.99
    assert config.diagnosis_threshold == 0.90


def test_er0_environment_rejects_er1_only_hypothesis() -> None:
    with pytest.raises(ValueError, match="ER-1-only"):
        BinaryMachine().reset(FailureMode.NO_STRUCTURAL_CHANGE)


@pytest.mark.parametrize("hypothesis", ER1_HYPOTHESES)
@pytest.mark.parametrize("context", tuple(Context))
def test_er1_conditional_distributions_normalize(
    model: StochasticLikelihoodModel,
    hypothesis: FailureMode,
    context: Context,
) -> None:
    assert sum(model.probability_y(y, hypothesis, context) for y in (0, 1)) == 1
    assert sum(
        model.probability_primary_observation(o, hypothesis, context)
        for o in (0, 1)
    ) == pytest.approx(1.0)
    assert sum(
        model.probability_trusted_observation(t, hypothesis, context)
        for t in (0, 1)
    ) == pytest.approx(1.0)


def test_physical_and_sensor_probabilities_distinguish_er1_causes(
    model: StochasticLikelihoodModel,
) -> None:
    no_change = FailureMode.NO_STRUCTURAL_CHANGE
    world = FailureMode.WORLD_SHIFT
    sensor = FailureMode.SENSOR_CORRUPTION
    latent = FailureMode.MISSING_LATENT_VARIABLE

    assert model.probability_y(1, no_change, Context.B) == pytest.approx(0.90)
    assert model.probability_y(1, world, Context.B) == pytest.approx(0.10)
    assert model.probability_y(1, sensor, Context.B) == pytest.approx(0.90)
    assert model.probability_o_given_y(0, 1, sensor) == pytest.approx(0.90)
    assert model.probability_o_given_y(1, 1, world) == pytest.approx(0.95)
    assert model.probability_y(1, latent, Context.A) == pytest.approx(0.90)
    assert model.probability_y(1, latent, Context.B) == pytest.approx(0.10)
    for hypothesis in (no_change, world, sensor):
        assert model.probability_y(1, hypothesis, Context.A) == pytest.approx(
            model.probability_y(1, hypothesis, Context.B)
        )


def test_trusted_sensor_is_highly_reliable_but_not_perfect(
    model: StochasticLikelihoodModel,
) -> None:
    reliability = model.config.trusted_sensor_reliability
    assert reliability == 0.99
    assert 0.0 < 1.0 - reliability < 1.0


def test_trusted_sensor_sampling_matches_hidden_y_at_configured_rate() -> None:
    env = StochasticBinaryMachine()
    env.reset(FailureMode.NO_STRUCTURAL_CHANGE, episode_seed=2026)
    matches = 0
    samples = 8000
    for _ in range(samples):
        result = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)
        matches += result.trusted_t == env.get_ground_truth().y
    assert matches / samples == pytest.approx(0.99, abs=0.008)


def test_initial_anomaly_likelihoods_and_conditioning_math(
    model: StochasticLikelihoodModel,
) -> None:
    expected_likelihoods = {
        FailureMode.NO_STRUCTURAL_CHANGE: 0.14,
        FailureMode.WORLD_SHIFT: 0.86,
        FailureMode.SENSOR_CORRUPTION: 0.82,
        FailureMode.MISSING_LATENT_VARIABLE: 0.86,
    }
    for hypothesis, expected in expected_likelihoods.items():
        assert model.initial_anomaly_likelihood(hypothesis) == pytest.approx(expected)

    beliefs = model.conditioned_initial_beliefs()
    denominator = sum(expected_likelihoods.values())
    for hypothesis in ER1_HYPOTHESES:
        assert beliefs.probability(hypothesis) == pytest.approx(
            expected_likelihoods[hypothesis] / denominator
        )
    assert sum(beliefs.values()) == pytest.approx(1.0)
    assert len(set(round(value, 12) for value in beliefs.values())) > 1


def _trajectory(seed: int) -> tuple[object, ...]:
    env = StochasticBinaryMachine()
    env.reset(FailureMode.MISSING_LATENT_VARIABLE, episode_seed=seed)
    values: list[object] = [env.initial_anomaly()]
    values.append(env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1))
    values.append(env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1))
    values.append(
        env.run_experiment(
            DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A
        )
    )
    return tuple(values)


def test_seed_reproduces_complete_action_trajectory() -> None:
    assert _trajectory(12345) == _trajectory(12345)


def test_different_seeds_can_produce_different_trajectories() -> None:
    assert len({_trajectory(seed) for seed in range(20)}) > 1


def test_reset_reinitializes_rng_and_parallel_environments_do_not_interfere() -> None:
    env = StochasticBinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION, episode_seed=72)
    first = (
        env.initial_anomaly(),
        env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1),
    )
    env.reset(FailureMode.SENSOR_CORRUPTION, episode_seed=72)
    second = (
        env.initial_anomaly(),
        env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1),
    )
    assert first == second

    left = StochasticBinaryMachine()
    right = StochasticBinaryMachine()
    left.reset(FailureMode.WORLD_SHIFT, episode_seed=9)
    right.reset(FailureMode.WORLD_SHIFT, episode_seed=9)
    assert left.initial_anomaly() == right.initial_anomaly()
    left.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1)
    assert right.run_experiment(
        DiagnosticAction.REPEAT_TRIAL, x=1
    ) == _trajectory_for_world_first_repeat(9)


def _trajectory_for_world_first_repeat(seed: int) -> object:
    env = StochasticBinaryMachine()
    env.reset(FailureMode.WORLD_SHIFT, episode_seed=seed)
    env.initial_anomaly()
    return env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1)


def test_trusted_action_exposes_t_not_hidden_y() -> None:
    env = StochasticBinaryMachine()
    env.reset(FailureMode.NO_STRUCTURAL_CHANGE, episode_seed=4)
    env.initial_anomaly()
    result = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)
    assert isinstance(result, StochasticTrustedSensorResult)
    assert hasattr(result, "trusted_t")
    assert not hasattr(result, "trusted_y")
    assert "y" not in result.__dataclass_fields__


@pytest.mark.parametrize("hypothesis", ER1_HYPOTHESES)
def test_seeded_sampling_matches_analytic_primary_likelihoods(
    model: StochasticLikelihoodModel,
    hypothesis: FailureMode,
) -> None:
    env = StochasticBinaryMachine()
    env.reset(hypothesis, episode_seed=811)
    samples = 6000
    zeros = sum(
        env.run_experiment(DiagnosticAction.REPEAT_TRIAL, x=1).o == 0
        for _ in range(samples)
    )
    expected = model.probability_primary_observation(0, hypothesis, Context.B)
    assert zeros / samples == pytest.approx(expected, abs=0.025)


def test_seeded_trusted_sampling_matches_analytic_likelihood(
    model: StochasticLikelihoodModel,
) -> None:
    env = StochasticBinaryMachine()
    hypothesis = FailureMode.NO_STRUCTURAL_CHANGE
    env.reset(hypothesis, episode_seed=120)
    samples = 6000
    ones = sum(
        env.run_experiment(
            DiagnosticAction.USE_TRUSTED_SENSOR, x=1
        ).trusted_t
        for _ in range(samples)
    )
    expected = model.probability_trusted_observation(1, hypothesis, Context.B)
    assert ones / samples == pytest.approx(expected, abs=0.025)


def test_soft_bayesian_updates_accumulate_and_can_conflict(
    model: StochasticLikelihoodModel,
) -> None:
    beliefs = model.conditioned_initial_beliefs()
    primary_one = StochasticExperimentOutcome(
        StochasticOutcomeSignal.PRIMARY_O, 1
    )
    for _ in range(3):
        beliefs = model.update(
            beliefs, DiagnosticAction.REPEAT_TRIAL, primary_one, Context.B
        )
    assert beliefs.probability(FailureMode.NO_STRUCTURAL_CHANGE) > 0.3
    assert beliefs.confidence() < 1.0

    confidence_before_conflict = beliefs.confidence()
    primary_zero = StochasticExperimentOutcome(
        StochasticOutcomeSignal.PRIMARY_O, 0
    )
    beliefs = model.update(
        beliefs, DiagnosticAction.REPEAT_TRIAL, primary_zero, Context.B
    )
    assert beliefs.confidence() < confidence_before_conflict


def test_structural_hypotheses_gain_support_from_matching_evidence(
    model: StochasticLikelihoodModel,
) -> None:
    beliefs = model.conditioned_initial_beliefs()
    prior_world = beliefs.probability(FailureMode.WORLD_SHIFT)
    trusted_zero = StochasticExperimentOutcome(
        StochasticOutcomeSignal.TRUSTED_T, 0
    )
    beliefs = model.update(
        beliefs,
        DiagnosticAction.USE_TRUSTED_SENSOR,
        trusted_zero,
        Context.B,
    )
    assert beliefs.probability(FailureMode.WORLD_SHIFT) > prior_world
    assert beliefs.confidence() < 1.0


def test_expected_information_gain_is_probabilistic_and_nonnegative(
    model: StochasticLikelihoodModel,
) -> None:
    beliefs = model.conditioned_initial_beliefs()
    scores = stochastic_information_gains(beliefs, model, Context.B)
    assert all(value >= -1e-12 for _, value in scores.items())
    assert scores.repeat_trial > 0.0
    assert scores.use_trusted_sensor > scores.repeat_trial
    assert scores.change_context > scores.repeat_trial
    for action, expected in scores.items():
        assert expected == pytest.approx(
            stochastic_expected_information_gain(
                beliefs, action, model, Context.B
            )
        )


def test_information_gain_changes_with_beliefs_and_context(
    model: StochasticLikelihoodModel,
) -> None:
    initial = model.conditioned_initial_beliefs()
    context_b = stochastic_information_gains(initial, model, Context.B)
    context_a = stochastic_information_gains(initial, model, Context.A)
    assert context_a.items() != context_b.items()

    concentrated = StochasticHypothesisBeliefs.from_weights(
        {
            FailureMode.NO_STRUCTURAL_CHANGE: 0.7,
            FailureMode.WORLD_SHIFT: 0.1,
            FailureMode.SENSOR_CORRUPTION: 0.1,
            FailureMode.MISSING_LATENT_VARIABLE: 0.1,
        }
    )
    assert stochastic_information_gains(
        concentrated, model, Context.B
    ).items() != context_b.items()


def test_certainty_like_beliefs_have_low_information_gain(
    model: StochasticLikelihoodModel,
) -> None:
    beliefs = StochasticHypothesisBeliefs.from_weights(
        {
            FailureMode.NO_STRUCTURAL_CHANGE: 0.999999,
            FailureMode.WORLD_SHIFT: 0.0000003,
            FailureMode.SENSOR_CORRUPTION: 0.0000003,
            FailureMode.MISSING_LATENT_VARIABLE: 0.0000004,
        }
    )
    scores = stochastic_information_gains(beliefs, model, Context.B)
    assert scores.best_value() < 0.0001


def test_oracle_selects_maximum_expected_information_gain_without_truth() -> None:
    model = StochasticLikelihoodModel()
    beliefs = model.conditioned_initial_beliefs()
    view = StochasticOraclePolicyView(
        beliefs=beliefs,
        current_context=Context.B,
        available_actions=(
            DiagnosticAction.REPEAT_TRIAL,
            DiagnosticAction.USE_TRUSTED_SENSOR,
            DiagnosticAction.CHANGE_CONTEXT,
        ),
        likelihood_model=model,
    )
    policy = StochasticOracleInformationGainPolicy()
    scores = stochastic_information_gains(beliefs, model, Context.B)
    action = policy.choose_action(view)
    assert scores.for_action(action) == scores.best_value()
    source = inspect.getsource(policy.choose_action).lower()
    assert "ground_truth" not in source
    assert "failure_mode" not in source


@pytest.mark.parametrize("hypothesis", ER1_HYPOTHESES)
def test_seeded_oracle_is_reproducible_respects_budget_and_has_zero_regret(
    hypothesis: FailureMode,
) -> None:
    runner = StochasticDiagnosticEpisodeRunner(max_experiments=5)
    policy = StochasticOracleInformationGainPolicy()
    first = runner.run(
        StochasticBinaryMachine(), hypothesis, policy, episode_seed=44
    )
    second = runner.run(
        StochasticBinaryMachine(), hypothesis, policy, episode_seed=44
    )
    assert first.trace == second.trace
    assert first.experiments_used <= 5
    assert first.cumulative_action_regret == pytest.approx(0.0, abs=1e-12)
    if first.experiments_used < 5:
        assert first.reached_threshold


def test_random_policy_is_seeded_and_has_no_likelihood_access() -> None:
    runner = StochasticDiagnosticEpisodeRunner(max_experiments=3)
    first = runner.run(
        StochasticBinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        StochasticRandomDiagnosticPolicy(seed=8),
        episode_seed=99,
    )
    second = runner.run(
        StochasticBinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        StochasticRandomDiagnosticPolicy(seed=8),
        episode_seed=99,
    )
    assert first.trace == second.trace
    assert "likelihood" not in inspect.getsource(
        StochasticRandomDiagnosticPolicy.choose_action
    ).lower()


def test_er1_repair_mapping_assigns_no_repair_to_no_change() -> None:
    assert (
        repair_for_failure(FailureMode.NO_STRUCTURAL_CHANGE)
        is RepairOperator.NO_REPAIR
    )


def test_er1_structural_error_metrics() -> None:
    runner = StochasticDiagnosticEpisodeRunner(max_experiments=1)
    policy = StochasticOracleInformationGainPolicy()
    no_change = runner.run(
        StochasticBinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        policy,
        episode_seed=0,
    )
    structural = runner.run(
        StochasticBinaryMachine(),
        FailureMode.WORLD_SHIFT,
        policy,
        episode_seed=1,
    )
    false_structural = replace(
        no_change,
        predicted_diagnosis=FailureMode.WORLD_SHIFT,
        success_within_budget=False,
    )
    missed = replace(
        structural,
        predicted_diagnosis=FailureMode.NO_STRUCTURAL_CHANGE,
        success_within_budget=False,
    )
    summary = summarize_stochastic_results([false_structural, missed])
    assert summary.false_structural_diagnosis_rate == 1.0
    assert summary.missed_structural_failure_rate == 1.0
    assert summary.diagnosis_accuracy == 0.0


def test_custom_probability_config_is_validated() -> None:
    with pytest.raises(ValueError):
        ER1ProbabilityConfig(trusted_sensor_reliability=1.0)
    assert isclose(
        ER1ProbabilityConfig(diagnosis_threshold=0.8).diagnosis_threshold,
        0.8,
    )


def test_er1_budget_evaluation_supports_requested_budget_set() -> None:
    evaluations = evaluate_stochastic_policy_budgets(
        StochasticOracleInformationGainPolicy,
        budgets=(1, 2, 3, 5, 8),
        base_episode_seed=50,
    )
    assert tuple(item.max_experiments for item in evaluations) == (1, 2, 3, 5, 8)
    assert all(item.summary.episode_count == 4 for item in evaluations)
