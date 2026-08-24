"""Deterministic tests for beliefs, policies, episode traces, and metrics."""

from dataclasses import fields
from math import log2

import pytest

from epistemic_repair import (
    BENCHMARK_ACTIONS,
    ActionInformationGains,
    BenchmarkAgentView,
    BenchmarkDiagnosticPolicy,
    BinaryMachine,
    Context,
    DeterministicLikelihoodModel,
    DiagnosticAction,
    DiagnosticEpisodeRunner,
    ExperimentOutcome,
    FailureMode,
    GroundTruth,
    HYPOTHESES,
    HypothesisBeliefs,
    ImpossibleObservationError,
    OracleInformationGainPolicy,
    OraclePolicyView,
    OutcomeSignal,
    RandomDiagnosticPolicy,
    expected_information_gain,
    information_gains,
    summarize_results,
)


def test_uniform_prior_entropy_is_log2_three() -> None:
    beliefs = HypothesisBeliefs.uniform()

    assert beliefs.entropy() == pytest.approx(log2(3))
    assert all(
        beliefs.probability(hypothesis) == pytest.approx(1.0 / 3.0)
        for hypothesis in HYPOTHESES
    )


def test_initial_expected_information_gains() -> None:
    beliefs = HypothesisBeliefs.uniform()
    model = DeterministicLikelihoodModel()

    repeat_gain = expected_information_gain(
        beliefs, DiagnosticAction.REPEAT_TRIAL, model, Context.B
    )
    trusted_gain = expected_information_gain(
        beliefs, DiagnosticAction.USE_TRUSTED_SENSOR, model, Context.B
    )
    context_gain = expected_information_gain(
        beliefs, DiagnosticAction.CHANGE_CONTEXT, model, Context.B
    )

    assert repeat_gain == 0.0
    assert trusted_gain == pytest.approx(0.9182958340544894)
    assert context_gain == pytest.approx(0.9182958340544894)
    assert trusted_gain > 0.0


def test_trusted_y_one_identifies_sensor_corruption() -> None:
    prior = HypothesisBeliefs.uniform()
    model = DeterministicLikelihoodModel()
    outcome = ExperimentOutcome(OutcomeSignal.TRUSTED_Y, 1)

    posterior = model.update(
        prior,
        DiagnosticAction.USE_TRUSTED_SENSOR,
        outcome,
        Context.B,
    )

    assert posterior.probability(FailureMode.WORLD_SHIFT) == 0.0
    assert posterior.probability(FailureMode.SENSOR_CORRUPTION) == 1.0
    assert posterior.probability(FailureMode.MISSING_LATENT_VARIABLE) == 0.0


def test_trusted_y_zero_leaves_world_shift_and_latent_equally_likely() -> None:
    prior = HypothesisBeliefs.uniform()
    model = DeterministicLikelihoodModel()
    outcome = ExperimentOutcome(OutcomeSignal.TRUSTED_Y, 0)

    posterior = model.update(
        prior,
        DiagnosticAction.USE_TRUSTED_SENSOR,
        outcome,
        Context.B,
    )

    assert posterior.probability(FailureMode.WORLD_SHIFT) == 0.5
    assert posterior.probability(FailureMode.SENSOR_CORRUPTION) == 0.0
    assert posterior.probability(FailureMode.MISSING_LATENT_VARIABLE) == 0.5


def test_context_change_o_one_identifies_missing_latent_variable() -> None:
    prior = HypothesisBeliefs.uniform()
    model = DeterministicLikelihoodModel()
    outcome = ExperimentOutcome(OutcomeSignal.PRIMARY_O, 1)

    posterior = model.update(
        prior,
        DiagnosticAction.CHANGE_CONTEXT,
        outcome,
        Context.B,
    )

    assert posterior.probability(FailureMode.WORLD_SHIFT) == 0.0
    assert posterior.probability(FailureMode.SENSOR_CORRUPTION) == 0.0
    assert posterior.probability(FailureMode.MISSING_LATENT_VARIABLE) == 1.0


def test_impossible_observation_is_explicitly_rejected() -> None:
    model = DeterministicLikelihoodModel()
    certain_sensor = HypothesisBeliefs.from_weights(
        {
            FailureMode.WORLD_SHIFT: 0.0,
            FailureMode.SENSOR_CORRUPTION: 1.0,
            FailureMode.MISSING_LATENT_VARIABLE: 0.0,
        }
    )

    with pytest.raises(ImpossibleObservationError, match="zero probability"):
        model.update(
            certain_sensor,
            DiagnosticAction.USE_TRUSTED_SENSOR,
            ExperimentOutcome(OutcomeSignal.TRUSTED_Y, 0),
            Context.B,
        )


def test_oracle_chooses_a_maximum_information_gain_action() -> None:
    beliefs = HypothesisBeliefs.uniform()
    model = DeterministicLikelihoodModel()
    oracle = OracleInformationGainPolicy()
    scores = information_gains(beliefs, model, Context.B)

    chosen = oracle.choose_action(
        OraclePolicyView(
            beliefs=beliefs,
            current_context=Context.B,
            available_actions=BENCHMARK_ACTIONS,
            likelihood_model=model,
        )
    )

    assert scores.for_action(chosen) == scores.best_value()
    assert chosen is DiagnosticAction.USE_TRUSTED_SENSOR


@pytest.mark.parametrize("mode", HYPOTHESES)
def test_oracle_diagnoses_every_failure_within_budget(mode: FailureMode) -> None:
    runner = DiagnosticEpisodeRunner(max_experiments=3)

    result = runner.run(BinaryMachine(), mode, OracleInformationGainPolicy())

    assert result.predicted_diagnosis is mode
    assert result.diagnosis_correct
    assert result.success_within_budget
    assert result.experiments_used <= 2


@pytest.mark.parametrize("mode", HYPOTHESES)
def test_oracle_action_regret_is_zero(mode: FailureMode) -> None:
    result = DiagnosticEpisodeRunner(max_experiments=3).run(
        BinaryMachine(), mode, OracleInformationGainPolicy()
    )

    assert result.cumulative_action_regret == pytest.approx(0.0)
    assert all(step.action_regret == pytest.approx(0.0) for step in result.trace)


def test_random_policy_is_reproducible_with_fixed_seed() -> None:
    first = RandomDiagnosticPolicy(seed=2026)
    second = RandomDiagnosticPolicy(seed=2026)
    view = BenchmarkAgentView(
        initial_history=(),
        experiment_history=(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=20,
    )

    first_actions = [
        first.choose_action(view)
        for _ in range(20)
    ]
    second_actions = [
        second.choose_action(view)
        for _ in range(20)
    ]

    assert first_actions == second_actions


def test_latent_inspection_is_excluded_from_benchmark_actions() -> None:
    assert BENCHMARK_ACTIONS == (
        DiagnosticAction.REPEAT_TRIAL,
        DiagnosticAction.USE_TRUSTED_SENSOR,
        DiagnosticAction.CHANGE_CONTEXT,
    )
    assert DiagnosticAction.INSPECT_LATENT_VARIABLE not in BENCHMARK_ACTIONS


class RecordingPolicy(BenchmarkDiagnosticPolicy):
    """Test policy that records exactly what the runner supplies."""

    def __init__(self) -> None:
        self.calls: list[BenchmarkAgentView] = []

    def choose_action(self, view: BenchmarkAgentView) -> DiagnosticAction:
        self.calls.append(view)
        return DiagnosticAction.USE_TRUSTED_SENSOR


def test_policy_inputs_do_not_contain_evaluation_ground_truth() -> None:
    first_calls = []

    for mode in HYPOTHESES:
        policy = RecordingPolicy()
        DiagnosticEpisodeRunner(max_experiments=1).run(BinaryMachine(), mode, policy)
        assert len(policy.calls) == 1
        view = policy.calls[0]
        assert not hasattr(view, "ground_truth")
        assert not hasattr(view, "failure_mode")
        assert not hasattr(view, "beliefs")
        assert not hasattr(view, "likelihood_model")
        first_calls.append(view)

    assert first_calls == [first_calls[0]] * len(HYPOTHESES)


class GroundTruthAuditMachine(BinaryMachine):
    """Count evaluation-only metadata reads during an episode."""

    def __init__(self) -> None:
        super().__init__()
        self.ground_truth_reads = 0

    def get_ground_truth(self) -> GroundTruth:
        self.ground_truth_reads += 1
        return super().get_ground_truth()


def test_oracle_episode_reads_ground_truth_only_at_evaluation_boundary() -> None:
    env = GroundTruthAuditMachine()

    result = DiagnosticEpisodeRunner(max_experiments=3).run(
        env,
        FailureMode.WORLD_SHIFT,
        OracleInformationGainPolicy(),
    )

    assert result.diagnosis_correct
    assert env.ground_truth_reads == 1


def test_trace_steps_do_not_contain_hidden_failure_or_repair_labels() -> None:
    result = DiagnosticEpisodeRunner(max_experiments=2).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        OracleInformationGainPolicy(),
    )

    for step in result.trace:
        step_fields = {field.name for field in fields(step)}
        result_fields = {field.name for field in fields(step.experiment_result)}
        assert "ground_truth" not in step_fields
        assert "failure_mode" not in step_fields
        assert "correct_repair" not in step_fields
        assert "failure_mode" not in result_fields
        assert "correct_repair" not in result_fields


def test_episode_summary_reports_required_metrics() -> None:
    runner = DiagnosticEpisodeRunner(max_experiments=3)
    results = [
        runner.run(BinaryMachine(), mode, OracleInformationGainPolicy())
        for mode in HYPOTHESES
    ]

    summary = summarize_results(results)

    assert summary.diagnosis_accuracy == 1.0
    assert summary.success_rate_within_budget == 1.0
    assert summary.total_experiments == 5
    assert summary.mean_experiments == pytest.approx(5.0 / 3.0)
    assert summary.total_information_gain > 0.0
    assert summary.total_action_regret == pytest.approx(0.0)


def test_action_information_gain_type_exposes_only_benchmark_actions() -> None:
    scores = ActionInformationGains(0.0, 0.5, 0.5)

    with pytest.raises(ValueError, match="not available"):
        scores.for_action(DiagnosticAction.INSPECT_LATENT_VARIABLE)
