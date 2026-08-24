"""Research-invariant hardening tests for boundaries, numerics, and isolation."""

from dataclasses import fields, is_dataclass
from math import log2

import pytest

from epistemic_repair import (
    BENCHMARK_ACTIONS,
    AgentExperimentRecord,
    BenchmarkAgentView,
    BenchmarkDiagnosticPolicy,
    BinaryMachine,
    Context,
    DeterministicLikelihoodModel,
    DiagnosticAction,
    DiagnosticEpisodeRunner,
    ExperimentOutcome,
    FailureMode,
    HYPOTHESES,
    HypothesisBeliefs,
    OracleInformationGainPolicy,
    OraclePolicyView,
    OutcomeSignal,
    RandomDiagnosticPolicy,
    RepairOperator,
    RepeatTrialResult,
    TrustedSensorResult,
    evaluate_policy_budgets,
    expected_information_gain,
)


def test_arbitrary_positive_weights_are_normalized() -> None:
    beliefs = HypothesisBeliefs.from_weights(
        {
            FailureMode.WORLD_SHIFT: 2.0,
            FailureMode.SENSOR_CORRUPTION: 3.0,
            FailureMode.MISSING_LATENT_VARIABLE: 5.0,
        }
    )

    assert beliefs.values() == pytest.approx((0.2, 0.3, 0.5))
    assert sum(beliefs.values()) == pytest.approx(1.0)


def test_zero_entries_and_certainty_have_zero_entropy() -> None:
    beliefs = HypothesisBeliefs.from_weights(
        {
            FailureMode.WORLD_SHIFT: 0.0,
            FailureMode.SENSOR_CORRUPTION: 4.0,
            FailureMode.MISSING_LATENT_VARIABLE: 0.0,
        }
    )

    assert beliefs.entropy() == 0.0
    assert beliefs.confidence() == 1.0


@pytest.mark.parametrize(
    "probabilities",
    [
        (-0.1, 0.5, 0.6),
        (float("inf"), 0.0, 0.0),
        (True, 0.0, 0.0),
    ],
)
def test_malformed_direct_beliefs_are_rejected(
    probabilities: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        HypothesisBeliefs(*probabilities)


def test_all_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        HypothesisBeliefs.from_weights({hypothesis: 0.0 for hypothesis in HYPOTHESES})


def test_belief_weights_require_exact_hypothesis_keys() -> None:
    with pytest.raises(ValueError, match="exactly"):
        HypothesisBeliefs.from_weights(
            {
                FailureMode.WORLD_SHIFT: 1.0,
                FailureMode.SENSOR_CORRUPTION: 1.0,
            }
        )


def test_probability_sum_accepts_only_small_numerical_tolerance() -> None:
    beliefs = HypothesisBeliefs(0.1, 0.2, 0.7000000000005)
    assert sum(beliefs.values()) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="sum to 1"):
        HypothesisBeliefs(0.1, 0.2, 0.700001)


@pytest.mark.parametrize(
    ("beliefs", "expected"),
    [
        (HypothesisBeliefs(0.5, 0.5, 0.0), FailureMode.WORLD_SHIFT),
        (HypothesisBeliefs(0.0, 0.5, 0.5), FailureMode.SENSOR_CORRUPTION),
    ],
)
def test_argmax_tie_breaking_follows_canonical_hypothesis_order(
    beliefs: HypothesisBeliefs,
    expected: FailureMode,
) -> None:
    assert beliefs.most_likely() is expected


def test_repeated_compatible_updates_preserve_normalized_certainty() -> None:
    model = DeterministicLikelihoodModel()
    beliefs = HypothesisBeliefs.uniform()
    trusted_y_one = ExperimentOutcome(OutcomeSignal.TRUSTED_Y, 1)

    for _ in range(100):
        beliefs = model.update(
            beliefs,
            DiagnosticAction.USE_TRUSTED_SENSOR,
            trusted_y_one,
            Context.B,
        )

    assert beliefs.values() == (0.0, 1.0, 0.0)
    assert sum(beliefs.values()) == 1.0


def test_world_latent_posterior_information_gains_are_zero_zero_one() -> None:
    beliefs = HypothesisBeliefs(0.5, 0.0, 0.5)
    model = DeterministicLikelihoodModel()

    assert expected_information_gain(
        beliefs, DiagnosticAction.REPEAT_TRIAL, model, Context.B
    ) == 0.0
    assert expected_information_gain(
        beliefs, DiagnosticAction.USE_TRUSTED_SENSOR, model, Context.B
    ) == 0.0
    assert expected_information_gain(
        beliefs, DiagnosticAction.CHANGE_CONTEXT, model, Context.B
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("action", BENCHMARK_ACTIONS)
def test_expected_information_gain_is_zero_after_certainty(
    action: DiagnosticAction,
) -> None:
    beliefs = HypothesisBeliefs(0.0, 0.0, 1.0)
    gain = expected_information_gain(
        beliefs,
        action,
        DeterministicLikelihoodModel(),
        Context.B,
    )
    assert gain == 0.0


def test_deterministic_expected_entropy_never_increases() -> None:
    model = DeterministicLikelihoodModel()
    belief_states = (
        HypothesisBeliefs.uniform(),
        HypothesisBeliefs(0.5, 0.5, 0.0),
        HypothesisBeliefs(0.5, 0.0, 0.5),
        HypothesisBeliefs(0.0, 1.0, 0.0),
    )

    for beliefs in belief_states:
        for context in Context:
            for action in BENCHMARK_ACTIONS:
                gain = expected_information_gain(beliefs, action, model, context)
                assert gain >= -1e-12
                assert beliefs.entropy() - gain <= beliefs.entropy() + 1e-12


class CapturingBenchmarkPolicy(BenchmarkDiagnosticPolicy):
    """Record restricted views while following a fixed safe action sequence."""

    def __init__(self, actions: tuple[DiagnosticAction, ...]) -> None:
        self._actions = actions
        self.views: list[BenchmarkAgentView] = []

    def choose_action(self, view: BenchmarkAgentView) -> DiagnosticAction:
        self.views.append(view)
        return self._actions[len(self.views) - 1]


def _walk_dataclass_values(value: object):  # type: ignore[no-untyped-def]
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield field.name, getattr(value, field.name)
            yield from _walk_dataclass_values(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_dataclass_values(item)


def test_benchmark_policy_view_contains_no_privileged_or_hidden_objects() -> None:
    policy = CapturingBenchmarkPolicy(
        (DiagnosticAction.REPEAT_TRIAL, DiagnosticAction.USE_TRUSTED_SENSOR)
    )
    DiagnosticEpisodeRunner(max_experiments=2).run(
        BinaryMachine(), FailureMode.SENSOR_CORRUPTION, policy
    )

    assert len(policy.views) == 2
    for view in policy.views:
        assert not hasattr(view, "beliefs")
        assert not hasattr(view, "likelihood_model")
        assert DiagnosticAction.INSPECT_LATENT_VARIABLE not in view.available_actions
        for field_name, value in _walk_dataclass_values(view):
            assert field_name not in {
                "failure_mode",
                "correct_repair",
                "ground_truth",
                "z",
                "y",
            }
            assert not isinstance(
                value,
                (FailureMode, RepairOperator, DeterministicLikelihoodModel),
            )

    second_history = policy.views[1].experiment_history
    assert second_history == (
        AgentExperimentRecord(
            step_number=1,
            action=DiagnosticAction.REPEAT_TRIAL,
            result=RepeatTrialResult(x=1, o=0),
            context_before=Context.B,
            context_after=Context.B,
        ),
    )


@pytest.mark.parametrize("view_type", (BenchmarkAgentView, OraclePolicyView))
def test_policy_views_reject_internal_inspection_action(view_type: type) -> None:
    actions = BENCHMARK_ACTIONS + (DiagnosticAction.INSPECT_LATENT_VARIABLE,)
    with pytest.raises(ValueError, match="BENCHMARK_ACTIONS"):
        if view_type is BenchmarkAgentView:
            BenchmarkAgentView((), (), Context.B, actions, 1)
        else:
            OraclePolicyView(
                HypothesisBeliefs.uniform(),
                Context.B,
                actions,
                DeterministicLikelihoodModel(),
            )


def test_agent_history_rejects_result_from_wrong_action_channel() -> None:
    with pytest.raises(TypeError, match="does not match"):
        AgentExperimentRecord(
            step_number=1,
            action=DiagnosticAction.REPEAT_TRIAL,
            result=TrustedSensorResult(x=1, trusted_y=1),
            context_before=Context.B,
            context_after=Context.B,
        )


def test_different_random_seeds_can_produce_different_sequences() -> None:
    view = BenchmarkAgentView((), (), Context.B, BENCHMARK_ACTIONS, 20)
    first = RandomDiagnosticPolicy(seed=1)
    second = RandomDiagnosticPolicy(seed=2)

    assert [first.choose_action(view) for _ in range(20)] != [
        second.choose_action(view) for _ in range(20)
    ]


def test_oracle_budget_sweep_has_expected_minimal_sufficiency() -> None:
    evaluations = evaluate_policy_budgets(
        OracleInformationGainPolicy,
        budgets=(1, 2, 3, 5),
    )
    by_budget = {item.max_experiments: item.summary for item in evaluations}

    assert by_budget[1].diagnosis_accuracy == pytest.approx(2.0 / 3.0)
    assert by_budget[1].success_rate_within_budget == pytest.approx(1.0 / 3.0)
    for budget in (2, 3, 5):
        assert by_budget[budget].diagnosis_accuracy == 1.0
        assert by_budget[budget].success_rate_within_budget == 1.0
        assert by_budget[budget].mean_experiments == pytest.approx(5.0 / 3.0)
        assert by_budget[budget].total_action_regret == 0.0


def test_seeded_random_budget_sweep_is_reproducible_without_fixed_accuracy_claim() -> None:
    first = evaluate_policy_budgets(
        lambda: RandomDiagnosticPolicy(seed=17),
        budgets=(1, 2, 3, 5),
    )
    second = evaluate_policy_budgets(
        lambda: RandomDiagnosticPolicy(seed=17),
        budgets=(1, 2, 3, 5),
    )

    assert first == second
    assert tuple(item.max_experiments for item in first) == (1, 2, 3, 5)
    assert all(item.summary.mean_experiments > 0.0 for item in first)
    assert all(item.summary.total_action_regret >= 0.0 for item in first)


def test_runner_stops_immediately_when_threshold_is_reached() -> None:
    result = DiagnosticEpisodeRunner(max_experiments=5).run(
        BinaryMachine(),
        FailureMode.SENSOR_CORRUPTION,
        OracleInformationGainPolicy(),
    )

    assert result.reached_threshold
    assert result.experiments_used == 1
    assert len(result.trace) == 1


@pytest.mark.parametrize("budget", (0, -1, 1.5, True))
def test_invalid_experiment_budgets_are_rejected(budget: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        DiagnosticEpisodeRunner(max_experiments=budget)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", (0.0, -0.1, 1.01))
def test_invalid_diagnosis_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DiagnosticEpisodeRunner(diagnosis_threshold=threshold)


def test_trusted_sensor_does_not_change_primary_sensor_state() -> None:
    env = BinaryMachine()
    env.reset(FailureMode.SENSOR_CORRUPTION)

    trusted = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)
    assert trusted == TrustedSensorResult(x=1, trusted_y=1)
    assert env.step(1).o == 0


def test_multiple_environments_keep_context_and_failure_state_independent() -> None:
    latent_env = BinaryMachine()
    sensor_env = BinaryMachine()
    latent_env.reset(FailureMode.MISSING_LATENT_VARIABLE)
    sensor_env.reset(FailureMode.SENSOR_CORRUPTION)

    latent_env.run_experiment(
        DiagnosticAction.CHANGE_CONTEXT, x=1, context=Context.A
    )

    assert latent_env.step(1).o == 1
    assert sensor_env.step(1).o == 0
    assert latent_env.get_ground_truth().context is Context.A
    assert sensor_env.get_ground_truth().context is Context.B


def test_runner_history_is_fresh_for_each_episode() -> None:
    policy = CapturingBenchmarkPolicy(
        (DiagnosticAction.REPEAT_TRIAL, DiagnosticAction.REPEAT_TRIAL)
    )
    runner = DiagnosticEpisodeRunner(max_experiments=1)

    runner.run(BinaryMachine(), FailureMode.WORLD_SHIFT, policy)
    runner.run(BinaryMachine(), FailureMode.SENSOR_CORRUPTION, policy)

    assert policy.views[0].experiment_history == ()
    assert policy.views[1].experiment_history == ()


@pytest.mark.parametrize("bad_x", (-1, 2, True, 0.5))
def test_likelihood_model_rejects_invalid_binary_inputs(bad_x: object) -> None:
    with pytest.raises(ValueError, match="integer 0 or 1"):
        DeterministicLikelihoodModel().predict_outcome(
            DiagnosticAction.REPEAT_TRIAL,
            FailureMode.WORLD_SHIFT,
            Context.B,
            x=bad_x,  # type: ignore[arg-type]
        )


def test_likelihood_model_rejects_invalid_context_and_internal_action() -> None:
    model = DeterministicLikelihoodModel()
    with pytest.raises(TypeError, match="Context"):
        model.predict_outcome(
            DiagnosticAction.REPEAT_TRIAL,
            FailureMode.WORLD_SHIFT,
            "B",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="not available"):
        model.predict_outcome(
            DiagnosticAction.INSPECT_LATENT_VARIABLE,
            FailureMode.WORLD_SHIFT,
            Context.B,
        )
