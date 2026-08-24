"""Exhaustive cross-checks between simulator outcomes and normative predictions."""

import pytest

from epistemic_repair import (
    BENCHMARK_ACTIONS,
    BinaryMachine,
    ChangeContextResult,
    Context,
    DeterministicLikelihoodModel,
    DiagnosticAction,
    FailureMode,
    HYPOTHESES,
    RepeatTrialResult,
    TrustedSensorResult,
)


@pytest.mark.parametrize("failure_mode", HYPOTHESES)
@pytest.mark.parametrize("action", BENCHMARK_ACTIONS)
@pytest.mark.parametrize("x", (0, 1))
@pytest.mark.parametrize("current_context", (Context.A, Context.B))
def test_environment_outcome_has_unit_likelihood_under_matching_hypothesis(
    failure_mode: FailureMode,
    action: DiagnosticAction,
    x: int,
    current_context: Context,
) -> None:
    """Compare implementations directly for every V0 experimental combination."""
    env = BinaryMachine()
    model = DeterministicLikelihoodModel()
    env.reset(failure_mode)

    if current_context is Context.A:
        setup = env.run_experiment(
            DiagnosticAction.CHANGE_CONTEXT,
            x=0,
            context=Context.A,
        )
        assert isinstance(setup, ChangeContextResult)

    if action is DiagnosticAction.REPEAT_TRIAL:
        result = env.run_experiment(action, x=x)
        assert isinstance(result, RepeatTrialResult)
    elif action is DiagnosticAction.USE_TRUSTED_SENSOR:
        result = env.run_experiment(action, x=x)
        assert isinstance(result, TrustedSensorResult)
    else:
        target_context = model.target_context(current_context)
        result = env.run_experiment(action, x=x, context=target_context)
        assert isinstance(result, ChangeContextResult)

    actual_outcome = model.outcome_from_result(action, result, current_context)
    predicted_outcome = model.predict_outcome(
        action,
        failure_mode,
        current_context,
        x=x,
    )

    assert actual_outcome == predicted_outcome
    assert (
        model.likelihood(
            actual_outcome,
            failure_mode,
            action,
            current_context,
            x=x,
        )
        == 1.0
    )

