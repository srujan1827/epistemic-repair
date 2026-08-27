"""No-network tests for V2 prompts, LLM views, and benchmark selection."""

import json
import inspect
import sys

import pytest

from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, Context
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.llm_policy import (
    ER1V2FullAutonomousLLMPolicy,
    ER1V2PlannerOnlyLLMPolicy,
)
from epistemic_repair.er1_v2.llm_runner import (
    ER1V2LLMEpisodeRunner,
    evaluate_er1_v2_diagnosis,
)
from epistemic_repair.evaluation.llm_runner import LLMTerminationReason
from epistemic_repair.er1_v2.llm_smoke import run_er1_v2_llm_smoke
from epistemic_repair.evaluation.er1_llm_smoke import run_er1_llm_smoke
from epistemic_repair.evaluation.llm_smoke import run_llm_smoke
from epistemic_repair.er1_v2.trigger_model import TriggerLikelihoodModel
from epistemic_repair.er1_v2.views import ER1V2BenchmarkAgentView
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient, LLMRequest, LLMResponse
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import LLMCondition
from epistemic_repair.prompts.binary_er1 import BINARY_ER1_PROMPT_VERSION
from epistemic_repair.prompts.binary_er1_v2 import (
    BINARY_ER1_V2_PROMPT_VERSION,
    build_er1_v2_full_autonomous_prompt,
    build_er1_v2_planner_only_prompt,
)
from scripts.demo_llm_agent import main, parse_args, selected_failure_modes


class CapturingClient(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=json.dumps(self.responses.pop(0)))


def _view() -> ER1V2BenchmarkAgentView:
    return ER1V2BenchmarkAgentView(
        trigger_history=(Observation(x=1, o=0),),
        experiment_history=(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=2,
    )


def _experiment(action: str, *, autonomous: bool) -> dict[str, object]:
    response: dict[str, object] = {
        "decision": "RUN_EXPERIMENT",
        "action": action,
        "diagnosis": None,
        "reason_summary": "Collect persistent evidence.",
    }
    if autonomous:
        response["beliefs"] = {
            "NO_STRUCTURAL_CHANGE": 0.25,
            "WORLD_SHIFT": 0.25,
            "SENSOR_CORRUPTION": 0.25,
            "MISSING_LATENT_VARIABLE": 0.25,
        }
        response["confidence"] = None
    return response


def _diagnose(label: str, *, autonomous: bool) -> dict[str, object]:
    response: dict[str, object] = {
        "decision": "DIAGNOSE",
        "action": None,
        "diagnosis": label,
        "reason_summary": "The persistent evidence is decisive.",
    }
    if autonomous:
        response["beliefs"] = {
            hypothesis.value: 1.0 if hypothesis.value == label else 0.0
            for hypothesis in (
                FailureMode.NO_STRUCTURAL_CHANGE,
                FailureMode.WORLD_SHIFT,
                FailureMode.SENSOR_CORRUPTION,
                FailureMode.MISSING_LATENT_VARIABLE,
            )
        }
        response["confidence"] = 1.0
    return response


def test_full_prompt_is_qualitative_and_contains_no_privileged_parameters() -> None:
    prompt = build_er1_v2_full_autonomous_prompt(_view())
    assert BINARY_ER1_V2_PROMPT_VERSION in prompt
    assert "transient trigger anomaly" in prompt
    assert "persistent investigation dynamics" in prompt
    for forbidden in (
        "0.30", "0.70", "0.65", "0.95", "0.90", "0.99",
        "UPDATE_WORLD_MODEL", "RECALIBRATE_SENSOR", "ADD_LATENT_VARIABLE",
        "expected information gain", "oracle action", "hidden Y", "hidden Z",
    ):
        assert forbidden not in prompt
    assert "Authoritative current probabilities" not in prompt


def test_planner_prompt_contains_posterior_but_no_likelihoods() -> None:
    prompt = build_er1_v2_planner_only_prompt(
        _view(), TriggerLikelihoodModel().conditioned_beliefs()
    )
    assert "Authoritative current probabilities" in prompt
    assert "NO_STRUCTURAL_CHANGE: 0.127660" in prompt
    assert "P(A0" not in prompt
    assert "0.300000" not in prompt
    assert "0.950000" not in prompt


@pytest.mark.parametrize("condition", (LLMCondition.FULL_AUTONOMOUS, LLMCondition.PLANNER_ONLY))
def test_v2_runner_passes_only_exact_v2_agent_view(condition: LLMCondition) -> None:
    autonomous = condition is LLMCondition.FULL_AUTONOMOUS
    client = CapturingClient([
        _experiment("USE_TRUSTED_SENSOR", autonomous=autonomous),
        _diagnose("NO_STRUCTURAL_CHANGE", autonomous=autonomous),
    ])
    config = LLMConfig(max_decision_calls=2, max_retries=0)
    policy = (
        ER1V2FullAutonomousLLMPolicy(client, config)
        if autonomous
        else ER1V2PlannerOnlyLLMPolicy(client, config)
    )
    result = ER1V2LLMEpisodeRunner(
        condition=condition, experiment_budget=1, episode_seed=0
    ).run(ER1V2BinaryMachine(), FailureMode.NO_STRUCTURAL_CHANGE, policy)
    assert all(type(turn.agent_view) is ER1V2BenchmarkAgentView for turn in result.trace)
    assert result.run_metadata.prompt_version == BINARY_ER1_V2_PROMPT_VERSION
    assert result.run_metadata.benchmark_version == "binary_er1_v2"
    assert all("INSPECT_LATENT_VARIABLE" not in request.prompt for request in client.requests)


def test_v1_prompt_version_is_unchanged() -> None:
    assert BINARY_ER1_PROMPT_VERSION == "binary_er1_001"


def test_benchmark_selector_preserves_er1_alias_and_adds_explicit_versions() -> None:
    assert parse_args(["--benchmark", "er1"]).benchmark == "er1"
    assert parse_args(["--benchmark", "er1_v1"]).benchmark == "er1_v1"
    assert parse_args(["--benchmark", "er1_v2"]).benchmark == "er1_v2"
    assert selected_failure_modes("all", "er1") == selected_failure_modes("all", "er1_v1")
    assert selected_failure_modes("all", "er1_v1") == selected_failure_modes("all", "er1_v2")


def test_diagnosis_threshold_cli_default_and_override() -> None:
    assert parse_args([]).diagnosis_threshold == 0.90
    args = parse_args([
        "--benchmark", "er1_v2",
        "--diagnosis-threshold", "0.95",
        "--max-decision-calls", "9",
    ])
    assert args.diagnosis_threshold == 0.95
    assert args.max_decision_calls == 9
    assert LLMConfig().max_decision_calls == 4


@pytest.mark.parametrize("value", ("0", "-0.1", "1.01", "nan", "inf", "not-a-number"))
def test_invalid_diagnosis_threshold_cli_values_are_rejected(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--diagnosis-threshold", value])


def test_v2_smoke_passes_threshold_to_runner_metadata() -> None:
    client = CapturingClient([
        _diagnose("NO_STRUCTURAL_CHANGE", autonomous=True),
    ])
    result = run_er1_v2_llm_smoke(
        client,
        LLMConfig(max_retries=0, max_decision_calls=1),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        repetitions=1,
        experiment_budget=8,
        diagnosis_threshold=0.95,
        failure_modes=(FailureMode.NO_STRUCTURAL_CHANGE,),
    )[0].episodes[0]
    assert result.evaluation_metadata.diagnosis_threshold == 0.95
    assert result.run_metadata.experiment_budget == 8


def test_historical_smoke_signatures_remain_threshold_free() -> None:
    assert "diagnosis_threshold" not in inspect.signature(run_llm_smoke).parameters
    assert "diagnosis_threshold" not in inspect.signature(run_er1_llm_smoke).parameters
    assert inspect.signature(run_er1_v2_llm_smoke).parameters[
        "diagnosis_threshold"
    ].default == 0.90


def test_v2_cli_prints_threshold_without_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "demo_llm_agent.py",
        "--benchmark", "er1_v2",
        "--mock",
        "--condition", "full",
        "--failure", "no_structural_change",
        "--budget", "2",
        "--diagnosis-threshold", "0.95",
        "--repetitions", "1",
        "--max-decision-calls", "3",
    ])
    main()
    output = capsys.readouterr().out
    assert "diagnosis threshold: 0.95" in output
    assert "diagnosis_correct=True" in output
    assert "within_budget=True" in output
    assert "threshold_success=False" in output
    assert "premature=True" in output
    assert "aggregate: accuracy=1.000 within_budget=1.000" in output
    assert "threshold_success=0.000 premature=1.000" in output


def _normative_beliefs(
    *,
    no_change: float,
    world: float,
    sensor: float,
    latent: float,
) -> StochasticHypothesisBeliefs:
    return StochasticHypothesisBeliefs.from_weights({
        FailureMode.NO_STRUCTURAL_CHANGE: no_change,
        FailureMode.WORLD_SHIFT: world,
        FailureMode.SENSOR_CORRUPTION: sensor,
        FailureMode.MISSING_LATENT_VARIABLE: latent,
    })


def test_correct_diagnosis_below_threshold_is_premature_not_threshold_success() -> None:
    evaluation = evaluate_er1_v2_diagnosis(
        final_diagnosis=FailureMode.NO_STRUCTURAL_CHANGE,
        hidden_failure_mode=FailureMode.NO_STRUCTURAL_CHANGE,
        normative_beliefs=_normative_beliefs(
            no_change=0.85, world=0.05, sensor=0.05, latent=0.05
        ),
        diagnosis_threshold=0.95,
        termination_reason=LLMTerminationReason.DIAGNOSED,
    )
    assert evaluation.diagnosis_correct
    assert evaluation.diagnosed_correctly_within_budget
    assert not evaluation.threshold_qualified_success
    assert evaluation.premature_diagnosis
    assert evaluation.normative_probability_of_final_diagnosis == pytest.approx(0.85)


def test_correct_diagnosis_at_threshold_is_qualified_and_not_premature() -> None:
    evaluation = evaluate_er1_v2_diagnosis(
        final_diagnosis=FailureMode.NO_STRUCTURAL_CHANGE,
        hidden_failure_mode=FailureMode.NO_STRUCTURAL_CHANGE,
        normative_beliefs=_normative_beliefs(
            no_change=0.96, world=0.01, sensor=0.01, latent=0.02
        ),
        diagnosis_threshold=0.95,
        termination_reason=LLMTerminationReason.DIAGNOSED,
    )
    assert evaluation.diagnosis_correct
    assert evaluation.threshold_qualified_success
    assert not evaluation.premature_diagnosis


def test_wrong_diagnosis_is_premature_when_only_another_hypothesis_is_supported() -> None:
    evaluation = evaluate_er1_v2_diagnosis(
        final_diagnosis=FailureMode.SENSOR_CORRUPTION,
        hidden_failure_mode=FailureMode.WORLD_SHIFT,
        normative_beliefs=_normative_beliefs(
            no_change=0.01, world=0.96, sensor=0.01, latent=0.02
        ),
        diagnosis_threshold=0.95,
        termination_reason=LLMTerminationReason.DIAGNOSED,
    )
    assert not evaluation.diagnosis_correct
    assert not evaluation.threshold_qualified_success
    assert evaluation.premature_diagnosis
    assert evaluation.normative_probability_of_final_diagnosis == pytest.approx(0.01)


def test_wrong_but_normatively_supported_diagnosis_is_not_premature() -> None:
    evaluation = evaluate_er1_v2_diagnosis(
        final_diagnosis=FailureMode.SENSOR_CORRUPTION,
        hidden_failure_mode=FailureMode.WORLD_SHIFT,
        normative_beliefs=_normative_beliefs(
            no_change=0.01, world=0.01, sensor=0.96, latent=0.02
        ),
        diagnosis_threshold=0.95,
        termination_reason=LLMTerminationReason.DIAGNOSED,
    )
    assert not evaluation.diagnosis_correct
    assert not evaluation.threshold_qualified_success
    assert not evaluation.premature_diagnosis


@pytest.mark.parametrize(
    "termination",
    (
        LLMTerminationReason.MODEL_FAILURE,
        LLMTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED,
    ),
)
def test_no_diagnosis_has_no_normative_support_or_threshold_success(
    termination: LLMTerminationReason,
) -> None:
    evaluation = evaluate_er1_v2_diagnosis(
        final_diagnosis=None,
        hidden_failure_mode=FailureMode.NO_STRUCTURAL_CHANGE,
        normative_beliefs=TriggerLikelihoodModel().conditioned_beliefs(),
        diagnosis_threshold=0.95,
        termination_reason=termination,
    )
    assert not evaluation.diagnosis_correct
    assert not evaluation.diagnosed_correctly_within_budget
    assert not evaluation.threshold_qualified_success
    assert not evaluation.premature_diagnosis
    assert evaluation.normative_probability_of_final_diagnosis is None


def test_first_live_episode_pattern_is_correct_but_premature() -> None:
    client = CapturingClient([
        _experiment("REPEAT_TRIAL", autonomous=True),
        _diagnose("NO_STRUCTURAL_CHANGE", autonomous=True),
    ])
    result = run_er1_v2_llm_smoke(
        client,
        LLMConfig(max_retries=0, max_decision_calls=2),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        repetitions=1,
        experiment_budget=8,
        diagnosis_threshold=0.95,
        base_episode_seed=0,
        failure_modes=(FailureMode.NO_STRUCTURAL_CHANGE,),
    )[0]
    episode = result.episodes[0]
    assert episode.diagnosis_correct
    assert episode.diagnosed_correctly_within_budget
    assert not episode.threshold_qualified_success
    assert episode.premature_diagnosis
    assert episode.normative_probability_of_final_diagnosis == pytest.approx(
        0.4861235452103849
    )
    assert result.summary.diagnosis_accuracy == 1.0
    assert result.summary.diagnosed_correctly_within_budget == 1.0
    assert result.summary.threshold_qualified_success == 0.0
    assert result.summary.premature_diagnosis_rate == 1.0


def test_model_failure_has_no_threshold_success_or_diagnosis_support() -> None:
    result = run_er1_v2_llm_smoke(
        CapturingClient([{}]),
        LLMConfig(max_retries=0, max_decision_calls=1),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        repetitions=1,
        experiment_budget=8,
        diagnosis_threshold=0.95,
        failure_modes=(FailureMode.NO_STRUCTURAL_CHANGE,),
    )[0].episodes[0]
    assert result.termination_reason is LLMTerminationReason.MODEL_FAILURE
    assert result.final_diagnosis is None
    assert not result.threshold_qualified_success
    assert result.normative_probability_of_final_diagnosis is None


def test_budget_exhaustion_has_no_threshold_success_or_diagnosis_support() -> None:
    result = run_er1_v2_llm_smoke(
        CapturingClient([
            _experiment("REPEAT_TRIAL", autonomous=True),
            _experiment("USE_TRUSTED_SENSOR", autonomous=True),
        ]),
        LLMConfig(max_retries=0, max_decision_calls=2),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        repetitions=1,
        experiment_budget=1,
        diagnosis_threshold=0.95,
        failure_modes=(FailureMode.NO_STRUCTURAL_CHANGE,),
    )[0].episodes[0]
    assert result.termination_reason is LLMTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
    assert result.final_diagnosis is None
    assert not result.threshold_qualified_success
    assert result.normative_probability_of_final_diagnosis is None
