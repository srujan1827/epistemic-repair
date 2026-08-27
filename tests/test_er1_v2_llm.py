"""No-network tests for V2 prompts, LLM views, and benchmark selection."""

import json

import pytest

from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, Context
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.llm_policy import (
    ER1V2FullAutonomousLLMPolicy,
    ER1V2PlannerOnlyLLMPolicy,
)
from epistemic_repair.er1_v2.llm_runner import ER1V2LLMEpisodeRunner
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
from scripts.demo_llm_agent import parse_args, selected_failure_modes


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
