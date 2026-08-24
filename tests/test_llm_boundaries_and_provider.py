"""Information-boundary, configuration, and Gemini-adapter tests without network."""

import inspect
import json
from pathlib import Path

import pytest

from epistemic_repair import (
    BENCHMARK_ACTIONS,
    AgentExperimentRecord,
    BenchmarkAgentView,
    BinaryMachine,
    Context,
    DeterministicLikelihoodModel,
    DiagnosticAction,
    FailureMode,
    FullAutonomousLLMPolicy,
    GeminiLLMClient,
    HypothesisBeliefs,
    LLMClient,
    LLMCondition,
    LLMConfig,
    LLMConfigurationError,
    LLMEpisodeRunner,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    Observation,
    OraclePolicyView,
    PlannerOnlyLLMPolicy,
    RepeatTrialResult,
    TrustedSensorResult,
)
from epistemic_repair.llm.config import (
    load_dotenv_if_present,
    require_gemini_api_key,
)
from epistemic_repair.prompts import (
    build_full_autonomous_prompt,
    build_planner_only_prompt,
)


class RecordingClient(LLMClient):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(self.response_text, provider_request_id="fake-1")


def _initial_view() -> BenchmarkAgentView:
    return BenchmarkAgentView(
        initial_history=(Observation(x=1, o=0),),
        experiment_history=(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=2,
    )


def _full_diagnosis(label: str = "WORLD_SHIFT") -> str:
    beliefs = {
        "WORLD_SHIFT": 1.0 if label == "WORLD_SHIFT" else 0.0,
        "SENSOR_CORRUPTION": 1.0 if label == "SENSOR_CORRUPTION" else 0.0,
        "MISSING_LATENT_VARIABLE": (
            1.0 if label == "MISSING_LATENT_VARIABLE" else 0.0
        ),
    }
    return json.dumps(
        {
            "decision": "DIAGNOSE",
            "diagnosis": label,
            "beliefs": beliefs,
            "confidence": 1.0,
            "reason_summary": "The available evidence supports this diagnosis.",
        }
    )


def test_full_autonomous_policy_accepts_only_benchmark_agent_view() -> None:
    client = RecordingClient(_full_diagnosis())
    policy = FullAutonomousLLMPolicy(client, LLMConfig(max_retries=0))
    oracle_view = OraclePolicyView(
        HypothesisBeliefs.uniform(),
        Context.B,
        BENCHMARK_ACTIONS,
        DeterministicLikelihoodModel(),
    )

    with pytest.raises(TypeError, match="BenchmarkAgentView"):
        policy.decide(oracle_view)  # type: ignore[arg-type]
    assert client.requests == []


def test_autonomous_prompt_excludes_privileged_and_hidden_metadata() -> None:
    prompt = build_full_autonomous_prompt(_initial_view())

    for forbidden in (
        "get_ground_truth",
        "RepairOperator",
        "correct_repair",
        "DeterministicLikelihoodModel",
        "expected information gain",
        "oracle action",
        "hidden Z",
    ):
        assert forbidden not in prompt
    assert "INSPECT_LATENT_VARIABLE" not in prompt
    assert "trusted Y=" not in prompt


def test_trusted_y_appears_only_after_legitimate_trusted_result() -> None:
    view = BenchmarkAgentView(
        initial_history=(Observation(1, 0),),
        experiment_history=(
            AgentExperimentRecord(
                1,
                DiagnosticAction.USE_TRUSTED_SENSOR,
                TrustedSensorResult(x=1, trusted_y=1),
                Context.B,
                Context.B,
            ),
        ),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=1,
    )

    prompt = build_full_autonomous_prompt(view)
    assert "trusted Y=1" in prompt


def test_planner_prompt_contains_probabilities_but_no_oracle_plan_or_likelihoods() -> None:
    prompt = build_planner_only_prompt(
        _initial_view(), HypothesisBeliefs(0.5, 0.0, 0.5)
    )

    assert "WORLD_SHIFT: 0.500000" in prompt
    assert "SENSOR_CORRUPTION: 0.000000" in prompt
    assert "MISSING_LATENT_VARIABLE: 0.500000" in prompt
    assert "likelihood" not in prompt.lower()
    assert "expected information gain" not in prompt.lower()
    assert "oracle action" not in prompt.lower()


def test_initial_llm_request_is_identical_across_hidden_failure_modes() -> None:
    prompts = []
    for mode in (
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        client = RecordingClient(_full_diagnosis())
        policy = FullAutonomousLLMPolicy(client, LLMConfig(max_retries=0))
        LLMEpisodeRunner(
            condition=LLMCondition.FULL_AUTONOMOUS,
            experiment_budget=2,
        ).run(BinaryMachine(), mode, policy)
        prompts.append(client.requests[0].prompt)

    assert prompts == [prompts[0]] * 3


def test_dotenv_loading_preserves_shell_precedence(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=file-secret\nOTHER=value\n", encoding="utf-8")
    environ = {"GEMINI_API_KEY": "shell-secret"}

    assert load_dotenv_if_present(dotenv, environ=environ)
    assert environ["GEMINI_API_KEY"] == "shell-secret"
    assert environ["OTHER"] == "value"


def test_missing_gemini_key_has_clear_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        require_gemini_api_key(
            dotenv_path=tmp_path / "missing.env",
            environ={},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": ""},
        {"max_output_tokens": 0},
        {"request_timeout_seconds": 0},
        {"max_retries": -1},
        {"max_decision_calls": 0},
    ],
)
def test_invalid_llm_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LLMConfig(**kwargs)  # type: ignore[arg-type]


class FakeSDKModels:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeSDKClient:
    def __init__(self, models: FakeSDKModels) -> None:
        self.models = models


class FakeSDKResponse:
    text = '{"decision":"DIAGNOSE"}'
    response_id = "gemini-fake-id"


def test_gemini_adapter_uses_configured_model_schema_and_no_tools() -> None:
    models = FakeSDKModels(response=FakeSDKResponse())
    config = LLMConfig(model_id="named-test-model", thinking_level="high")
    client = GeminiLLMClient(
        config,
        api_key="test-only-key",
        sdk_client=FakeSDKClient(models),
    )

    response = client.generate(
        LLMRequest(prompt="safe prompt", response_schema={"type": "object"})
    )

    call = models.calls[0]
    assert call["model"] == "named-test-model"
    assert call["contents"] == "safe prompt"
    assert "tools" not in call
    generation = call["config"]
    assert isinstance(generation, dict)
    assert generation["max_output_tokens"] == config.max_output_tokens
    assert generation["thinking_config"] == {
        "thinking_level": "high",
        "include_thoughts": False,
    }
    assert response.provider_request_id == "gemini-fake-id"


def test_gemini_adapter_maps_timeout_without_network() -> None:
    client = GeminiLLMClient(
        LLMConfig(),
        api_key="test-only-key",
        sdk_client=FakeSDKClient(FakeSDKModels(error=TimeoutError("slow"))),
    )

    with pytest.raises(LLMTimeoutError, match="timed out"):
        client.generate(LLMRequest("prompt", {"type": "object"}))


def test_unknown_provider_error_never_falls_back() -> None:
    class BadRequest(Exception):
        code = 404

    client = GeminiLLMClient(
        LLMConfig(model_id="unavailable-model"),
        api_key="test-only-key",
        sdk_client=FakeSDKClient(FakeSDKModels(error=BadRequest("not found"))),
    )

    with pytest.raises(LLMProviderError, match="model_id"):
        client.generate(LLMRequest("prompt", {"type": "object"}))


def test_gemini_rate_limit_is_reported_without_fallback() -> None:
    class RateLimited(Exception):
        code = 429

    client = GeminiLLMClient(
        LLMConfig(),
        api_key="test-only-key",
        sdk_client=FakeSDKClient(FakeSDKModels(error=RateLimited("quota"))),
    )

    with pytest.raises(LLMRateLimitError, match="rate limit"):
        client.generate(LLMRequest("prompt", {"type": "object"}))


def test_provider_code_is_absent_from_environment_and_normative_modules() -> None:
    environment_source = inspect.getsource(BinaryMachine)
    likelihood_source = inspect.getsource(DeterministicLikelihoodModel)

    assert "gemini" not in environment_source.lower()
    assert "gemini" not in likelihood_source.lower()
