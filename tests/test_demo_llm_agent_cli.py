"""Tests for selecting failures in the LLM smoke-demo CLI."""

import pytest

from epistemic_repair import (
    BinaryMachine,
    DeterministicMockLLMClient,
    FailureMode,
    LLMClient,
    LLMCondition,
    LLMConfig,
    LLMEpisodeRunner,
    LLMRequest,
    LLMResponse,
    PlannerOnlyLLMPolicy,
    run_llm_smoke,
)
from scripts.demo_llm_agent import (
    parse_args,
    print_attempt_history,
    selected_failure_modes,
)


class InvalidResponseClient(LLMClient):
    """Return a fixed malformed response without network access."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            self.response,
            provider_request_id=(
                f"fake-request-{self.call_count}-{self.response}"
            ),
        )


@pytest.mark.parametrize(
    ("selection", "expected"),
    (
        ("world_shift", (FailureMode.WORLD_SHIFT,)),
        ("sensor_corruption", (FailureMode.SENSOR_CORRUPTION,)),
        (
            "missing_latent_variable",
            (FailureMode.MISSING_LATENT_VARIABLE,),
        ),
        (
            "all",
            (
                FailureMode.WORLD_SHIFT,
                FailureMode.SENSOR_CORRUPTION,
                FailureMode.MISSING_LATENT_VARIABLE,
            ),
        ),
    ),
)
def test_failure_selector_maps_to_expected_modes(
    selection: str,
    expected: tuple[FailureMode, ...],
) -> None:
    assert selected_failure_modes(selection) == expected


def test_failure_selector_defaults_to_all() -> None:
    assert parse_args([]).failure == "all"


def test_verbose_attempts_is_optional() -> None:
    assert not parse_args([]).verbose_attempts
    assert parse_args(["--verbose-attempts"]).verbose_attempts


@pytest.mark.parametrize(
    "conditions",
    (
        (LLMCondition.FULL_AUTONOMOUS,),
        (LLMCondition.PLANNER_ONLY,),
        (LLMCondition.FULL_AUTONOMOUS, LLMCondition.PLANNER_ONLY),
    ),
)
def test_selected_failure_applies_to_each_condition(
    conditions: tuple[LLMCondition, ...],
) -> None:
    repetitions = 2
    results = run_llm_smoke(
        DeterministicMockLLMClient(),
        LLMConfig(max_retries=0),
        conditions=conditions,
        repetitions=repetitions,
        failure_modes=selected_failure_modes("sensor_corruption"),
    )

    assert tuple(result.condition for result in results) == conditions
    assert all(len(result.episodes) == repetitions for result in results)
    assert all(
        episode.evaluation_metadata.hidden_failure_mode
        is FailureMode.SENSOR_CORRUPTION
        for result in results
        for episode in result.episodes
    )


def test_failed_attempts_print_automatically_with_sanitized_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "never-print-this-api-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=2,
    ).run(
        BinaryMachine(),
        FailureMode.SENSOR_CORRUPTION,
        PlannerOnlyLLMPolicy(
            InvalidResponseClient(secret),
            LLMConfig(max_retries=1),
        ),
    )

    print_attempt_history(result)
    output = capsys.readouterr().out

    assert "termination_reason=MODEL_FAILURE" in output
    assert "call_number=1 attempt_number=1 status=INVALID_FORMAT" in output
    assert "call_number=1 attempt_number=2 status=INVALID_FORMAT" in output
    assert "error_type=StructuredResponseError" in output
    assert "error_message=" in output
    assert "raw_output='[REDACTED]'" in output
    assert "provider_request_id=fake-request-2-[REDACTED]" in output
    assert secret not in output


def test_successful_attempts_print_only_when_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_llm_smoke(
        DeterministicMockLLMClient(),
        LLMConfig(max_retries=0),
        conditions=(LLMCondition.PLANNER_ONLY,),
        repetitions=1,
        failure_modes=(FailureMode.SENSOR_CORRUPTION,),
    )[0].episodes[0]

    print_attempt_history(result)
    assert capsys.readouterr().out == ""

    print_attempt_history(result, verbose=True)
    output = capsys.readouterr().out
    assert "termination_reason=DIAGNOSED" in output
    assert "status=VALID" in output
