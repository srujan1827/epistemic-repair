"""Tests for selecting failures in the LLM smoke-demo CLI."""

import pytest

from epistemic_repair import (
    DeterministicMockLLMClient,
    FailureMode,
    LLMCondition,
    LLMConfig,
    run_llm_smoke,
)
from scripts.demo_llm_agent import parse_args, selected_failure_modes


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
