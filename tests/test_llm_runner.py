"""Fake-client tests for LLM interaction, retries, budgets, traces, and metrics."""

from datetime import datetime, timezone
import json

import pytest

from epistemic_repair import (
    BinaryMachine,
    DecisionType,
    DeterministicMockLLMClient,
    DiagnosticAction,
    FailureMode,
    FullAutonomousDecision,
    FullAutonomousLLMPolicy,
    LLMClient,
    LLMCondition,
    LLMConfig,
    LLMEpisodeRunner,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTerminationReason,
    LLMTimeoutError,
    PlannerOnlyLLMPolicy,
    summarize_llm_results,
    run_llm_smoke,
)
from epistemic_repair.policies.llm import LLMAttemptStatus


def _full_run(action: str, beliefs: tuple[float, float, float]) -> str:
    return json.dumps(
        {
            "decision": "RUN_EXPERIMENT",
            "action": action,
            "beliefs": {
                "WORLD_SHIFT": beliefs[0],
                "SENSOR_CORRUPTION": beliefs[1],
                "MISSING_LATENT_VARIABLE": beliefs[2],
            },
            "reason_summary": "Run this diagnostic experiment next.",
        }
    )


def _full_diagnose(label: str, beliefs: tuple[float, float, float]) -> str:
    return json.dumps(
        {
            "decision": "DIAGNOSE",
            "diagnosis": label,
            "beliefs": {
                "WORLD_SHIFT": beliefs[0],
                "SENSOR_CORRUPTION": beliefs[1],
                "MISSING_LATENT_VARIABLE": beliefs[2],
            },
            "confidence": max(beliefs),
            "reason_summary": "The accumulated evidence is decisive.",
        }
    )


def _planner_run(action: str) -> str:
    return json.dumps(
        {
            "decision": "RUN_EXPERIMENT",
            "action": action,
            "reason_summary": "This experiment should reduce uncertainty.",
        }
    )


def _planner_diagnose(label: str) -> str:
    return json.dumps(
        {
            "decision": "DIAGNOSE",
            "diagnosis": label,
            "reason_summary": "The supplied posterior is decisive.",
        }
    )


class QueueClient(LLMClient):
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("fake response queue exhausted")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(outcome, provider_request_id=f"fake-{len(self.requests)}")


def test_full_autonomous_multistep_episode_uses_only_safe_results() -> None:
    client = QueueClient(
        [
            _full_run("USE_TRUSTED_SENSOR", (1 / 3, 1 / 3, 1 / 3)),
            _full_diagnose("SENSOR_CORRUPTION", (0.0, 1.0, 0.0)),
        ]
    )
    policy = FullAutonomousLLMPolicy(client, LLMConfig(max_retries=0))

    result = LLMEpisodeRunner(
        condition=LLMCondition.FULL_AUTONOMOUS,
        experiment_budget=2,
    ).run(BinaryMachine(), FailureMode.SENSOR_CORRUPTION, policy)

    assert result.final_diagnosis is FailureMode.SENSOR_CORRUPTION
    assert result.success_within_budget
    assert result.experiments_used == 1
    assert result.decision_calls == 2
    assert result.cumulative_action_regret == 0.0
    assert "trusted Y=1" in client.requests[1].prompt
    assert isinstance(result.trace[1].policy_result.decision, FullAutonomousDecision)


def test_planner_only_multistep_episode_uses_normative_beliefs() -> None:
    client = QueueClient(
        [
            _planner_run("USE_TRUSTED_SENSOR"),
            _planner_run("CHANGE_CONTEXT"),
            _planner_diagnose("WORLD_SHIFT"),
        ]
    )
    policy = PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=0))

    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=2,
    ).run(BinaryMachine(), FailureMode.WORLD_SHIFT, policy)

    assert result.final_diagnosis is FailureMode.WORLD_SHIFT
    assert result.experiments_used == 2
    assert "WORLD_SHIFT: 0.500000" in client.requests[1].prompt
    assert "MISSING_LATENT_VARIABLE: 0.500000" in client.requests[1].prompt
    assert "expected information gain" not in client.requests[1].prompt.lower()


def test_conditions_remain_distinct_in_metadata() -> None:
    full = LLMEpisodeRunner(
        condition=LLMCondition.FULL_AUTONOMOUS,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        FullAutonomousLLMPolicy(
            QueueClient([_full_diagnose("WORLD_SHIFT", (1.0, 0.0, 0.0))]),
            LLMConfig(max_retries=0),
        ),
    )
    planner = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(
            QueueClient([_planner_diagnose("WORLD_SHIFT")]),
            LLMConfig(max_retries=0),
        ),
    )

    assert full.run_metadata.condition is LLMCondition.FULL_AUTONOMOUS
    assert planner.run_metadata.condition is LLMCondition.PLANNER_ONLY
    with pytest.raises(ValueError, match="cannot be pooled"):
        summarize_llm_results([full, planner])


def test_transient_timeout_is_retried_once_and_recorded() -> None:
    client = QueueClient(
        [
            LLMTimeoutError("fake timeout"),
            _planner_diagnose("WORLD_SHIFT"),
        ]
    )
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=1)),
    )

    assert result.total_retries == 1
    assert len(client.requests) == 2
    assert result.trace[0].policy_result.attempts[0].status is LLMAttemptStatus.TRANSIENT_ERROR
    assert result.trace[0].policy_result.attempts[1].status is LLMAttemptStatus.VALID


def test_format_failure_is_retried_when_configured() -> None:
    client = QueueClient(["not json", _planner_diagnose("WORLD_SHIFT")])
    policy = PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=1))

    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(BinaryMachine(), FailureMode.WORLD_SHIFT, policy)

    assert result.total_retries == 1
    assert result.trace[0].policy_result.attempts[0].status is LLMAttemptStatus.INVALID_FORMAT


def test_bad_scientific_diagnosis_is_not_retried() -> None:
    client = QueueClient(
        [
            _planner_diagnose("SENSOR_CORRUPTION"),
            _planner_diagnose("WORLD_SHIFT"),
        ]
    )
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=1)),
    )

    assert len(client.requests) == 1
    assert result.total_retries == 0
    assert not result.diagnosis_correct


def test_nontransient_provider_error_is_not_retried() -> None:
    client = QueueClient([LLMProviderError("model unavailable")])
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=3)),
    )

    assert len(client.requests) == 1
    assert result.termination_reason is LLMTerminationReason.MODEL_FAILURE
    assert result.trace[0].policy_result.attempts[0].status is LLMAttemptStatus.PROVIDER_ERROR


def test_invalid_response_after_retry_is_explicit_model_failure() -> None:
    client = QueueClient(["", "still not json"])
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=1)),
    )

    assert result.final_diagnosis is None
    assert result.termination_reason is LLMTerminationReason.MODEL_FAILURE
    assert result.total_retries == 1


def test_secret_is_redacted_from_audit_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "do-not-record-this-secret"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    client = QueueClient([secret])
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=0)),
    )

    attempt = result.trace[0].policy_result.attempts[0]
    assert secret not in (attempt.raw_output or "")
    assert attempt.raw_output == "[REDACTED]"


def test_experiment_budget_is_enforced_even_if_model_requests_more() -> None:
    client = QueueClient(
        [_planner_run("REPEAT_TRIAL"), _planner_run("REPEAT_TRIAL")]
    )
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=0)),
    )

    assert result.experiments_used == 1
    assert result.decision_calls == 2
    assert result.termination_reason is LLMTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
    assert "must return DIAGNOSE" in client.requests[1].prompt


def test_model_decision_call_budget_is_enforced() -> None:
    client = QueueClient([_planner_run("REPEAT_TRIAL"), _planner_diagnose("WORLD_SHIFT")])
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=2,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(
            client,
            LLMConfig(max_retries=0, max_decision_calls=1),
        ),
    )

    assert result.decision_calls == 1
    assert len(client.requests) == 1
    assert result.termination_reason is LLMTerminationReason.DECISION_CALL_BUDGET_EXHAUSTED


def test_same_fake_sequence_produces_reproducible_behavior() -> None:
    fixed_time = lambda: datetime(2026, 8, 22, tzinfo=timezone.utc)

    def run_once():  # type: ignore[no-untyped-def]
        client = QueueClient(
            [
                _planner_run("USE_TRUSTED_SENSOR"),
                _planner_diagnose("SENSOR_CORRUPTION"),
            ]
        )
        return LLMEpisodeRunner(
            condition=LLMCondition.PLANNER_ONLY,
            experiment_budget=2,
            episode_seed=77,
            timestamp_factory=fixed_time,
        ).run(
            BinaryMachine(),
            FailureMode.SENSOR_CORRUPTION,
            PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=0)),
        )

    assert run_once() == run_once()


def test_ground_truth_is_only_on_outer_evaluation_metadata() -> None:
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(
            QueueClient([_planner_diagnose("WORLD_SHIFT")]),
            LLMConfig(max_retries=0),
        ),
    )

    assert result.evaluation_metadata.hidden_failure_mode is FailureMode.WORLD_SHIFT
    assert not hasattr(result.run_metadata, "hidden_failure_mode")
    assert not hasattr(result.trace[0].agent_view, "hidden_failure_mode")
    assert "ground_truth" not in result.trace[0].policy_result.prompt


def test_prompt_version_and_reproducibility_fields_are_recorded() -> None:
    config = LLMConfig(model_id="configured-model", thinking_level="low")
    result = LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=1,
        episode_seed=42,
    ).run(
        BinaryMachine(),
        FailureMode.WORLD_SHIFT,
        PlannerOnlyLLMPolicy(
            QueueClient([_planner_diagnose("WORLD_SHIFT")]), config
        ),
    )

    metadata = result.run_metadata
    assert metadata.model_id == "configured-model"
    assert metadata.thinking_level == "low"
    assert metadata.prompt_version == "binary_v0_001"
    assert metadata.episode_seed == 42
    assert metadata.benchmark_version == "binary_v0"
    assert metadata.timestamp_utc


def test_llm_metrics_include_success_and_oracle_agreement() -> None:
    episodes = []
    for mode, diagnosis in (
        (FailureMode.WORLD_SHIFT, "WORLD_SHIFT"),
        (FailureMode.SENSOR_CORRUPTION, "SENSOR_CORRUPTION"),
    ):
        episodes.append(
            LLMEpisodeRunner(
                condition=LLMCondition.PLANNER_ONLY,
                experiment_budget=1,
            ).run(
                BinaryMachine(),
                mode,
                PlannerOnlyLLMPolicy(
                    QueueClient([_planner_diagnose(diagnosis)]),
                    LLMConfig(max_retries=0),
                ),
            )
        )

    summary = summarize_llm_results(episodes)
    assert summary.diagnosis_accuracy == 1.0
    assert summary.success_at_1 == 1.0
    assert summary.success_at_2 == 1.0
    assert summary.mean_experiments == 0.0
    assert summary.valid_decision_rate == 1.0


def test_mocked_smoke_runs_both_conditions_without_network() -> None:
    results = run_llm_smoke(
        DeterministicMockLLMClient(),
        LLMConfig(max_retries=0),
        repetitions=1,
        experiment_budget=2,
    )

    assert [result.condition for result in results] == [
        LLMCondition.FULL_AUTONOMOUS,
        LLMCondition.PLANNER_ONLY,
    ]
    assert all(len(result.episodes) == 3 for result in results)
    assert all(result.summary.diagnosis_accuracy == 1.0 for result in results)
    assert all(result.summary.success_within_budget == 1.0 for result in results)
    assert results[0].summary.belief_normalization_validity == 1.0
    assert results[1].summary.belief_normalization_validity is None
