"""ER-1 LLM schemas, prompts, boundaries, runner, metrics, and CLI tests."""

from datetime import datetime, timezone
import json

import pytest

from epistemic_repair.beliefs.stochastic_likelihoods import StochasticLikelihoodModel
from epistemic_repair.diagnostics.actions import BENCHMARK_ACTIONS, Context, DiagnosticAction
from epistemic_repair.diagnostics.results import StochasticTrustedSensorResult
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.evaluation.er1_llm_metrics import summarize_er1_llm_results
from epistemic_repair.evaluation.er1_llm_runner import (
    ER1LLMEpisodeRunner,
    LLMTerminationReason,
)
from epistemic_repair.evaluation.er1_llm_smoke import run_er1_llm_smoke
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient, LLMRequest, LLMResponse
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.mock import DeterministicMockLLMClient
from epistemic_repair.llm.schemas import (
    DecisionType,
    ER1FullAutonomousDecision,
    LLMCondition,
    StructuredResponseError,
    er1_full_autonomous_json_schema,
    er1_planner_only_json_schema,
    parse_er1_full_autonomous_response,
    parse_er1_planner_only_response,
)
from epistemic_repair.policies.llm import (
    ER1FullAutonomousLLMPolicy,
    ER1PlannerOnlyLLMPolicy,
)
from epistemic_repair.policies.stochastic_views import (
    StochasticAgentExperimentRecord,
    StochasticBenchmarkAgentView,
)
from epistemic_repair.prompts.binary_er1 import (
    BINARY_ER1_PROMPT_VERSION,
    build_er1_full_autonomous_prompt,
    build_er1_planner_only_prompt,
)
from epistemic_repair.prompts.binary_v0 import (
    BINARY_V0_PROMPT_VERSION,
    build_full_autonomous_prompt,
)
from epistemic_repair.policies.views import BenchmarkAgentView
from scripts.demo_llm_agent import parse_args, selected_failure_modes


class QueueClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("fake response queue exhausted")
        return LLMResponse(
            self.responses.pop(0),
            provider_request_id=f"er1-fake-{len(self.requests)}",
        )


def _beliefs(label: str | None = None) -> dict[str, float]:
    if label is None:
        return {hypothesis.value: 0.25 for hypothesis in ER1_HYPOTHESES}
    return {
        hypothesis.value: 1.0 if hypothesis.value == label else 0.0
        for hypothesis in ER1_HYPOTHESES
    }


def _full_run(action: str) -> str:
    return json.dumps(
        {
            "decision": "RUN_EXPERIMENT",
            "action": action,
            "diagnosis": None,
            "beliefs": _beliefs(),
            "confidence": None,
            "reason_summary": "Collect another noisy observation.",
        }
    )


def _full_diagnose(label: str) -> str:
    return json.dumps(
        {
            "decision": "DIAGNOSE",
            "action": None,
            "diagnosis": label,
            "beliefs": _beliefs(label),
            "confidence": 1.0,
            "reason_summary": "The accumulated evidence supports this diagnosis.",
        }
    )


def _planner_diagnose(label: str) -> str:
    return json.dumps(
        {
            "decision": "DIAGNOSE",
            "action": None,
            "diagnosis": label,
            "reason_summary": "Use the supplied posterior.",
        }
    )


def _initial_view() -> StochasticBenchmarkAgentView:
    return StochasticBenchmarkAgentView(
        initial_history=(Observation(1, 0),),
        experiment_history=(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=5,
    )


def test_er1_schemas_contain_four_hypotheses_and_safe_actions() -> None:
    full = er1_full_autonomous_json_schema()
    planner = er1_planner_only_json_schema()
    diagnosis_enum = full["properties"]["diagnosis"]["enum"]
    assert set(diagnosis_enum) == {
        None,
        *(hypothesis.value for hypothesis in ER1_HYPOTHESES),
    }
    assert "NO_STRUCTURAL_CHANGE" in diagnosis_enum
    assert "INSPECT_LATENT_VARIABLE" not in str(full)
    assert "INSPECT_LATENT_VARIABLE" not in str(planner)
    assert full["required"] == [
        "decision",
        "action",
        "diagnosis",
        "beliefs",
        "confidence",
        "reason_summary",
    ]


def test_er1_autonomous_and_planner_decisions_parse_strictly() -> None:
    run = parse_er1_full_autonomous_response(
        _full_run("USE_TRUSTED_SENSOR")
    )
    diagnosis = parse_er1_planner_only_response(
        _planner_diagnose("NO_STRUCTURAL_CHANGE")
    )
    assert run.decision is DecisionType.RUN_EXPERIMENT
    assert run.action is DiagnosticAction.USE_TRUSTED_SENSOR
    assert diagnosis.diagnosis is FailureMode.NO_STRUCTURAL_CHANGE


@pytest.mark.parametrize(
    "payload",
    (
        {
            "decision": "DIAGNOSE",
            "action": None,
            "diagnosis": "NORMAL",
            "beliefs": _beliefs(),
            "confidence": 0.5,
            "reason_summary": "Invalid ER-0 label.",
        },
        {
            "decision": "RUN_EXPERIMENT",
            "action": "INSPECT_LATENT_VARIABLE",
            "diagnosis": None,
            "beliefs": _beliefs(),
            "confidence": None,
            "reason_summary": "Invalid action.",
        },
    ),
)
def test_er1_parser_rejects_nonbenchmark_labels(payload: dict[str, object]) -> None:
    with pytest.raises(StructuredResponseError):
        parse_er1_full_autonomous_response(json.dumps(payload))


def test_er1_autonomous_prompt_is_qualitative_and_has_no_privileged_values() -> None:
    prompt = build_er1_full_autonomous_prompt(_initial_view())
    for hypothesis in ER1_HYPOTHESES:
        assert hypothesis.value in prompt
    for forbidden in (
        "0.90",
        "0.95",
        "0.99",
        "StochasticLikelihoodModel",
        "expected information gain",
        "oracle action",
        "get_ground_truth",
        "hidden Y",
    ):
        assert forbidden not in prompt
    assert "NO_STRUCTURAL_CHANGE" in prompt
    assert BINARY_ER1_PROMPT_VERSION in prompt


def test_er1_planner_prompt_contains_beliefs_but_no_likelihood_table() -> None:
    beliefs = StochasticLikelihoodModel().conditioned_initial_beliefs()
    prompt = build_er1_planner_only_prompt(_initial_view(), beliefs)
    for hypothesis in ER1_HYPOTHESES:
        assert (
            f"{hypothesis.value}: {beliefs.probability(hypothesis):.6f}"
            in prompt
        )
    assert "likelihood" not in prompt.lower()
    assert "oracle action" not in prompt.lower()


def test_er1_trusted_history_formats_t_without_exposing_y() -> None:
    view = StochasticBenchmarkAgentView(
        initial_history=(Observation(1, 0),),
        experiment_history=(
            StochasticAgentExperimentRecord(
                step_number=1,
                action=DiagnosticAction.USE_TRUSTED_SENSOR,
                result=StochasticTrustedSensorResult(x=1, trusted_t=1),
                context_before=Context.B,
                context_after=Context.B,
            ),
        ),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=4,
    )
    prompt = build_er1_full_autonomous_prompt(view)
    assert "trusted T=1" in prompt
    assert "trusted Y=" not in prompt


def test_er0_prompt_version_and_hypothesis_set_remain_unchanged() -> None:
    view = BenchmarkAgentView(
        initial_history=(Observation(1, 0),),
        experiment_history=(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=2,
    )
    prompt = build_full_autonomous_prompt(view)
    assert BINARY_V0_PROMPT_VERSION == "binary_v0_001"
    assert "deterministic binary machine" in prompt
    assert "NO_STRUCTURAL_CHANGE" not in prompt


def test_er1_policy_accepts_only_er1_agent_view() -> None:
    policy = ER1FullAutonomousLLMPolicy(
        QueueClient([_full_diagnose("NO_STRUCTURAL_CHANGE")]),
        LLMConfig(max_retries=0),
    )
    er0_view = BenchmarkAgentView(
        initial_history=(Observation(1, 0),),
        experiment_history=(),
        current_context=Context.B,
        available_actions=BENCHMARK_ACTIONS,
        steps_remaining=2,
    )
    with pytest.raises(TypeError, match="StochasticBenchmarkAgentView"):
        policy.decide(er0_view)  # type: ignore[arg-type]


def test_er1_full_runner_multistep_trace_and_metadata() -> None:
    client = QueueClient(
        [
            _full_run("USE_TRUSTED_SENSOR"),
            _full_diagnose("NO_STRUCTURAL_CHANGE"),
        ]
    )
    result = ER1LLMEpisodeRunner(
        condition=LLMCondition.FULL_AUTONOMOUS,
        experiment_budget=5,
        episode_seed=13,
        timestamp_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    ).run(
        StochasticBinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        ER1FullAutonomousLLMPolicy(client, LLMConfig(max_retries=0)),
    )
    assert result.experiments_used == 1
    assert result.final_diagnosis is FailureMode.NO_STRUCTURAL_CHANGE
    assert result.termination_reason is LLMTerminationReason.DIAGNOSED
    assert result.run_metadata.benchmark_version == "binary_er1"
    assert result.run_metadata.prompt_version == BINARY_ER1_PROMPT_VERSION
    assert result.evaluation_metadata.episode_seed == 13
    assert result.trace[0].agent_view.initial_history == (Observation(1, 0),)
    assert all("NO_STRUCTURAL_CHANGE" in request.prompt for request in client.requests)
    assert all("hidden_failure_mode" not in request.prompt for request in client.requests)


def test_er1_planner_receives_normative_beliefs_and_can_diagnose_no_change() -> None:
    client = QueueClient([_planner_diagnose("NO_STRUCTURAL_CHANGE")])
    result = ER1LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        experiment_budget=5,
        episode_seed=3,
    ).run(
        StochasticBinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        ER1PlannerOnlyLLMPolicy(client, LLMConfig(max_retries=0)),
    )
    assert result.final_diagnosis is FailureMode.NO_STRUCTURAL_CHANGE
    assert "Authoritative current probabilities:" in client.requests[0].prompt
    assert result.premature_diagnosis


def test_er1_mock_smoke_is_reproducible_and_keeps_conditions_separate() -> None:
    config = LLMConfig(max_retries=0, max_decision_calls=5)
    first = run_er1_llm_smoke(
        DeterministicMockLLMClient(),
        config,
        repetitions=1,
        experiment_budget=5,
        base_episode_seed=20,
    )
    second = run_er1_llm_smoke(
        DeterministicMockLLMClient(),
        config,
        repetitions=1,
        experiment_budget=5,
        base_episode_seed=20,
    )
    assert [result.condition for result in first] == [
        LLMCondition.FULL_AUTONOMOUS,
        LLMCondition.PLANNER_ONLY,
    ]
    assert [
        (episode.final_diagnosis, episode.experiments_used)
        for result in first
        for episode in result.episodes
    ] == [
        (episode.final_diagnosis, episode.experiments_used)
        for result in second
        for episode in result.episodes
    ]


def test_er1_llm_metrics_measure_false_missed_and_premature_diagnoses() -> None:
    config = LLMConfig(max_retries=0)
    false_result = ER1LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        episode_seed=1,
    ).run(
        StochasticBinaryMachine(),
        FailureMode.NO_STRUCTURAL_CHANGE,
        ER1PlannerOnlyLLMPolicy(
            QueueClient([_planner_diagnose("WORLD_SHIFT")]), config
        ),
    )
    missed_result = ER1LLMEpisodeRunner(
        condition=LLMCondition.PLANNER_ONLY,
        episode_seed=2,
    ).run(
        StochasticBinaryMachine(),
        FailureMode.WORLD_SHIFT,
        ER1PlannerOnlyLLMPolicy(
            QueueClient([_planner_diagnose("NO_STRUCTURAL_CHANGE")]), config
        ),
    )
    summary = summarize_er1_llm_results([false_result, missed_result])
    assert summary.false_structural_diagnosis_rate == 1.0
    assert summary.missed_structural_failure_rate == 1.0
    assert summary.premature_diagnosis_rate == 1.0


def test_cli_defaults_to_er0_and_selects_all_four_er1_modes() -> None:
    assert parse_args([]).benchmark == "er0"
    assert selected_failure_modes("all", "er1") == ER1_HYPOTHESES
    assert selected_failure_modes("no_structural_change", "er1") == (
        FailureMode.NO_STRUCTURAL_CHANGE,
    )
    with pytest.raises(ValueError):
        selected_failure_modes("no_structural_change", "er0")
