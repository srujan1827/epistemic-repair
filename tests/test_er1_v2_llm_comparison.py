"""No-network tests for the ER-1 V2 matched comparison harness."""

from __future__ import annotations

import csv
import json

import pytest

import epistemic_repair.evaluation.er1_v2_llm_comparison as comparison_module

from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.llm_policy import ER1V2FullAutonomousLLMPolicy
from epistemic_repair.er1_v2.llm_runner import ER1V2LLMEpisodeRunner
from epistemic_repair.evaluation.er1_v2_llm_comparison import (
    ComparisonCell,
    ComparisonConfig,
    PacedLLMClient,
    comparison_cells,
    confusion_matrix,
    episode_row,
    failure_record,
    format_progress,
    paired_comparison,
    parse_gemini_retry_delay_seconds,
    run_er1_v2_llm_comparison,
    sanitized_episode_trace,
    summarize_rows,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import (
    LLMClient,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.mock import DeterministicMockLLMClient
from epistemic_repair.llm.schemas import LLMCondition
from scripts.run_er1_v2_llm_comparison import parse_args, parse_seeds


def _base_row(
    condition: LLMCondition,
    hypothesis: FailureMode,
    seed: int,
    *,
    correct: bool = False,
) -> dict[str, object]:
    return {
        "condition": condition.value,
        "true_hypothesis": hypothesis.value,
        "seed": seed,
        "final_diagnosis": hypothesis.value if correct else None,
        "diagnosis_correct": correct,
        "diagnosed_correctly_within_budget": correct,
        "threshold_qualified_success": correct,
        "premature_diagnosis": not correct,
        "experiments_used": 2,
        "decision_calls": 3,
        "total_retries": 1,
        "cumulative_action_regret": 0.4,
        "oracle_action_agreements": 1,
        "oracle_action_agreement_rate": 0.5,
        "mean_autonomous_belief_l1_error": (
            0.2 if condition is LLMCondition.FULL_AUTONOMOUS else None
        ),
        "provider_failure_flag": False,
    }


def _real_episode(cell: ComparisonCell, client: LLMClient | None = None):
    config = LLMConfig(max_retries=0, max_decision_calls=3)
    model_client = client or DeterministicMockLLMClient()
    policy = ER1V2FullAutonomousLLMPolicy(model_client, config)
    runner = ER1V2LLMEpisodeRunner(
        condition=LLMCondition.FULL_AUTONOMOUS,
        experiment_budget=2,
        episode_seed=cell.seed,
    )
    return runner.run(ER1V2BinaryMachine(), cell.hypothesis, policy)


def test_default_grid_has_matched_seeds_and_40_cells_per_condition() -> None:
    cells = comparison_cells(ComparisonConfig())

    assert len(cells) == 80
    for condition in (LLMCondition.FULL_AUTONOMOUS, LLMCondition.PLANNER_ONLY):
        selected = [cell for cell in cells if cell.condition is condition]
        assert len(selected) == 40
        assert {
            (cell.hypothesis, cell.seed) for cell in selected
        } == {
            (hypothesis, seed)
            for hypothesis in ER1_HYPOTHESES
            for seed in range(10)
        }
    assert cells[0].key[1:] == cells[1].key[1:]


def test_seed_parser_and_cli_defaults() -> None:
    assert parse_seeds(("0..9",)) == tuple(range(10))
    assert parse_seeds(("0,2", "4")) == (0, 2, 4)
    args = parse_args([])
    assert args.model == "gemini-3.6-flash"
    assert args.budget == 8
    assert args.diagnosis_threshold == 0.95
    assert args.min_request_interval_seconds == 3.5
    assert args.rate_limit_backoff_seconds == 10.0
    assert args.episode_cooldown_seconds == 0.0
    assert args.resume_provider_failures is True
    assert tuple(args.conditions) == ("full", "planner")
    assert parse_args(
        ["--conditions", "threshold_aware", "--seeds", "0..1"]
    ).conditions == ["threshold_aware"]
    assert parse_args(
        ["--episode-cooldown-seconds", "90"]
    ).episode_cooldown_seconds == 90.0


def test_resume_skips_every_completed_cell(tmp_path) -> None:
    config = ComparisonConfig(
        seeds=(0,), conditions=(LLMCondition.FULL_AUTONOMOUS,)
    )
    first_calls: list[ComparisonCell] = []

    def first_executor(cell: ComparisonCell):
        first_calls.append(cell)
        return _real_episode(cell)

    run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(), config, tmp_path, episode_executor=first_executor,
        progress=lambda _: None,
    )
    config_path = tmp_path / "run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config.pop("min_request_interval_seconds")
    old_config.pop("rate_limit_backoff_seconds")
    old_config.pop("episode_cooldown_seconds")
    config_path.write_text(json.dumps(old_config), encoding="utf-8")
    second_calls: list[ComparisonCell] = []
    rows = run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(), config, tmp_path,
        episode_executor=lambda cell: second_calls.append(cell),
        progress=lambda _: None,
    )

    assert len(first_calls) == 4
    assert second_calls == []
    assert len(rows) == 4
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["min_request_interval_seconds"] == 3.5
    assert migrated["rate_limit_backoff_seconds"] == 10.0
    assert migrated["episode_cooldown_seconds"] == 0.0


def test_pairing_uses_hypothesis_and_seed_and_computes_deltas() -> None:
    full = _base_row(LLMCondition.FULL_AUTONOMOUS, FailureMode.WORLD_SHIFT, 7)
    planner = _base_row(
        LLMCondition.PLANNER_ONLY, FailureMode.WORLD_SHIFT, 7, correct=True
    )
    planner["experiments_used"] = 1
    planner["cumulative_action_regret"] = 0.1

    pairs = paired_comparison([planner, full])

    assert len(pairs) == 1
    assert pairs[0]["pair_key"] == "WORLD_SHIFT:7"
    assert pairs[0]["outcome_category"] == "PLANNER_CORRECT_FULL_WRONG"
    assert pairs[0]["delta_experiments_planner_minus_full"] == -1
    assert pairs[0]["delta_regret_planner_minus_full"] == pytest.approx(-0.3)


def test_aggregate_math_and_null_planner_belief_error() -> None:
    rows = [
        _base_row(LLMCondition.FULL_AUTONOMOUS, FailureMode.WORLD_SHIFT, 0, correct=True),
        _base_row(LLMCondition.FULL_AUTONOMOUS, FailureMode.WORLD_SHIFT, 1),
        _base_row(LLMCondition.PLANNER_ONLY, FailureMode.WORLD_SHIFT, 0, correct=True),
    ]

    summaries = {row["condition"]: row for row in summarize_rows(rows)}

    assert summaries[LLMCondition.FULL_AUTONOMOUS.value]["diagnosis_accuracy"] == 0.5
    assert summaries[LLMCondition.FULL_AUTONOMOUS.value]["mean_decision_calls"] == 3.0
    assert summaries[LLMCondition.FULL_AUTONOMOUS.value]["oracle_action_agreement_rate"] == 0.5
    assert summaries[LLMCondition.PLANNER_ONLY.value]["mean_autonomous_belief_l1_error"] is None


def test_confusion_accounts_for_all_diagnoses_and_no_diagnosis() -> None:
    correct = _base_row(
        LLMCondition.FULL_AUTONOMOUS, FailureMode.SENSOR_CORRUPTION, 0, correct=True
    )
    missing = _base_row(
        LLMCondition.FULL_AUTONOMOUS, FailureMode.SENSOR_CORRUPTION, 1
    )
    matrix = confusion_matrix([correct, missing], LLMCondition.FULL_AUTONOMOUS)

    counts = matrix["counts"][FailureMode.SENSOR_CORRUPTION.value]
    assert counts[FailureMode.SENSOR_CORRUPTION.value] == 1
    assert counts["NO_DIAGNOSIS"] == 1
    assert sum(counts.values()) == 2


def test_progress_contains_cell_timing_retries_and_failures() -> None:
    cell = ComparisonCell(
        LLMCondition.PLANNER_ONLY, FailureMode.MISSING_LATENT_VARIABLE, 3
    )
    message = format_progress(
        4, 80, cell, 38.24,
        {
            "total_retries": 1,
            "provider_failure_flag": True,
            "provider_rate_limit_failure_flag": True,
            "scientific_model_failure_flag": False,
        },
    )
    assert "[4/80] PLANNER_ONLY MISSING_LATENT_VARIABLE seed=3" in message
    assert "38.2s" in message
    assert "retries=1 provider_failure=True rate_limit_failure=True" in message
    assert "scientific_model_failure=False" in message


def test_provider_failure_is_checkpointed_and_later_cells_continue(tmp_path) -> None:
    config = ComparisonConfig(
        seeds=(0,), conditions=(LLMCondition.FULL_AUTONOMOUS,)
    )
    calls = 0

    def executor(cell: ComparisonCell):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMTimeoutError("mock timeout")
        return _real_episode(cell)

    rows = run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(), config, tmp_path,
        episode_executor=executor, progress=lambda _: None,
    )

    assert calls == 4
    assert len(rows) == 4
    assert rows[0]["provider_failure_flag"] is True
    assert rows[0]["termination_reason"] == "MODEL_FAILURE"
    assert rows[1]["provider_failure_flag"] is False

    resume_calls: list[ComparisonCell] = []

    def resume_executor(cell: ComparisonCell):
        resume_calls.append(cell)
        return _real_episode(cell)

    resumed = run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(), config, tmp_path,
        episode_executor=resume_executor, progress=lambda _: None,
    )
    assert [cell.key for cell in resume_calls] == [
        (
            LLMCondition.FULL_AUTONOMOUS.value,
            FailureMode.NO_STRUCTURAL_CHANGE.value,
            0,
        )
    ]
    assert all(row["provider_failure_flag"] is False for row in resumed)
    with (tmp_path / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert len((tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_csv_order_is_deterministic_even_when_checkpoints_complete_out_of_order(tmp_path) -> None:
    config = ComparisonConfig(
        seeds=(1, 0), conditions=(LLMCondition.FULL_AUTONOMOUS,)
    )
    run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(), config, tmp_path,
        episode_executor=_real_episode, progress=lambda _: None,
    )
    with (tmp_path / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    keys = [(row["true_hypothesis"], int(row["seed"])) for row in rows]
    assert keys[:4] == [
        (FailureMode.NO_STRUCTURAL_CHANGE.value, 0),
        (FailureMode.NO_STRUCTURAL_CHANGE.value, 1),
        (FailureMode.WORLD_SHIFT.value, 0),
        (FailureMode.WORLD_SHIFT.value, 1),
    ]


class _SecretReasonClient(LLMClient):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text=json.dumps({
            "decision": "DIAGNOSE",
            "action": None,
            "diagnosis": "NO_STRUCTURAL_CHANGE",
            "beliefs": {
                "NO_STRUCTURAL_CHANGE": 1.0,
                "WORLD_SHIFT": 0.0,
                "SENSOR_CORRUPTION": 0.0,
                "MISSING_LATENT_VARIABLE": 0.0,
            },
            "confidence": 1.0,
            "reason_summary": f"safe rationale {self.secret}",
        }))


def test_trace_sanitizes_reason_and_never_contains_prompt_or_raw_output(monkeypatch) -> None:
    secret = "unit-test-secret-value"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    cell = ComparisonCell(
        LLMCondition.FULL_AUTONOMOUS, FailureMode.NO_STRUCTURAL_CHANGE, 0
    )
    episode = _real_episode(cell, _SecretReasonClient(secret))

    trace = sanitized_episode_trace(episode)
    serialized = json.dumps(trace)

    assert secret not in serialized
    assert "[REDACTED]" in serialized
    assert "prompt" not in serialized
    assert "raw_output" not in serialized


def test_failure_record_sanitizes_exception_text(monkeypatch) -> None:
    secret = "exception-secret"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    cell = ComparisonCell(
        LLMCondition.FULL_AUTONOMOUS, FailureMode.WORLD_SHIFT, 0
    )
    row, trace = failure_record(cell, ComparisonConfig(), RuntimeError(secret))
    assert secret not in json.dumps({"row": row, "trace": trace})
    assert row["provider_error_message"] == "[REDACTED]"


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _RecordingClient(LLMClient):
    def __init__(self, fake_time: _FakeTime) -> None:
        self.fake_time = fake_time
        self.starts: list[float] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.starts.append(self.fake_time.clock())
        return LLMResponse(text="{}")


def test_global_pacing_uses_monotonic_start_times() -> None:
    fake_time = _FakeTime()
    underlying = _RecordingClient(fake_time)
    events: list[str] = []
    client = PacedLLMClient(
        underlying,
        min_request_interval_seconds=3.5,
        rate_limit_backoff_seconds=10.0,
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
        event_writer=events.append,
    )
    request = LLMRequest(prompt="safe", response_schema={})

    client.generate(request)
    client.generate(request)
    client.generate(request)

    assert underlying.starts == [0.0, 3.5, 7.0]
    assert fake_time.sleeps == [3.5, 3.5]
    assert events == [
        "waiting 3.5s for provider pacing",
        "waiting 3.5s for provider pacing",
    ]


class _RateLimitOnceClient(LLMClient):
    def __init__(self, fake_time: _FakeTime, message: str) -> None:
        self.fake_time = fake_time
        self.message = message
        self.starts: list[float] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.starts.append(self.fake_time.clock())
        if len(self.starts) == 1:
            raise LLMRateLimitError(self.message)
        return LLMResponse(text="{}")


def test_gemini_retry_hint_parser_and_safety_margin() -> None:
    message = "Quota exceeded. Please retry in 9.30403225s."
    assert parse_gemini_retry_delay_seconds(message) == pytest.approx(9.30403225)
    assert parse_gemini_retry_delay_seconds("retry later") is None

    fake_time = _FakeTime()
    underlying = _RateLimitOnceClient(fake_time, message)
    events: list[str] = []
    client = PacedLLMClient(
        underlying,
        min_request_interval_seconds=3.5,
        rate_limit_backoff_seconds=10.0,
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
        event_writer=events.append,
    )
    request = LLMRequest(prompt="safe", response_schema={})

    with pytest.raises(LLMRateLimitError):
        client.generate(request)
    client.generate(request)

    assert underlying.starts[1] == pytest.approx(9.80403225)
    assert fake_time.sleeps == pytest.approx([9.80403225])
    assert events == ["rate limited; waiting 9.8s before retry"]


def test_rate_limit_without_hint_uses_fallback_backoff() -> None:
    fake_time = _FakeTime()
    underlying = _RateLimitOnceClient(fake_time, "HTTP 429 without duration")
    client = PacedLLMClient(
        underlying,
        min_request_interval_seconds=3.5,
        rate_limit_backoff_seconds=10.0,
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
        event_writer=lambda _: None,
    )
    request = LLMRequest(prompt="safe", response_schema={})

    with pytest.raises(LLMRateLimitError):
        client.generate(request)
    client.generate(request)

    assert fake_time.sleeps == [10.0]


class _AlwaysRateLimitedClient(LLMClient):
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMRateLimitError("Please retry in 1s")


def test_429_is_provider_failure_not_scientific_model_failure() -> None:
    cell = ComparisonCell(
        LLMCondition.FULL_AUTONOMOUS, FailureMode.NO_STRUCTURAL_CHANGE, 0
    )
    result = _real_episode(cell, _AlwaysRateLimitedClient())
    row = episode_row(result)

    assert row["provider_failure_flag"] is True
    assert row["provider_transport_failure_flag"] is True
    assert row["provider_rate_limit_failure_flag"] is True
    assert row["scientific_model_failure_flag"] is False
    assert row["model_failure_flag"] is True  # historical termination label retained


class _MalformedClient(LLMClient):
    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="not-json")


@pytest.mark.parametrize(
    ("first_outcome", "expected_flag"),
    (
        ("success", None),
        ("provider_failure", "provider_failure_flag"),
        ("scientific_failure", "scientific_model_failure_flag"),
    ),
)
def test_episode_cooldown_after_every_processed_outcome(
    tmp_path,
    first_outcome: str,
    expected_flag: str | None,
) -> None:
    config = ComparisonConfig(
        seeds=(0,),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        min_request_interval_seconds=0.0,
        episode_cooldown_seconds=90.0,
    )
    fake_time = _FakeTime()
    starts: list[float] = []
    calls = 0

    def executor(cell: ComparisonCell):
        nonlocal calls
        starts.append(fake_time.clock())
        calls += 1
        if calls == 1 and first_outcome == "provider_failure":
            raise LLMTimeoutError("mock provider timeout")
        if calls == 1 and first_outcome == "scientific_failure":
            return _real_episode(cell, _MalformedClient())
        return _real_episode(cell)

    events: list[str] = []
    rows = run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(),
        config,
        tmp_path,
        episode_executor=executor,
        progress=events.append,
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
    )

    assert starts == [0.0, 90.0, 180.0, 270.0]
    assert fake_time.sleeps == [90.0, 90.0, 90.0]
    assert events.count("cooldown: waiting 90.0s before next episode") == 3
    if expected_flag is not None:
        assert rows[0][expected_flag] is True


def test_skipped_cells_do_not_cool_down_and_resume_is_unchanged(tmp_path) -> None:
    config = ComparisonConfig(
        seeds=(0,),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        min_request_interval_seconds=0.0,
        episode_cooldown_seconds=90.0,
    )
    first_time = _FakeTime()
    run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(),
        config,
        tmp_path,
        episode_executor=_real_episode,
        progress=lambda _: None,
        clock=first_time.clock,
        sleeper=first_time.sleep,
    )
    resume_time = _FakeTime()
    resume_calls: list[ComparisonCell] = []

    run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(),
        config,
        tmp_path,
        episode_executor=lambda cell: resume_calls.append(cell),
        progress=lambda _: None,
        clock=resume_time.clock,
        sleeper=resume_time.sleep,
    )

    assert resume_calls == []
    assert resume_time.sleeps == []


def test_artifacts_are_rebuilt_after_each_completed_cell(
    tmp_path,
    monkeypatch,
) -> None:
    config = ComparisonConfig(
        seeds=(0,),
        conditions=(LLMCondition.FULL_AUTONOMOUS,),
        min_request_interval_seconds=0.0,
    )
    observed_counts: list[int] = []
    real_writer = comparison_module._write_current_artifacts

    def observing_writer(destination, records, run_config, **kwargs):
        observed_counts.append(len(records))
        return real_writer(destination, records, run_config, **kwargs)

    monkeypatch.setattr(
        comparison_module,
        "_write_current_artifacts",
        observing_writer,
    )
    run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(),
        config,
        tmp_path,
        episode_executor=_real_episode,
        progress=lambda _: None,
    )

    assert observed_counts[:4] == [1, 2, 3, 4]
    assert observed_counts[-1] == 4
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Run progress" in report
    assert "Completed valid cells: **4**" in report


def test_threshold_aware_condition_runs_and_is_summarized_without_network(
    tmp_path,
) -> None:
    config = ComparisonConfig(
        seeds=(0,),
        conditions=(LLMCondition.THRESHOLD_AWARE_AUTONOMOUS,),
        experiment_budget=2,
        max_decision_calls=3,
        min_request_interval_seconds=0.0,
    )

    rows = run_er1_v2_llm_comparison(
        DeterministicMockLLMClient(),
        config,
        tmp_path,
        progress=lambda _: None,
    )

    assert len(rows) == 4
    assert {
        row["condition"] for row in rows
    } == {LLMCondition.THRESHOLD_AWARE_AUTONOMOUS.value}
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 1
    assert summaries[0]["condition"] == "THRESHOLD_AWARE_AUTONOMOUS"
    confusion = json.loads(
        (tmp_path / "confusion_threshold_aware.json").read_text(
            encoding="utf-8"
        )
    )
    assert confusion["condition"] == "THRESHOLD_AWARE_AUTONOMOUS"
    run_config = json.loads(
        (tmp_path / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["conditions"] == ["THRESHOLD_AWARE_AUTONOMOUS"]
    assert run_config["prompt_version"] == "binary_er1_v2_threshold_aware_001"
