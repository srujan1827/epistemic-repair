"""Run a bounded live or mocked LLM smoke experiment."""

import argparse
from collections.abc import Sequence

from epistemic_repair import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_ID,
    DeterministicMockLLMClient,
    FailureMode,
    HYPOTHESES,
    LLMCondition,
    LLMConfig,
    LLMConfigurationError,
    LLMEpisodeResult,
    create_llm_client,
    run_llm_smoke,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.evaluation.er1_llm_runner import ER1LLMEpisodeResult
from epistemic_repair.evaluation.er1_llm_smoke import run_er1_llm_smoke
from epistemic_repair.er1_v2.llm_runner import ER1V2LLMEpisodeResult
from epistemic_repair.er1_v2.llm_smoke import run_er1_v2_llm_smoke
from epistemic_repair.llm.sanitize import sanitize_text


FAILURE_SELECTIONS: dict[str, tuple[FailureMode, ...]] = {
    "world_shift": (FailureMode.WORLD_SHIFT,),
    "sensor_corruption": (FailureMode.SENSOR_CORRUPTION,),
    "missing_latent_variable": (FailureMode.MISSING_LATENT_VARIABLE,),
    "all": HYPOTHESES,
}
ER1_FAILURE_SELECTIONS: dict[str, tuple[FailureMode, ...]] = {
    "no_structural_change": (FailureMode.NO_STRUCTURAL_CHANGE,),
    "world_shift": (FailureMode.WORLD_SHIFT,),
    "sensor_corruption": (FailureMode.SENSOR_CORRUPTION,),
    "missing_latent_variable": (FailureMode.MISSING_LATENT_VARIABLE,),
    "all": ER1_HYPOTHESES,
}


def selected_failure_modes(
    selection: str,
    benchmark: str = "er0",
) -> tuple[FailureMode, ...]:
    """Return the failure modes selected by a CLI value."""
    selections = (
        ER1_FAILURE_SELECTIONS
        if benchmark in ("er1", "er1_v1", "er1_v2")
        else FAILURE_SELECTIONS
    )
    if selection not in selections:
        raise ValueError(f"{selection} is not available for {benchmark}")
    return selections[selection]


def _diagnosis_threshold(value: str) -> float:
    """Parse a posterior threshold constrained to the interval (0, 1]."""
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "diagnosis threshold must be a number"
        ) from error
    if not 0.0 < threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "diagnosis threshold must be in (0, 1]"
        )
    return threshold


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse provider, model, condition, and call-control configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("er0", "er1", "er1_v1", "er1_v2"),
        default="er0",
        help="Benchmark version; er1 remains an alias for historical er1_v1.",
    )
    parser.add_argument("--provider", default="gemini", choices=("gemini",))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--condition", choices=("full", "planner", "both"), default="both")
    parser.add_argument(
        "--failure",
        choices=(
            "world_shift",
            "sensor_corruption",
            "missing_latent_variable",
            "no_structural_change",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument(
        "--diagnosis-threshold",
        type=_diagnosis_threshold,
        default=0.90,
        help=(
            "ER-1 V2 normative diagnosis threshold; historical benchmarks "
            "retain their existing behavior."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--thinking-level", default="medium")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-decision-calls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verbose-attempts",
        action="store_true",
        help="Print sanitized LLM attempt details for successful episodes too.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic no-network client and require no API key.",
    )
    return parser.parse_args(argv)


def print_attempt_history(
    episode: LLMEpisodeResult | ER1LLMEpisodeResult | ER1V2LLMEpisodeResult,
    *,
    verbose: bool = False,
) -> None:
    """Print stored sanitized attempts for failures, or all episodes if verbose."""
    if episode.final_diagnosis is not None and not verbose:
        return

    print(f"    termination_reason={episode.termination_reason.value}")
    for turn in episode.trace:
        for attempt in turn.policy_result.attempts:
            error_type = attempt.error_type or "NONE"
            error_message = (
                repr(sanitize_text(attempt.error_message))
                if attempt.error_message is not None
                else "NONE"
            )
            print(
                f"    call_number={turn.call_number} "
                f"attempt_number={attempt.attempt_number} "
                f"status={attempt.status.value}"
            )
            print(f"      error_type={error_type}")
            print(f"      error_message={error_message}")
            if attempt.raw_output is not None:
                print(f"      raw_output={sanitize_text(attempt.raw_output)!r}")
            if attempt.provider_request_id is not None:
                print(
                    "      provider_request_id="
                    f"{sanitize_text(attempt.provider_request_id)}"
                )


def main() -> None:
    """Run the requested small smoke experiment and print concise traces."""
    args = parse_args()
    config = LLMConfig(
        provider=args.provider,
        model_id=args.model,
        thinking_level=args.thinking_level,
        max_output_tokens=args.max_output_tokens,
        request_timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        max_decision_calls=args.max_decision_calls,
    )
    conditions = {
        "full": (LLMCondition.FULL_AUTONOMOUS,),
        "planner": (LLMCondition.PLANNER_ONLY,),
        "both": (
            LLMCondition.FULL_AUTONOMOUS,
            LLMCondition.PLANNER_ONLY,
        ),
    }[args.condition]
    try:
        failure_modes = selected_failure_modes(args.failure, args.benchmark)
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    episode_count = len(failure_modes) * args.repetitions * len(conditions)

    print(f"provider: {config.provider}{' (mocked transport)' if args.mock else ''}")
    print(f"model: {config.model_id}")
    print(f"benchmark: {args.benchmark}")
    print(f"condition: {args.condition}")
    print(f"failure: {args.failure}")
    print(f"budget: {args.budget}")
    if args.benchmark == "er1_v2":
        print(f"diagnosis threshold: {args.diagnosis_threshold}")
    print(f"number of episodes: {episode_count}")

    try:
        client = DeterministicMockLLMClient() if args.mock else create_llm_client(config)
    except LLMConfigurationError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    if args.benchmark in ("er1", "er1_v1"):
        results = run_er1_llm_smoke(
            client,
            config,
            conditions=conditions,
            repetitions=args.repetitions,
            experiment_budget=args.budget,
            diagnosis_threshold=args.diagnosis_threshold,
            base_episode_seed=args.seed,
            failure_modes=failure_modes,
        )
    elif args.benchmark == "er1_v2":
        results = run_er1_v2_llm_smoke(
            client,
            config,
            conditions=conditions,
            repetitions=args.repetitions,
            experiment_budget=args.budget,
            base_episode_seed=args.seed,
            failure_modes=failure_modes,
        )
    else:
        results = run_llm_smoke(
            client,
            config,
            conditions=conditions,
            repetitions=args.repetitions,
            experiment_budget=args.budget,
            base_episode_seed=args.seed,
            failure_modes=failure_modes,
        )
    for condition_result in results:
        print(f"\n{condition_result.condition.value}")
        for episode in condition_result.episodes:
            actions = [
                turn.experiment_record.action.value
                for turn in episode.trace
                if turn.experiment_record is not None
            ]
            truth = episode.evaluation_metadata.hidden_failure_mode.value
            diagnosis = (
                episode.final_diagnosis.value
                if episode.final_diagnosis is not None
                else "NONE"
            )
            if args.benchmark == "er1_v2":
                assert isinstance(episode, ER1V2LLMEpisodeResult)
                print(
                    f"  seed={episode.run_metadata.episode_seed} "
                    f"truth={truth} actions={actions} diagnosis={diagnosis} "
                    f"diagnosis_correct={episode.diagnosis_correct} "
                    f"within_budget={episode.diagnosed_correctly_within_budget} "
                    f"threshold_success={episode.threshold_qualified_success} "
                    f"premature={episode.premature_diagnosis} "
                    "normative_diagnosis_probability="
                    f"{episode.normative_probability_of_final_diagnosis} "
                    f"experiments={episode.experiments_used} "
                    f"regret={episode.cumulative_action_regret:.6f} "
                    f"retries={episode.total_retries}"
                )
            else:
                print(
                    f"  seed={episode.run_metadata.episode_seed} "
                    f"truth={truth} actions={actions} diagnosis={diagnosis} "
                    f"success={episode.success_within_budget} "
                    f"experiments={episode.experiments_used} "
                    f"regret={episode.cumulative_action_regret:.6f} "
                    f"retries={episode.total_retries}"
                )
            print_attempt_history(episode, verbose=args.verbose_attempts)
        summary = condition_result.summary
        if args.benchmark == "er1_v2":
            print(
                f"  aggregate: accuracy={summary.diagnosis_accuracy:.3f} "
                "within_budget="
                f"{summary.diagnosed_correctly_within_budget:.3f} "
                "threshold_success="
                f"{summary.threshold_qualified_success:.3f} "
                f"premature={summary.premature_diagnosis_rate:.3f} "
                f"mean_experiments={summary.mean_experiments:.3f} "
                f"mean_regret={summary.mean_action_regret:.6f} "
                f"oracle_agreement={summary.oracle_action_agreement:.3f}"
            )
        else:
            print(
                f"  aggregate: accuracy={summary.diagnosis_accuracy:.3f} "
                f"success={summary.success_within_budget:.3f} "
                f"mean_experiments={summary.mean_experiments:.3f} "
                f"mean_regret={summary.mean_action_regret:.6f} "
                f"oracle_agreement={summary.oracle_action_agreement:.3f}"
            )
        if args.benchmark in ("er1", "er1_v1", "er1_v2"):
            suffix = (
                ""
                if args.benchmark == "er1_v2"
                else f" premature={summary.premature_diagnosis_rate:.3f}"
            )
            print(
                "  structural metrics: "
                f"false_structural={summary.false_structural_diagnosis_rate:.3f} "
                f"missed_structural={summary.missed_structural_failure_rate:.3f}"
                f"{suffix}"
            )


if __name__ == "__main__":
    main()
