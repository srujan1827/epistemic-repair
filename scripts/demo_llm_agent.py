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
from epistemic_repair.llm.sanitize import sanitize_text


FAILURE_SELECTIONS: dict[str, tuple[FailureMode, ...]] = {
    "world_shift": (FailureMode.WORLD_SHIFT,),
    "sensor_corruption": (FailureMode.SENSOR_CORRUPTION,),
    "missing_latent_variable": (FailureMode.MISSING_LATENT_VARIABLE,),
    "all": HYPOTHESES,
}


def selected_failure_modes(selection: str) -> tuple[FailureMode, ...]:
    """Return the failure modes selected by a CLI value."""
    return FAILURE_SELECTIONS[selection]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse provider, model, condition, and call-control configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gemini", choices=("gemini",))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--condition", choices=("full", "planner", "both"), default="both")
    parser.add_argument(
        "--failure",
        choices=(
            "world_shift",
            "sensor_corruption",
            "missing_latent_variable",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--budget", type=int, default=2)
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
    episode: LLMEpisodeResult,
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
    failure_modes = selected_failure_modes(args.failure)
    episode_count = len(failure_modes) * args.repetitions * len(conditions)

    print(f"provider: {config.provider}{' (mocked transport)' if args.mock else ''}")
    print(f"model: {config.model_id}")
    print(f"condition: {args.condition}")
    print(f"failure: {args.failure}")
    print(f"budget: {args.budget}")
    print(f"number of episodes: {episode_count}")

    try:
        client = DeterministicMockLLMClient() if args.mock else create_llm_client(config)
    except LLMConfigurationError as error:
        raise SystemExit(f"Configuration error: {error}") from error

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
        print(
            f"  aggregate: accuracy={summary.diagnosis_accuracy:.3f} "
            f"success={summary.success_within_budget:.3f} "
            f"mean_experiments={summary.mean_experiments:.3f} "
            f"mean_regret={summary.mean_action_regret:.6f} "
            f"oracle_agreement={summary.oracle_action_agreement:.3f}"
        )


if __name__ == "__main__":
    main()
