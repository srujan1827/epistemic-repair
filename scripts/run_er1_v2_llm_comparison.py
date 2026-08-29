"""Run a resumable matched-seed ER-1 V2 LLM-condition comparison."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.evaluation.er1_v2_llm_comparison import (
    ComparisonConfig,
    run_er1_v2_llm_comparison,
)
from epistemic_repair.llm import LLMConfigurationError, create_llm_client
from epistemic_repair.llm.schemas import LLMCondition


DEFAULT_OUTPUT_DIR = Path("results/er1_v2_gemini_3_6_flash_low_seed0_9")
CONDITION_SELECTIONS = {
    "full": LLMCondition.FULL_AUTONOMOUS,
    "planner": LLMCondition.PLANNER_ONLY,
    "threshold_aware": LLMCondition.THRESHOLD_AWARE_AUTONOMOUS,
}


def parse_seeds(values: Sequence[str]) -> tuple[int, ...]:
    """Parse individual, comma-separated, or inclusive ``start..end`` seeds."""
    seeds: list[int] = []
    try:
        for value in values:
            for item in value.split(","):
                item = item.strip()
                if not item:
                    raise ValueError
                if ".." in item:
                    start_text, end_text = item.split("..", 1)
                    start, end = int(start_text), int(end_text)
                    if end < start:
                        raise ValueError
                    seeds.extend(range(start, end + 1))
                else:
                    seeds.append(int(item))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be integers, comma-separated integers, or inclusive ranges such as 0..9"
        ) from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return tuple(seeds)


def _threshold(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("diagnosis threshold must be a number") from error
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("diagnosis threshold must be in (0, 1]")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse exact provider, scientific, call-control, and output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("gemini",), default="gemini")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITION_SELECTIONS),
        default=("full", "planner"),
        help="LLM conditions to run (default: full planner).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=("0..9",),
        metavar="SEED",
        help="Seeds as integers, comma lists, or inclusive ranges (default: 0..9).",
    )
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--diagnosis-threshold", type=_threshold, default=0.95)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-decision-calls", type=int, default=9)
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=3.5,
        help="Minimum wall-clock interval between starts of provider attempts.",
    )
    parser.add_argument(
        "--rate-limit-backoff-seconds",
        type=float,
        default=10.0,
        help="Fallback delay when a rate-limit response has no parseable retry hint.",
    )
    parser.add_argument(
        "--episode-cooldown-seconds",
        type=float,
        default=0.0,
        help="Delay after a processed cell when another runnable cell remains.",
    )
    parser.add_argument(
        "--resume-provider-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerun provider-failed checkpoints while preserving successful cells.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace known artifacts and checkpoints in the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create one named client, then run or resume all matched cells."""
    args = parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds)
        config = ComparisonConfig(
            provider=args.provider,
            model_id=args.model,
            thinking_level=args.thinking_level,
            conditions=tuple(
                CONDITION_SELECTIONS[value] for value in args.conditions
            ),
            seeds=seeds,
            experiment_budget=args.budget,
            diagnosis_threshold=args.diagnosis_threshold,
            timeout_seconds=args.timeout,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            max_decision_calls=args.max_decision_calls,
            min_request_interval_seconds=args.min_request_interval_seconds,
            rate_limit_backoff_seconds=args.rate_limit_backoff_seconds,
            episode_cooldown_seconds=args.episode_cooldown_seconds,
        )
        client = create_llm_client(config.llm_config())
    except (LLMConfigurationError, ValueError, argparse.ArgumentTypeError) as error:
        raise SystemExit(f"Configuration error: {error}") from error

    episode_count = 4 * len(config.seeds) * len(config.conditions)
    print(f"benchmark: binary_er1_v2", flush=True)
    print(f"provider: {config.provider}", flush=True)
    print(f"model: {config.model_id}", flush=True)
    print(f"thinking level: {config.thinking_level}", flush=True)
    print(f"conditions: {' + '.join(args.conditions)}", flush=True)
    print(f"seeds per hypothesis: {','.join(map(str, config.seeds))}", flush=True)
    print(f"budget: {config.experiment_budget}", flush=True)
    print(f"diagnosis threshold: {config.diagnosis_threshold}", flush=True)
    print(f"max decision calls: {config.max_decision_calls}", flush=True)
    print(
        f"minimum request interval: {config.min_request_interval_seconds}s",
        flush=True,
    )
    print(
        f"rate-limit fallback backoff: {config.rate_limit_backoff_seconds}s",
        flush=True,
    )
    print(f"episode cooldown: {config.episode_cooldown_seconds}s", flush=True)
    print(
        f"resume provider failures: {args.resume_provider_failures}",
        flush=True,
    )
    print(f"planned episodes: {episode_count}", flush=True)
    print(f"output directory: {args.output_dir}", flush=True)

    run_er1_v2_llm_comparison(
        client,
        config,
        args.output_dir,
        overwrite=args.overwrite,
        retry_provider_failures=args.resume_provider_failures,
        progress=lambda message: print(message, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
