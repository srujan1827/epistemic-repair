"""Run or preflight the prospective ER-1 V2 -> ER-2 end-to-end study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.er2.end_to_end_study import (
    EndToEndStudyConfig,
    run_end_to_end_study,
    write_end_to_end_preflight,
)
from epistemic_repair.llm import LLMConfigurationError, create_llm_client


DEFAULT_OUTPUT_DIR = Path(
    "results/er1_v2_er2_end_to_end_gemini_3_6_flash_low_seed0_9_tier1"
)


def parse_seeds(values: Sequence[str]) -> tuple[int, ...]:
    """Parse integers, comma lists, and inclusive ranges such as ``0..9``."""
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
            "seeds must be integers, comma lists, or inclusive ranges such as 0..9"
        ) from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return tuple(seeds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("gemini",), default="gemini")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument("--seeds", nargs="+", default=("0..9",), metavar="SEED")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--diagnosis-threshold", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-decision-calls", type=int, default=9)
    parser.add_argument("--min-request-interval-seconds", type=float, default=2.0)
    parser.add_argument("--rate-limit-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--episode-cooldown-seconds", type=float, default=5.0)
    parser.add_argument(
        "--resume-provider-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write boundary/method artifacts without creating a provider client.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = EndToEndStudyConfig(
            provider=args.provider,
            model_id=args.model,
            thinking_level=args.thinking_level,
            seeds=parse_seeds(args.seeds),
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
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise SystemExit(f"Configuration error: {error}") from error

    print("benchmark: binary_er1_v2_er2_end_to_end_v0", flush=True)
    print("investigation condition: FULL_AUTONOMOUS", flush=True)
    print("repair condition: EVIDENCE_ONLY_NEUTRAL_REPAIR_SELECTION", flush=True)
    print(f"provider: {config.provider}", flush=True)
    print(f"model: {config.model_id}", flush=True)
    print(f"thinking level: {config.thinking_level}", flush=True)
    print(f"seeds: {','.join(map(str, config.seeds))}", flush=True)
    print(f"budget: {config.experiment_budget}", flush=True)
    print(f"diagnosis threshold: {config.diagnosis_threshold}", flush=True)
    print(f"max decision calls: {config.max_decision_calls}", flush=True)
    print(f"planned episodes: {config.episode_count}", flush=True)
    print(f"output directory: {args.output_dir}", flush=True)

    if args.preflight_only:
        write_end_to_end_preflight(args.output_dir, config, overwrite=args.overwrite)
        print("preflight complete; provider client was not created", flush=True)
        return 0
    try:
        client = create_llm_client(config.llm_config())
    except LLMConfigurationError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    run_end_to_end_study(
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
