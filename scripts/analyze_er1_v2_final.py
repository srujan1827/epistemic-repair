"""Combine ER-1 V2 pilot and replication artifacts without provider calls."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.evaluation.er1_v2_final_analysis import combine_final_analysis
from epistemic_repair.evaluation.er1_v2_three_condition_analysis import (
    IncompatibleAnalysisInput,
)


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ranges such as ``0..9``."""
    seeds: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if ".." in token:
            start_text, end_text = token.split("..", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError("seed ranges must be ascending")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(token))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return tuple(sorted(seeds))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="Source result directory; provide once per dataset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seed_spec, default=parse_seed_spec("0..9"))
    parser.add_argument(
        "--pilot-seeds", type=parse_seed_spec, default=parse_seed_spec("0..1")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known final-analysis artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = combine_final_analysis(
            args.input_dir,
            args.output_dir,
            expected_seeds=args.seeds,
            pilot_seeds=args.pilot_seeds,
            overwrite=args.overwrite,
        )
    except (IncompatibleAnalysisInput, FileExistsError) as error:
        raise SystemExit(f"Analysis configuration error: {error}") from error
    valid = sum(row["protocol_status"] == "VALID_EPISODE" for row in result["rows"])
    print(f"planned cells: {len(result['rows'])}")
    print(f"valid episodes: {valid}")
    print(f"matched triplets: {len(result['matched'])}")
    print(f"output directory: {args.output_dir}")
    print("provider calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
