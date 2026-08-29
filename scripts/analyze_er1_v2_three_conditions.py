"""Combine completed ER-1 V2 condition runs without making provider calls."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.evaluation.er1_v2_three_condition_analysis import (
    IncompatibleAnalysisInput,
    combine_three_condition_results,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="Completed comparison directory; provide this option more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known combined-analysis artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = combine_three_condition_results(
            args.input_dir,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (IncompatibleAnalysisInput, FileExistsError) as error:
        raise SystemExit(f"Analysis configuration error: {error}") from error
    print(f"combined episodes: {len(result['rows'])}")
    print(f"matched triplets: {len(result['matched'])}")
    print(f"output directory: {args.output_dir}")
    print("provider calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
