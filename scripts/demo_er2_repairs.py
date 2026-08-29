"""Generate the deterministic ER-2 repair matrix and baseline report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.er2.report import generate_er2_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/er2_deterministic"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_er2_report(args.output_dir, overwrite=args.overwrite)
    print(f"repair-matrix cells: {len(result['matrix'])}")
    print(f"baselines: {len(result['baselines'])}")
    print(f"output directory: {args.output_dir}")
    print("provider calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
