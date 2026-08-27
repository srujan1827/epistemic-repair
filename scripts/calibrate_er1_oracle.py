"""Run the oracle-only ER-1 statistical calibration sweep."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.evaluation.er1_calibration import (
    DEFAULT_CALIBRATION_BUDGETS,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_CALIBRATION_THRESHOLDS,
    run_oracle_calibration,
    write_calibration_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bounded oracle calibration grid and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_CALIBRATION_SEEDS)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_CALIBRATION_BUDGETS,
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_CALIBRATION_THRESHOLDS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main() -> None:
    """Run the requested no-network oracle sweep and save aggregate artifacts."""
    args = parse_args()
    episode_count = 4 * len(args.budgets) * len(args.thresholds) * args.seeds
    print("ER-1 oracle calibration (no LLM, no network)")
    print(f"seeds per condition: {args.seeds}")
    print(f"budgets: {args.budgets}")
    print(f"thresholds: {args.thresholds}")
    print(f"episodes: {episode_count}")
    study = run_oracle_calibration(
        seed_count=args.seeds,
        budgets=args.budgets,
        thresholds=args.thresholds,
    )
    artifacts = write_calibration_artifacts(study, args.output_dir)
    print(f"runtime_seconds: {study.runtime_seconds:.3f}")
    print(f"calibration_csv: {artifacts.calibration_csv}")
    print(f"overall_csv: {artifacts.overall_csv}")
    print(f"confusion_json: {artifacts.confusion_json}")
    print(f"hard_cases_json: {artifacts.hard_cases_json}")
    print(f"report_markdown: {artifacts.report_markdown}")


if __name__ == "__main__":
    main()
