"""Run the fixed-grid, oracle-only ER-1 V2 calibration."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.evaluation.er1_v2_calibration import (
    ER1_V2_CALIBRATION_BUDGETS,
    ER1_V2_CALIBRATION_SEEDS,
    ER1_V2_CALIBRATION_THRESHOLDS,
    attach_v1_comparison,
    run_er1_v2_oracle_calibration,
    write_er1_v2_calibration_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=ER1_V2_CALIBRATION_SEEDS)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=ER1_V2_CALIBRATION_BUDGETS
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=ER1_V2_CALIBRATION_THRESHOLDS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--v1-calibration",
        type=Path,
        default=Path("results/er1_oracle_calibration.csv"),
    )
    parser.add_argument(
        "--v1-overall",
        type=Path,
        default=Path("results/er1_oracle_overall.csv"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    episode_count = 4 * len(args.budgets) * len(args.thresholds) * args.seeds
    print("ER-1 V2 oracle calibration (no LLM, no network)")
    print(f"seeds per condition: {args.seeds}")
    print(f"budgets: {args.budgets}")
    print(f"thresholds: {args.thresholds}")
    print(f"episodes: {episode_count}")
    study = run_er1_v2_oracle_calibration(
        seed_count=args.seeds,
        budgets=args.budgets,
        thresholds=args.thresholds,
    )
    study = attach_v1_comparison(
        study,
        v1_calibration_csv=args.v1_calibration,
        v1_overall_csv=args.v1_overall,
    )
    artifacts = write_er1_v2_calibration_artifacts(study, args.output_dir)
    print(f"runtime_seconds: {study.runtime_seconds:.3f}")
    print(f"calibration_csv: {artifacts.calibration_csv}")
    print(f"overall_csv: {artifacts.overall_csv}")
    print(f"confusion_json: {artifacts.confusion_json}")
    print(f"hard_cases_json: {artifacts.hard_cases_json}")
    print(f"report_markdown: {artifacts.report_markdown}")


if __name__ == "__main__":
    main()
