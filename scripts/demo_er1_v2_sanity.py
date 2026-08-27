"""Run the bounded 800-episode ER-1 V2 oracle sanity study."""

import argparse
from pathlib import Path

from epistemic_repair.er1_v2.sanity import (
    ER1_V2_SANITY_SEEDS,
    run_er1_v2_sanity_study,
    write_er1_v2_sanity_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=ER1_V2_SANITY_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("ER-1 V2 oracle sanity study (no LLM, no network)")
    print(f"seeds per hypothesis/budget: {args.seeds}")
    print(f"episodes: {4 * 2 * args.seeds}")
    study = run_er1_v2_sanity_study(seed_count=args.seeds)
    artifacts = write_er1_v2_sanity_artifacts(study, args.output_dir)
    print(f"runtime_seconds: {study.runtime_seconds:.3f}")
    print(f"metrics_csv: {artifacts.metrics_csv}")
    print(f"confusion_json: {artifacts.confusion_json}")
    print(f"traces_json: {artifacts.traces_json}")
    print(f"report_markdown: {artifacts.report_markdown}")


if __name__ == "__main__":
    main()
