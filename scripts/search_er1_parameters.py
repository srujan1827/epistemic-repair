"""Run the staged oracle-only search for ER-1 V2 candidate parameters."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from epistemic_repair.evaluation.er1_parameter_search import (
    run_parameter_search,
    write_parameter_search_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-seeds", type=int, default=250)
    parser.add_argument("--phase2-seeds", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print("ER-1 V2 parameter search (oracle only; no LLM, no network)")
    print(f"phase1 seeds per hypothesis/cell: {args.phase1_seeds}")
    print(f"phase2 seeds per hypothesis/cell: {args.phase2_seeds}")
    study = run_parameter_search(
        phase1_seed_count=args.phase1_seeds,
        phase2_seed_count=args.phase2_seeds,
    )
    artifacts = write_parameter_search_artifacts(study, args.output_dir)
    print(f"promising Stage-A normal accuracies: {study.promising_normal_values}")
    print(
        "top three: "
        + ", ".join(item.candidate_id for item in study.top_three_parameters)
    )
    print(f"recommended (not applied): {study.recommended_candidate_id}")
    print(f"runtime_seconds: {study.runtime_seconds:.3f}")
    print(f"stage_a_csv: {artifacts.stage_a_csv}")
    print(f"stage_b_csv: {artifacts.stage_b_csv}")
    print(f"finalists_csv: {artifacts.finalists_csv}")
    print(f"phase2_cells_csv: {artifacts.phase2_cells_csv}")
    print(f"report_markdown: {artifacts.report_markdown}")


if __name__ == "__main__":
    main()
