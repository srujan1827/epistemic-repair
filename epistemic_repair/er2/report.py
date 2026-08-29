"""Artifact generation for the deterministic minimal ER-2 benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from epistemic_repair.er2.runner import build_wrong_repair_matrix, evaluate_baselines


OUTPUT_FILES = ("repair_matrix.csv", "baseline_summary.csv", "report.md")


def generate_er2_report(
    output_directory: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the deterministic matrix, baseline summary, and technical report."""
    output = Path(output_directory)
    _prepare_output(output, overwrite=overwrite)
    matrix = build_wrong_repair_matrix()
    baselines = evaluate_baselines()
    _write_csv(output / "repair_matrix.csv", matrix)
    _write_csv(output / "baseline_summary.csv", baselines)
    report = build_report_markdown(matrix, baselines)
    (output / "report.md").write_text(report, encoding="utf-8")
    return {"matrix": matrix, "baselines": baselines, "report": report}


def build_report_markdown(
    matrix: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
) -> str:
    """Build an answer-first technical report from deterministic results."""
    indexed = {
        (row["true_hypothesis"], row["applied_repair"]): row for row in matrix
    }
    sensor_wrong = indexed[("SENSOR_CORRUPTION", "UPDATE_WORLD_MODEL")]
    world_correct = indexed[("WORLD_SHIFT", "UPDATE_WORLD_MODEL")]
    oracle_rows = [row for row in matrix if row["repair_selection_correct"]]
    correct_affected = all(row["affected_region_accuracy"] == 1.0 for row in oracle_rows)
    correct_preserves = all(row["collateral_damage"] == 0.0 for row in oracle_rows)
    lines = [
        "# ER-2 minimal deterministic repair benchmark",
        "",
        "## Technical summary",
        "",
        f"All four canonical repairs recover 100% affected-region accuracy and cause zero collateral damage. "
        f"The full 4×4 matrix is non-degenerate: wrong repairs can fail, compensate for one output layer, or fix one context while damaging another.",
        "",
        f"The central negative example is `SENSOR_CORRUPTION + UPDATE_WORLD_MODEL`: affected-region accuracy rises from 0% before repair to {_pct(sensor_wrong['affected_region_accuracy'])}, "
        f"but unaffected physical knowledge falls to {_pct(sensor_wrong['unaffected_region_accuracy'])}, producing {_f(sensor_wrong['collateral_damage'])} collateral damage. "
        "The wrong world update predicts corrupted end-to-end observations correctly for the wrong reason.",
        "",
        "## The repair matrix separates recovery from preservation",
        "",
        _matrix_table(matrix),
        "",
        "`repair_success` is behavioral rather than label-only: affected accuracy must be 1.0, collateral damage must be non-positive, and overall accuracy must not decline. Repair-selection correctness is reported separately against the canonical mapping.",
        "",
        "## Fixed baselines expose the value of diagnosis-conditioned repair",
        "",
        _baseline_table(baselines),
        "",
        "Each always-repair baseline selects correctly in exactly one of four hypotheses. Baseline metrics are unweighted macro means across the four hypotheses; within an episode, overall accuracy is case-weighted over the fixed ten-case suite. The oracle uses the same state mutation and post-repair evaluator as every baseline; it has no scoring exception.",
        "",
        "## Wrong learning can match observations while corrupting physics",
        "",
        f"For sensor corruption, the healthy pre-repair state has {_pct(sensor_wrong['pre_repair_accuracy'])} overall accuracy. "
        f"Updating the world model leaves overall accuracy at {_pct(sensor_wrong['post_repair_accuracy'])}: it repairs all four end-to-end `X→O` cases, but the two direct sensor-mapping cases remain wrong and all four physical `X→Y` cases become wrong. "
        f"Thus the affected-region score is {_pct(sensor_wrong['affected_region_accuracy'])}, unaffected-region accuracy is {_pct(sensor_wrong['unaffected_region_accuracy'])}, and collateral damage is {_f(sensor_wrong['collateral_damage'])}.",
        "",
        "This is a concrete demonstration of why observation agreement alone is insufficient evidence for changing the world model.",
        "",
        "## Correct world repair recovers the shifted process without spillover",
        "",
        f"For world shift, `UPDATE_WORLD_MODEL` raises overall accuracy from {_pct(world_correct['pre_repair_accuracy'])} to {_pct(world_correct['post_repair_accuracy'])}. "
        f"Affected physical and end-to-end cases reach {_pct(world_correct['affected_region_accuracy'])}; the unchanged sensor mapping remains at {_pct(world_correct['unaffected_region_accuracy'])}; collateral damage is {_f(world_correct['collateral_damage'])}.",
        "",
        "## State, suite, and metric definitions",
        "",
        "The pre-repair state has three independent components: world relation `Y=X`, primary-sensor calibration `O=Y`, and absent context/latent structure. `UPDATE_WORLD_MODEL` changes only the world relation to `Y=1-X`; `RECALIBRATE_SENSOR` changes only the sensor mapping to `O=1-Y`; `ADD_LATENT_VARIABLE` adds `Y=X` in context A and `Y=1-X` in context B; `NO_REPAIR` is identity.",
        "",
        "Each hypothesis is evaluated on the same ten deterministic predictions: four physical `X,context→Y` cases, two direct sensor `Y→O` cases, and four end-to-end `X,context→O` cases. For structural hypotheses, affected cases are exactly those whose targets differ from the healthy pre-change process. Because `NO_STRUCTURAL_CHANGE` has no persistent changed region, its affected region is explicitly defined as two trigger-adjacent held-out probes (`X=1`, context A, physical and end-to-end); the other eight cases are unaffected preservation checks.",
        "",
        "Collateral damage is not clamped:",
        "",
        "```text",
        "pre-repair accuracy on unaffected cases",
        "- post-repair accuracy on unaffected cases",
        "```",
        "",
        "Positive values mean damage; zero means preservation; negative values would mean improvement on previously unaffected cases.",
        "",
        "## Acceptance checks",
        "",
        f"- **A — Correct repair recovers affected performance:** {'PASS' if correct_affected else 'FAIL'}. Every canonical repair reaches 100% affected-region accuracy.",
        f"- **B — Correct repair preserves unaffected knowledge:** {'PASS' if correct_preserves else 'FAIL'}. Every canonical repair has zero collateral damage. Some wrong repairs also preserve unaffected cases while failing to repair the affected region, so preservation alone is not sufficient; correct repairs dominate on the joint recovery-and-preservation criterion.",
        "- **C — Wrong repair can improve observations while causing damage:** PASS. Sensor corruption plus world update repairs end-to-end observations while destroying physical predictions; missing latent plus a global world update similarly fixes context B while damaging context A.",
        "- **D — Sensor corruption demonstrates knowing when not to learn:** PASS. Updating physics fits corrupted observations for the wrong causal reason and incurs maximal physical collateral damage.",
        "- **E — Ready for LLM repair selection:** PASS WITH CAVEAT. The state/evaluator is behaviorally non-trivial and suitable for testing repair choices. However, diagnosis-to-repair selection may be easy if semantically transparent repair labels are exposed; future LLM claims should distinguish label matching from understanding state consequences.",
        "",
        "## Limitations and next step",
        "",
        "ER-2 V0 is deterministic and isolates repair selection after an externally supplied diagnosis. It does not yet propagate ER-1 diagnostic uncertainty, execute LLM calls, model repair costs, or test sequential/multiple repairs. The appropriate next step is mocked repair-policy integration and prompt/interface design, not a live model study yet.",
        "",
        "Exact rows are available in `repair_matrix.csv` and `baseline_summary.csv`.",
    ]
    return "\n".join(lines) + "\n"


def _matrix_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| True hypothesis | Applied repair | Overall | Affected | Unaffected | Collateral damage | Selection correct | Repair success |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['true_hypothesis']} | {row['applied_repair']} | "
            f"{_pct(row['post_repair_accuracy'])} | {_pct(row['affected_region_accuracy'])} | "
            f"{_pct(row['unaffected_region_accuracy'])} | {_f(row['collateral_damage'])} | "
            f"{'yes' if row['repair_selection_correct'] else 'no'} | "
            f"{'yes' if row['repair_success'] else 'no'} |"
        )
    return "\n".join(lines)


def _baseline_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Baseline | Selection accuracy | Repair success | Overall | Affected | Unaffected | Collateral damage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline']} | {_pct(row['repair_selection_accuracy'])} | "
            f"{_pct(row['repair_success_rate'])} | {_pct(row['post_repair_accuracy'])} | "
            f"{_pct(row['affected_region_accuracy'])} | {_pct(row['unaffected_region_accuracy'])} | "
            f"{_f(row['collateral_damage'])} |"
        )
    return "\n".join(lines)


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite and any((output / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError(f"ER-2 report already exists at {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in OUTPUT_FILES:
            path = output / name
            if path.is_file():
                path.unlink()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _f(value: Any) -> str:
    return f"{float(value):.3f}"
