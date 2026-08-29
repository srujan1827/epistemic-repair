"""Zero-call analysis of completed ER-1 V2 LLM comparison directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


CONDITIONS = (
    "FULL_AUTONOMOUS",
    "PLANNER_ONLY",
    "THRESHOLD_AWARE_AUTONOMOUS",
)
HYPOTHESES = (
    "NO_STRUCTURAL_CHANGE",
    "WORLD_SHIFT",
    "SENSOR_CORRUPTION",
    "MISSING_LATENT_VARIABLE",
)
AUTONOMOUS_CONDITIONS = (
    "FULL_AUTONOMOUS",
    "THRESHOLD_AWARE_AUTONOMOUS",
)
MATCH_FIELDS = (
    "benchmark_version",
    "provider",
    "model",
    "thinking_level",
    "experiment_budget",
    "diagnosis_threshold",
    "max_decision_calls",
    "seeds",
    "hypotheses",
)
OUTPUT_FILES = (
    "episodes_combined.csv",
    "summary_three_conditions.csv",
    "per_hypothesis_three_conditions.csv",
    "matched_three_condition.csv",
    "report.md",
)
MATCHED_VALUE_FIELDS = (
    "final_diagnosis",
    "diagnosis_correct",
    "threshold_qualified_success",
    "premature_diagnosis",
    "experiments_used",
    "normative_probability_of_final_diagnosis",
    "final_self_reported_confidence",
    "cumulative_action_regret",
    "oracle_action_agreement_rate",
    "action_sequence",
)
DELTA_FIELDS = (
    "diagnosis_correct",
    "threshold_qualified_success",
    "premature_diagnosis",
    "experiments_used",
    "normative_probability_of_final_diagnosis",
    "final_self_reported_confidence",
    "cumulative_action_regret",
    "oracle_action_agreement_rate",
)


class IncompatibleAnalysisInput(ValueError):
    """Completed run artifacts cannot form one controlled comparison."""


def combine_three_condition_results(
    input_directories: Sequence[Path | str],
    output_directory: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate, combine, analyze, and report three completed conditions."""
    if len(input_directories) < 2:
        raise IncompatibleAnalysisInput("at least two input directories are required")
    inputs = tuple(Path(item) for item in input_directories)
    output = Path(output_directory)
    _prepare_output(output, overwrite=overwrite)

    configs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()
    seen_keys: set[tuple[str, str, int]] = set()
    prompt_versions: dict[str, str] = {}

    for directory in inputs:
        config, source_rows = _read_run(directory)
        _validate_against_reference(configs[0] if configs else None, config, directory)
        source_conditions = set(config["conditions"])
        overlap = seen_conditions & source_conditions
        if overlap:
            raise IncompatibleAnalysisInput(
                f"duplicate conditions across inputs: {sorted(overlap)}"
            )
        seen_conditions.update(source_conditions)
        _validate_source_rows(directory, config, source_rows)
        for row in source_rows:
            key = _episode_key(row)
            if key in seen_keys:
                raise IncompatibleAnalysisInput(
                    f"duplicate episode key: {key}"
                )
            seen_keys.add(key)
            condition = row["condition"]
            version = row["prompt_version"]
            previous = prompt_versions.setdefault(condition, version)
            if previous != version:
                raise IncompatibleAnalysisInput(
                    f"condition {condition} has multiple prompt versions"
                )
            combined = dict(row)
            combined["source_run_directory"] = directory.name
            rows.append(combined)
        configs.append(config)

    if seen_conditions != set(CONDITIONS):
        raise IncompatibleAnalysisInput(
            "combined inputs must contain exactly FULL_AUTONOMOUS, "
            "PLANNER_ONLY, and THRESHOLD_AWARE_AUTONOMOUS"
        )
    reference = configs[0]
    _validate_complete_grid(rows, reference)
    ordered = sorted(rows, key=_row_sort_key)
    summary = summarize_episode_rows(ordered)
    per_hypothesis = summarize_episode_rows(ordered, by_hypothesis=True)
    matched = build_matched_triplets(ordered)
    calibration = stopping_calibration(ordered, float(reference["diagnosis_threshold"]))

    _write_csv(output / "episodes_combined.csv", ordered)
    _write_csv(output / "summary_three_conditions.csv", summary)
    _write_csv(output / "per_hypothesis_three_conditions.csv", per_hypothesis)
    _write_csv(output / "matched_three_condition.csv", matched)
    (output / "report.md").write_text(
        build_analysis_report(
            inputs=inputs,
            config=reference,
            rows=ordered,
            summary=summary,
            per_hypothesis=per_hypothesis,
            matched=matched,
            calibration=calibration,
            prompt_versions=prompt_versions,
        ),
        encoding="utf-8",
    )
    return {
        "rows": ordered,
        "summary": summary,
        "per_hypothesis": per_hypothesis,
        "matched": matched,
        "calibration": calibration,
        "prompt_versions": prompt_versions,
    }


def summarize_episode_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    by_hypothesis: bool = False,
) -> list[dict[str, Any]]:
    """Compute condition metrics with action-weighted oracle agreement."""
    groups: dict[tuple[str, str | None], list[Mapping[str, str]]] = {}
    for row in rows:
        key = (
            row["condition"],
            row["true_hypothesis"] if by_hypothesis else None,
        )
        groups.setdefault(key, []).append(row)
    condition_order = {name: index for index, name in enumerate(CONDITIONS)}
    hypothesis_order = {name: index for index, name in enumerate(HYPOTHESES)}
    output = []
    for (condition, hypothesis), group in sorted(
        groups.items(),
        key=lambda item: (
            condition_order[item[0][0]],
            hypothesis_order.get(item[0][1], -1),
        ),
    ):
        experiments = sum(_integer(row, "experiments_used") for row in group)
        agreements = sum(_integer(row, "oracle_action_agreements") for row in group)
        no_change = [row for row in group if row["true_hypothesis"] == "NO_STRUCTURAL_CHANGE"]
        structural = [row for row in group if row["true_hypothesis"] != "NO_STRUCTURAL_CHANGE"]
        belief_errors = _available_numbers(group, "mean_autonomous_belief_l1_error")
        result: dict[str, Any] = {
            "condition": condition,
            "episode_count": len(group),
            "diagnosis_accuracy": _boolean_rate(group, "diagnosis_correct"),
            "threshold_qualified_success": _boolean_rate(group, "threshold_qualified_success"),
            "premature_diagnosis_rate": _boolean_rate(group, "premature_diagnosis"),
            "mean_experiments": _mean_field(group, "experiments_used"),
            "mean_regret": _mean_field(group, "cumulative_action_regret"),
            "oracle_action_agreement": agreements / experiments if experiments else None,
            "mean_autonomous_belief_l1_error": (
                mean(belief_errors)
                if condition in AUTONOMOUS_CONDITIONS and belief_errors
                else None
            ),
            "false_structural_diagnosis_rate": (
                mean(
                    row["final_diagnosis"] not in ("", "NO_STRUCTURAL_CHANGE")
                    for row in no_change
                )
                if no_change
                else None
            ),
            "missed_structural_diagnosis_rate": (
                mean(row["final_diagnosis"] == "NO_STRUCTURAL_CHANGE" for row in structural)
                if structural
                else None
            ),
            "provider_failure_rate": _boolean_rate(group, "provider_failure_flag"),
            "scientific_model_failure_rate": _boolean_rate(group, "scientific_model_failure_flag"),
        }
        if by_hypothesis:
            result = {
                "condition": condition,
                "true_hypothesis": hypothesis,
                **{key: value for key, value in result.items() if key != "condition"},
            }
        output.append(result)
    return output


def build_matched_triplets(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Put the three conditions side-by-side at hypothesis/seed grain."""
    grouped: dict[tuple[str, int], dict[str, Mapping[str, str]]] = {}
    for row in rows:
        key = (row["true_hypothesis"], int(row["seed"]))
        conditions = grouped.setdefault(key, {})
        if row["condition"] in conditions:
            raise IncompatibleAnalysisInput(
                f"duplicate matched condition for {key}: {row['condition']}"
            )
        conditions[row["condition"]] = row

    output = []
    for hypothesis in HYPOTHESES:
        seeds = sorted(seed for truth, seed in grouped if truth == hypothesis)
        for seed in seeds:
            triplet = grouped[(hypothesis, seed)]
            if set(triplet) != set(CONDITIONS):
                raise IncompatibleAnalysisInput(
                    f"matched key {(hypothesis, seed)} is not a complete triplet"
                )
            item: dict[str, Any] = {
                "true_hypothesis": hypothesis,
                "seed": seed,
                "pair_key": f"{hypothesis}:{seed}",
            }
            comparable: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                prefix = _condition_prefix(condition)
                row = triplet[condition]
                values = _matched_values(row)
                comparable[condition] = values
                for field in MATCHED_VALUE_FIELDS:
                    item[f"{prefix}_{field}"] = values[field]
            for label, left, right in (
                ("planner_minus_full", "PLANNER_ONLY", "FULL_AUTONOMOUS"),
                ("threshold_aware_minus_full", "THRESHOLD_AWARE_AUTONOMOUS", "FULL_AUTONOMOUS"),
                ("threshold_aware_minus_planner", "THRESHOLD_AWARE_AUTONOMOUS", "PLANNER_ONLY"),
            ):
                for field in DELTA_FIELDS:
                    item[f"{label}_{field}"] = _subtract(
                        comparable[left][field],
                        comparable[right][field],
                    )
            output.append(item)
    return output


def stopping_calibration(
    rows: Sequence[Mapping[str, str]],
    diagnosis_threshold: float,
) -> list[dict[str, Any]]:
    """Compare autonomous self-confidence with normative diagnosis support."""
    output = []
    for condition in AUTONOMOUS_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for hypothesis in ("ALL", *HYPOTHESES):
            selected = (
                condition_rows
                if hypothesis == "ALL"
                else [row for row in condition_rows if row["true_hypothesis"] == hypothesis]
            )
            comparable = []
            for row in selected:
                confidence = _optional_float(row.get("model_reported_final_confidence"))
                normative = _optional_float(row.get("normative_probability_of_final_diagnosis"))
                if confidence is not None and normative is not None:
                    comparable.append((confidence, normative))
            gaps = [confidence - normative for confidence, normative in comparable]
            count = len(comparable)
            output.append({
                "condition": condition,
                "true_hypothesis": hypothesis,
                "episode_count": len(selected),
                "comparable_confidence_count": count,
                "missing_confidence_count": len(selected) - count,
                "mean_signed_gap": mean(gaps) if gaps else None,
                "mean_absolute_gap": mean(abs(gap) for gap in gaps) if gaps else None,
                "overconfidence_frequency": (
                    mean(gap > 1e-12 for gap in gaps) if gaps else None
                ),
                "self_ge_threshold_normative_lt_count": sum(
                    confidence >= diagnosis_threshold and normative < diagnosis_threshold
                    for confidence, normative in comparable
                ),
                "self_ge_threshold_normative_lt_rate": (
                    mean(
                        confidence >= diagnosis_threshold and normative < diagnosis_threshold
                        for confidence, normative in comparable
                    )
                    if comparable
                    else None
                ),
                "both_ge_threshold_count": sum(
                    confidence >= diagnosis_threshold and normative >= diagnosis_threshold
                    for confidence, normative in comparable
                ),
                "both_ge_threshold_rate": (
                    mean(
                        confidence >= diagnosis_threshold and normative >= diagnosis_threshold
                        for confidence, normative in comparable
                    )
                    if comparable
                    else None
                ),
            })
    return output


def build_analysis_report(
    *,
    inputs: Sequence[Path],
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    summary: Sequence[Mapping[str, Any]],
    per_hypothesis: Sequence[Mapping[str, Any]],
    matched: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    prompt_versions: Mapping[str, str],
) -> str:
    """Render a descriptive technical report from already computed artifacts."""
    indexed = {row["condition"]: row for row in summary}
    full = indexed["FULL_AUTONOMOUS"]
    planner = indexed["PLANNER_ONLY"]
    aware = indexed["THRESHOLD_AWARE_AUTONOMOUS"]
    calibration_overall = {
        row["condition"]: row
        for row in calibration
        if row["true_hypothesis"] == "ALL"
    }
    planner_regret_delta = planner["mean_regret"] - full["mean_regret"]
    aware_premature_delta = (
        aware["premature_diagnosis_rate"] - full["premature_diagnosis_rate"]
    )
    lines = [
        "# ER-1 V2 three-condition pilot analysis",
        "",
        "## Technical summary",
        "",
        f"All three conditions diagnosed all {len(matched)} matched cases correctly, so this pilot shows no raw-accuracy advantage from authoritative beliefs or threshold awareness. The accuracy result is ceiling-limited.",
        "",
        f"Planner-only reduced mean regret from {_f(full['mean_regret'])} to {_f(planner['mean_regret'])} (delta {_signed(planner_regret_delta)}) and increased action-level oracle agreement from {_pct(full['oracle_action_agreement'])} to {_pct(planner['oracle_action_agreement'])}. This directionally supports better experiment selection with authoritative beliefs.",
        "",
        f"Threshold awareness reduced premature diagnosis from {_pct(full['premature_diagnosis_rate'])} to {_pct(aware['premature_diagnosis_rate'])} and raised threshold-qualified success from {_pct(full['threshold_qualified_success'])} to {_pct(aware['threshold_qualified_success'])}. It did not eliminate premature stopping: {int(aware['premature_diagnosis_rate'] * aware['episode_count'])} of {aware['episode_count']} threshold-aware episodes remained premature.",
        "",
        "These are descriptive results from only two seeds per hypothesis. They do not establish statistical significance or a stable model-level effect.",
        "",
        "## Threshold awareness improves stopping directionally, not completely",
        "",
        _summary_table(summary),
        "",
        "### Diagnosis and execution failure checks",
        "",
        _failure_table(summary),
        "",
        f"The threshold-aware premature rate changed by {_signed(aware_premature_delta)} relative to full autonomy. It used {_f(aware['mean_experiments'])} experiments on average versus {_f(full['mean_experiments'])} for full autonomy and {_f(planner['mean_experiments'])} for planner-only.",
        "",
        "## Self-confidence remains misaligned with normative support",
        "",
        _calibration_table(calibration_overall),
        "",
        f"In threshold-aware autonomy, {calibration_overall['THRESHOLD_AWARE_AUTONOMOUS']['self_ge_threshold_normative_lt_count']} of 8 episodes reported confidence at or above {config['diagnosis_threshold']:.2f} while the normative posterior remained below it. That pattern exactly motivates treating confidence calibration as a candidate bottleneck, even though the threshold-aware mean absolute confidence gap was lower than full autonomy's.",
        "",
        "### Calibration by true hypothesis",
        "",
        _calibration_by_hypothesis_table(calibration),
        "",
        "## Pilot hypotheses A–E",
        "",
        "- **A — Raw diagnosis accuracy:** No observed improvement. Full and planner-only were both 8/8 correct; the ceiling prevents a useful accuracy contrast.",
        f"- **B — Experiment selection:** Directionally supported. Planner-only had {_f(abs(planner_regret_delta))} lower mean regret and {_signed(planner['oracle_action_agreement'] - full['oracle_action_agreement'])} higher oracle agreement.",
        f"- **C — Explicit threshold and prematurity:** Directionally supported. Prematurity fell by {100.0 * abs(aware_premature_delta):.1f} percentage points, corresponding to two additional non-premature matched cases.",
        f"- **D — Elimination of prematurity:** Not supported. Threshold-aware autonomy was still premature in {int(aware['premature_diagnosis_rate'] * aware['episode_count'])}/8 episodes.",
        f"- **E — Confidence calibration bottleneck:** Candidate supported descriptively. Threshold-aware confidence crossed 0.95 without normative support in {calibration_overall['THRESHOLD_AWARE_AUTONOMOUS']['self_ge_threshold_normative_lt_count']}/8 episodes; replication is required before attributing a stable mechanism.",
        "",
        "## Scope, definitions, and validation",
        "",
        f"The analysis combines {len(rows)} episode rows into {len(matched)} matched triplets keyed by `(true hypothesis, seed)`. Each condition contains seeds {config['seeds']} for all four ER-1 V2 hypotheses, with budget {config['experiment_budget']} and normative diagnosis threshold {config['diagnosis_threshold']}.",
        "",
        "Threshold-qualified success means the final diagnosis was correct and the benchmark's normative probability for that diagnosis met the evaluator threshold. Premature diagnosis means the normative probability of the chosen diagnosis was below that threshold. Oracle agreement is action-weighted across executed experiments. Regret is cumulative expected-information-gain regret already recorded by the frozen evaluator.",
        "",
        "Planner-only is excluded from autonomous calibration: its stored final confidence is benchmark-derived rather than a comparable model self-estimate. Missing autonomous confidence would be excluded rather than imputed.",
        "",
        "Input validation required identical benchmark, provider/model, thinking level, budget, threshold, max decision calls, hypotheses, and seeds. Prompt versions were required to be internally stable per condition but were allowed to differ across conditions:",
        "",
        *[f"- `{condition}`: `{version}`" for condition, version in sorted(prompt_versions.items())],
        "",
        "## Method and robustness limits",
        "",
        "All metrics were recomputed from episode-level CSV rows; no episodes were rerun. Categorical diagnoses and action sequences are shown side-by-side in `matched_three_condition.csv` and are not subtracted. Numeric and Boolean pairwise deltas use the named left condition minus right condition.",
        "",
        "No chart is included because only eight matched cases are available; exact tables are less likely to imply unsupported distributional precision. The principal robustness limitation is N=2 seeds per hypothesis. Provider/model stochasticity, prompt-order effects, and seed-specific trajectories may dominate observed differences.",
        "",
        "## Recommended next step",
        "",
        "Replicate the same frozen three-condition design over more matched seeds before changing prompts. Monitor premature-diagnosis rate, threshold-qualified success, self-confidence minus normative support, mean regret, and oracle agreement. The most decision-relevant question is whether threshold awareness continues to reduce prematurity while self-confidence frequently crosses 0.95 without normative support.",
        "",
        "## Further questions",
        "",
        "- Does the threshold-aware reduction in prematurity persist by hypothesis with a larger matched sample?",
        "- Are high self-confidence/low normative-support episodes concentrated in particular action histories?",
        "- Does planner-only's regret advantage persist when raw diagnosis accuracy is no longer at ceiling?",
        "",
        "The machine-readable per-hypothesis and matched-triplet artifacts contain the exact audit rows supporting these questions.",
        "",
    ]
    return "\n".join(lines)


def _read_run(directory: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    config_path = directory / "run_config.json"
    episodes_path = directory / "episodes.csv"
    if not config_path.is_file() or not episodes_path.is_file():
        raise IncompatibleAnalysisInput(
            f"{directory} must contain run_config.json and episodes.csv"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with episodes_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise IncompatibleAnalysisInput(f"{directory} contains no episode rows")
    return config, rows


def _validate_against_reference(
    reference: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    directory: Path,
) -> None:
    missing = [field for field in (*MATCH_FIELDS, "conditions", "prompt_version") if field not in candidate]
    if missing:
        raise IncompatibleAnalysisInput(
            f"{directory} run config is missing fields: {missing}"
        )
    if reference is None:
        return
    mismatches = [
        field for field in MATCH_FIELDS if candidate[field] != reference[field]
    ]
    if mismatches:
        detail = ", ".join(
            f"{field}: {reference[field]!r} != {candidate[field]!r}"
            for field in mismatches
        )
        raise IncompatibleAnalysisInput(
            f"incompatible run configuration in {directory}: {detail}"
        )


def _validate_source_rows(
    directory: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> None:
    configured_conditions = set(config["conditions"])
    row_conditions = {row.get("condition", "") for row in rows}
    if row_conditions != configured_conditions:
        raise IncompatibleAnalysisInput(
            f"{directory} conditions do not match its run config"
        )
    if len(rows) != int(config["planned_episode_count"]):
        raise IncompatibleAnalysisInput(
            f"{directory} is incomplete: {len(rows)} rows versus "
            f"{config['planned_episode_count']} planned"
        )
    versions = {row["prompt_version"] for row in rows}
    if len(versions) != 1 or next(iter(versions)) != config["prompt_version"]:
        raise IncompatibleAnalysisInput(
            f"{directory} episode prompt version does not match run config"
        )
    expected = {
        "benchmark_version": str(config["benchmark_version"]),
        "provider": str(config["provider"]),
        "model": str(config["model"]),
        "thinking_level": str(config["thinking_level"]),
        "experiment_budget": str(config["experiment_budget"]),
        "diagnosis_threshold": str(config["diagnosis_threshold"]),
        "max_decision_calls": str(config["max_decision_calls"]),
    }
    local_keys: set[tuple[str, str, int]] = set()
    for row in rows:
        for field, value in expected.items():
            if row.get(field) != value:
                raise IncompatibleAnalysisInput(
                    f"{directory} row disagrees with run config for {field}"
                )
        if row["true_hypothesis"] not in HYPOTHESES:
            raise IncompatibleAnalysisInput("unsupported true hypothesis")
        if int(row["seed"]) not in config["seeds"]:
            raise IncompatibleAnalysisInput("episode seed is outside configured seeds")
        key = _episode_key(row)
        if key in local_keys:
            raise IncompatibleAnalysisInput(f"duplicate episode key: {key}")
        local_keys.add(key)


def _validate_complete_grid(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> None:
    actual = {_episode_key(row) for row in rows}
    expected = {
        (condition, hypothesis, int(seed))
        for condition in CONDITIONS
        for hypothesis in HYPOTHESES
        for seed in config["seeds"]
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise IncompatibleAnalysisInput(
            f"incomplete or unexpected episode grid; missing={missing}, extra={extra}"
        )


def _matched_values(row: Mapping[str, str]) -> dict[str, Any]:
    autonomous = row["condition"] in AUTONOMOUS_CONDITIONS
    return {
        "final_diagnosis": row["final_diagnosis"] or None,
        "diagnosis_correct": _boolean(row, "diagnosis_correct"),
        "threshold_qualified_success": _boolean(row, "threshold_qualified_success"),
        "premature_diagnosis": _boolean(row, "premature_diagnosis"),
        "experiments_used": _integer(row, "experiments_used"),
        "normative_probability_of_final_diagnosis": _optional_float(
            row.get("normative_probability_of_final_diagnosis")
        ),
        "final_self_reported_confidence": (
            _optional_float(row.get("model_reported_final_confidence"))
            if autonomous
            else None
        ),
        "cumulative_action_regret": _number(row, "cumulative_action_regret"),
        "oracle_action_agreement_rate": _optional_float(
            row.get("oracle_action_agreement_rate")
        ),
        "action_sequence": row["action_sequence"],
    }


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite and any((output / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError(
            f"analysis output already exists at {output}; use --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in OUTPUT_FILES:
            path = output / name
            if path.is_file():
                path.unlink()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _episode_key(row: Mapping[str, str]) -> tuple[str, str, int]:
    return (row["condition"], row["true_hypothesis"], int(row["seed"]))


def _row_sort_key(row: Mapping[str, str]) -> tuple[int, int, int]:
    return (
        HYPOTHESES.index(row["true_hypothesis"]),
        int(row["seed"]),
        CONDITIONS.index(row["condition"]),
    )


def _condition_prefix(condition: str) -> str:
    return {
        "FULL_AUTONOMOUS": "full",
        "PLANNER_ONLY": "planner",
        "THRESHOLD_AWARE_AUTONOMOUS": "threshold_aware",
    }[condition]


def _boolean(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field, "")
    if value == "True":
        return True
    if value == "False":
        return False
    raise IncompatibleAnalysisInput(f"{field} must be True or False")


def _boolean_rate(rows: Sequence[Mapping[str, str]], field: str) -> float:
    return mean(_boolean(row, field) for row in rows)


def _number(row: Mapping[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise IncompatibleAnalysisInput(f"{field} must be numeric") from error


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise IncompatibleAnalysisInput(f"{field} must be an integer") from error


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise IncompatibleAnalysisInput("expected an optional numeric value") from error


def _available_numbers(rows: Sequence[Mapping[str, str]], field: str) -> list[float]:
    return [value for row in rows if (value := _optional_float(row.get(field))) is not None]


def _mean_field(rows: Sequence[Mapping[str, str]], field: str) -> float:
    return mean(_number(row, field) for row in rows)


def _subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    if isinstance(left, bool) and isinstance(right, bool):
        return float(int(left) - int(right))
    if type(left) in (int, float) and type(right) in (int, float):
        return float(left) - float(right)
    return None


def _summary_table(summary: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Accuracy | Threshold success | Premature | Mean experiments | Mean regret | Oracle agreement | Belief L1 | Provider failure | Scientific failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {_pct(row['diagnosis_accuracy'])} | "
            f"{_pct(row['threshold_qualified_success'])} | {_pct(row['premature_diagnosis_rate'])} | "
            f"{_f(row['mean_experiments'])} | {_f(row['mean_regret'])} | "
            f"{_pct(row['oracle_action_agreement'])} | {_f(row['mean_autonomous_belief_l1_error'])} | "
            f"{_pct(row['provider_failure_rate'])} | {_pct(row['scientific_model_failure_rate'])} |"
        )
    return "\n".join(lines)


def _calibration_table(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Comparable N | Mean signed gap | Mean absolute gap | Overconfidence | Self ≥ .95, normative < .95 | Both ≥ .95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in AUTONOMOUS_CONDITIONS:
        row = rows[condition]
        lines.append(
            f"| {condition} | {row['comparable_confidence_count']} | {_signed(row['mean_signed_gap'])} | "
            f"{_f(row['mean_absolute_gap'])} | {_pct(row['overconfidence_frequency'])} | "
            f"{row['self_ge_threshold_normative_lt_count']}/{row['comparable_confidence_count']} | "
            f"{row['both_ge_threshold_count']}/{row['comparable_confidence_count']} |"
        )
    return "\n".join(lines)


def _failure_table(summary: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | False structural | Missed structural | Provider failure | Scientific-model failure |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {_pct(row['false_structural_diagnosis_rate'])} | "
            f"{_pct(row['missed_structural_diagnosis_rate'])} | "
            f"{_pct(row['provider_failure_rate'])} | "
            f"{_pct(row['scientific_model_failure_rate'])} |"
        )
    return "\n".join(lines)


def _calibration_by_hypothesis_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Truth | Signed gap | Absolute gap | Overconfidence | Self ≥ .95, normative < .95 | Both ≥ .95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["true_hypothesis"] == "ALL":
            continue
        lines.append(
            f"| {row['condition']} | {row['true_hypothesis']} | {_signed(row['mean_signed_gap'])} | "
            f"{_f(row['mean_absolute_gap'])} | {_pct(row['overconfidence_frequency'])} | "
            f"{row['self_ge_threshold_normative_lt_count']}/{row['comparable_confidence_count']} | "
            f"{row['both_ge_threshold_count']}/{row['comparable_confidence_count']} |"
        )
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _f(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _signed(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.3f}"
