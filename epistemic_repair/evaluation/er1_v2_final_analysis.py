"""Offline final analysis for disjoint ER-1 V2 three-condition datasets."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from epistemic_repair.evaluation.er1_v2_three_condition_analysis import (
    AUTONOMOUS_CONDITIONS,
    CONDITIONS,
    HYPOTHESES,
    IncompatibleAnalysisInput,
)


FINAL_OUTPUT_FILES = (
    "episodes_final.csv",
    "summary_final.csv",
    "per_hypothesis_final.csv",
    "matched_final.csv",
    "calibration_final.csv",
    "failure_analysis.csv",
    "report.md",
)
COMMON_CONFIGURATION_FIELDS = (
    "benchmark_version",
    "provider",
    "model",
    "thinking_level",
    "experiment_budget",
    "diagnosis_threshold",
    "max_decision_calls",
)
CELL_CATEGORIES = (
    "VALID_EPISODE",
    "DIAGNOSTIC_ERROR",
    "SCIENTIFIC_MODEL_FAILURE",
    "PROVIDER_FAILURE",
    "RATE_LIMIT_FAILURE",
)
MATCHED_FIELDS = (
    "protocol_status",
    "analysis_category",
    "final_diagnosis",
    "diagnosis_correct",
    "threshold_qualified_success",
    "premature_diagnosis",
    "experiments_used",
    "normative_probability_of_final_diagnosis",
    "self_reported_confidence",
    "cumulative_action_regret",
    "oracle_action_agreement_rate",
    "action_sequence",
)
PAIRINGS = (
    ("planner_vs_full", "PLANNER_ONLY", "FULL_AUTONOMOUS"),
    (
        "threshold_aware_vs_full",
        "THRESHOLD_AWARE_AUTONOMOUS",
        "FULL_AUTONOMOUS",
    ),
    (
        "threshold_aware_vs_planner",
        "THRESHOLD_AWARE_AUTONOMOUS",
        "PLANNER_ONLY",
    ),
)


def combine_final_analysis(
    input_directories: Sequence[Path | str],
    output_directory: Path | str,
    *,
    expected_seeds: Sequence[int] = tuple(range(10)),
    pilot_seeds: Sequence[int] = (0, 1),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate disjoint source datasets and produce the final offline analysis."""
    if len(input_directories) < 2:
        raise IncompatibleAnalysisInput("at least two source datasets are required")
    expected_seed_tuple = tuple(sorted(set(int(seed) for seed in expected_seeds)))
    pilot_seed_set = frozenset(int(seed) for seed in pilot_seeds)
    if not expected_seed_tuple or not pilot_seed_set <= set(expected_seed_tuple):
        raise IncompatibleAnalysisInput("pilot seeds must be a nonempty subset of expected seeds")

    inputs = tuple(Path(directory) for directory in input_directories)
    output = Path(output_directory)
    _prepare_output(output, overwrite=overwrite)

    all_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    reference: dict[str, Any] | None = None
    prompt_versions: dict[str, str] = {}
    source_validations: list[dict[str, Any]] = []

    for directory in inputs:
        source = _read_source(directory)
        if reference is None:
            reference = source["configuration"]
        else:
            _validate_compatible(reference, source["configuration"], directory)
        for condition, version in source["prompt_versions"].items():
            prior = prompt_versions.setdefault(condition, version)
            if prior != version:
                raise IncompatibleAnalysisInput(
                    f"condition {condition} has incompatible prompt versions: "
                    f"{prior!r} and {version!r}"
                )
        for source_row in source["rows"]:
            key = _episode_key(source_row)
            if key in seen_keys:
                raise IncompatibleAnalysisInput(f"duplicate episode key: {key}")
            seen_keys.add(key)
            row = dict(source_row)
            protocol_status, category = classify_episode(row)
            row["protocol_status"] = protocol_status
            row["analysis_category"] = category
            row["source_dataset"] = directory.name
            all_rows.append(row)
        source_validations.append({
            "directory": directory.name,
            "row_count": len(source["rows"]),
            "conditions": sorted(source["conditions"]),
            "seeds": sorted(source["seeds"]),
        })

    assert reference is not None
    if set(prompt_versions) != set(CONDITIONS):
        raise IncompatibleAnalysisInput(
            f"prompt versions missing for conditions: {sorted(set(CONDITIONS) - set(prompt_versions))}"
        )
    _validate_final_grid(all_rows, expected_seed_tuple)
    ordered = sorted(all_rows, key=_row_sort_key)

    split_seeds = {
        "PILOT": tuple(seed for seed in expected_seed_tuple if seed in pilot_seed_set),
        "REPLICATION": tuple(seed for seed in expected_seed_tuple if seed not in pilot_seed_set),
        "COMBINED": expected_seed_tuple,
    }
    summary = build_split_summary(ordered, split_seeds)
    per_hypothesis = summarize_groups(
        ordered,
        analysis_split="COMBINED",
        by_hypothesis=True,
    )
    matched = build_matched_final(ordered)
    threshold = float(reference["diagnosis_threshold"])
    calibration = build_calibration(ordered, threshold)
    failure_analysis = build_failure_analysis(ordered, split_seeds)

    _write_csv(output / "episodes_final.csv", ordered)
    _write_csv(output / "summary_final.csv", summary)
    _write_csv(output / "per_hypothesis_final.csv", per_hypothesis)
    _write_csv(output / "matched_final.csv", matched)
    _write_csv(output / "calibration_final.csv", calibration)
    _write_csv(output / "failure_analysis.csv", failure_analysis)
    report = build_final_report(
        inputs=inputs,
        configuration=reference,
        rows=ordered,
        summary=summary,
        per_hypothesis=per_hypothesis,
        calibration=calibration,
        prompt_versions=prompt_versions,
        source_validations=source_validations,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return {
        "rows": ordered,
        "summary": summary,
        "per_hypothesis": per_hypothesis,
        "matched": matched,
        "calibration": calibration,
        "failure_analysis": failure_analysis,
        "prompt_versions": prompt_versions,
        "source_validations": source_validations,
    }


def classify_episode(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return protocol validity and a mutually exclusive analysis category."""
    diagnosed = str(row.get("termination_reason", "")) == "DIAGNOSED"
    diagnosis = str(row.get("final_diagnosis", ""))
    if diagnosed and diagnosis in HYPOTHESES:
        correct = _boolean(row, "diagnosis_correct")
        return "VALID_EPISODE", "VALID_EPISODE" if correct else "DIAGNOSTIC_ERROR"

    rate_limited = _boolean(row, "provider_rate_limit_failure_flag", default=False)
    provider_failed = _boolean(row, "provider_failure_flag", default=False)
    scientific_failed = _boolean(row, "scientific_model_failure_flag", default=False)
    if scientific_failed and provider_failed:
        raise IncompatibleAnalysisInput(
            "a cell cannot be both scientific-model and provider failure"
        )
    if rate_limited and not provider_failed:
        raise IncompatibleAnalysisInput("rate-limit failure must also be a provider failure")
    if rate_limited:
        return "RATE_LIMIT_FAILURE", "RATE_LIMIT_FAILURE"
    if provider_failed:
        return "PROVIDER_FAILURE", "PROVIDER_FAILURE"
    if scientific_failed:
        return "SCIENTIFIC_MODEL_FAILURE", "SCIENTIFIC_MODEL_FAILURE"
    raise IncompatibleAnalysisInput(
        "unclassified non-provider episode without an evaluable diagnosis"
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    if total == 0:
        return None, None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return center - half_width, center + half_width


def build_split_summary(
    rows: Sequence[Mapping[str, Any]],
    split_seeds: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Summarize pilot, replication, and combined cells independently."""
    output: list[dict[str, Any]] = []
    for split_name in ("PILOT", "REPLICATION", "COMBINED"):
        seeds = set(split_seeds[split_name])
        selected = [row for row in rows if int(row["seed"]) in seeds]
        output.extend(summarize_groups(selected, analysis_split=split_name))
    return output


def summarize_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_split: str,
    by_hypothesis: bool = False,
) -> list[dict[str, Any]]:
    """Compute planned-cell and valid-episode metrics with Wilson intervals."""
    groups: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["true_hypothesis"]) if by_hypothesis else None,
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        hypotheses: Iterable[str | None] = HYPOTHESES if by_hypothesis else (None,)
        for hypothesis in hypotheses:
            group = groups.get((condition, hypothesis), [])
            if not group:
                continue
            result = _summarize_group(group)
            prefix = {
                "analysis_split": analysis_split,
                "condition": condition,
            }
            if by_hypothesis:
                prefix["true_hypothesis"] = hypothesis
            output.append({**prefix, **result})
    return output


def _summarize_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    planned = len(group)
    valid = [row for row in group if row["protocol_status"] == "VALID_EPISODE"]
    correct = sum(_boolean(row, "diagnosis_correct") for row in valid)
    diagnostic_errors = len(valid) - correct
    scientific = sum(row["analysis_category"] == "SCIENTIFIC_MODEL_FAILURE" for row in group)
    provider = sum(row["analysis_category"] == "PROVIDER_FAILURE" for row in group)
    rate_limit = sum(row["analysis_category"] == "RATE_LIMIT_FAILURE" for row in group)
    threshold_success_all = sum(_boolean(row, "threshold_qualified_success", default=False) for row in group)
    threshold_success_valid = sum(_boolean(row, "threshold_qualified_success", default=False) for row in valid)
    premature_all = sum(_boolean(row, "premature_diagnosis", default=False) for row in group)
    premature_valid = sum(_boolean(row, "premature_diagnosis", default=False) for row in valid)
    experiments_all = sum(_integer(row, "experiments_used", default=0) for row in group)
    experiments_valid = sum(_integer(row, "experiments_used", default=0) for row in valid)
    agreements_all = sum(_integer(row, "oracle_action_agreements", default=0) for row in group)
    agreements_valid = sum(_integer(row, "oracle_action_agreements", default=0) for row in valid)
    no_change = [row for row in valid if row["true_hypothesis"] == "NO_STRUCTURAL_CHANGE"]
    structural = [row for row in valid if row["true_hypothesis"] != "NO_STRUCTURAL_CHANGE"]
    beliefs = _available_numbers(valid, "mean_autonomous_belief_l1_error")
    result: dict[str, Any] = {
        "planned_cells": planned,
        "valid_episodes": len(valid),
        "diagnostic_errors": diagnostic_errors,
        "scientific_model_failures": scientific,
        "provider_failures": provider,
        "rate_limit_failures": rate_limit,
        "planned_cell_accuracy": correct / planned,
        "valid_episode_accuracy": _rate(correct, len(valid)),
        "threshold_qualified_success_all_cells": threshold_success_all / planned,
        "threshold_success_valid_episodes": _rate(threshold_success_valid, len(valid)),
        "premature_diagnosis_all_cells": premature_all / planned,
        "premature_diagnosis_valid_episodes": _rate(premature_valid, len(valid)),
        "mean_experiments_all_cells": _mean_numeric(group, "experiments_used"),
        "mean_experiments_valid_episodes": _mean_numeric(valid, "experiments_used"),
        "mean_regret_all_cells": _mean_numeric(group, "cumulative_action_regret"),
        "mean_regret_valid_episodes": _mean_numeric(valid, "cumulative_action_regret"),
        "oracle_agreement_all_actions": _rate(agreements_all, experiments_all),
        "oracle_agreement_valid_actions": _rate(agreements_valid, experiments_valid),
        "false_structural_diagnosis_valid_rate": (
            mean(str(row.get("final_diagnosis", "")) != "NO_STRUCTURAL_CHANGE" for row in no_change)
            if no_change else None
        ),
        "missed_structural_diagnosis_valid_rate": (
            mean(str(row.get("final_diagnosis", "")) == "NO_STRUCTURAL_CHANGE" for row in structural)
            if structural else None
        ),
        "mean_autonomous_belief_l1_error_valid": mean(beliefs) if beliefs else None,
    }
    for name, successes, total in (
        ("planned_cell_accuracy", correct, planned),
        ("valid_episode_accuracy", correct, len(valid)),
        ("threshold_qualified_success_all_cells", threshold_success_all, planned),
        ("premature_diagnosis_all_cells", premature_all, planned),
        ("scientific_model_failure_rate", scientific, planned),
        ("threshold_success_valid_episodes", threshold_success_valid, len(valid)),
        ("premature_diagnosis_valid_episodes", premature_valid, len(valid)),
    ):
        low, high = wilson_interval(successes, total)
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high
    result["scientific_model_failure_rate"] = scientific / planned
    result["provider_failure_rate_excluding_rate_limits"] = provider / planned
    result["rate_limit_failure_rate"] = rate_limit / planned
    return result


def build_matched_final(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create 40 matched cases and protocol-aware pairwise comparisons."""
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["true_hypothesis"]), int(row["seed"]))
        conditions = grouped.setdefault(key, {})
        condition = str(row["condition"])
        if condition in conditions:
            raise IncompatibleAnalysisInput(f"duplicate matched cell: {key + (condition,)}")
        conditions[condition] = row

    output: list[dict[str, Any]] = []
    for hypothesis in HYPOTHESES:
        seeds = sorted(seed for truth, seed in grouped if truth == hypothesis)
        for seed in seeds:
            cells = grouped[(hypothesis, seed)]
            if set(cells) != set(CONDITIONS):
                raise IncompatibleAnalysisInput(
                    f"matched key {(hypothesis, seed)} is not a complete triplet"
                )
            item: dict[str, Any] = {
                "true_hypothesis": hypothesis,
                "seed": seed,
                "match_key": f"{hypothesis}:{seed}",
            }
            values: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                prefix = _condition_prefix(condition)
                values[condition] = _matched_values(cells[condition])
                for field in MATCHED_FIELDS:
                    item[f"{prefix}_{field}"] = values[condition][field]
            for label, left_name, right_name in PAIRINGS:
                left = values[left_name]
                right = values[right_name]
                left_valid = left["protocol_status"] == "VALID_EPISODE"
                right_valid = right["protocol_status"] == "VALID_EPISODE"
                item[f"{label}_protocol_comparison"] = _comparison_status(
                    left_valid, right_valid
                )
                if left_valid and right_valid:
                    item[f"{label}_diagnosis_correct_delta"] = (
                        int(left["diagnosis_correct"]) - int(right["diagnosis_correct"])
                    )
                    item[f"{label}_diagnostic_outcome"] = _diagnostic_outcome(left, right)
                    for field in (
                        "threshold_qualified_success",
                        "premature_diagnosis",
                        "experiments_used",
                        "cumulative_action_regret",
                        "oracle_action_agreement_rate",
                    ):
                        item[f"{label}_{field}_delta"] = _numeric_delta(
                            left[field], right[field]
                        )
                else:
                    item[f"{label}_diagnosis_correct_delta"] = None
                    item[f"{label}_diagnostic_outcome"] = "NOT_COMPARABLE_PROTOCOL_FAILURE"
                    for field in (
                        "threshold_qualified_success",
                        "premature_diagnosis",
                        "experiments_used",
                        "cumulative_action_regret",
                        "oracle_action_agreement_rate",
                    ):
                        item[f"{label}_{field}_delta"] = None
            output.append(item)
    return output


def build_calibration(
    rows: Sequence[Mapping[str, Any]], diagnosis_threshold: float
) -> list[dict[str, Any]]:
    """Summarize confidence calibration and stopping only on valid episodes."""
    output: list[dict[str, Any]] = []
    for condition in AUTONOMOUS_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for hypothesis in ("ALL", *HYPOTHESES):
            selected = [
                row for row in condition_rows
                if hypothesis == "ALL" or row["true_hypothesis"] == hypothesis
            ]
            valid = [row for row in selected if row["protocol_status"] == "VALID_EPISODE"]
            comparable: list[tuple[float, float]] = []
            for row in valid:
                self_confidence = _optional_float(row.get("model_reported_final_confidence"))
                normative = _optional_float(row.get("normative_probability_of_final_diagnosis"))
                if self_confidence is not None and normative is not None:
                    comparable.append((self_confidence, normative))
            gaps = [self_confidence - normative for self_confidence, normative in comparable]
            over = sum(gap > 1e-12 for gap in gaps)
            under = sum(gap < -1e-12 for gap in gaps)
            self_only = sum(
                self_confidence >= diagnosis_threshold and normative < diagnosis_threshold
                for self_confidence, normative in comparable
            )
            both = sum(
                self_confidence >= diagnosis_threshold and normative >= diagnosis_threshold
                for self_confidence, normative in comparable
            )
            normative_only = sum(
                normative >= diagnosis_threshold and self_confidence < diagnosis_threshold
                for self_confidence, normative in comparable
            )
            premature = sum(_boolean(row, "premature_diagnosis", default=False) for row in valid)
            threshold_success = sum(
                _boolean(row, "threshold_qualified_success", default=False) for row in valid
            )
            output.append({
                "condition": condition,
                "true_hypothesis": hypothesis,
                "planned_cells": len(selected),
                "valid_episodes": len(valid),
                "comparable_confidence_count": len(comparable),
                "missing_confidence_on_valid_count": len(valid) - len(comparable),
                "mean_signed_gap": mean(gaps) if gaps else None,
                "mean_absolute_gap": mean(abs(gap) for gap in gaps) if gaps else None,
                "overconfidence_count": over,
                "overconfidence_frequency": _rate(over, len(comparable)),
                "underconfidence_count": under,
                "underconfidence_frequency": _rate(under, len(comparable)),
                "self_ge_threshold_normative_lt_count": self_only,
                "self_ge_threshold_normative_lt_rate": _rate(self_only, len(comparable)),
                "both_ge_threshold_count": both,
                "both_ge_threshold_rate": _rate(both, len(comparable)),
                "normative_ge_threshold_self_lt_count": normative_only,
                "normative_ge_threshold_self_lt_rate": _rate(normative_only, len(comparable)),
                "premature_diagnosis_valid_count": premature,
                "premature_diagnosis_valid_rate": _rate(premature, len(valid)),
                "threshold_success_valid_count": threshold_success,
                "threshold_success_valid_rate": _rate(threshold_success, len(valid)),
            })
    return output


def build_failure_analysis(
    rows: Sequence[Mapping[str, Any]],
    split_seeds: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Report the mutually exclusive cell taxonomy by split and hypothesis."""
    output: list[dict[str, Any]] = []
    for split_name in ("PILOT", "REPLICATION", "COMBINED"):
        seeds = set(split_seeds[split_name])
        split_rows = [row for row in rows if int(row["seed"]) in seeds]
        for condition in CONDITIONS:
            condition_rows = [row for row in split_rows if row["condition"] == condition]
            for hypothesis in ("ALL", *HYPOTHESES):
                selected = [
                    row for row in condition_rows
                    if hypothesis == "ALL" or row["true_hypothesis"] == hypothesis
                ]
                valid = sum(row["protocol_status"] == "VALID_EPISODE" for row in selected)
                counts = {
                    category: sum(row["analysis_category"] == category for row in selected)
                    for category in CELL_CATEGORIES
                }
                output.append({
                    "analysis_split": split_name,
                    "condition": condition,
                    "true_hypothesis": hypothesis,
                    "planned_cells": len(selected),
                    "valid_episodes_including_diagnostic_errors": valid,
                    "valid_correct_episodes": counts["VALID_EPISODE"],
                    "diagnostic_errors": counts["DIAGNOSTIC_ERROR"],
                    "scientific_model_failures": counts["SCIENTIFIC_MODEL_FAILURE"],
                    "provider_failures_excluding_rate_limits": counts["PROVIDER_FAILURE"],
                    "rate_limit_failures": counts["RATE_LIMIT_FAILURE"],
                })
    return output


def build_final_report(
    *,
    inputs: Sequence[Path],
    configuration: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    per_hypothesis: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    prompt_versions: Mapping[str, str],
    source_validations: Sequence[Mapping[str, Any]],
) -> str:
    """Render a technical report that labels observations and interpretations."""
    combined = _index_summary(summary, "COMBINED")
    pilot = _index_summary(summary, "PILOT")
    replication = _index_summary(summary, "REPLICATION")
    calibration_overall = {
        row["condition"]: row for row in calibration if row["true_hypothesis"] == "ALL"
    }
    total_valid = sum(row["valid_episodes"] for row in combined.values())
    total_scientific = sum(row["scientific_model_failures"] for row in combined.values())
    total_provider = sum(row["provider_failures"] for row in combined.values())
    total_rate_limit = sum(row["rate_limit_failures"] for row in combined.values())
    planner = combined["PLANNER_ONLY"]
    full = combined["FULL_AUTONOMOUS"]
    aware = combined["THRESHOLD_AWARE_AUTONOMOUS"]
    planner_regret_delta = planner["mean_regret_valid_episodes"] - full["mean_regret_valid_episodes"]
    planner_oracle_delta = planner["oracle_agreement_valid_actions"] - full["oracle_agreement_valid_actions"]
    aware_premature_delta = (
        aware["premature_diagnosis_valid_episodes"]
        - full["premature_diagnosis_valid_episodes"]
    )
    aware_calibration = calibration_overall["THRESHOLD_AWARE_AUTONOMOUS"]

    lines = [
        "# ER-1 V2 seeds 0–9 final three-condition analysis",
        "",
        "## Technical summary",
        "",
        f"The frozen analysis contains {len(rows)} planned cells, {total_valid} valid episodes, "
        f"{total_scientific} scientific-model failures, {total_provider} non-rate-limit provider failures, "
        f"and {total_rate_limit} rate-limit failures. Scientific-model failures remain in all planned-cell denominators.",
        "",
        f"**Directly observed:** planner-only changed valid-episode mean regret by {_signed(planner_regret_delta)} "
        f"and action-weighted oracle agreement by {_signed(planner_oracle_delta)} relative to full autonomy. "
        f"Threshold awareness changed valid-episode premature diagnosis by {_signed(aware_premature_delta)}.",
        "",
        f"**Directional interpretation:** authoritative beliefs primarily affect experiment selection, while explicit threshold knowledge affects stopping. "
        f"Threshold awareness did not eliminate premature stopping: {_count_rate(aware, 'premature_diagnosis_valid_episodes', 'valid_episodes')} of its valid episodes remained premature.",
        "",
        f"**Requires more replication:** confidence calibration is a plausible bottleneck, but the combined sample is still small and comes from one model/configuration. "
        f"Threshold-aware autonomy reported confidence ≥ {float(configuration['diagnosis_threshold']):.2f} without matching normative support in "
        f"{aware_calibration['self_ge_threshold_normative_lt_count']}/{aware_calibration['comparable_confidence_count']} comparable valid episodes.",
        "",
        "## Combined outcomes retain protocol failures in the denominator",
        "",
        _overall_table(combined),
        "",
        "Planned-cell accuracy treats every protocol/provider failure as unsuccessful. Valid-episode accuracy conditions on completing the protocol with an evaluable diagnosis; a wrong diagnosis is a diagnostic error within that valid set. Provider failures exclude rate limits so all failure counts are mutually exclusive.",
        "",
        "### Wilson 95% intervals for key combined proportions",
        "",
        _interval_table(combined),
        "",
        "The intervals quantify binomial uncertainty only. Their overlap or non-overlap is not used as a significance test, and the matched design is not modeled by these marginal intervals.",
        "",
        "## Calibration and stopping remain distinct",
        "",
        _calibration_table(calibration_overall),
        "",
        "Self-reported confidence minus normative probability is computed only for valid autonomous episodes with both values present. Planner-only is excluded because its stored confidence is benchmark-derived rather than an autonomous self-estimate.",
        "",
        "### Calibration and stopping by hypothesis",
        "",
        _calibration_hypothesis_table(calibration),
        "",
        "## Pilot and replication consistency",
        "",
        _split_table(summary, calibration, rows),
        "",
        *_directional_consistency_lines(pilot, replication, combined, rows),
        "",
        "## Hypothesis-level difficulty",
        "",
        _hypothesis_table(per_hypothesis),
        "",
        *_hardest_hypothesis_lines(per_hypothesis),
        "",
        "## Answers to the scientific questions",
        "",
        f"- **A. Does authoritative normative belief information improve diagnosis?** Directly observed planned-cell accuracy was {_pct(planner['planned_cell_accuracy'])} for planner-only and {_pct(full['planned_cell_accuracy'])} for full autonomy; valid-episode accuracy was {_pct(planner['valid_episode_accuracy'])} versus {_pct(full['valid_episode_accuracy'])}. This is descriptive, not a causal or significant difference claim.",
        f"- **B. Does it improve experiment selection?** Directionally, {_direction_word(planner_regret_delta < 0 and planner_oracle_delta > 0)}: planner-only had {_f(abs(planner_regret_delta))} lower valid-episode mean regret and {_pct(abs(planner_oracle_delta))} higher oracle agreement than full autonomy." if planner_regret_delta < 0 and planner_oracle_delta > 0 else f"- **B. Does it improve experiment selection?** The two experiment-quality measures did not both improve: regret delta {_signed(planner_regret_delta)}, oracle-agreement delta {_signed(planner_oracle_delta)}.",
        f"- **C. Does explicit threshold awareness reduce premature stopping?** Directly observed valid-episode prematurity changed from {_pct(full['premature_diagnosis_valid_episodes'])} to {_pct(aware['premature_diagnosis_valid_episodes'])}; the directional change is {_signed(aware_premature_delta)}.",
        f"- **D. Does threshold awareness eliminate premature stopping?** No. {_count_rate(aware, 'premature_diagnosis_valid_episodes', 'valid_episodes')} valid threshold-aware episodes were premature.",
        f"- **E. Is confidence calibration plausibly a bottleneck?** Plausibly, yes: the mean absolute gap was {_f(aware_calibration['mean_absolute_gap'])}, and {aware_calibration['self_ge_threshold_normative_lt_count']} episodes crossed the self-reported threshold without normative support. This mechanism claim requires further replication.",
        f"- **F. Are protocol failures concentrated?** {_protocol_concentration(rows)}",
        "- **G. Which hypothesis appears hardest?** The hypothesis table and notes above separate diagnostic difficulty from experiment-selection difficulty; ties are retained rather than forced into a single winner.",
        "- **H. Which weaknesses are separable?** Planner-only versus full autonomy isolates belief support for planning; threshold-aware versus full autonomy isolates threshold knowledge for stopping; calibration gaps isolate confidence alignment; and the explicit protocol taxonomy isolates structured-output compliance. The design makes these descriptive contrasts separable, but it does not by itself identify a unique causal mechanism.",
        "",
        "## Scope, validation, and metric definitions",
        "",
        f"The final grid is exactly 10 seeds × 4 hypotheses × 3 conditions = {len(rows)} cells. "
        f"All rows match benchmark `{configuration['benchmark_version']}`, provider/model `{configuration['provider']}/{configuration['model']}`, "
        f"thinking level `{configuration['thinking_level']}`, budget {configuration['experiment_budget']}, diagnosis threshold {configuration['diagnosis_threshold']}, and max decision calls {configuration['max_decision_calls']}.",
        "",
        "The hypotheses are `NO_STRUCTURAL_CHANGE`, `WORLD_SHIFT`, `SENSOR_CORRUPTION`, and `MISSING_LATENT_VARIABLE`. Matched comparisons use `(true_hypothesis, seed)` and require all three conditions. Duplicate cell keys are rejected.",
        "",
        "Prompt versions were validated per condition:",
        "",
        *[f"- `{condition}`: `{prompt_versions[condition]}`" for condition in CONDITIONS],
        "",
        "Source validation:",
        "",
        *[
            f"- `{item['directory']}`: {item['row_count']} rows, seeds {item['seeds']}, conditions {item['conditions']}"
            for item in source_validations
        ],
        "",
        "Threshold-qualified success requires a correct diagnosis whose normative support reaches the frozen evaluator threshold. Premature diagnosis means the final diagnosis was issued below that normative threshold. Regret and oracle agreement use the frozen episode-level evaluator outputs; oracle agreement is action-weighted. Wilson intervals use the stated binomial denominator and do not account for matched-cell dependence.",
        "",
        "## Limitations and next steps",
        "",
        "This is a single-model, single-thinking-level study with ten seeds per hypothesis. The analysis is descriptive; no prompt, benchmark, evaluator, or episode was changed, and no API call was made. Three strict protocol failures reduce the valid threshold-aware sample and must not be reinterpreted as diagnostic errors.",
        "",
        "A defensible next step is to freeze this artifact and pre-specify any larger replication or second-model-family study before collecting more data. Do not tune prompts or thresholds against these results without labeling that work as a new experimental stage.",
        "",
        "## Further questions",
        "",
        "- Do the planning and stopping patterns persist across another named model family?",
        "- Do confidence gaps predict premature stopping within matched cases when the sample is larger?",
        "- Are strict structured-output failures stable across repeated runs, or are they transient model-interface behavior?",
        "",
        "Exact row-level evidence is preserved in `episodes_final.csv`, `matched_final.csv`, `calibration_final.csv`, and `failure_analysis.csv`.",
    ]
    return "\n".join(lines) + "\n"


def _read_source(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise IncompatibleAnalysisInput(f"source directory does not exist: {directory}")
    combined_path = directory / "episodes_combined.csv"
    episode_path = directory / "episodes.csv"
    if combined_path.is_file():
        data_path = combined_path
    elif episode_path.is_file():
        data_path = episode_path
    else:
        raise IncompatibleAnalysisInput(
            f"{directory} contains neither episodes_combined.csv nor episodes.csv"
        )
    with data_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise IncompatibleAnalysisInput(f"{data_path} contains no rows")
    required = {
        *COMMON_CONFIGURATION_FIELDS,
        "prompt_version",
        "condition",
        "true_hypothesis",
        "seed",
        "termination_reason",
        "final_diagnosis",
        "diagnosis_correct",
        "scientific_model_failure_flag",
        "provider_failure_flag",
        "provider_rate_limit_failure_flag",
    }
    missing = required - set(rows[0])
    if missing:
        raise IncompatibleAnalysisInput(f"{data_path} is missing columns: {sorted(missing)}")

    config_path = directory / "run_config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    configuration = _derive_configuration(rows)
    conditions = {str(row["condition"]) for row in rows}
    hypotheses = {str(row["true_hypothesis"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    prompt_versions = _derive_prompt_versions(rows)
    if not conditions <= set(CONDITIONS):
        raise IncompatibleAnalysisInput(f"{directory} contains unsupported conditions")
    if hypotheses != set(HYPOTHESES):
        raise IncompatibleAnalysisInput(
            f"{directory} hypothesis definitions are incompatible: {sorted(hypotheses)}"
        )
    _validate_source_grid(rows, conditions, seeds, directory)
    if run_config is not None:
        _validate_run_config(
            directory, run_config, configuration, conditions, hypotheses, seeds, len(rows)
        )
    return {
        "rows": rows,
        "configuration": configuration,
        "conditions": conditions,
        "hypotheses": hypotheses,
        "seeds": seeds,
        "prompt_versions": prompt_versions,
    }


def _derive_configuration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for field in COMMON_CONFIGURATION_FIELDS:
        values = {str(row.get(field, "")) for row in rows}
        if len(values) != 1:
            raise IncompatibleAnalysisInput(f"source rows disagree on {field}: {sorted(values)}")
        value = values.pop()
        if field in {"experiment_budget", "max_decision_calls"}:
            config[field] = int(value)
        elif field == "diagnosis_threshold":
            config[field] = float(value)
        else:
            config[field] = value
    return config


def _derive_prompt_versions(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        condition = str(row["condition"])
        version = str(row["prompt_version"])
        prior = output.setdefault(condition, version)
        if prior != version:
            raise IncompatibleAnalysisInput(
                f"condition {condition} has multiple prompt versions within a source"
            )
    return output


def _validate_run_config(
    directory: Path,
    run_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
    conditions: set[str],
    hypotheses: set[str],
    seeds: set[int],
    row_count: int,
) -> None:
    for field in COMMON_CONFIGURATION_FIELDS:
        if field not in run_config:
            raise IncompatibleAnalysisInput(f"{directory} run_config missing {field}")
        if str(run_config[field]) != str(configuration[field]):
            raise IncompatibleAnalysisInput(
                f"{directory} run_config disagrees with rows for {field}"
            )
    for name, actual, expected in (
        ("conditions", set(run_config.get("conditions", [])), conditions),
        ("hypotheses", set(run_config.get("hypotheses", [])), hypotheses),
        ("seeds", {int(seed) for seed in run_config.get("seeds", [])}, seeds),
    ):
        if actual != expected:
            raise IncompatibleAnalysisInput(f"{directory} run_config disagrees on {name}")
    if int(run_config.get("planned_episode_count", -1)) != row_count:
        raise IncompatibleAnalysisInput(
            f"{directory} planned episode count does not equal CSV row count"
        )


def _validate_compatible(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], directory: Path
) -> None:
    for field in COMMON_CONFIGURATION_FIELDS:
        if candidate[field] != reference[field]:
            raise IncompatibleAnalysisInput(
                f"{directory} is incompatible for {field}: "
                f"{candidate[field]!r} != {reference[field]!r}"
            )


def _validate_source_grid(
    rows: Sequence[Mapping[str, Any]],
    conditions: set[str],
    seeds: set[int],
    directory: Path,
) -> None:
    actual = {_episode_key(row) for row in rows}
    if len(actual) != len(rows):
        raise IncompatibleAnalysisInput(f"{directory} contains duplicate episode keys")
    expected = {
        (condition, hypothesis, seed)
        for condition in conditions
        for hypothesis in HYPOTHESES
        for seed in seeds
    }
    if actual != expected:
        raise IncompatibleAnalysisInput(
            f"{directory} does not contain a complete local condition/hypothesis/seed grid"
        )


def _validate_final_grid(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> None:
    actual = {_episode_key(row) for row in rows}
    expected = {
        (condition, hypothesis, seed)
        for condition in CONDITIONS
        for hypothesis in HYPOTHESES
        for seed in seeds
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise IncompatibleAnalysisInput(
            f"final grid mismatch; missing={missing}, extra={extra}"
        )


def _matched_values(row: Mapping[str, Any]) -> dict[str, Any]:
    autonomous = row["condition"] in AUTONOMOUS_CONDITIONS
    return {
        "protocol_status": row["protocol_status"],
        "analysis_category": row["analysis_category"],
        "final_diagnosis": row.get("final_diagnosis") or None,
        "diagnosis_correct": (
            _boolean(row, "diagnosis_correct")
            if row["protocol_status"] == "VALID_EPISODE" else None
        ),
        "threshold_qualified_success": (
            _boolean(row, "threshold_qualified_success", default=False)
            if row["protocol_status"] == "VALID_EPISODE" else None
        ),
        "premature_diagnosis": (
            _boolean(row, "premature_diagnosis", default=False)
            if row["protocol_status"] == "VALID_EPISODE" else None
        ),
        "experiments_used": _integer(row, "experiments_used", default=0),
        "normative_probability_of_final_diagnosis": _optional_float(
            row.get("normative_probability_of_final_diagnosis")
        ),
        "self_reported_confidence": (
            _optional_float(row.get("model_reported_final_confidence"))
            if autonomous else None
        ),
        "cumulative_action_regret": _optional_float(row.get("cumulative_action_regret")),
        "oracle_action_agreement_rate": _optional_float(row.get("oracle_action_agreement_rate")),
        "action_sequence": row.get("action_sequence", ""),
    }


def _comparison_status(left_valid: bool, right_valid: bool) -> str:
    if left_valid and right_valid:
        return "BOTH_VALID"
    if left_valid:
        return "RIGHT_PROTOCOL_FAILURE"
    if right_valid:
        return "LEFT_PROTOCOL_FAILURE"
    return "BOTH_PROTOCOL_FAILURE"


def _diagnostic_outcome(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    left_correct = bool(left["diagnosis_correct"])
    right_correct = bool(right["diagnosis_correct"])
    if left_correct and right_correct:
        return "BOTH_CORRECT"
    if left_correct:
        return "LEFT_ONLY_CORRECT"
    if right_correct:
        return "RIGHT_ONLY_CORRECT"
    return "BOTH_INCORRECT"


def _numeric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _index_summary(
    summary: Sequence[Mapping[str, Any]], split: str
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["condition"]): row
        for row in summary
        if row["analysis_split"] == split
    }


def _overall_table(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Planned | Valid | Diagnostic errors | Scientific failures | Provider failures | Rate limits | Planned accuracy | Valid accuracy | Threshold success (all) | Premature (valid) | Mean experiments (all) | Mean regret (all) | Oracle agreement (all actions) | False structural | Missed structural | Belief L1 (valid) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        row = rows[condition]
        lines.append(
            f"| {condition} | {row['planned_cells']} | {row['valid_episodes']} | {row['diagnostic_errors']} | "
            f"{row['scientific_model_failures']} | {row['provider_failures']} | {row['rate_limit_failures']} | "
            f"{_pct(row['planned_cell_accuracy'])} | {_pct(row['valid_episode_accuracy'])} | "
            f"{_pct(row['threshold_qualified_success_all_cells'])} | {_pct(row['premature_diagnosis_valid_episodes'])} | "
            f"{_f(row['mean_experiments_all_cells'])} | {_f(row['mean_regret_all_cells'])} | "
            f"{_pct(row['oracle_agreement_all_actions'])} | {_pct(row['false_structural_diagnosis_valid_rate'])} | "
            f"{_pct(row['missed_structural_diagnosis_valid_rate'])} | {_f(row['mean_autonomous_belief_l1_error_valid'])} |"
        )
    return "\n".join(lines)


def _interval_table(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Planned accuracy | Valid accuracy | Threshold success (all cells) | Premature (all cells) | Scientific failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        row = rows[condition]
        lines.append(
            f"| {condition} | {_pci(row, 'planned_cell_accuracy')} | {_pci(row, 'valid_episode_accuracy')} | "
            f"{_pci(row, 'threshold_qualified_success_all_cells')} | {_pci(row, 'premature_diagnosis_all_cells')} | "
            f"{_pci(row, 'scientific_model_failure_rate')} |"
        )
    return "\n".join(lines)


def _calibration_table(rows: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Valid | Comparable | Signed gap | Absolute gap | Overconfident | Underconfident | Self ≥ .95, normative < .95 | Both ≥ .95 | Normative ≥ .95, self < .95 | Premature (valid) | Threshold success (valid) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in AUTONOMOUS_CONDITIONS:
        row = rows[condition]
        denominator = row["comparable_confidence_count"]
        lines.append(
            f"| {condition} | {row['valid_episodes']} | {denominator} | {_signed(row['mean_signed_gap'])} | "
            f"{_f(row['mean_absolute_gap'])} | {_pct(row['overconfidence_frequency'])} | "
            f"{_pct(row['underconfidence_frequency'])} | {row['self_ge_threshold_normative_lt_count']}/{denominator} | "
            f"{row['both_ge_threshold_count']}/{denominator} | {row['normative_ge_threshold_self_lt_count']}/{denominator} | "
            f"{_pct(row['premature_diagnosis_valid_rate'])} | {_pct(row['threshold_success_valid_rate'])} |"
        )
    return "\n".join(lines)


def _calibration_hypothesis_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Hypothesis | Valid/comparable | Signed gap | Absolute gap | Overconfident | Underconfident | Self-only threshold | Both threshold | Normative-only threshold | Premature valid | Threshold success valid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["true_hypothesis"] == "ALL":
            continue
        lines.append(
            f"| {row['condition']} | {row['true_hypothesis']} | {row['valid_episodes']}/{row['comparable_confidence_count']} | "
            f"{_signed(row['mean_signed_gap'])} | {_f(row['mean_absolute_gap'])} | "
            f"{_pct(row['overconfidence_frequency'])} | {_pct(row['underconfidence_frequency'])} | "
            f"{row['self_ge_threshold_normative_lt_count']} | {row['both_ge_threshold_count']} | "
            f"{row['normative_ge_threshold_self_lt_count']} | {_pct(row['premature_diagnosis_valid_rate'])} | "
            f"{_pct(row['threshold_success_valid_rate'])} |"
        )
    return "\n".join(lines)


def _split_table(
    summary: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
) -> str:
    calibration_by_split: dict[tuple[str, str], float | None] = {}
    for split, seeds in (("PILOT", {0, 1}), ("REPLICATION", set(range(2, 10))), ("COMBINED", set(range(10)))):
        split_calibration = build_calibration(
            [row for row in all_rows if int(row["seed"]) in seeds], 0.95
        )
        for row in split_calibration:
            if row["true_hypothesis"] == "ALL":
                calibration_by_split[(split, row["condition"])] = row["mean_signed_gap"]
    lines = [
        "| Split | Condition | Planned accuracy | Valid accuracy | Threshold success (all) | Premature (valid) | Regret (valid) | Oracle agreement (valid) | Calibration signed gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("PILOT", "REPLICATION", "COMBINED"):
        for condition in CONDITIONS:
            row = next(
                item for item in summary
                if item["analysis_split"] == split and item["condition"] == condition
            )
            lines.append(
                f"| {split} | {condition} | {_pct(row['planned_cell_accuracy'])} | {_pct(row['valid_episode_accuracy'])} | "
                f"{_pct(row['threshold_qualified_success_all_cells'])} | {_pct(row['premature_diagnosis_valid_episodes'])} | "
                f"{_f(row['mean_regret_valid_episodes'])} | {_pct(row['oracle_agreement_valid_actions'])} | "
                f"{_signed(calibration_by_split.get((split, condition)))} |"
            )
    return "\n".join(lines)


def _directional_consistency_lines(
    pilot: Mapping[str, Mapping[str, Any]],
    replication: Mapping[str, Mapping[str, Any]],
    combined: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    patterns = []
    for label, metric, left, right, desirable_sign in (
        ("planner-only lower regret", "mean_regret_valid_episodes", "PLANNER_ONLY", "FULL_AUTONOMOUS", -1),
        ("planner-only higher oracle agreement", "oracle_agreement_valid_actions", "PLANNER_ONLY", "FULL_AUTONOMOUS", 1),
        ("threshold awareness lower prematurity", "premature_diagnosis_valid_episodes", "THRESHOLD_AWARE_AUTONOMOUS", "FULL_AUTONOMOUS", -1),
    ):
        pilot_delta = pilot[left][metric] - pilot[right][metric]
        replication_delta = replication[left][metric] - replication[right][metric]
        replicated = pilot_delta * desirable_sign > 0 and replication_delta * desirable_sign > 0
        patterns.append(
            f"- **{label}:** {'replicated directionally' if replicated else 'did not replicate directionally'} "
            f"(pilot delta {_signed(pilot_delta)}, replication delta {_signed(replication_delta)}, combined delta {_signed(combined[left][metric] - combined[right][metric])})."
        )
    pilot_cal = {
        row["condition"]: row for row in build_calibration(
            [row for row in rows if int(row["seed"]) in {0, 1}], 0.95
        ) if row["true_hypothesis"] == "ALL"
    }
    rep_cal = {
        row["condition"]: row for row in build_calibration(
            [row for row in rows if int(row["seed"]) in set(range(2, 10))], 0.95
        ) if row["true_hypothesis"] == "ALL"
    }
    pilot_delta = (
        pilot_cal["THRESHOLD_AWARE_AUTONOMOUS"]["mean_absolute_gap"]
        - pilot_cal["FULL_AUTONOMOUS"]["mean_absolute_gap"]
    )
    replication_delta = (
        rep_cal["THRESHOLD_AWARE_AUTONOMOUS"]["mean_absolute_gap"]
        - rep_cal["FULL_AUTONOMOUS"]["mean_absolute_gap"]
    )
    patterns.append(
        f"- **threshold-aware absolute calibration gap:** "
        f"{'replicated directionally' if pilot_delta < 0 and replication_delta < 0 else 'did not replicate directionally'} "
        f"(pilot delta {_signed(pilot_delta)}, replication delta {_signed(replication_delta)})."
    )
    patterns.append(
        "These are matched descriptive consistency checks, not statistical significance tests."
    )
    return patterns


def _hypothesis_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Condition | Hypothesis | Planned/valid | Planned accuracy | Valid accuracy | Threshold success (all) | Premature (valid) | Scientific failures | Regret (valid) | Oracle agreement (valid) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['true_hypothesis']} | {row['planned_cells']}/{row['valid_episodes']} | "
            f"{_pct(row['planned_cell_accuracy'])} | {_pct(row['valid_episode_accuracy'])} | "
            f"{_pct(row['threshold_qualified_success_all_cells'])} | {_pct(row['premature_diagnosis_valid_episodes'])} | "
            f"{row['scientific_model_failures']} | {_f(row['mean_regret_valid_episodes'])} | "
            f"{_pct(row['oracle_agreement_valid_actions'])} |"
        )
    return "\n".join(lines)


def _hardest_hypothesis_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        minimum_accuracy = min(row["planned_cell_accuracy"] for row in selected)
        diagnostic_hardest = [
            row["true_hypothesis"] for row in selected
            if row["planned_cell_accuracy"] == minimum_accuracy
        ]
        maximum_regret = max(row["mean_regret_valid_episodes"] for row in selected if row["mean_regret_valid_episodes"] is not None)
        planning_hardest = [
            row["true_hypothesis"] for row in selected
            if row["mean_regret_valid_episodes"] == maximum_regret
        ]
        lines.append(
            f"- **{condition}:** lowest planned accuracy: {', '.join(diagnostic_hardest)} ({_pct(minimum_accuracy)}); "
            f"highest valid-episode mean regret: {', '.join(planning_hardest)} ({_f(maximum_regret)})."
        )
    return lines


def _protocol_concentration(rows: Sequence[Mapping[str, Any]]) -> str:
    failures = [row for row in rows if row["protocol_status"] != "VALID_EPISODE"]
    if not failures:
        return "No protocol/provider failures were observed."
    counts: dict[tuple[str, str, str], int] = {}
    for row in failures:
        key = (
            str(row["analysis_category"]),
            str(row["condition"]),
            str(row["true_hypothesis"]),
        )
        counts[key] = counts.get(key, 0) + 1
    details = "; ".join(
        f"{count} {category} in {condition}/{hypothesis}"
        for (category, condition, hypothesis), count in sorted(counts.items())
    )
    return f"Yes: {details}."


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite and any((output / name).exists() for name in FINAL_OUTPUT_FILES):
        raise FileExistsError(f"final analysis already exists at {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in FINAL_OUTPUT_FILES:
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


def _episode_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["condition"]), str(row["true_hypothesis"]), int(row["seed"])


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        HYPOTHESES.index(str(row["true_hypothesis"])),
        int(row["seed"]),
        CONDITIONS.index(str(row["condition"])),
    )


def _condition_prefix(condition: str) -> str:
    return {
        "FULL_AUTONOMOUS": "full",
        "PLANNER_ONLY": "planner",
        "THRESHOLD_AWARE_AUTONOMOUS": "threshold_aware",
    }[condition]


def _boolean(row: Mapping[str, Any], field: str, *, default: bool | None = None) -> bool:
    value = row.get(field)
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    if value in (None, "") and default is not None:
        return default
    raise IncompatibleAnalysisInput(f"{field} must be Boolean")


def _integer(row: Mapping[str, Any], field: str, *, default: int | None = None) -> int:
    value = row.get(field)
    if value in (None, "") and default is not None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise IncompatibleAnalysisInput(f"{field} must be an integer") from error


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise IncompatibleAnalysisInput("expected an optional numeric value") from error
    if not math.isfinite(number):
        raise IncompatibleAnalysisInput("numeric analysis values must be finite")
    return number


def _available_numbers(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = _optional_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _mean_numeric(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = _available_numbers(rows, field)
    return mean(values) if values else None


def _rate(successes: int, total: int) -> float | None:
    return successes / total if total else None


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _f(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _signed(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.3f}"


def _pci(row: Mapping[str, Any], field: str) -> str:
    value = row[field]
    low = row[f"{field}_ci95_low"]
    high = row[f"{field}_ci95_high"]
    return f"{_pct(value)} [{_pct(low)}, {_pct(high)}]"


def _count_rate(row: Mapping[str, Any], rate_field: str, denominator_field: str) -> str:
    denominator = int(row[denominator_field])
    count = round(float(row[rate_field]) * denominator) if row[rate_field] is not None else 0
    return f"{count}/{denominator}"


def _direction_word(value: bool) -> str:
    return "yes"
