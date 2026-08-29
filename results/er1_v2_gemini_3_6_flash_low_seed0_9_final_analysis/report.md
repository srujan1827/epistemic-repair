# ER-1 V2 seeds 0–9 final three-condition analysis

## Technical summary

The frozen analysis contains 120 planned cells, 117 valid episodes, 3 scientific-model failures, 0 non-rate-limit provider failures, and 0 rate-limit failures. Scientific-model failures remain in all planned-cell denominators.

**Directly observed:** planner-only changed valid-episode mean regret by -0.087 and action-weighted oracle agreement by +0.074 relative to full autonomy. Threshold awareness changed valid-episode premature diagnosis by -0.141.

**Directional interpretation:** authoritative beliefs primarily affect experiment selection, while explicit threshold knowledge affects stopping. Threshold awareness did not eliminate premature stopping: 29/37 of its valid episodes remained premature.

**Requires more replication:** confidence calibration is a plausible bottleneck, but the combined sample is still small and comes from one model/configuration. Threshold-aware autonomy reported confidence ≥ 0.95 without matching normative support in 29/37 comparable valid episodes.

## Combined outcomes retain protocol failures in the denominator

| Condition | Planned | Valid | Diagnostic errors | Scientific failures | Provider failures | Rate limits | Planned accuracy | Valid accuracy | Threshold success (all) | Premature (valid) | Mean experiments (all) | Mean regret (all) | Oracle agreement (all actions) | False structural | Missed structural | Belief L1 (valid) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_AUTONOMOUS | 40 | 40 | 8 | 0 | 0 | 0 | 80.0% | 80.0% | 7.5% | 92.5% | 2.525 | 0.168 | 63.4% | 10.0% | 0.0% | 0.367 |
| PLANNER_ONLY | 40 | 40 | 5 | 0 | 0 | 0 | 87.5% | 87.5% | 10.0% | 90.0% | 2.650 | 0.081 | 70.8% | 10.0% | 3.3% | — |
| THRESHOLD_AWARE_AUTONOMOUS | 40 | 37 | 5 | 3 | 0 | 0 | 80.0% | 86.5% | 20.0% | 78.4% | 3.175 | 0.151 | 65.4% | 0.0% | 7.4% | 0.380 |

Planned-cell accuracy treats every protocol/provider failure as unsuccessful. Valid-episode accuracy conditions on completing the protocol with an evaluable diagnosis; a wrong diagnosis is a diagnostic error within that valid set. Provider failures exclude rate limits so all failure counts are mutually exclusive.

### Wilson 95% intervals for key combined proportions

| Condition | Planned accuracy | Valid accuracy | Threshold success (all cells) | Premature (all cells) | Scientific failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL_AUTONOMOUS | 80.0% [65.2%, 89.5%] | 80.0% [65.2%, 89.5%] | 7.5% [2.6%, 19.9%] | 92.5% [80.1%, 97.4%] | 0.0% [0.0%, 8.8%] |
| PLANNER_ONLY | 87.5% [73.9%, 94.5%] | 87.5% [73.9%, 94.5%] | 10.0% [4.0%, 23.1%] | 90.0% [76.9%, 96.0%] | 0.0% [0.0%, 8.8%] |
| THRESHOLD_AWARE_AUTONOMOUS | 80.0% [65.2%, 89.5%] | 86.5% [72.0%, 94.1%] | 20.0% [10.5%, 34.8%] | 72.5% [57.2%, 83.9%] | 7.5% [2.6%, 19.9%] |

The intervals quantify binomial uncertainty only. Their overlap or non-overlap is not used as a significance test, and the matched design is not modeled by these marginal intervals.

## Calibration and stopping remain distinct

| Condition | Valid | Comparable | Signed gap | Absolute gap | Overconfident | Underconfident | Self ≥ .95, normative < .95 | Both ≥ .95 | Normative ≥ .95, self < .95 | Premature (valid) | Threshold success (valid) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_AUTONOMOUS | 40 | 40 | +0.121 | 0.134 | 87.5% | 12.5% | 4/40 | 1/40 | 2/40 | 92.5% | 7.5% |
| THRESHOLD_AWARE_AUTONOMOUS | 37 | 37 | +0.106 | 0.110 | 83.8% | 16.2% | 29/37 | 8/37 | 0/37 | 78.4% | 21.6% |

Self-reported confidence minus normative probability is computed only for valid autonomous episodes with both values present. Planner-only is excluded because its stored confidence is benchmark-derived rather than an autonomous self-estimate.

### Calibration and stopping by hypothesis

| Condition | Hypothesis | Valid/comparable | Signed gap | Absolute gap | Overconfident | Underconfident | Self-only threshold | Both threshold | Normative-only threshold | Premature valid | Threshold success valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_AUTONOMOUS | NO_STRUCTURAL_CHANGE | 10/10 | +0.151 | 0.157 | 90.0% | 10.0% | 0 | 0 | 0 | 100.0% | 0.0% |
| FULL_AUTONOMOUS | WORLD_SHIFT | 10/10 | +0.113 | 0.130 | 80.0% | 20.0% | 1 | 1 | 1 | 80.0% | 20.0% |
| FULL_AUTONOMOUS | SENSOR_CORRUPTION | 10/10 | +0.089 | 0.120 | 80.0% | 20.0% | 0 | 0 | 1 | 90.0% | 10.0% |
| FULL_AUTONOMOUS | MISSING_LATENT_VARIABLE | 10/10 | +0.131 | 0.131 | 100.0% | 0.0% | 3 | 0 | 0 | 100.0% | 0.0% |
| THRESHOLD_AWARE_AUTONOMOUS | NO_STRUCTURAL_CHANGE | 10/10 | +0.054 | 0.054 | 100.0% | 0.0% | 10 | 0 | 0 | 100.0% | 0.0% |
| THRESHOLD_AWARE_AUTONOMOUS | WORLD_SHIFT | 7/7 | +0.150 | 0.164 | 42.9% | 57.1% | 3 | 4 | 0 | 42.9% | 57.1% |
| THRESHOLD_AWARE_AUTONOMOUS | SENSOR_CORRUPTION | 10/10 | +0.084 | 0.089 | 80.0% | 20.0% | 6 | 4 | 0 | 60.0% | 40.0% |
| THRESHOLD_AWARE_AUTONOMOUS | MISSING_LATENT_VARIABLE | 10/10 | +0.151 | 0.151 | 100.0% | 0.0% | 10 | 0 | 0 | 100.0% | 0.0% |

## Pilot and replication consistency

| Split | Condition | Planned accuracy | Valid accuracy | Threshold success (all) | Premature (valid) | Regret (valid) | Oracle agreement (valid) | Calibration signed gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PILOT | FULL_AUTONOMOUS | 100.0% | 100.0% | 12.5% | 87.5% | 0.232 | 60.9% | +0.055 |
| PILOT | PLANNER_ONLY | 100.0% | 100.0% | 12.5% | 87.5% | 0.049 | 68.4% | — |
| PILOT | THRESHOLD_AWARE_AUTONOMOUS | 100.0% | 100.0% | 37.5% | 62.5% | 0.153 | 72.0% | +0.039 |
| REPLICATION | FULL_AUTONOMOUS | 75.0% | 75.0% | 6.2% | 93.8% | 0.152 | 64.1% | +0.137 |
| REPLICATION | PLANNER_ONLY | 84.4% | 84.4% | 9.4% | 90.6% | 0.089 | 71.3% | — |
| REPLICATION | THRESHOLD_AWARE_AUTONOMOUS | 75.0% | 82.8% | 15.6% | 82.8% | 0.149 | 61.5% | +0.125 |
| COMBINED | FULL_AUTONOMOUS | 80.0% | 80.0% | 7.5% | 92.5% | 0.168 | 63.4% | +0.121 |
| COMBINED | PLANNER_ONLY | 87.5% | 87.5% | 10.0% | 90.0% | 0.081 | 70.8% | — |
| COMBINED | THRESHOLD_AWARE_AUTONOMOUS | 80.0% | 86.5% | 20.0% | 78.4% | 0.150 | 63.8% | +0.106 |

- **planner-only lower regret:** replicated directionally (pilot delta -0.183, replication delta -0.063, combined delta -0.087).
- **planner-only higher oracle agreement:** replicated directionally (pilot delta +0.076, replication delta +0.072, combined delta +0.074).
- **threshold awareness lower prematurity:** replicated directionally (pilot delta -0.250, replication delta -0.110, combined delta -0.141).
- **threshold-aware absolute calibration gap:** replicated directionally (pilot delta -0.045, replication delta -0.017).
These are matched descriptive consistency checks, not statistical significance tests.

## Hypothesis-level difficulty

| Condition | Hypothesis | Planned/valid | Planned accuracy | Valid accuracy | Threshold success (all) | Premature (valid) | Scientific failures | Regret (valid) | Oracle agreement (valid) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_AUTONOMOUS | NO_STRUCTURAL_CHANGE | 10/10 | 90.0% | 90.0% | 0.0% | 100.0% | 0 | 0.080 | 43.5% |
| FULL_AUTONOMOUS | WORLD_SHIFT | 10/10 | 70.0% | 70.0% | 20.0% | 80.0% | 0 | 0.134 | 84.6% |
| FULL_AUTONOMOUS | SENSOR_CORRUPTION | 10/10 | 80.0% | 80.0% | 10.0% | 90.0% | 0 | 0.260 | 50.0% |
| FULL_AUTONOMOUS | MISSING_LATENT_VARIABLE | 10/10 | 80.0% | 80.0% | 0.0% | 100.0% | 0 | 0.199 | 75.0% |
| PLANNER_ONLY | NO_STRUCTURAL_CHANGE | 10/10 | 90.0% | 90.0% | 0.0% | 100.0% | 0 | 0.045 | 45.8% |
| PLANNER_ONLY | WORLD_SHIFT | 10/10 | 80.0% | 80.0% | 40.0% | 60.0% | 0 | 0.205 | 81.6% |
| PLANNER_ONLY | SENSOR_CORRUPTION | 10/10 | 80.0% | 80.0% | 0.0% | 100.0% | 0 | 0.054 | 52.4% |
| PLANNER_ONLY | MISSING_LATENT_VARIABLE | 10/10 | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0.021 | 95.7% |
| THRESHOLD_AWARE_AUTONOMOUS | NO_STRUCTURAL_CHANGE | 10/10 | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0.131 | 34.4% |
| THRESHOLD_AWARE_AUTONOMOUS | WORLD_SHIFT | 10/7 | 50.0% | 71.4% | 40.0% | 42.9% | 3 | 0.099 | 86.4% |
| THRESHOLD_AWARE_AUTONOMOUS | SENSOR_CORRUPTION | 10/10 | 90.0% | 90.0% | 40.0% | 60.0% | 0 | 0.191 | 65.6% |
| THRESHOLD_AWARE_AUTONOMOUS | MISSING_LATENT_VARIABLE | 10/10 | 80.0% | 80.0% | 0.0% | 100.0% | 0 | 0.162 | 76.7% |

- **FULL_AUTONOMOUS:** lowest planned accuracy: WORLD_SHIFT (70.0%); highest valid-episode mean regret: SENSOR_CORRUPTION (0.260).
- **PLANNER_ONLY:** lowest planned accuracy: WORLD_SHIFT, SENSOR_CORRUPTION (80.0%); highest valid-episode mean regret: WORLD_SHIFT (0.205).
- **THRESHOLD_AWARE_AUTONOMOUS:** lowest planned accuracy: WORLD_SHIFT (50.0%); highest valid-episode mean regret: SENSOR_CORRUPTION (0.191).

## Answers to the scientific questions

- **A. Does authoritative normative belief information improve diagnosis?** Directly observed planned-cell accuracy was 87.5% for planner-only and 80.0% for full autonomy; valid-episode accuracy was 87.5% versus 80.0%. This is descriptive, not a causal or significant difference claim.
- **B. Does it improve experiment selection?** Directionally, yes: planner-only had 0.087 lower valid-episode mean regret and 7.4% higher oracle agreement than full autonomy.
- **C. Does explicit threshold awareness reduce premature stopping?** Directly observed valid-episode prematurity changed from 92.5% to 78.4%; the directional change is -0.141.
- **D. Does threshold awareness eliminate premature stopping?** No. 29/37 valid threshold-aware episodes were premature.
- **E. Is confidence calibration plausibly a bottleneck?** Plausibly, yes: the mean absolute gap was 0.110, and 29 episodes crossed the self-reported threshold without normative support. This mechanism claim requires further replication.
- **F. Are protocol failures concentrated?** Yes: 3 SCIENTIFIC_MODEL_FAILURE in THRESHOLD_AWARE_AUTONOMOUS/WORLD_SHIFT.
- **G. Which hypothesis appears hardest?** The hypothesis table and notes above separate diagnostic difficulty from experiment-selection difficulty; ties are retained rather than forced into a single winner.
- **H. Which weaknesses are separable?** Planner-only versus full autonomy isolates belief support for planning; threshold-aware versus full autonomy isolates threshold knowledge for stopping; calibration gaps isolate confidence alignment; and the explicit protocol taxonomy isolates structured-output compliance. The design makes these descriptive contrasts separable, but it does not by itself identify a unique causal mechanism.

## Scope, validation, and metric definitions

The final grid is exactly 10 seeds × 4 hypotheses × 3 conditions = 120 cells. All rows match benchmark `binary_er1_v2`, provider/model `gemini/gemini-3.6-flash`, thinking level `low`, budget 8, diagnosis threshold 0.95, and max decision calls 9.

The hypotheses are `NO_STRUCTURAL_CHANGE`, `WORLD_SHIFT`, `SENSOR_CORRUPTION`, and `MISSING_LATENT_VARIABLE`. Matched comparisons use `(true_hypothesis, seed)` and require all three conditions. Duplicate cell keys are rejected.

Prompt versions were validated per condition:

- `FULL_AUTONOMOUS`: `binary_er1_v2_001`
- `PLANNER_ONLY`: `binary_er1_v2_001`
- `THRESHOLD_AWARE_AUTONOMOUS`: `binary_er1_v2_threshold_aware_001`

Source validation:

- `er1_v2_gemini_3_6_flash_low_seed0_1_three_condition_analysis`: 24 rows, seeds [0, 1], conditions ['FULL_AUTONOMOUS', 'PLANNER_ONLY', 'THRESHOLD_AWARE_AUTONOMOUS']
- `er1_v2_gemini_3_6_flash_low_seed2_9_three_condition_tier1`: 96 rows, seeds [2, 3, 4, 5, 6, 7, 8, 9], conditions ['FULL_AUTONOMOUS', 'PLANNER_ONLY', 'THRESHOLD_AWARE_AUTONOMOUS']

Threshold-qualified success requires a correct diagnosis whose normative support reaches the frozen evaluator threshold. Premature diagnosis means the final diagnosis was issued below that normative threshold. Regret and oracle agreement use the frozen episode-level evaluator outputs; oracle agreement is action-weighted. Wilson intervals use the stated binomial denominator and do not account for matched-cell dependence.

## Limitations and next steps

This is a single-model, single-thinking-level study with ten seeds per hypothesis. The analysis is descriptive; no prompt, benchmark, evaluator, or episode was changed, and no API call was made. Three strict protocol failures reduce the valid threshold-aware sample and must not be reinterpreted as diagnostic errors.

A defensible next step is to freeze this artifact and pre-specify any larger replication or second-model-family study before collecting more data. Do not tune prompts or thresholds against these results without labeling that work as a new experimental stage.

## Further questions

- Do the planning and stopping patterns persist across another named model family?
- Do confidence gaps predict premature stopping within matched cases when the sample is larger?
- Are strict structured-output failures stable across repeated runs, or are they transient model-interface behavior?

Exact row-level evidence is preserved in `episodes_final.csv`, `matched_final.csv`, `calibration_final.csv`, and `failure_analysis.csv`.
