# ER-1 Oracle Calibration

## Executive summary

This oracle-only sweep contains 60,000 episodes (1000 seeds per hypothesis/condition) and took 96.647 seconds. MAP accuracy and threshold-qualified success are reported separately.

**C — One or more hypotheses remain statistically difficult at practical budgets; revisit the generative probabilities before freezing ER-1.** Even budget 8 / threshold 0.90 does not deliver uniformly strong oracle resolution.

## Overall results

| Budget | Threshold | MAP accuracy (95% CI) | Success@threshold (95% CI) | Mean experiments | False structural (95% CI) | Missed structural (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.80 | 0.445 [0.430, 0.460] | 0.000 [0.000, 0.001] | 1.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 1 | 0.90 | 0.445 [0.430, 0.460] | 0.000 [0.000, 0.001] | 1.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 1 | 0.95 | 0.445 [0.430, 0.460] | 0.000 [0.000, 0.001] | 1.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 2 | 0.80 | 0.601 [0.586, 0.616] | 0.387 [0.372, 0.402] | 2.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 2 | 0.90 | 0.601 [0.586, 0.616] | 0.000 [0.000, 0.001] | 2.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 2 | 0.95 | 0.601 [0.586, 0.616] | 0.000 [0.000, 0.001] | 2.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 3 | 0.80 | 0.632 [0.617, 0.647] | 0.559 [0.544, 0.575] | 2.351 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 3 | 0.90 | 0.641 [0.626, 0.656] | 0.503 [0.488, 0.519] | 3.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 3 | 0.95 | 0.641 [0.626, 0.656] | 0.334 [0.320, 0.349] | 3.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 5 | 0.80 | 0.674 [0.659, 0.688] | 0.647 [0.633, 0.662] | 2.629 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 5 | 0.90 | 0.824 [0.811, 0.835] | 0.757 [0.744, 0.771] | 3.831 | 0.507 [0.476, 0.538] | 0.002 [0.001, 0.004] |
| 5 | 0.95 | 0.824 [0.811, 0.835] | 0.570 [0.555, 0.586] | 4.098 | 0.507 [0.476, 0.538] | 0.002 [0.001, 0.004] |
| 8 | 0.80 | 0.684 [0.669, 0.698] | 0.682 [0.667, 0.696] | 2.713 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 8 | 0.90 | 0.891 [0.881, 0.901] | 0.863 [0.852, 0.873] | 4.174 | 0.296 [0.269, 0.325] | 0.003 [0.002, 0.006] |
| 8 | 0.95 | 0.901 [0.892, 0.910] | 0.843 [0.831, 0.853] | 4.784 | 0.297 [0.269, 0.326] | 0.003 [0.001, 0.005] |

## Per-hypothesis difficulty: budget 5, threshold 0.90

| Hypothesis | MAP accuracy (95% CI) | Success@threshold (95% CI) | Mean experiments | Mean true posterior | Threshold reached |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.493 [0.462, 0.524] | 0.493 [0.462, 0.524] | 4.596 | 0.554 | 0.741 |
| `WORLD_SHIFT` | 0.947 [0.931, 0.959] | 0.910 [0.891, 0.926] | 3.475 | 0.900 | 0.941 |
| `SENSOR_CORRUPTION` | 0.930 [0.912, 0.944] | 0.816 [0.791, 0.839] | 3.644 | 0.861 | 0.856 |
| `MISSING_LATENT_VARIABLE` | 0.924 [0.906, 0.939] | 0.811 [0.786, 0.834] | 3.608 | 0.864 | 0.841 |

## Per-hypothesis difficulty: budget 8, threshold 0.90

| Hypothesis | MAP accuracy (95% CI) | Success@threshold (95% CI) | Mean experiments | Mean true posterior | Threshold reached |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.704 [0.675, 0.731] | 0.646 [0.616, 0.675] | 5.218 | 0.650 | 0.929 |
| `WORLD_SHIFT` | 0.960 [0.946, 0.970] | 0.958 [0.944, 0.969] | 3.586 | 0.923 | 0.993 |
| `SENSOR_CORRUPTION` | 0.948 [0.932, 0.960] | 0.902 [0.882, 0.919] | 3.989 | 0.903 | 0.945 |
| `MISSING_LATENT_VARIABLE` | 0.954 [0.939, 0.965] | 0.945 [0.929, 0.958] | 3.902 | 0.899 | 0.982 |

## Threshold sensitivity at budget 5

| Threshold | MAP accuracy | Success@threshold | Mean experiments | False structural | Missed structural |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 0.674 | 0.647 | 2.629 | 1.000 | 0.000 |
| 0.90 | 0.824 | 0.757 | 3.831 | 0.507 | 0.002 |
| 0.95 | 0.824 | 0.570 | 4.098 | 0.507 | 0.002 |

## Final true-hypothesis posterior: budget 5, threshold 0.90

| Hypothesis | Mean | Median | p10 | p25 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.554 | 0.429 | 0.019 | 0.028 | 0.936 | 0.936 |
| `WORLD_SHIFT` | 0.900 | 0.968 | 0.914 | 0.968 | 0.968 | 0.968 |
| `SENSOR_CORRUPTION` | 0.861 | 0.954 | 0.576 | 0.954 | 0.954 | 0.962 |
| `MISSING_LATENT_VARIABLE` | 0.864 | 0.940 | 0.850 | 0.940 | 0.940 | 0.940 |

## Confusion matrix: budget 5, threshold 0.90

Rows are truth; columns are final MAP diagnosis. Cells show count (row %).

| Truth | N | W | S | L |
| --- | ---: | ---: | ---: | ---: |
| N | 493 (49.3%) | 7 (0.7%) | 381 (38.1%) | 119 (11.9%) |
| W | 0 (0.0%) | 947 (94.7%) | 12 (1.2%) | 41 (4.1%) |
| S | 3 (0.3%) | 16 (1.6%) | 930 (93.0%) | 51 (5.1%) |
| L | 3 (0.3%) | 52 (5.2%) | 21 (2.1%) | 924 (92.4%) |

N=no change, W=world shift, S=sensor corruption, L=missing latent.

## Experiment selection: budget 5, threshold 0.90

| Hypothesis | First trusted | Mean repeat | Mean trusted | Mean context changes |
| --- | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 1.000 | 0.800 | 2.244 | 1.552 |
| `WORLD_SHIFT` | 1.000 | 0.009 | 2.477 | 0.989 |
| `SENSOR_CORRUPTION` | 1.000 | 0.345 | 2.298 | 1.001 |
| `MISSING_LATENT_VARIABLE` | 1.000 | 0.113 | 2.500 | 0.995 |

Top five action sequences by hypothesis:

- `NO_STRUCTURAL_CHANGE`:
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → REPEAT_TRIAL → CHANGE_CONTEXT` — 562 episodes (56.2%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 116 episodes (11.6%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → REPEAT_TRIAL → REPEAT_TRIAL` — 109 episodes (10.9%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 86 episodes (8.6%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 74 episodes (7.4%)
- `WORLD_SHIFT`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 706 episodes (70.6%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 103 episodes (10.3%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 87 episodes (8.7%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 80 episodes (8.0%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 11 episodes (1.1%)
- `SENSOR_CORRUPTION`:
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 647 episodes (64.7%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → REPEAT_TRIAL → REPEAT_TRIAL` — 129 episodes (12.9%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 76 episodes (7.6%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 74 episodes (7.4%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 31 episodes (3.1%)
- `MISSING_LATENT_VARIABLE`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 693 episodes (69.3%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 107 episodes (10.7%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 90 episodes (9.0%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 87 episodes (8.7%)
  `USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 11 episodes (1.1%)

## Representative hard cases

### NO_STRUCTURAL_CHANGE

- `quick_threshold_success` seed 0: USE_TRUSTED_SENSOR trusted_t=1 → [N=0.120,W=0.089,S=0.702,L=0.089]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.142,W=0.013,S=0.832,L=0.013]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.429,W=0.006,S=0.526,L=0.039]; REPEAT_TRIAL primary_o=1 → [N=0.741,W=0.002,S=0.190,L=0.067]; CHANGE_CONTEXT context=B,primary_o=1 → [N=0.936,W=0.000,S=0.050,L=0.014]; final `NO_STRUCTURAL_CHANGE`, THRESHOLD_REACHED.
- `correct_map_without_threshold`: no matching seed in the sweep.
- `incorrect_map` seed 2: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.016,W=0.135,S=0.020,L=0.829]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.019,W=0.019,S=0.023,L=0.940]; final `MISSING_LATENT_VARIABLE`, THRESHOLD_REACHED.

### WORLD_SHIFT

- `quick_threshold_success` seed 0: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.000,W=0.968,S=0.013,L=0.019]; final `WORLD_SHIFT`, THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 15: USE_TRUSTED_SENSOR trusted_t=1 → [N=0.120,W=0.089,S=0.702,L=0.089]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.052,W=0.321,S=0.306,L=0.321]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.120,W=0.089,S=0.702,L=0.089]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.052,W=0.321,S=0.306,L=0.321]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; final `WORLD_SHIFT`, BUDGET_EXHAUSTED.
- `incorrect_map` seed 3: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.008,W=0.305,S=0.277,L=0.410]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.008,W=0.305,S=0.277,L=0.410]; final `MISSING_LATENT_VARIABLE`, BUDGET_EXHAUSTED.

### SENSOR_CORRUPTION

- `quick_threshold_success` seed 0: USE_TRUSTED_SENSOR trusted_t=1 → [N=0.120,W=0.089,S=0.702,L=0.089]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.142,W=0.013,S=0.832,L=0.013]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.028,W=0.015,S=0.954,L=0.003]; final `SENSOR_CORRUPTION`, THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 2: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.008,W=0.305,S=0.277,L=0.410]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.011,W=0.050,S=0.379,L=0.560]; REPEAT_TRIAL primary_o=0 → [N=0.004,W=0.100,S=0.716,L=0.181]; final `SENSOR_CORRUPTION`, BUDGET_EXHAUSTED.
- `incorrect_map` seed 5: USE_TRUSTED_SENSOR trusted_t=1 → [N=0.120,W=0.089,S=0.702,L=0.089]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.052,W=0.321,S=0.306,L=0.321]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.008,W=0.305,S=0.277,L=0.410]; final `MISSING_LATENT_VARIABLE`, BUDGET_EXHAUSTED.

### MISSING_LATENT_VARIABLE

- `quick_threshold_success` seed 0: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.016,W=0.135,S=0.020,L=0.829]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.019,W=0.019,S=0.023,L=0.940]; final `MISSING_LATENT_VARIABLE`, THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 14: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.008,W=0.305,S=0.277,L=0.410]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.011,W=0.050,S=0.379,L=0.560]; REPEAT_TRIAL primary_o=1 → [N=0.017,W=0.012,S=0.120,L=0.850]; final `MISSING_LATENT_VARIABLE`, BUDGET_EXHAUSTED.
- `incorrect_map` seed 5: USE_TRUSTED_SENSOR trusted_t=0 → [N=0.009,W=0.468,S=0.054,L=0.468]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.784,S=0.086,L=0.128]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.000,W=0.968,S=0.013,L=0.019]; final `WORLD_SHIFT`, THRESHOLD_REACHED.


## Recommendation

**C — One or more hypotheses remain statistically difficult at practical budgets; revisit the generative probabilities before freezing ER-1.** Even budget 8 / threshold 0.90 does not deliver uniformly strong oracle resolution.

No ER-1 probabilities, defaults, prompts, or action semantics were changed by this calibration.
