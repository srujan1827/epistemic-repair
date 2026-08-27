# ER-1 V2 Full Oracle Calibration

This is a fixed-parameter measurement run. No benchmark probability, threshold, prompt, action, or architecture was changed.

Episodes: 60,000; seeds per cell: 1000; runtime: 97.537 seconds.

MAP accuracy and Success@threshold are deliberately reported separately. Oracle premature diagnosis is not applicable: the oracle stops only at its configured posterior threshold or budget exhaustion.

## Overall results

| Budget | Threshold | MAP (95% CI) | Success (95% CI) | Mean exp. | False structural (95% CI) | Missed structural (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.80 | 0.456 [0.440, 0.471] | 0.000 [0.000, 0.001] | 1.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 1 | 0.90 | 0.456 [0.440, 0.471] | 0.000 [0.000, 0.001] | 1.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 1 | 0.95 | 0.456 [0.440, 0.471] | 0.000 [0.000, 0.001] | 1.000 | 1.000 [0.996, 1.000] | 0.000 [0.000, 0.001] |
| 2 | 0.80 | 0.795 [0.782, 0.807] | 0.585 [0.569, 0.600] | 2.000 | 0.159 [0.138, 0.183] | 0.080 [0.071, 0.090] |
| 2 | 0.90 | 0.795 [0.782, 0.807] | 0.000 [0.000, 0.001] | 2.000 | 0.159 [0.138, 0.183] | 0.080 [0.071, 0.090] |
| 2 | 0.95 | 0.795 [0.782, 0.807] | 0.000 [0.000, 0.001] | 2.000 | 0.159 [0.138, 0.183] | 0.080 [0.071, 0.090] |
| 3 | 0.80 | 0.802 [0.789, 0.814] | 0.772 [0.759, 0.785] | 2.270 | 0.251 [0.225, 0.279] | 0.011 [0.008, 0.015] |
| 3 | 0.90 | 0.818 [0.806, 0.830] | 0.713 [0.699, 0.727] | 3.000 | 0.251 [0.225, 0.279] | 0.011 [0.008, 0.015] |
| 3 | 0.95 | 0.818 [0.806, 0.830] | 0.174 [0.163, 0.186] | 3.000 | 0.251 [0.225, 0.279] | 0.011 [0.008, 0.015] |
| 5 | 0.80 | 0.830 [0.818, 0.842] | 0.809 [0.797, 0.821] | 2.420 | 0.173 [0.151, 0.198] | 0.021 [0.017, 0.027] |
| 5 | 0.90 | 0.898 [0.888, 0.907] | 0.860 [0.848, 0.870] | 3.404 | 0.173 [0.151, 0.198] | 0.021 [0.017, 0.027] |
| 5 | 0.95 | 0.925 [0.916, 0.933] | 0.741 [0.727, 0.754] | 4.138 | 0.057 [0.044, 0.073] | 0.023 [0.019, 0.029] |
| 8 | 0.80 | 0.838 [0.826, 0.849] | 0.835 [0.823, 0.846] | 2.470 | 0.175 [0.153, 0.200] | 0.014 [0.010, 0.018] |
| 8 | 0.90 | 0.912 [0.903, 0.920] | 0.903 [0.894, 0.912] | 3.539 | 0.177 [0.155, 0.202] | 0.013 [0.009, 0.017] |
| 8 | 0.95 | 0.958 [0.951, 0.964] | 0.928 [0.919, 0.935] | 4.596 | 0.046 [0.035, 0.061] | 0.006 [0.004, 0.009] |

## Per-hypothesis: budget 5, threshold 0.90

| Hypothesis | Episodes | MAP (95% CI) | Success (95% CI) | Mean / median / SD exp. | Threshold reached | Budget exhausted | Cumulative / mean-action regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 1000 | 0.827 [0.802, 0.849] | 0.749 [0.721, 0.775] | 3.206 / 3.000 / 0.606 | 0.919 | 0.081 | 0.000000 / 0.000000 |
| `WORLD_SHIFT` | 1000 | 0.961 [0.947, 0.971] | 0.930 [0.912, 0.944] | 3.401 / 3.000 / 0.690 | 0.961 | 0.039 | 0.000000 / 0.000000 |
| `SENSOR_CORRUPTION` | 1000 | 0.931 [0.914, 0.945] | 0.897 [0.877, 0.914] | 3.445 / 3.000 / 0.829 | 0.941 | 0.059 | 0.000000 / 0.000000 |
| `MISSING_LATENT_VARIABLE` | 1000 | 0.873 [0.851, 0.892] | 0.862 [0.839, 0.882] | 3.562 / 3.000 / 0.890 | 0.938 | 0.062 | 0.000000 / 0.000000 |

### Final true-hypothesis posterior: budget 5, threshold 0.90

| Hypothesis | Mean | Median | p10 | p25 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.734 | 0.911 | 0.047 | 0.579 | 0.911 | 0.911 |
| `WORLD_SHIFT` | 0.924 | 0.977 | 0.916 | 0.977 | 0.977 | 0.977 |
| `SENSOR_CORRUPTION` | 0.857 | 0.922 | 0.622 | 0.922 | 0.922 | 0.922 |
| `MISSING_LATENT_VARIABLE` | 0.832 | 0.942 | 0.241 | 0.934 | 0.942 | 0.942 |

## Per-hypothesis: budget 8, threshold 0.90

| Hypothesis | Episodes | MAP (95% CI) | Success (95% CI) | Mean / median / SD exp. | Threshold reached | Budget exhausted | Cumulative / mean-action regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 1000 | 0.823 [0.798, 0.845] | 0.812 [0.787, 0.835] | 3.384 / 3.000 / 1.174 | 0.985 | 0.015 | 0.000000 / 0.000000 |
| `WORLD_SHIFT` | 1000 | 0.964 [0.951, 0.974] | 0.963 [0.949, 0.973] | 3.468 / 3.000 / 0.909 | 0.997 | 0.003 | 0.000000 / 0.000000 |
| `SENSOR_CORRUPTION` | 1000 | 0.950 [0.935, 0.962] | 0.935 [0.918, 0.949] | 3.594 / 3.000 / 1.233 | 0.981 | 0.019 | 0.000000 / 0.000000 |
| `MISSING_LATENT_VARIABLE` | 1000 | 0.911 [0.892, 0.927] | 0.904 [0.884, 0.921] | 3.711 / 3.000 / 1.254 | 0.989 | 0.011 | 0.000000 / 0.000000 |

### Final true-hypothesis posterior: budget 8, threshold 0.90

| Hypothesis | Mean | Median | p10 | p25 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.759 | 0.911 | 0.047 | 0.911 | 0.911 | 0.911 |
| `WORLD_SHIFT` | 0.936 | 0.977 | 0.916 | 0.977 | 0.977 | 0.977 |
| `SENSOR_CORRUPTION` | 0.877 | 0.922 | 0.902 | 0.922 | 0.922 | 0.939 |
| `MISSING_LATENT_VARIABLE` | 0.859 | 0.942 | 0.905 | 0.942 | 0.942 | 0.942 |

## Threshold sensitivity at budget 5

| Threshold | MAP | Success | Mean exp. | False structural | Missed structural |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 0.830 | 0.809 | 2.420 | 0.173 | 0.021 |
| 0.90 | 0.898 | 0.860 | 3.404 | 0.173 | 0.021 |
| 0.95 | 0.925 | 0.741 | 4.138 | 0.057 | 0.023 |

## Confusion matrix: budget 5, threshold 0.90

Rows are truth; columns are N/W/S/L. Cells are count (row percentage).

| Truth | N | W | S | L |
| --- | ---: | ---: | ---: | ---: |
| N | 827 (82.7%) | 5 (0.5%) | 106 (10.6%) | 62 (6.2%) |
| W | 4 (0.4%) | 961 (96.1%) | 14 (1.4%) | 21 (2.1%) |
| S | 30 (3.0%) | 16 (1.6%) | 931 (93.1%) | 23 (2.3%) |
| L | 30 (3.0%) | 52 (5.2%) | 45 (4.5%) | 873 (87.3%) |

## Action selection: budget 5, threshold 0.90

| Hypothesis | First R/T/C | Overall R/T/C | Mean count R/T/C |
| --- | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.000/1.000/0.000 | 0.001/0.396/0.603 | 0.003/1.270/1.933 |
| `WORLD_SHIFT` | 0.000/1.000/0.000 | 0.001/0.696/0.303 | 0.005/2.366/1.030 |
| `SENSOR_CORRUPTION` | 0.000/1.000/0.000 | 0.013/0.623/0.364 | 0.045/2.147/1.253 |
| `MISSING_LATENT_VARIABLE` | 0.000/1.000/0.000 | 0.027/0.646/0.327 | 0.097/2.301/1.164 |

Top five action sequences:

- `NO_STRUCTURAL_CHANGE`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT` — 749 (74.9%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 147 (14.7%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 92 (9.2%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 7 (0.7%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 3 (0.3%)
- `WORLD_SHIFT`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 713 (71.3%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 165 (16.5%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 100 (10.0%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 13 (1.3%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 5 (0.5%)
- `SENSOR_CORRUPTION`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 762 (76.2%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 120 (12.0%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 55 (5.5%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 45 (4.5%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT` — 13 (1.3%)
- `MISSING_LATENT_VARIABLE`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 695 (69.5%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 102 (10.2%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 97 (9.7%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 74 (7.4%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT` — 16 (1.6%)

## Confusion matrix: budget 8, threshold 0.90

Rows are truth; columns are N/W/S/L. Cells are count (row percentage).

| Truth | N | W | S | L |
| --- | ---: | ---: | ---: | ---: |
| N | 823 (82.3%) | 3 (0.3%) | 109 (10.9%) | 65 (6.5%) |
| W | 4 (0.4%) | 964 (96.4%) | 15 (1.5%) | 17 (1.7%) |
| S | 16 (1.6%) | 10 (1.0%) | 950 (95.0%) | 24 (2.4%) |
| L | 18 (1.8%) | 33 (3.3%) | 38 (3.8%) | 911 (91.1%) |

## Action selection: budget 8, threshold 0.90

| Hypothesis | First R/T/C | Overall R/T/C | Mean count R/T/C |
| --- | ---: | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.000/1.000/0.000 | 0.025/0.379/0.596 | 0.084/1.283/2.017 |
| `WORLD_SHIFT` | 0.000/1.000/0.000 | 0.004/0.699/0.297 | 0.014/2.424/1.030 |
| `SENSOR_CORRUPTION` | 0.000/1.000/0.000 | 0.029/0.614/0.358 | 0.104/2.205/1.285 |
| `MISSING_LATENT_VARIABLE` | 0.000/1.000/0.000 | 0.041/0.641/0.317 | 0.153/2.380/1.178 |

Top five action sequences:

- `NO_STRUCTURAL_CHANGE`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT` — 749 (74.9%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 147 (14.7%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → REPEAT_TRIAL` — 63 (6.3%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 13 (1.3%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → REPEAT_TRIAL → REPEAT_TRIAL` — 7 (0.7%)
- `WORLD_SHIFT`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 713 (71.3%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 165 (16.5%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 71 (7.1%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 20 (2.0%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 7 (0.7%)
- `SENSOR_CORRUPTION`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 762 (76.2%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 99 (9.9%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 51 (5.1%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL → REPEAT_TRIAL → USE_TRUSTED_SENSOR` — 26 (2.6%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 15 (1.5%)
- `MISSING_LATENT_VARIABLE`:
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR` — 695 (69.5%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → REPEAT_TRIAL` — 80 (8.0%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR → USE_TRUSTED_SENSOR` — 79 (7.9%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT → USE_TRUSTED_SENSOR → CHANGE_CONTEXT` — 52 (5.2%)
  `USE_TRUSTED_SENSOR → CHANGE_CONTEXT → CHANGE_CONTEXT` — 16 (1.6%)

## Representative hard cases: budget 5, threshold 0.90

### NO_STRUCTURAL_CHANGE

- `quick_threshold_success` seed 0; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.270,W=0.072,S=0.585,L=0.072]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.613,W=0.025,S=0.205,L=0.156]; CHANGE_CONTEXT context=B,primary_o=1 → [N=0.911,W=0.006,S=0.047,L=0.036]; final `NO_STRUCTURAL_CHANGE` vs truth `NO_STRUCTURAL_CHANGE`; THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 19; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.270,W=0.072,S=0.585,L=0.072]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.613,W=0.025,S=0.205,L=0.156]; CHANGE_CONTEXT context=B,primary_o=0 → [N=0.149,W=0.056,S=0.452,L=0.343]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.230,W=0.010,S=0.699,L=0.061]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.579,W=0.004,S=0.272,L=0.145]; final `NO_STRUCTURAL_CHANGE` vs truth `NO_STRUCTURAL_CHANGE`; BUDGET_EXHAUSTED.
- `incorrect_map` seed 2; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.025,W=0.135,S=0.008,L=0.832]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.030,W=0.019,S=0.010,L=0.942]; final `MISSING_LATENT_VARIABLE` vs truth `NO_STRUCTURAL_CHANGE`; THRESHOLD_REACHED.

### WORLD_SHIFT

- `quick_threshold_success` seed 0; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.815,S=0.050,L=0.133]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.000,W=0.977,S=0.004,L=0.019]; final `WORLD_SHIFT` vs truth `WORLD_SHIFT`; THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 24; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.025,W=0.135,S=0.008,L=0.832]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.007,W=0.568,S=0.002,L=0.423]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.014,W=0.137,S=0.005,L=0.843]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.004,W=0.570,S=0.001,L=0.424]; final `WORLD_SHIFT` vs truth `WORLD_SHIFT`; BUDGET_EXHAUSTED.
- `incorrect_map` seed 45; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.025,W=0.135,S=0.008,L=0.832]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.030,W=0.019,S=0.010,L=0.942]; final `MISSING_LATENT_VARIABLE` vs truth `WORLD_SHIFT`; THRESHOLD_REACHED.

### SENSOR_CORRUPTION

- `quick_threshold_success` seed 0; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.270,W=0.072,S=0.585,L=0.072]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.043,W=0.103,S=0.837,L=0.017]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.047,W=0.013,S=0.922,L=0.018]; final `SENSOR_CORRUPTION` vs truth `SENSOR_CORRUPTION`; THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 22; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.815,S=0.050,L=0.133]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.009,W=0.344,S=0.184,L=0.463]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.014,W=0.059,S=0.274,L=0.653]; REPEAT_TRIAL primary_o=0 → [N=0.004,W=0.133,S=0.622,L=0.241]; final `SENSOR_CORRUPTION` vs truth `SENSOR_CORRUPTION`; BUDGET_EXHAUSTED.
- `incorrect_map` seed 2; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.025,W=0.135,S=0.008,L=0.832]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.030,W=0.019,S=0.010,L=0.942]; final `MISSING_LATENT_VARIABLE` vs truth `SENSOR_CORRUPTION`; THRESHOLD_REACHED.

### MISSING_LATENT_VARIABLE

- `quick_threshold_success` seed 0; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.025,W=0.135,S=0.008,L=0.832]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.030,W=0.019,S=0.010,L=0.942]; final `MISSING_LATENT_VARIABLE` vs truth `MISSING_LATENT_VARIABLE`; THRESHOLD_REACHED.
- `correct_map_without_threshold` seed 107; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=0 → [N=0.003,W=0.815,S=0.050,L=0.133]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.009,W=0.344,S=0.184,L=0.463]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.002,W=0.833,S=0.029,L=0.136]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.006,W=0.375,S=0.116,L=0.504]; final `MISSING_LATENT_VARIABLE` vs truth `MISSING_LATENT_VARIABLE`; BUDGET_EXHAUSTED.
- `incorrect_map` seed 15; initial [N=0.128,W=0.298,S=0.277,L=0.298]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.014,W=0.479,S=0.029,L=0.479]; CHANGE_CONTEXT context=A,primary_o=1 → [N=0.025,W=0.135,S=0.008,L=0.832]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.007,W=0.568,S=0.002,L=0.423]; USE_TRUSTED_SENSOR trusted_t=1 → [N=0.014,W=0.137,S=0.005,L=0.843]; USE_TRUSTED_SENSOR trusted_t=0 → [N=0.004,W=0.570,S=0.001,L=0.424]; final `WORLD_SHIFT` vs truth `MISSING_LATENT_VARIABLE`; BUDGET_EXHAUSTED.


## Direct V1 versus V2 comparison

All deltas are V2 minus V1.

### Budget 5, threshold 0.90

| Scope | Metric | V1 | V2 | Delta |
| --- | --- | ---: | ---: | ---: |
| `Overall` | `map_accuracy` | 0.824 | 0.898 | +0.075 |
| `Overall` | `success_at_threshold` | 0.757 | 0.860 | +0.102 |
| `Overall` | `mean_experiments` | 3.831 | 3.404 | -0.427 |
| `Overall` | `false_structural_diagnosis_rate` | 0.507 | 0.173 | -0.334 |
| `Overall` | `missed_structural_failure_rate` | 0.002 | 0.021 | +0.019 |
| `NO_STRUCTURAL_CHANGE` | `map_accuracy` | 0.493 | 0.827 | +0.334 |
| `WORLD_SHIFT` | `map_accuracy` | 0.947 | 0.961 | +0.014 |
| `SENSOR_CORRUPTION` | `map_accuracy` | 0.930 | 0.931 | +0.001 |
| `MISSING_LATENT_VARIABLE` | `map_accuracy` | 0.924 | 0.873 | -0.051 |

### Budget 8, threshold 0.90

| Scope | Metric | V1 | V2 | Delta |
| --- | --- | ---: | ---: | ---: |
| `Overall` | `map_accuracy` | 0.891 | 0.912 | +0.021 |
| `Overall` | `success_at_threshold` | 0.863 | 0.903 | +0.041 |
| `Overall` | `mean_experiments` | 4.174 | 3.539 | -0.635 |
| `Overall` | `false_structural_diagnosis_rate` | 0.296 | 0.177 | -0.119 |
| `Overall` | `missed_structural_failure_rate` | 0.003 | 0.013 | +0.010 |
| `NO_STRUCTURAL_CHANGE` | `map_accuracy` | 0.704 | 0.823 | +0.119 |
| `WORLD_SHIFT` | `map_accuracy` | 0.960 | 0.964 | +0.004 |
| `SENSOR_CORRUPTION` | `map_accuracy` | 0.948 | 0.950 | +0.002 |
| `MISSING_LATENT_VARIABLE` | `map_accuracy` | 0.954 | 0.911 | -0.043 |

### Budget 8, threshold 0.95

| Scope | Metric | V1 | V2 | Delta |
| --- | --- | ---: | ---: | ---: |
| `Overall` | `map_accuracy` | 0.901 | 0.958 | +0.057 |
| `Overall` | `success_at_threshold` | 0.843 | 0.928 | +0.085 |
| `Overall` | `mean_experiments` | 4.784 | 4.596 | -0.188 |
| `Overall` | `false_structural_diagnosis_rate` | 0.297 | 0.046 | -0.251 |
| `Overall` | `missed_structural_failure_rate` | 0.003 | 0.006 | +0.003 |
| `NO_STRUCTURAL_CHANGE` | `map_accuracy` | 0.703 | 0.954 | +0.251 |
| `WORLD_SHIFT` | `map_accuracy` | 0.980 | 0.971 | -0.009 |
| `SENSOR_CORRUPTION` | `map_accuracy` | 0.965 | 0.963 | -0.002 |
| `MISSING_LATENT_VARIABLE` | `map_accuracy` | 0.957 | 0.944 | -0.013 |

### Budget 5 threshold sensitivity: overall V2−V1 deltas

| Threshold | MAP | Success | Mean exp. | False structural | Missed structural |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | +0.157 | +0.162 | -0.209 | -0.827 | +0.021 |
| 0.90 | +0.075 | +0.102 | -0.427 | -0.334 | +0.019 |
| 0.95 | +0.101 | +0.171 | +0.040 | -0.450 | +0.021 |

## Acceptance questions

- **A — No-change recoverability:** Yes. Budget-5 no-change MAP changed by +0.334 versus V1.
- **B — False structural diagnosis:** Yes, materially lower. The budget-5 change is -0.334.
- **C — Structural performance:** Healthy; the lowest structural MAP at budget 5 is 0.873.
- **D — Balance:** At the recommended budget-8/0.95 point, easiest is WORLD_SHIFT (0.971) and hardest is MISSING_LATENT_VARIABLE (0.944); no hypothesis is unidentifiable or pathologically easy.
- **E — Budget 5:** Useful but not the strongest primary setting: overall MAP=0.898, success=0.860.
- **F — Budget 8:** Improves overall MAP by +0.014 and success by +0.044, at +0.136 mean experiments.
- **G — Threshold:** 0.95 is the best primary operating point when paired with budget 8. Relative to budget-8/0.90 it changes MAP by +0.046, success by +0.024, false structural by -0.131, and mean experiments by +1.057. Threshold 0.90 remains appropriate for the constrained budget-5 condition.
- **H — Identifiability:** Preserved. W/L still require context evidence in context B, and N/S require primary-sensor evidence beyond trusted measurements, but the complete intervention signatures remain distinct.

## Final recommendation

**B — KEEP ARCHITECTURE, CHANGE EVALUATION SETTING.** Freeze the V2 probabilities and architecture, but use budget 8 / threshold 0.95 as the primary operating point; retain budget 5 / threshold 0.90 as a constrained-efficiency condition.

No provider or LLM was invoked. No commit or push was performed.
