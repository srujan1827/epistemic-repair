# ER-1 V2 Analytic Validation and Small Oracle Sanity Study

This is an implementation diagnostic, not a final scientific calibration.

## Transient trigger anomaly

| Hypothesis | P(A0|H) | P(H|A0) |
| --- | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | 0.3000000000 | 0.1276595745 |
| `WORLD_SHIFT` | 0.7000000000 | 0.2978723404 |
| `SENSOR_CORRUPTION` | 0.6500000000 | 0.2765957447 |
| `MISSING_LATENT_VARIABLE` | 0.7000000000 | 0.2978723404 |

## Persistent investigation likelihoods

The trigger probabilities do not appear in this table.

| Current | Action | Signal | Hypothesis | Effective | P(0) | P(1) |
| --- | --- | --- | --- | --- | ---: | ---: |
| A | `REPEAT_TRIAL` | PRIMARY_O | `NO_STRUCTURAL_CHANGE` | A | 0.095000 | 0.905000 |
| A | `REPEAT_TRIAL` | PRIMARY_O | `WORLD_SHIFT` | A | 0.860000 | 0.140000 |
| A | `REPEAT_TRIAL` | PRIMARY_O | `SENSOR_CORRUPTION` | A | 0.860000 | 0.140000 |
| A | `REPEAT_TRIAL` | PRIMARY_O | `MISSING_LATENT_VARIABLE` | A | 0.140000 | 0.860000 |
| A | `USE_TRUSTED_SENSOR` | TRUSTED_T | `NO_STRUCTURAL_CHANGE` | A | 0.059000 | 0.941000 |
| A | `USE_TRUSTED_SENSOR` | TRUSTED_T | `WORLD_SHIFT` | A | 0.892000 | 0.108000 |
| A | `USE_TRUSTED_SENSOR` | TRUSTED_T | `SENSOR_CORRUPTION` | A | 0.059000 | 0.941000 |
| A | `USE_TRUSTED_SENSOR` | TRUSTED_T | `MISSING_LATENT_VARIABLE` | A | 0.108000 | 0.892000 |
| A | `CHANGE_CONTEXT` | PRIMARY_O | `NO_STRUCTURAL_CHANGE` | B | 0.095000 | 0.905000 |
| A | `CHANGE_CONTEXT` | PRIMARY_O | `WORLD_SHIFT` | B | 0.860000 | 0.140000 |
| A | `CHANGE_CONTEXT` | PRIMARY_O | `SENSOR_CORRUPTION` | B | 0.860000 | 0.140000 |
| A | `CHANGE_CONTEXT` | PRIMARY_O | `MISSING_LATENT_VARIABLE` | B | 0.860000 | 0.140000 |
| B | `REPEAT_TRIAL` | PRIMARY_O | `NO_STRUCTURAL_CHANGE` | B | 0.095000 | 0.905000 |
| B | `REPEAT_TRIAL` | PRIMARY_O | `WORLD_SHIFT` | B | 0.860000 | 0.140000 |
| B | `REPEAT_TRIAL` | PRIMARY_O | `SENSOR_CORRUPTION` | B | 0.860000 | 0.140000 |
| B | `REPEAT_TRIAL` | PRIMARY_O | `MISSING_LATENT_VARIABLE` | B | 0.860000 | 0.140000 |
| B | `USE_TRUSTED_SENSOR` | TRUSTED_T | `NO_STRUCTURAL_CHANGE` | B | 0.059000 | 0.941000 |
| B | `USE_TRUSTED_SENSOR` | TRUSTED_T | `WORLD_SHIFT` | B | 0.892000 | 0.108000 |
| B | `USE_TRUSTED_SENSOR` | TRUSTED_T | `SENSOR_CORRUPTION` | B | 0.059000 | 0.941000 |
| B | `USE_TRUSTED_SENSOR` | TRUSTED_T | `MISSING_LATENT_VARIABLE` | B | 0.892000 | 0.108000 |
| B | `CHANGE_CONTEXT` | PRIMARY_O | `NO_STRUCTURAL_CHANGE` | A | 0.095000 | 0.905000 |
| B | `CHANGE_CONTEXT` | PRIMARY_O | `WORLD_SHIFT` | A | 0.860000 | 0.140000 |
| B | `CHANGE_CONTEXT` | PRIMARY_O | `SENSOR_CORRUPTION` | A | 0.860000 | 0.140000 |
| B | `CHANGE_CONTEXT` | PRIMARY_O | `MISSING_LATENT_VARIABLE` | A | 0.140000 | 0.860000 |

## Initial expected information gain

| Action | EIG (bits) |
| --- | ---: |
| `REPEAT_TRIAL` | 0.223648936 |
| `USE_TRUSTED_SENSOR` | 0.566200312 |
| `CHANGE_CONTEXT` | 0.425899649 |

## 800-episode sanity results

Runtime: 1.764 seconds; 100 seeds per hypothesis/budget.

| Budget | Hypothesis | MAP | Success@0.90 | Mean experiments | Mean true posterior |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | `NO_STRUCTURAL_CHANGE` | 0.820 | 0.730 | 3.260 | 0.724 |
| 5 | `WORLD_SHIFT` | 0.960 | 0.930 | 3.500 | 0.924 |
| 5 | `SENSOR_CORRUPTION` | 0.920 | 0.910 | 3.430 | 0.854 |
| 5 | `MISSING_LATENT_VARIABLE` | 0.850 | 0.850 | 3.610 | 0.827 |
| 8 | `NO_STRUCTURAL_CHANGE` | 0.810 | 0.790 | 3.490 | 0.746 |
| 8 | `WORLD_SHIFT` | 0.960 | 0.960 | 3.570 | 0.935 |
| 8 | `SENSOR_CORRUPTION` | 0.940 | 0.920 | 3.520 | 0.866 |
| 8 | `MISSING_LATENT_VARIABLE` | 0.920 | 0.920 | 3.820 | 0.867 |

## Structural safety rates

- Budget 5: overall MAP=0.887, Success@0.90=0.855, false structural=0.180, missed structural=0.013, mean experiments=3.450.
- Budget 8: overall MAP=0.907, Success@0.90=0.897, false structural=0.190, missed structural=0.003, mean experiments=3.600.

## Confusion matrix: budget 5

| Truth | N | W | S | L |
| --- | ---: | ---: | ---: | ---: |
| N | 82 (82.0%) | 1 (1.0%) | 12 (12.0%) | 5 (5.0%) |
| W | 0 (0.0%) | 96 (96.0%) | 0 (0.0%) | 4 (4.0%) |
| S | 3 (3.0%) | 2 (2.0%) | 92 (92.0%) | 3 (3.0%) |
| L | 1 (1.0%) | 7 (7.0%) | 7 (7.0%) | 85 (85.0%) |

## Confusion matrix: budget 8

| Truth | N | W | S | L |
| --- | ---: | ---: | ---: | ---: |
| N | 81 (81.0%) | 0 (0.0%) | 13 (13.0%) | 6 (6.0%) |
| W | 0 (0.0%) | 96 (96.0%) | 1 (1.0%) | 3 (3.0%) |
| S | 1 (1.0%) | 2 (2.0%) | 94 (94.0%) | 3 (3.0%) |
| L | 0 (0.0%) | 3 (3.0%) | 5 (5.0%) | 92 (92.0%) |

## Representative seed-0 traces

### NO_STRUCTURAL_CHANGE

Initial: [N=0.1277, W=0.2979, S=0.2766, L=0.2979]
- `USE_TRUSTED_SENSOR` {'trusted_t': 1} → [N=0.2701, W=0.0723, S=0.5852, L=0.0723]
- `CHANGE_CONTEXT` {'context': 'A', 'primary_o': 1} → [N=0.6131, W=0.0254, S=0.2055, L=0.1560]
- `CHANGE_CONTEXT` {'context': 'B', 'primary_o': 1} → [N=0.9111, W=0.0058, S=0.0472, L=0.0359]
- Final: `NO_STRUCTURAL_CHANGE`; THRESHOLD_REACHED.

### WORLD_SHIFT

Initial: [N=0.1277, W=0.2979, S=0.2766, L=0.2979]
- `USE_TRUSTED_SENSOR` {'trusted_t': 0} → [N=0.0136, W=0.4785, S=0.0294, L=0.4785]
- `CHANGE_CONTEXT` {'context': 'A', 'primary_o': 0} → [N=0.0026, W=0.8148, S=0.0500, L=0.1326]
- `USE_TRUSTED_SENSOR` {'trusted_t': 0} → [N=0.0002, W=0.9766, S=0.0040, L=0.0192]
- Final: `WORLD_SHIFT`; THRESHOLD_REACHED.

### SENSOR_CORRUPTION

Initial: [N=0.1277, W=0.2979, S=0.2766, L=0.2979]
- `USE_TRUSTED_SENSOR` {'trusted_t': 1} → [N=0.2701, W=0.0723, S=0.5852, L=0.0723]
- `CHANGE_CONTEXT` {'context': 'A', 'primary_o': 0} → [N=0.0427, W=0.1035, S=0.8370, L=0.0168]
- `USE_TRUSTED_SENSOR` {'trusted_t': 1} → [N=0.0470, W=0.0131, S=0.9223, L=0.0176]
- Final: `SENSOR_CORRUPTION`; THRESHOLD_REACHED.

### MISSING_LATENT_VARIABLE

Initial: [N=0.1277, W=0.2979, S=0.2766, L=0.2979]
- `USE_TRUSTED_SENSOR` {'trusted_t': 0} → [N=0.0136, W=0.4785, S=0.0294, L=0.4785]
- `CHANGE_CONTEXT` {'context': 'A', 'primary_o': 1} → [N=0.0248, W=0.1354, S=0.0083, L=0.8315]
- `USE_TRUSTED_SENSOR` {'trusted_t': 1} → [N=0.0296, W=0.0186, S=0.0099, L=0.9419]
- Final: `MISSING_LATENT_VARIABLE`; THRESHOLD_REACHED.

## Interpretation

There is no exact full-signature non-identifiability: every hypothesis has a distinct distribution when all available actions and contexts are considered, and every ordinary binary outcome remains possible.

Important conditional ambiguities remain. In context B, WORLD_SHIFT and MISSING_LATENT_VARIABLE have identical repeat and trusted distributions, so a context intervention is necessary to distinguish that pair. NO_STRUCTURAL_CHANGE and SENSOR_CORRUPTION have identical trusted-sensor distributions, so trusted evidence alone cannot distinguish them; primary-sensor evidence is necessary.

A normal repeat (`O=1`) raises NO_STRUCTURAL_CHANGE from 0.127660 to 0.486124 in one update, a 3.81× posterior increase. This is strong positive rehabilitation evidence after the transient trigger.

USE_TRUSTED_SENSOR has the largest initial EIG (0.566200 bits versus 0.425900 for CHANGE_CONTEXT and 0.223649 for REPEAT_TRIAL), so the oracle chooses it first in every sanity episode. It does not dominate the whole trajectory: subsequent choices adapt between trusted measurements and context interventions. Oracle action regret and premature-diagnosis rate are both zero by construction.

No full calibration or parameter tuning was performed.
