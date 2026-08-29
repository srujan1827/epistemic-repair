# ER-2 LLM causal repair-selection study

## Technical summary

Completed 40 of 40 planned episodes. Valid structured choices: 40; repair-selection accuracy among valid choices: 1.000. Structured-output and provider failures are reported separately and are not relabeled as wrong repairs.

Among valid wrong choices, 0 cases partially improved affected behavior while causing positive collateral damage to unaffected knowledge.

## Per-hypothesis metrics

| Hypothesis | Valid | Selection accuracy | Repair success | Overall | Affected | Unaffected | Collateral damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO_STRUCTURAL_CHANGE | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| WORLD_SHIFT | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| SENSOR_CORRUPTION | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| MISSING_LATENT_VARIABLE | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Method and denominator

The externally supplied diagnosis equals the evaluation truth. The model sees only its fixed causal description and seed-permuted A/B/C/D consequence text. Trusted Python translates the option and invokes the unchanged deterministic ER-2 mutation/evaluator. Behavioral metric means use valid structured choices; completion and failure counts retain all planned episodes.
