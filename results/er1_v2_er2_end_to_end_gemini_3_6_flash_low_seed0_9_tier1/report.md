# ER-1 V2 to ER-2 end-to-end repair study

## Technical summary

Completed 40 of 40 planned episodes. Among valid end-to-end episodes, diagnosis accuracy is 0.750, repair selection accuracy is 0.550, and the joint correct-diagnosis/correct-repair rate is 0.500. Protocol failures remain outside valid-choice denominators and are reported separately.

The strongest wrong-learning pattern—sensor corruption followed by a world-model update—occurred in 0 completed episode(s). Exact cases and collateral outcomes are in `wrong_repair_analysis.csv`.

## Diagnosis and repair outcomes by hidden hypothesis

| Hypothesis | Valid | Diagnosis accuracy | Repair accuracy | Repair success | Post-repair accuracy | Collateral damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NO_STRUCTURAL_CHANGE | 10 | 0.900 | 0.300 | 0.300 | 0.700 | 0.362 |
| WORLD_SHIFT | 10 | 0.500 | 0.500 | 0.500 | 0.740 | 0.300 |
| SENSOR_CORRUPTION | 10 | 0.700 | 0.900 | 0.900 | 0.940 | 0.050 |
| MISSING_LATENT_VARIABLE | 10 | 0.900 | 0.500 | 0.500 | 0.760 | 0.267 |

## Failure chain remains decomposed

| Category | Count | Fraction of completed episodes |
| --- | ---: | ---: |
| CORRECT_DIAGNOSIS_CORRECT_REPAIR | 20 | 0.500 |
| CORRECT_DIAGNOSIS_WRONG_REPAIR | 10 | 0.250 |
| WRONG_DIAGNOSIS_REPAIR_CONSISTENT_WITH_WRONG_DIAGNOSIS | 5 | 0.125 |
| WRONG_DIAGNOSIS_BUT_CORRECT_REPAIR | 2 | 0.050 |
| WRONG_DIAGNOSIS_OTHER_WRONG_REPAIR | 3 | 0.075 |
| SCIENTIFIC_MODEL_FAILURE | 0 | 0.000 |
| PROVIDER_FAILURE | 0 | 0.000 |
| RATE_LIMIT_FAILURE | 0 | 0.000 |

## Scope, definitions, and experimental design

The cohort is the matched four-hypothesis by seed grid. Diagnosis comes from the unchanged ER-1 V2 FULL_AUTONOMOUS interaction. The repair request receives only its final agent-visible investigation record and frozen neutral A/B/C/D repair descriptions. Trusted Python translates the option and invokes the unchanged deterministic ER-2 mutation/evaluator.

Behavioral repair means use valid repair selections. Diagnosis accuracy uses episodes with a completed model diagnosis. Joint and conditional repair metrics use valid end-to-end episodes. Failure counts retain every completed planned cell.

## Counterfactual checks distinguish selection from benchmark degeneracy

For every completed investigation with a diagnosis, all four repairs are evaluated without additional model calls. `counterfactual_repairs.csv` records chosen, oracle, and alternative outcomes using the same held-out evaluator.

## Limitations and robustness

The provider interface is stateless across calls. To honor the strict no-diagnosis-label boundary, the repair request repeats the complete agent-visible evidence but does not re-inject the model's diagnosis label or rationale. The analysis therefore compares the recorded diagnosis with a repair chosen from the same evidence, rather than relying on hidden conversational memory. The supplied-diagnosis 40/40 control remains external and is not pooled into these denominators.

## Recommended next step

Run the frozen 40-cell Tier-1 command once, inspect protocol failures before any scientific interpretation, and do not change prompts based on individual outcomes.

## Further questions

If repair errors remain after correct diagnoses, compare their option positions and counterfactual collateral damage. If errors primarily follow wrong diagnoses, the bottleneck is evidence interpretation rather than repair semantics.
