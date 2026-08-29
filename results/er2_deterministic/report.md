# ER-2 minimal deterministic repair benchmark

## Technical summary

All four canonical repairs recover 100% affected-region accuracy and cause zero collateral damage. The full 4×4 matrix is non-degenerate: wrong repairs can fail, compensate for one output layer, or fix one context while damaging another.

The central negative example is `SENSOR_CORRUPTION + UPDATE_WORLD_MODEL`: affected-region accuracy rises from 0% before repair to 66.7%, but unaffected physical knowledge falls to 0.0%, producing 1.000 collateral damage. The wrong world update predicts corrupted end-to-end observations correctly for the wrong reason.

## The repair matrix separates recovery from preservation

| True hypothesis | Applied repair | Overall | Affected | Unaffected | Collateral damage | Selection correct | Repair success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NO_STRUCTURAL_CHANGE | NO_REPAIR | 100.0% | 100.0% | 100.0% | 0.000 | yes | yes |
| NO_STRUCTURAL_CHANGE | UPDATE_WORLD_MODEL | 20.0% | 0.0% | 25.0% | 0.750 | no | no |
| NO_STRUCTURAL_CHANGE | RECALIBRATE_SENSOR | 40.0% | 50.0% | 37.5% | 0.625 | no | no |
| NO_STRUCTURAL_CHANGE | ADD_LATENT_VARIABLE | 60.0% | 100.0% | 50.0% | 0.500 | no | no |
| WORLD_SHIFT | NO_REPAIR | 20.0% | 0.0% | 100.0% | 0.000 | no | no |
| WORLD_SHIFT | UPDATE_WORLD_MODEL | 100.0% | 100.0% | 100.0% | 0.000 | yes | yes |
| WORLD_SHIFT | RECALIBRATE_SENSOR | 40.0% | 50.0% | 0.0% | 1.000 | no | no |
| WORLD_SHIFT | ADD_LATENT_VARIABLE | 60.0% | 50.0% | 100.0% | 0.000 | no | no |
| SENSOR_CORRUPTION | NO_REPAIR | 40.0% | 0.0% | 100.0% | 0.000 | no | no |
| SENSOR_CORRUPTION | UPDATE_WORLD_MODEL | 40.0% | 66.7% | 0.0% | 1.000 | no | no |
| SENSOR_CORRUPTION | RECALIBRATE_SENSOR | 100.0% | 100.0% | 100.0% | 0.000 | yes | yes |
| SENSOR_CORRUPTION | ADD_LATENT_VARIABLE | 40.0% | 33.3% | 50.0% | 0.500 | no | no |
| MISSING_LATENT_VARIABLE | NO_REPAIR | 60.0% | 0.0% | 100.0% | 0.000 | no | no |
| MISSING_LATENT_VARIABLE | UPDATE_WORLD_MODEL | 60.0% | 100.0% | 33.3% | 0.667 | no | no |
| MISSING_LATENT_VARIABLE | RECALIBRATE_SENSOR | 40.0% | 50.0% | 33.3% | 0.667 | no | no |
| MISSING_LATENT_VARIABLE | ADD_LATENT_VARIABLE | 100.0% | 100.0% | 100.0% | 0.000 | yes | yes |

`repair_success` is behavioral rather than label-only: affected accuracy must be 1.0, collateral damage must be non-positive, and overall accuracy must not decline. Repair-selection correctness is reported separately against the canonical mapping.

## Fixed baselines expose the value of diagnosis-conditioned repair

| Baseline | Selection accuracy | Repair success | Overall | Affected | Unaffected | Collateral damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALWAYS_NO_REPAIR | 25.0% | 25.0% | 55.0% | 25.0% | 100.0% | 0.000 |
| ALWAYS_UPDATE_WORLD_MODEL | 25.0% | 25.0% | 55.0% | 66.7% | 39.6% | 0.604 |
| ALWAYS_RECALIBRATE_SENSOR | 25.0% | 25.0% | 55.0% | 62.5% | 42.7% | 0.573 |
| ALWAYS_ADD_LATENT_VARIABLE | 25.0% | 25.0% | 65.0% | 70.8% | 75.0% | 0.250 |
| ORACLE_REPAIR | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 |

Each always-repair baseline selects correctly in exactly one of four hypotheses. Baseline metrics are unweighted macro means across the four hypotheses; within an episode, overall accuracy is case-weighted over the fixed ten-case suite. The oracle uses the same state mutation and post-repair evaluator as every baseline; it has no scoring exception.

## Wrong learning can match observations while corrupting physics

For sensor corruption, the healthy pre-repair state has 40.0% overall accuracy. Updating the world model leaves overall accuracy at 40.0%: it repairs all four end-to-end `X→O` cases, but the two direct sensor-mapping cases remain wrong and all four physical `X→Y` cases become wrong. Thus the affected-region score is 66.7%, unaffected-region accuracy is 0.0%, and collateral damage is 1.000.

This is a concrete demonstration of why observation agreement alone is insufficient evidence for changing the world model.

## Correct world repair recovers the shifted process without spillover

For world shift, `UPDATE_WORLD_MODEL` raises overall accuracy from 20.0% to 100.0%. Affected physical and end-to-end cases reach 100.0%; the unchanged sensor mapping remains at 100.0%; collateral damage is 0.000.

## State, suite, and metric definitions

The pre-repair state has three independent components: world relation `Y=X`, primary-sensor calibration `O=Y`, and absent context/latent structure. `UPDATE_WORLD_MODEL` changes only the world relation to `Y=1-X`; `RECALIBRATE_SENSOR` changes only the sensor mapping to `O=1-Y`; `ADD_LATENT_VARIABLE` adds `Y=X` in context A and `Y=1-X` in context B; `NO_REPAIR` is identity.

Each hypothesis is evaluated on the same ten deterministic predictions: four physical `X,context→Y` cases, two direct sensor `Y→O` cases, and four end-to-end `X,context→O` cases. For structural hypotheses, affected cases are exactly those whose targets differ from the healthy pre-change process. Because `NO_STRUCTURAL_CHANGE` has no persistent changed region, its affected region is explicitly defined as two trigger-adjacent held-out probes (`X=1`, context A, physical and end-to-end); the other eight cases are unaffected preservation checks.

Collateral damage is not clamped:

```text
pre-repair accuracy on unaffected cases
- post-repair accuracy on unaffected cases
```

Positive values mean damage; zero means preservation; negative values would mean improvement on previously unaffected cases.

## Acceptance checks

- **A — Correct repair recovers affected performance:** PASS. Every canonical repair reaches 100% affected-region accuracy.
- **B — Correct repair preserves unaffected knowledge:** PASS. Every canonical repair has zero collateral damage. Some wrong repairs also preserve unaffected cases while failing to repair the affected region, so preservation alone is not sufficient; correct repairs dominate on the joint recovery-and-preservation criterion.
- **C — Wrong repair can improve observations while causing damage:** PASS. Sensor corruption plus world update repairs end-to-end observations while destroying physical predictions; missing latent plus a global world update similarly fixes context B while damaging context A.
- **D — Sensor corruption demonstrates knowing when not to learn:** PASS. Updating physics fits corrupted observations for the wrong causal reason and incurs maximal physical collateral damage.
- **E — Ready for LLM repair selection:** PASS WITH CAVEAT. The state/evaluator is behaviorally non-trivial and suitable for testing repair choices. However, diagnosis-to-repair selection may be easy if semantically transparent repair labels are exposed; future LLM claims should distinguish label matching from understanding state consequences.

## Limitations and next step

ER-2 V0 is deterministic and isolates repair selection after an externally supplied diagnosis. It does not yet propagate ER-1 diagnostic uncertainty, execute LLM calls, model repair costs, or test sequential/multiple repairs. The appropriate next step is mocked repair-policy integration and prompt/interface design, not a live model study yet.

Exact rows are available in `repair_matrix.csv` and `baseline_summary.csv`.
