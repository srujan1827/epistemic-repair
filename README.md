# Epistemic Repair

**Epistemic Repair** studies what an AI system should do when its predictions stop matching observations. A mismatch does not necessarily mean that the world model is wrong: the world may have changed, the observation channel may be corrupted, important context may be missing, or there may be no persistent structural change at all.

The central question is whether an agent can diagnose what went wrong, actively gather evidence, and apply the smallest appropriate repair without damaging knowledge that was already correct.

## The toy world

The benchmark uses a controlled binary system where `X` is the input, `Y` is the true physical outcome, `O` is the observation reported by the primary sensor, and `context` is an environmental condition such as A or B.

```text
X ---> Y ---> O
     world   sensor
```

Under normal behavior, the physical relation is approximately `Y = X`, the primary sensor usually reports `O = Y`, and context does not normally change the relation. Now suppose the system sees `X = 1` but `O = 0`. That observation alone does **not** reveal what failed: the world may have changed, the sensor may be corrupted, context may matter even though the model ignores it, or the event may have been transient with no persistent change.

| Failure mechanism | What changed in the toy world? | Correct repair |
| --- | --- | --- |
| `NO_STRUCTURAL_CHANGE` | The triggering anomaly was transient; the underlying world, sensor, and contextual structure remain healthy | `NO_REPAIR` |
| `WORLD_SHIFT` | The physical relationship changes from approximately `Y = X` to `Y = 1 - X` | `UPDATE_WORLD_MODEL` |
| `SENSOR_CORRUPTION` | The physical relationship remains healthy, but the primary sensor tends to report the opposite outcome | `RECALIBRATE_SENSOR` |
| `MISSING_LATENT_VARIABLE` | The physical relationship depends on context, so one global `X -> Y` rule is insufficient | `ADD_LATENT_VARIABLE` |

The same initial mismatch can therefore come from different underlying mechanisms, so the agent should investigate before deciding what to change. It can repeat the trial, use a trusted sensor, or change context.

## Benchmark stages

### ER-0: deterministic diagnosis

ER-0 uses this toy world to test whether different hidden failure mechanisms can actually be distinguished through interventions. It validates deterministic causal distinctions, information boundaries, and the basic diagnostic loop:

```text
Detect -> Hypothesize -> Experiment -> Diagnose
```

ER-0 includes a Bayesian likelihood model, an information-gain oracle, and a random diagnostic baseline.

### ER-1: active investigation under uncertainty

ER-1 turns diagnosis into a stochastic active-investigation problem. ER-1 V2 evaluates four hypotheses:

- `NO_STRUCTURAL_CHANGE`: the triggering anomaly was transient;
- `WORLD_SHIFT`: the physical input-output relationship changed;
- `SENSOR_CORRUPTION`: the physical relationship remains stable, but the primary sensor misreports it; and
- `MISSING_LATENT_VARIABLE`: behavior depends systematically on context.

The agent can repeat a trial, use a trusted but imperfect sensor, or change context. ER-1 V1 exposed a benchmark-design problem: the initial anomaly and persistent dynamics were not cleanly separated. V2 fixes this by treating the trigger as a transient event and using subsequent experiments to measure persistent behavior.

Before LLM evaluation, the benchmark was validated through Bayesian oracle simulation. The full oracle calibration sweep contained **60,000 simulated episodes**. Its primary operating-point cell—an experiment budget of 8 and a diagnosis threshold of 0.95—contained **4,000 simulated episodes** and achieved **95.8%** MAP diagnosis accuracy and **92.8%** threshold-qualified success. These are oracle-simulation results, not Gemini LLM or API runs ([calibration report](results/er1_v2_oracle_calibration.md), [overall results](results/er1_v2_oracle_overall.csv)).

The Gemini 3.6 Flash study compared `FULL_AUTONOMOUS`, `PLANNER_ONLY`, and `THRESHOLD_AWARE_AUTONOMOUS` over ten seeds per hypothesis. Planner-only had lower action regret and higher oracle-action agreement than full autonomy, while explicit threshold information reduced—but did not eliminate—premature diagnosis. These descriptive results suggest that better belief information helps planning, while stopping and confidence calibration remain important weaknesses. The sample uses one model and one reasoning configuration; it is not a statistical significance claim ([full analysis](results/er1_v2_gemini_3_6_flash_low_seed0_9_final_analysis/report.md)).

### ER-2: repair execution and collateral damage

ER-2 remains in the same controlled toy world and asks: given a diagnosis, what internal component should the toy agent actually change? It adds explicit repairs to independent components of the agent's predictive state:

| Diagnosed cause | Minimal repair |
| --- | --- |
| `NO_STRUCTURAL_CHANGE` | `NO_REPAIR` |
| `WORLD_SHIFT` | `UPDATE_WORLD_MODEL` |
| `SENSOR_CORRUPTION` | `RECALIBRATE_SENSOR` |
| `MISSING_LATENT_VARIABLE` | `ADD_LATENT_VARIABLE` |

These are benchmark state mutations, not real-world model repair. They are evaluated on affected-region recovery, preservation of unaffected knowledge, overall post-repair accuracy, and collateral damage. The deterministic 4×4 repair matrix demonstrates the core risk: a wrong repair can improve behavior in the region that triggered adaptation while damaging previously correct knowledge. For example, changing the world model after sensor corruption partially improves affected behavior but reduces unaffected-region accuracy to zero ([repair matrix](results/er2_deterministic/repair_matrix.csv)).

## Main results

All LLM results below used **Gemini 3.6 Flash** with low thinking. They are small, matched-seed studies in a toy benchmark.

| Study | Main result |
| --- | --- |
| ER-1 V2 oracle, budget 8 / threshold 0.95 | 95.8% MAP accuracy; 92.8% threshold success |
| ER-1 V2 LLM diagnosis | Planner-only reduced mean action regret from 0.168 to 0.081; threshold awareness reduced valid-episode premature diagnosis from 92.5% to 78.4% |
| ER-2 supplied causal diagnosis control | 40/40 correct repair selections |
| ER-1 → ER-2 end-to-end | 40/40 completed; 75% diagnosis accuracy; 55% repair accuracy; 50% correct diagnosis and repair |

### Supplied causal diagnosis control

The model received a correct causal description and seed-randomized neutral repair options labelled A–D. It selected the correct repair in **40/40 episodes**, with no protocol or provider failures ([control summary](results/er2_gemini_3_6_flash_low_seed0_9_causal_repair_tier1/summary.csv)). This shows that repair selection is easy for Gemini 3.6 Flash in this toy setting when the causal failure is already known. It does **not** solve the broader Epistemic Repair problem.

### End-to-end: no supplied diagnosis

In the end-to-end condition, the model first investigated through ER-1 evidence and then selected a neutral ER-2 repair without receiving the ground-truth diagnosis or a benchmark-written causal interpretation. All **40/40 episodes completed** with no provider, rate-limit, or scientific-model failures. Diagnosis accuracy was **75%**, repair-selection accuracy was **55%**, and the joint correct-diagnosis/correct-repair rate was **50%**. Mean collateral damage was **0.245**, showing that wrong repair choices measurably damaged unaffected knowledge ([end-to-end summary](results/er1_v2_er2_end_to_end_gemini_3_6_flash_low_seed0_9_tier1/summary.csv)).

The repair call is stateless: it reconstructs its choice from the agent-visible investigation record rather than carrying an internal belief state across provider calls. This limitation should be considered when interpreting diagnosis-repair consistency.

> **Main takeaway:** When the cause is explicitly known, repair selection is easy in this toy setting. The harder problem is determining the cause from noisy evidence and deciding what deserves adaptation. Wrong causal attribution can produce locally plausible but globally damaging repairs.

## Installation and use

The project requires Python 3.10 or later. Core benchmark code has no runtime dependencies; development tests use `pytest`, and live Gemini runs use the optional `google-genai` dependency.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Run the deterministic ER-2 matrix and report without network access:

```bash
python -m scripts.demo_er2_repairs --overwrite
```

For live LLM experiments, install the Gemini extra and set `GEMINI_API_KEY` in the environment or a local gitignored `.env` file. Never commit the key.

```bash
python -m pip install -e ".[dev,gemini]"
```

Supported experiment commands include:

```bash
python -m scripts.run_er1_v2_llm_comparison --provider gemini --model gemini-3.6-flash --thinking-level low --conditions full planner threshold_aware --seeds 0..9 --budget 8 --diagnosis-threshold 0.95 --max-decision-calls 9 --output-dir results/er1_v2_llm_run

python -m scripts.run_er2_llm_repair_selection --provider gemini --model gemini-3.6-flash --thinking-level low --seeds 0..9 --output-dir results/er2_repair_control

python -m scripts.run_er1_v2_er2_end_to_end --provider gemini --model gemini-3.6-flash --thinking-level low --seeds 0..9 --budget 8 --diagnosis-threshold 0.95 --max-decision-calls 9 --output-dir results/er1_v2_er2_end_to_end
```

These commands make live API calls and may encounter changing free-tier limits. Use `--preflight-only` on the ER-2 control or end-to-end command to validate prompts and permutations without creating a provider client.

## Repository structure

```text
epistemic_repair/
  er1_v2/       stochastic investigation environment and runner
  er2/          repair state, LLM selection, and held-out evaluation
  evaluation/   calibration, comparison, and reporting utilities
  llm/          provider-neutral interfaces and Gemini adapter
  prompts/      versioned ER-0 and ER-1 prompts
scripts/        demos, calibration, analysis, and live-run CLIs
tests/          deterministic and mocked-provider test suite
results/        committed calibration and experiment artifacts
```

The benchmark is intentionally small. It does not yet establish general performance across models, larger causal systems, or real-world repair tasks.
