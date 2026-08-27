# Epistemic Repair

This repository starts a research program around a simple question: can an AI
agent determine **why** a prediction became wrong before deciding what to learn
or repair?

Identical visible errors can have different epistemic causes. Updating the world
model is appropriate after a real world shift, but harmful when the sensor is
the faulty component. Likewise, an apparently broken rule may instead be an
incomplete rule that omits a relevant latent variable.

## ER-0: deterministic epistemic diagnosis

The binary machine accepts an agent-chosen bit `X`, computes a true physical
output `Y`, and returns sensor observation `O`. `Y` and optional latent variable
`Z` remain hidden from the ordinary agent observation.

V0 implements exactly four episode conditions:

| Condition | Physical rule | Sensor rule | Correct repair |
| --- | --- | --- | --- |
| `NORMAL` | `Y = X` | `O = Y` | `NO_REPAIR` |
| `WORLD_SHIFT` | `Y = 1 - X` | `O = Y` | `UPDATE_WORLD_MODEL` |
| `SENSOR_CORRUPTION` | `Y = X` | `O = 1 - Y` | `RECALIBRATE_SENSOR` |
| `MISSING_LATENT_VARIABLE` | `Y = X` if `Z=0`, otherwise `Y = 1-X` | `O = Y` | `ADD_LATENT_VARIABLE` |

For the latent-variable condition, `Z=0` reproduces the pre-anomaly behavior
and `Z=1` produces the anomaly. A missing-latent episode defaults to `Z=1`, but
either value can be selected explicitly at reset for controlled experiments.

All three failure conditions deliberately produce the same initial visible
anomaly for `X=1`:

```text
X=1 -> O=0
```

That observation alone cannot identify its cause. The simulator retains `Y`,
`Z`, the failure mode, and the correct repair behind the separate
`get_ground_truth()` evaluation interface. Agent code should receive only the
`Observation(x, o)` returned by `step()`.

ER-0 remains a standalone deterministic benchmark. Its environment,
likelihoods, three-hypothesis belief state, trusted-sensor semantics, prompts,
and evaluation behavior are unchanged by ER-1.

## ER-1 V1: single-process stochastic diagnosis

ER-1 asks a harder question: did anything structurally fail at all? In a noisy
world, adaptation can itself be harmful. An investigator must distinguish
evidence of a structural failure from ordinary variation—most importantly, it
must learn **when not to repair**.

ER-1 is implemented by separate stochastic environment, likelihood, belief,
policy-view, runner, and prompt modules. It adds the genuine competing
hypothesis `NO_STRUCTURAL_CHANGE`, whose correct repair label is `NO_REPAIR`.
The other repair mappings are unchanged.

The centralized default generative parameters are:

| Component | ER-1 probability |
| --- | ---: |
| Preferred physical relation | `0.90` |
| Reliable primary sensor reports `O=Y` | `0.95` |
| Corrupted primary sensor reports `O=1-Y` | `0.90` |
| Trusted sensor reports `T=Y` | `0.99` |
| Normative diagnosis threshold | `0.90` |

`NO_STRUCTURAL_CHANGE` and `SENSOR_CORRUPTION` retain a physical process that
prefers `Y=X`; `WORLD_SHIFT` prefers `Y=1-X`. `MISSING_LATENT_VARIABLE`
prefers `Y=X` in context A and `Y=1-X` in context B. The reliable primary
sensor applies to every hypothesis except sensor corruption, and all ordinary
binary outcomes have non-zero probability.

Every investigator is deliberately shown the same selected entry event,
`X=1, O=0`, in context B. The environment constructs that visible event
directly rather than rejection-sampling it. The normative initial belief state
explicitly accounts for the selection:

```text
P(H | X=1,O=0) ∝ P(O=0 | X=1,H,context=B) P(H)
```

With equal `0.25` base priors, the anomaly likelihoods and conditioned beliefs
are:

| Hypothesis | `P(O=0 | X=1,H,B)` | Conditioned belief |
| --- | ---: | ---: |
| `NO_STRUCTURAL_CHANGE` | `0.14` | `0.052239` |
| `WORLD_SHIFT` | `0.86` | `0.320896` |
| `SENSOR_CORRUPTION` | `0.82` | `0.305970` |
| `MISSING_LATENT_VARIABLE` | `0.86` | `0.320896` |

The hidden initial `Y` is sampled from its appropriate conditional
distribution given the selected observation. It is never agent-visible.
Episodes use a private random generator reinitialized from the recorded episode
seed, so the same seed and action sequence reproduce the same trajectory
without depending on global random state.

ER-1 changes the trusted action deliberately: `USE_TRUSTED_SENSOR` performs a
new physical trial and returns noisy trusted observation `T`, not hidden `Y`.
The `0.99` reliability makes it highly informative without allowing one
measurement to collapse the posterior mathematically. ER-0 continues to return
its existing perfect trusted measurement of `Y`.

## ER-1 V2: transient trigger, persistent investigation

ER-1 V1 remains the historical baseline: its selected trigger anomaly and its
later investigation evidence are explained by one stochastic process. V2 is
additive and separates those roles explicitly:

1. A one-time **transient trigger anomaly** model explains why the episode
   entered investigation after `X=1, O=0`.
2. A separate **persistent investigation dynamics** model generates every
   later `REPEAT_TRIAL`, `USE_TRUSTED_SENSOR`, and `CHANGE_CONTEXT` result.

This separation matters because making an anomaly plausible under
`NO_STRUCTURAL_CHANGE` should not require making every later healthy-system
observation noisier. With equal base priors, V2 conditions exactly once on the
trigger likelihoods `0.30`, `0.70`, `0.65`, and `0.70`, producing:

| Hypothesis | `P(H | A0)` |
| --- | ---: |
| `NO_STRUCTURAL_CHANGE` | `0.1276595745` |
| `WORLD_SHIFT` | `0.2978723404` |
| `SENSOR_CORRUPTION` | `0.2765957447` |
| `MISSING_LATENT_VARIABLE` | `0.2978723404` |

The trigger is constructed directly, consumes no random draw, and has no
hidden trigger `Y`. Every diagnostic experiment is then a fresh trial from the
persistent process. The V2 oracle starts from the trigger-conditioned belief
state, uses only persistent likelihoods for Bayesian updates and expected
information gain, and never receives hidden ground truth.

The V2 LLM prompt is versioned as `binary_er1_v2_001`. It explains the
transient-versus-persistent distinction qualitatively but exposes no trigger
probabilities, investigation parameters, likelihood tables, information-gain
values, repairs, or hidden state. Planner-only receives the current normative
posterior; full-autonomous does not. V1 retains `binary_er1_001` unchanged.

## ER-0 active diagnostic experiments

The environment implements four typed diagnostic actions:

| Action | Required arguments | Agent-visible result |
| --- | --- | --- |
| `REPEAT_TRIAL` | `x` | repeated primary-sensor `X` and `O` |
| `USE_TRUSTED_SENSOR` | `x` | trusted independent measurement of `Y` |
| `CHANGE_CONTEXT` | `x`, target `Context` | target context plus primary-sensor `X` and `O` |
| `INSPECT_LATENT_VARIABLE` | none | whether `Z` exists and its value when available |

`INSPECT_LATENT_VARIABLE` is retained strictly for debugging and internal
evaluation. It is intentionally excluded from `BENCHMARK_ACTIONS` because its
availability flag would reveal the missing-latent condition directly. Policies
can choose exactly these three agent-visible benchmark actions:

```text
REPEAT_TRIAL
USE_TRUSTED_SENSOR
CHANGE_CONTEXT
```

Contexts use an explicit deterministic mechanism: `Context.A` maps to `Z=0`
and `Context.B` maps to `Z=1` only in the missing-latent condition. In the world
shift and sensor-corruption conditions, changing context has no physical effect
and no causally meaningful `Z` is created.

The trusted sensor separates sensor corruption (`Y=1` for `X=1`) from the other
two failures (`Y=0`). A subsequent context change separates a global world shift
from missing latent dependence: the shifted world stays at `O=0`, while the
latent world changes between `O=1` in context A and `O=0` in context B.

No action result contains a hidden failure label or repair label.

## Architecture and information boundaries

- **Environment:** the hidden ground-truth generator. It owns physical `Y`,
  optional internal `Z`, failure mode, and correct repair metadata.
- **Diagnostics:** legitimate interventions and measurements. Their typed
  results expose only evidence produced by the selected action.
- **Beliefs and likelihoods:** normative Bayesian machinery used to define the
  reference investigator and score experiment informativeness.
- **Oracle:** a privileged reference policy. `OraclePolicyView` grants it the
  current belief state, normative likelihood model, observable context, and
  benchmark action set.
- **Benchmark agent:** a restricted policy. `BenchmarkAgentView` contains only
  initial observable history, safe experiment history, current context,
  benchmark actions, and steps remaining.
- **Evaluation:** runs the interaction and attaches ground truth only after
  policy decisions, outside the agent boundary.

The normative `DeterministicLikelihoodModel` is **not available to future LLM
agents**. It is oracle-privileged benchmark machinery. Restricted policies also
cannot access `get_ground_truth()`, hidden `Y` except through a trusted-sensor
result, internal `Z`, failure labels, repair labels, or evaluation traces.

## LLM investigator smoke layer

The LLM is treated as an **investigator policy**, never as the environment. It
receives a typed `BenchmarkAgentView`, returns one schema-constrained decision,
and cannot invoke diagnostics itself. Trusted Python validates the response and
executes the selected safe action. No model tools, browsing, file access, code
execution, or direct `BinaryMachine` reference are provided.

Two experimental conditions remain separate in code, traces, and summaries:

- `FULL_AUTONOMOUS`: receives only the task description, observable history,
  context, safe actions, and remaining experiment budget. It states and updates
  its own beliefs, selects experiments, chooses when to stop, and diagnoses.
- `PLANNER_ONLY`: receives the same safe view plus current normalized Bayesian
  hypothesis probabilities. The probabilities are authoritative; the model is
  evaluated only as an experiment planner and stopping/diagnosis selector.

Planner-only prompts contain no likelihood tables, expected information gains,
oracle action, or evaluation truth. Autonomous prompts contain no normative
posterior at all. The benchmark may compare stated autonomous beliefs with the
normative posterior after the fact, but never feeds that comparison back.

ER-0 prompts remain versioned as `binary_v0_001`; ER-1 uses the separate
`binary_er1_001` methodology. The ER-1 prompt describes process and sensor
noise qualitatively, includes all four hypotheses, and warns against premature
diagnosis. It does not reveal the `0.90`/`0.95`/`0.99` likelihood parameters.
Planner-only may receive the current four-way posterior, but never a likelihood
table or oracle recommendation. Prompts request only a short
`reason_summary`; private chain-of-thought and provider thinking tokens are not
collected. Structured responses may request one of the three benchmark actions
or issue a diagnosis valid for the selected benchmark. Malformed JSON, invalid probabilities,
unsupported actions, contradictory fields, timeouts, rate limits, and provider
errors are recorded explicitly rather than guessed around.

Provider schemas require a stable structural envelope with nullable
branch-specific fields. The strict Python parser remains responsible for
decision-dependent semantics, such as requiring an action only for
`RUN_EXPERIMENT` and a diagnosis (plus autonomous confidence) for `DIAGNOSE`.

### Provider architecture

Benchmark code depends only on `LLMClient.generate(LLMRequest)`. The sole real
adapter currently implemented is `GeminiLLMClient`; no automatic router or
fallback model exists. Provider, model ID, thinking level, output-token bound,
timeout, retries, and decision-call budget are configurable. Gemini-specific
SDK calls stay under `epistemic_repair/llm/gemini.py`.

The default is `gemini-3.7-flash` with `medium` thinking. The adapter uses
structured JSON output, supplies no tools, requests no thought summaries, and
does not send deprecated sampling settings. See the official
[Gemini 3.7 Flash documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash)
and [structured-output guide](https://ai.google.dev/gemini-api/docs/structured-output).

Structured LLM responses default to `1024` maximum output tokens. Live Gemini
smoke testing showed that `256` could truncate the autonomous response
envelope; the limit remains configurable with `--max-output-tokens`.

Automatic free-model routing is unsuitable for these experiments because a
different underlying model between requests would invalidate attribution and
reproducibility. Free-tier availability and rate limits can change; failures
are surfaced without silently switching provider or model.

### Credentials and local development

Install the optional official SDK and set the key in the shell:

```bash
python -m pip install -e ".[dev,gemini]"
$env:GEMINI_API_KEY = "your-local-key"  # PowerShell
```

Alternatively, copy `.env.example` to a local `.env` and fill it locally.
`.env` and `.env.*` are gitignored, existing shell variables take precedence,
and keys are never printed or recorded. Missing `GEMINI_API_KEY` causes a clear
configuration error. Never commit the local `.env` file.

### Small smoke runs

The no-network deterministic smoke demonstration is:

```bash
python -m scripts.demo_llm_agent --mock --condition both --budget 2 --repetitions 1
```

A deliberately tiny live call set can be started explicitly after configuring
the key:

```bash
python -m scripts.demo_llm_agent --provider gemini --model gemini-3.7-flash --condition full --budget 2 --repetitions 1
```

The default repetition count is three. This is only an integration smoke stage,
not a sample size suitable for scientific claims. No live API call is made by
the unit tests.

Select benchmark history explicitly with `--benchmark er0`,
`--benchmark er1_v1`, or `--benchmark er1_v2`. The older `--benchmark er1`
spelling remains an alias for V1 and is never silently redirected to V2. A
no-network V2 interface smoke run is:

```bash
python -m scripts.demo_llm_agent --mock --benchmark er1_v2 --condition both --budget 2 --repetitions 1
```

## Beliefs and deterministic likelihoods

`HypothesisBeliefs` is an immutable normalized distribution over exactly three
hypotheses: world shift, sensor corruption, and missing latent variable. It
supports probability lookup, entropy in bits, canonical tie-breaking, and
construction from normalized weights. The V0 prior is uniform, with entropy
`log2(3) ≈ 1.584963` bits.

`DeterministicLikelihoodModel` is the single normative outcome model used for
Bayesian updates and experiment selection. It predicts each action's observable
outcome under every hypothesis and current context. Updates use
`P(H | r, E) ∝ P(r | H, E)P(H)`; impossible observations raise
`ImpossibleObservationError` rather than producing invalid beliefs.

At the uniform prior in context B, the expected information gains are:

| Action | Expected information gain |
| --- | ---: |
| `REPEAT_TRIAL` | `0` bits |
| `USE_TRUSTED_SENSOR` | `0.918296` bits |
| `CHANGE_CONTEXT` | `0.918296` bits |

The change action deterministically switches to the other known context. Thus
the initial transition is B to A, matching the benchmark likelihood table; a
later change switches A back to B.

## ER-1 likelihoods, policies, and metrics

`StochasticLikelihoodModel` analytically computes every binary outcome
distribution, soft Bayesian update, and expected information gain. It does not
use realized entropy reduction to select actions. At the anomaly-conditioned
initial posterior in context B, the default expected information gains are:

| Action | Expected information gain |
| --- | ---: |
| `REPEAT_TRIAL` | `0.087597` bits |
| `USE_TRUSTED_SENSOR` | `0.470190` bits |
| `CHANGE_CONTEXT` | `0.368306` bits |

Unlike ER-0, repeating a primary-sensor trial is informative. The stochastic
oracle chooses maximum expected information gain with deterministic action-order
tie-breaking and stops when posterior confidence reaches the configurable
threshold or its budget is exhausted. It receives beliefs and the normative
likelihood model, but never hidden ground truth. The restricted random baseline
uses its own seed and receives only the safe ER-1 agent view.

ER-1 supports budgets `1`, `2`, `3`, `5`, and `8` by default. Alongside
accuracy, success, experiment count, action regret, and oracle-action agreement,
its summaries report:

- `false_structural_diagnosis_rate`: a structural diagnosis when truth is
  `NO_STRUCTURAL_CHANGE`;
- `missed_structural_failure_rate`: `NO_STRUCTURAL_CHANGE` diagnosed for a
  structural truth; and
- `premature_diagnosis_rate` for LLM episodes: diagnosis before the normative
  posterior reaches the configured threshold.

### ER-1 oracle calibration

The no-network calibration script measures the stochastic oracle over budgets,
confidence thresholds, hypotheses, and reproducible episode seeds. Its default
grid is 60,000 episodes: four hypotheses × five budgets × three thresholds ×
1,000 seeds.

```bash
python -m scripts.calibrate_er1_oracle
```

Use `--seeds`, `--budgets`, `--thresholds`, and `--output-dir` to run smaller
validation grids. The script reports MAP accuracy separately from reaching the
configured threshold on the correct hypothesis and writes aggregate CSV,
confusion-matrix JSON, representative hard-case JSON, and a Markdown report.
The calibration implementation contains no LLM/provider calls and makes no
network requests.

An experiment-only staged parameter search can evaluate possible ER-1 V2
generative constants without modifying the active V1 configuration:

```bash
python -m scripts.search_er1_parameters
```

It first screens ordinary no-change noise, then modest structural contrasts,
and finally runs the full calibration grid for the top three candidates plus
V1 when needed. Candidate parameters, likelihoods, and environments remain
isolated from `epistemic_repair/er1/config.py`.

## Policies and evaluation

`OracleInformationGainPolicy` chooses the action with maximum expected
information gain. Ties follow `BENCHMARK_ACTIONS` order, so the initial tie is
resolved in favor of `USE_TRUSTED_SENSOR`. `RandomDiagnosticPolicy` samples
uniformly from the same action tuple and accepts a seed for reproducibility.

The random baseline implements the restricted benchmark-policy interface and
receives no beliefs or likelihood model. Only the oracle receives the
privileged normative view. Both interfaces exclude the environment, hidden
failure mode, repair label, and evaluation metadata.

`DiagnosticEpisodeRunner` begins from `X=1 -> O=0`, updates beliefs after each
typed result, and stops at a confidence threshold or experiment budget. The
restricted agent receives only safe `AgentExperimentRecord` history. A separate
evaluation trace records priors, expected information gains, actions, results,
posteriors, realized information gain, and action regret. Ground truth is read
only after policy interaction and attached to the outer evaluation result.

Evaluation includes diagnosis accuracy, experiments used, success within
budget, cumulative information gain, and cumulative action regret. No repair
metrics or repair execution are present. `evaluate_policy_budgets()` supports
reproducible sweeps over budgets such as 1, 2, 3, and 5 experiments.

## Setup and usage

Requires Python 3.10 or later. There are no runtime dependencies.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m scripts.demo_binary_world
python -m scripts.demo_diagnostics
python -m scripts.demo_policy_evaluation
python -m scripts.demo_llm_agent --mock --condition both --budget 2 --repetitions 1
python -m scripts.demo_er1_oracle --budget 5 --threshold 0.90 --seed 0
python -m scripts.demo_llm_agent --mock --benchmark er1_v1 --condition both --budget 5 --repetitions 1
python -m scripts.demo_llm_agent --mock --benchmark er1_v2 --condition both --budget 5 --repetitions 1
python -m scripts.demo_er1_v2_sanity --seeds 100
```

For ER-1 V2 LLM runs, `--diagnosis-threshold` controls the normative threshold
used for premature-diagnosis evaluation and defaults to `0.90`. It is threaded
only into the V2 runner, so ER-0 and ER-1 V1 retain their historical behavior.
An episode budget of `B` can require up to `B+1` model decision calls when the
model uses all `B` experiments and diagnoses on the following turn; configure
`--max-decision-calls` independently when running that design.

V2 LLM evaluation keeps four concepts separate: raw diagnosis correctness,
correct diagnosis within the bounded episode, threshold-qualified success, and
premature diagnosis. Threshold qualification and prematurity use the normative
Bayesian probability of the model's chosen diagnosis at the diagnosis turn—not
the model's stated confidence and not the maximum posterior of some other
hypothesis. The historical `success_within_budget` field is preserved, with the
clearer V2 alias `diagnosed_correctly_within_budget`.

The full fixed-grid V2 oracle calibration is explicitly versioned and writes
only `er1_v2_oracle_*` artifacts. It compares matching cells against the
existing V1 CSV artifacts without modifying them:

```bash
python -m scripts.calibrate_er1_v2_oracle
```

The default grid is 60,000 no-network oracle episodes: four hypotheses, five
budgets, three thresholds, and 1,000 episode seeds. This command imports no LLM
or provider adapter and never makes a network request.

Minimal API example:

```python
from epistemic_repair import BinaryMachine, FailureMode

env = BinaryMachine()
env.reset(FailureMode.SENSOR_CORRUPTION)
observation = env.step(1)       # Observation(x=1, o=0): agent-visible
truth = env.get_ground_truth()  # evaluation only: Y=1, sensor is corrupted
```

Diagnostic API example:

```python
from epistemic_repair import Context, DiagnosticAction

trusted = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)
changed = env.run_experiment(
    DiagnosticAction.CHANGE_CONTEXT,
    x=1,
    context=Context.A,
)
```

## Scope

This repository contains both the standalone deterministic ER-0 benchmark and
the separate stochastic ER-1 diagnosis benchmark, including seeded
environments, normative beliefs, policy baselines, evaluation, and mocked/live
LLM transport support. It does not execute repairs, and no live provider is
called automatically. Planned research stages, not implemented here, include:

- selective repair; and
- larger causal environments.
