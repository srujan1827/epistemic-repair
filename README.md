# Epistemic Repair

This repository starts a research program around a simple question: can an AI
agent determine **why** a prediction became wrong before deciding what to learn
or repair?

Identical visible errors can have different epistemic causes. Updating the world
model is appropriate after a real world shift, but harmful when the sensor is
the faulty component. Likewise, an apparently broken rule may instead be an
incomplete rule that omits a relevant latent variable.

## Deterministic environment

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

## Active diagnostic experiments

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

Prompts are versioned as `binary_v0_001`. They request only a short
`reason_summary`; private chain-of-thought and provider thinking tokens are not
collected. Structured responses may request one of the three benchmark actions
or issue one of the three diagnoses. Malformed JSON, invalid probabilities,
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
```

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

This repository currently contains the deterministic environment, explicit
diagnostic experiment, normative belief, policy-baseline, evaluation, and LLM
smoke layers. It does not execute repairs. Planned research stages, not
implemented here, include:

- selective repair; and
- noisy and larger causal environments.
