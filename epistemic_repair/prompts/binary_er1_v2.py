"""Versioned ER-1 V2 prompts built only from the restricted V2 view."""

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.views import ER1V2BenchmarkAgentView


BINARY_ER1_V2_PROMPT_VERSION = "binary_er1_v2_001"
BINARY_ER1_V2_THRESHOLD_AWARE_PROMPT_VERSION = (
    "binary_er1_v2_threshold_aware_001"
)


_TASK_DESCRIPTION = """You are investigating a stochastic binary machine. Historically, input X usually produced physical output Y=X, and the ordinary primary sensor usually reported Y correctly.

This episode was triggered by one surprising observation, X=1 and primary O=0. That event may have been a transient trigger anomaly rather than evidence of a persistent failure. Evidence from later diagnostic experiments reflects the machine's persistent investigation dynamics. Do not assume that the single trigger implies a persistent structural failure.

The possible explanations are:
- NO_STRUCTURAL_CHANGE: the trigger was transient; the physical process and primary sensor remain structurally healthy.
- WORLD_SHIFT: the physical relationship has persistently reversed.
- SENSOR_CORRUPTION: the physical relationship remains healthy but the primary sensor is persistently corrupted.
- MISSING_LATENT_VARIABLE: environmental context persistently changes the physical relationship.

Available diagnostic actions have these meanings:
- REPEAT_TRIAL: run a fresh X=1 trial in the current context using the ordinary primary sensor.
- USE_TRUSTED_SENSOR: run a fresh X=1 trial and observe an independent, highly reliable but not infallible trusted sensor T.
- CHANGE_CONTEXT: test X=1 under the alternate environmental context using the primary sensor.

Use experiments to identify what persists. You have no tools, code execution, web access, files, or direct machine access. Return only the schema-constrained decision and a concise 1-3 sentence reason_summary; do not provide private chain-of-thought."""

_FULL_AUTONOMOUS_INSTRUCTIONS = (
    "Maintain normalized probabilities over all four explanations. Choose "
    "RUN_EXPERIMENT when more persistent evidence is needed, or DIAGNOSE when "
    "evidence is sufficient. Include all four current beliefs on every turn. "
    "A diagnosis requires confidence."
)


def build_er1_v2_full_autonomous_prompt(view: ER1V2BenchmarkAgentView) -> str:
    """Build the qualitative full-autonomous V2 prompt."""
    _validate_view(view)
    return _assemble(
        view,
        _FULL_AUTONOMOUS_INSTRUCTIONS,
        belief_block=None,
    )


def build_er1_v2_threshold_aware_autonomous_prompt(
    view: ER1V2BenchmarkAgentView,
    diagnosis_threshold: float,
) -> str:
    """Build an autonomous prompt that reveals only the stopping threshold."""
    _validate_view(view)
    if (
        not isinstance(diagnosis_threshold, (int, float))
        or isinstance(diagnosis_threshold, bool)
        or not 0.0 < diagnosis_threshold <= 1.0
    ):
        raise ValueError("diagnosis_threshold must be in (0, 1]")
    threshold = f"{float(diagnosis_threshold):.6g}"
    return _assemble(
        view,
        _FULL_AUTONOMOUS_INSTRUCTIONS
        + " Do not issue a final diagnosis until your confidence in one "
        "hypothesis is at least the configured diagnosis threshold of "
        f"{threshold}. If the experiment budget is exhausted before reaching "
        "that confidence, make your best-supported diagnosis.",
        belief_block=None,
        prompt_version=BINARY_ER1_V2_THRESHOLD_AWARE_PROMPT_VERSION,
    )


def build_er1_v2_planner_only_prompt(
    view: ER1V2BenchmarkAgentView,
    beliefs: StochasticHypothesisBeliefs,
) -> str:
    """Build the V2 planner prompt with authoritative beliefs only."""
    _validate_view(view)
    if not isinstance(beliefs, StochasticHypothesisBeliefs):
        raise TypeError("beliefs must be StochasticHypothesisBeliefs")
    block = "\n".join(
        f"- {hypothesis.value}: {beliefs.probability(hypothesis):.6f}"
        for hypothesis in ER1_HYPOTHESES
    )
    return _assemble(
        view,
        "The benchmark supplies authoritative current probabilities. Do not replace them for scoring. Choose RUN_EXPERIMENT when more persistent evidence is needed, or DIAGNOSE when the supplied posterior is sufficiently decisive.",
        belief_block="Authoritative current probabilities:\n" + block,
    )


def _assemble(
    view: ER1V2BenchmarkAgentView,
    instructions: str,
    *,
    belief_block: str | None,
    prompt_version: str = BINARY_ER1_V2_PROMPT_VERSION,
) -> str:
    sections = [
        f"Prompt version: {prompt_version}",
        _TASK_DESCRIPTION,
        instructions,
        _format_view(view),
    ]
    if belief_block is not None:
        sections.append(belief_block)
    if view.steps_remaining == 0:
        sections.append(
            "No diagnostic experiments remain. You must return DIAGNOSE, not RUN_EXPERIMENT."
        )
    return "\n\n".join(sections)


def _format_view(view: ER1V2BenchmarkAgentView) -> str:
    trigger = [
        f"- X={observation.x}, primary O={observation.o}"
        for observation in view.trigger_history
    ] or ["- none"]
    history = [
        f"- step {record.step_number}: {record.action.value} -> "
        f"{_format_result(record.result)}; context now {record.context_after.value}"
        for record in view.experiment_history
    ] or ["- none"]
    return "\n".join(
        [
            "Transient trigger observation:",
            *trigger,
            "Persistent experiment history:",
            *history,
            f"Current observable context: {view.current_context.value}",
            "Available actions: "
            + ", ".join(action.value for action in view.available_actions),
            f"Diagnostic experiments remaining: {view.steps_remaining}",
        ]
    )


def _format_result(result) -> str:
    if isinstance(result, RepeatTrialResult):
        return f"X={result.x}, primary O={result.o}"
    if isinstance(result, StochasticTrustedSensorResult):
        return f"X={result.x}, trusted T={result.trusted_t}"
    if isinstance(result, ChangeContextResult):
        return f"context={result.context.value}, X={result.x}, primary O={result.o}"
    raise TypeError("unsupported ER-1 V2 experiment result")


def _validate_view(view: ER1V2BenchmarkAgentView) -> None:
    if type(view) is not ER1V2BenchmarkAgentView:
        raise TypeError("ER-1 V2 prompts require exactly ER1V2BenchmarkAgentView")
    if DiagnosticAction.INSPECT_LATENT_VARIABLE in view.available_actions:
        raise ValueError("internal inspection action cannot appear in an LLM prompt")
