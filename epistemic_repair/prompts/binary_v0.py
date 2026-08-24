"""Versioned binary-world prompts built only from authorized policy views."""

from epistemic_repair.beliefs.state import HYPOTHESES, HypothesisBeliefs
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    TrustedSensorResult,
)
from epistemic_repair.policies.views import BenchmarkAgentView


BINARY_V0_PROMPT_VERSION = "binary_v0_001"


_TASK_DESCRIPTION = """You are investigating a deterministic binary machine.
Historically, input X produced physical output Y=X. The ordinary primary sensor reports observation O. An unexpected observation has occurred.

The possible explanations are:
- WORLD_SHIFT: the physical relationship may have changed.
- SENSOR_CORRUPTION: the physical relationship may be intact but the primary sensor may be unreliable.
- MISSING_LATENT_VARIABLE: an unobserved contextual factor may affect the physical output.

Available diagnostic actions have these meanings:
- REPEAT_TRIAL: repeat X=1 using the ordinary primary sensor.
- USE_TRUSTED_SENSOR: independently measure the true machine output for X=1.
- CHANGE_CONTEXT: test X=1 under the alternate environmental context.

You have no tools, code execution, web access, files, or direct machine access. Return only the schema-constrained decision. Give a concise reason_summary of about 1-3 sentences; do not provide private chain-of-thought."""


def build_full_autonomous_prompt(view: BenchmarkAgentView) -> str:
    """Build an autonomous-investigator prompt from the restricted view only."""
    _validate_view(view)
    instructions = """Maintain and update your own probabilities over all three explanations. Choose RUN_EXPERIMENT with one available action when more evidence is needed, or DIAGNOSE when you have enough evidence. Include your current normalized beliefs on every turn. A diagnosis requires a confidence value."""
    return _assemble_prompt(view, instructions=instructions, belief_block=None)


def build_planner_only_prompt(
    view: BenchmarkAgentView,
    normative_beliefs: HypothesisBeliefs,
) -> str:
    """Build a planning prompt with authoritative beliefs but no likelihood table."""
    _validate_view(view)
    if not isinstance(normative_beliefs, HypothesisBeliefs):
        raise TypeError("normative_beliefs must be HypothesisBeliefs")
    belief_block = "\n".join(
        f"- {hypothesis.value}: {normative_beliefs.probability(hypothesis):.6f}"
        for hypothesis in HYPOTHESES
    )
    instructions = """The benchmark supplies the authoritative current hypothesis probabilities below. Do not replace or update them for scoring. Choose RUN_EXPERIMENT with one available action when more evidence is needed, or DIAGNOSE when the supplied posterior is sufficiently decisive."""
    return _assemble_prompt(
        view,
        instructions=instructions,
        belief_block="Authoritative current probabilities:\n" + belief_block,
    )


def _assemble_prompt(
    view: BenchmarkAgentView,
    *,
    instructions: str,
    belief_block: str | None,
) -> str:
    sections = [
        f"Prompt version: {BINARY_V0_PROMPT_VERSION}",
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


def _format_view(view: BenchmarkAgentView) -> str:
    initial_lines = [
        f"- X={observation.x}, primary O={observation.o}"
        for observation in view.initial_history
    ] or ["- none"]
    history_lines = [
        f"- step {record.step_number}: {record.action.value} -> "
        f"{_format_result(record.result)}; context now {record.context_after.value}"
        for record in view.experiment_history
    ] or ["- none"]
    actions = ", ".join(action.value for action in view.available_actions)
    return "\n".join(
        [
            "Initial observable history:",
            *initial_lines,
            "Experiment history:",
            *history_lines,
            f"Current observable context: {view.current_context.value}",
            f"Available actions: {actions}",
            f"Diagnostic experiments remaining: {view.steps_remaining}",
        ]
    )


def _format_result(
    result: RepeatTrialResult | TrustedSensorResult | ChangeContextResult,
) -> str:
    if isinstance(result, RepeatTrialResult):
        return f"X={result.x}, primary O={result.o}"
    if isinstance(result, TrustedSensorResult):
        return f"X={result.x}, trusted Y={result.trusted_y}"
    if isinstance(result, ChangeContextResult):
        return (
            f"context={result.context.value}, X={result.x}, primary O={result.o}"
        )
    raise TypeError("unsupported benchmark experiment result")


def _validate_view(view: BenchmarkAgentView) -> None:
    if type(view) is not BenchmarkAgentView:
        raise TypeError("LLM prompts require exactly BenchmarkAgentView")
    if DiagnosticAction.INSPECT_LATENT_VARIABLE in view.available_actions:
        raise ValueError("internal inspection action cannot appear in an LLM prompt")

