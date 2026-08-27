"""Versioned ER-1 stochastic prompts built only from safe policy views."""

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.policies.stochastic_views import StochasticBenchmarkAgentView


BINARY_ER1_PROMPT_VERSION = "binary_er1_001"


_TASK_DESCRIPTION = """You are investigating a stochastic binary machine. Historically, input X usually produced physical output Y=X, and the ordinary primary sensor usually reported Y correctly. Both the physical process and sensors can occasionally produce noisy outcomes. One surprising observation does not by itself prove that anything structurally failed.

The possible explanations are:
- NO_STRUCTURAL_CHANGE: the world and sensor remain structurally intact; ordinary stochastic variation produced the surprise.
- WORLD_SHIFT: the physical relationship may have structurally reversed.
- SENSOR_CORRUPTION: the physical relationship may be intact but the primary sensor may have become mostly inverted.
- MISSING_LATENT_VARIABLE: environmental context may change the physical relationship.

Available diagnostic actions have these meanings:
- REPEAT_TRIAL: repeat X=1 in the current context using the ordinary primary sensor.
- USE_TRUSTED_SENSOR: perform another X=1 trial and observe an independent, highly reliable but not infallible trusted sensor T.
- CHANGE_CONTEXT: test X=1 under the alternate environmental context using the primary sensor.

Use experiments to accumulate evidence and avoid premature structural diagnosis. You have no tools, code execution, web access, files, or direct machine access. Return only the schema-constrained decision and a concise 1-3 sentence reason_summary; do not provide private chain-of-thought."""


def build_er1_full_autonomous_prompt(
    view: StochasticBenchmarkAgentView,
) -> str:
    """Build the qualitative-noise autonomous ER-1 prompt."""
    _validate_view(view)
    instructions = """Maintain normalized probabilities over all four explanations. Choose RUN_EXPERIMENT when more evidence is needed, or DIAGNOSE when evidence is sufficient. Include all four current beliefs on every turn. A diagnosis requires confidence."""
    return _assemble(view, instructions, belief_block=None)


def build_er1_planner_only_prompt(
    view: StochasticBenchmarkAgentView,
    beliefs: StochasticHypothesisBeliefs,
) -> str:
    """Build ER-1 planning prompt with authoritative beliefs but no likelihoods."""
    _validate_view(view)
    if not isinstance(beliefs, StochasticHypothesisBeliefs):
        raise TypeError("beliefs must be StochasticHypothesisBeliefs")
    block = "\n".join(
        f"- {hypothesis.value}: {beliefs.probability(hypothesis):.6f}"
        for hypothesis in ER1_HYPOTHESES
    )
    instructions = """The benchmark supplies authoritative current probabilities. Do not replace them for scoring. Choose RUN_EXPERIMENT when more evidence is needed, or DIAGNOSE when the supplied posterior is sufficiently decisive."""
    return _assemble(
        view,
        instructions,
        belief_block="Authoritative current probabilities:\n" + block,
    )


def _assemble(
    view: StochasticBenchmarkAgentView,
    instructions: str,
    *,
    belief_block: str | None,
) -> str:
    sections = [
        f"Prompt version: {BINARY_ER1_PROMPT_VERSION}",
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


def _format_view(view: StochasticBenchmarkAgentView) -> str:
    initial = [
        f"- X={observation.x}, primary O={observation.o}"
        for observation in view.initial_history
    ] or ["- none"]
    history = [
        f"- step {record.step_number}: {record.action.value} -> "
        f"{_format_result(record.result)}; context now {record.context_after.value}"
        for record in view.experiment_history
    ] or ["- none"]
    return "\n".join(
        [
            "Initial observable history:",
            *initial,
            "Experiment history:",
            *history,
            f"Current observable context: {view.current_context.value}",
            "Available actions: "
            + ", ".join(action.value for action in view.available_actions),
            f"Diagnostic experiments remaining: {view.steps_remaining}",
        ]
    )


def _format_result(
    result: RepeatTrialResult | StochasticTrustedSensorResult | ChangeContextResult,
) -> str:
    if isinstance(result, RepeatTrialResult):
        return f"X={result.x}, primary O={result.o}"
    if isinstance(result, StochasticTrustedSensorResult):
        return f"X={result.x}, trusted T={result.trusted_t}"
    if isinstance(result, ChangeContextResult):
        return f"context={result.context.value}, X={result.x}, primary O={result.o}"
    raise TypeError("unsupported ER-1 experiment result")


def _validate_view(view: StochasticBenchmarkAgentView) -> None:
    if type(view) is not StochasticBenchmarkAgentView:
        raise TypeError("ER-1 prompts require exactly StochasticBenchmarkAgentView")
    if DiagnosticAction.INSPECT_LATENT_VARIABLE in view.available_actions:
        raise ValueError("internal inspection action cannot appear in an LLM prompt")
