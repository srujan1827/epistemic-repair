"""Frozen evidence-only repair prompt for the ER-1 V2 -> ER-2 experiment."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.er1_v2.views import ER1V2BenchmarkAgentView
from epistemic_repair.er2.llm_prompts import (
    REPAIR_ACTION_DESCRIPTIONS,
    RepairOption,
    RepairOptionID,
    RepairOptionPermutation,
)


END_TO_END_REPAIR_PROMPT_VERSION = "binary_er1_v2_er2_repair_001"


@dataclass(frozen=True, slots=True)
class EndToEndRepairPromptView:
    """Provider-safe investigation evidence plus neutral repair descriptions."""

    investigation_view: ER1V2BenchmarkAgentView
    options: tuple[RepairOption, ...]

    def __post_init__(self) -> None:
        if type(self.investigation_view) is not ER1V2BenchmarkAgentView:
            raise TypeError("investigation_view must be exactly ER1V2BenchmarkAgentView")
        if tuple(option.option_id for option in self.options) != tuple(RepairOptionID):
            raise ValueError("options must be ordered A, B, C, D")
        if {option.description for option in self.options} != set(
            REPAIR_ACTION_DESCRIPTIONS.values()
        ):
            raise ValueError("options must use each frozen repair description once")


def end_to_end_repair_view(
    investigation_view: ER1V2BenchmarkAgentView,
    permutation: RepairOptionPermutation,
) -> EndToEndRepairPromptView:
    """Construct a safe view without diagnosis, truth, or hidden option mapping."""
    return EndToEndRepairPromptView(
        investigation_view=investigation_view,
        options=tuple(
            RepairOption(
                option_id=option,
                description=REPAIR_ACTION_DESCRIPTIONS[
                    permutation.repair_by_option[option]
                ],
            )
            for option in RepairOptionID
        ),
    )


def build_end_to_end_repair_prompt(view: EndToEndRepairPromptView) -> str:
    """Render evidence and neutral repairs without adding a causal summary."""
    if type(view) is not EndToEndRepairPromptView:
        raise TypeError("view must be exactly EndToEndRepairPromptView")
    options = "\n".join(
        f"{option.option_id.value}: {option.description}" for option in view.options
    )
    return (
        "ER-1 V2 to ER-2 end-to-end repair selection\n"
        f"Prompt version: {END_TO_END_REPAIR_PROMPT_VERSION}\n\n"
        "Task:\n"
        "The investigation phase is complete. Based only on its observation and "
        "experiment record below, choose exactly one candidate repair action. Do not "
        "repeat the diagnosis. Select one option; do not directly modify the system.\n\n"
        "Investigation record:\n"
        f"{format_investigation_evidence(view.investigation_view)}\n\n"
        "Candidate repair actions:\n"
        f"{options}\n\n"
        "Selection rule:\n"
        "Choose the minimal action justified by the investigation record, while "
        "avoiding changes to components that the evidence does not implicate.\n\n"
        "Output:\n"
        "Return only one JSON object with selected_option (A, B, C, or D), confidence "
        "from 0 to 1, and a short rationale."
    )


def format_investigation_evidence(view: ER1V2BenchmarkAgentView) -> str:
    """Render only fields present in the frozen ER-1 V2 benchmark-agent view."""
    if type(view) is not ER1V2BenchmarkAgentView:
        raise TypeError("view must be exactly ER1V2BenchmarkAgentView")
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
        ]
    )


def _format_result(result: object) -> str:
    if isinstance(result, RepeatTrialResult):
        return f"X={result.x}, primary O={result.o}"
    if isinstance(result, StochasticTrustedSensorResult):
        return f"X={result.x}, trusted T={result.trusted_t}"
    if isinstance(result, ChangeContextResult):
        return f"context={result.context.value}, X={result.x}, primary O={result.o}"
    raise TypeError("unsupported ER-1 V2 experiment result")
