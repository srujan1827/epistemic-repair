"""Deterministic no-network client for smoke demonstrations and tests."""

import json

from epistemic_repair.llm.base import LLMClient, LLMRequest, LLMResponse


class DeterministicMockLLMClient(LLMClient):
    """Produce sensible decisions using only evidence rendered in the safe prompt."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        planner_only = "Authoritative current probabilities:" in request.prompt
        decision = self._decide(request.prompt, planner_only=planner_only)
        return LLMResponse(
            text=json.dumps(decision),
            provider_request_id=f"mock-{self.call_count}",
        )

    @staticmethod
    def _decide(prompt: str, *, planner_only: bool) -> dict[str, object]:
        if (
            "binary_er1_001" in prompt
            or "binary_er1_v2_001" in prompt
            or "binary_er1_v2_threshold_aware_001" in prompt
        ):
            return _decide_er1(prompt, planner_only)
        if "trusted Y=1" in prompt:
            return _diagnosis("SENSOR_CORRUPTION", planner_only)
        if "CHANGE_CONTEXT -> context=A, X=1, primary O=1" in prompt:
            return _diagnosis("MISSING_LATENT_VARIABLE", planner_only)
        if (
            "trusted Y=0" in prompt
            and "CHANGE_CONTEXT -> context=A, X=1, primary O=0" in prompt
        ):
            return _diagnosis("WORLD_SHIFT", planner_only)
        if "trusted Y=0" in prompt:
            return _experiment("CHANGE_CONTEXT", (0.5, 0.0, 0.5), planner_only)
        if "CHANGE_CONTEXT -> context=A, X=1, primary O=0" in prompt:
            return _experiment("USE_TRUSTED_SENSOR", (0.5, 0.5, 0.0), planner_only)
        return _experiment(
            "USE_TRUSTED_SENSOR", (1 / 3, 1 / 3, 1 / 3), planner_only
        )


def _experiment(
    action: str,
    beliefs: tuple[float, float, float],
    planner_only: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "decision": "RUN_EXPERIMENT",
        "action": action,
        "diagnosis": None,
        "reason_summary": "This safe diagnostic should reduce the remaining ambiguity.",
    }
    if not planner_only:
        result["beliefs"] = _beliefs(*beliefs)
        result["confidence"] = None
    return result


def _diagnosis(label: str, planner_only: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "decision": "DIAGNOSE",
        "action": None,
        "diagnosis": label,
        "reason_summary": "The safe experiment history now supports one explanation.",
    }
    if not planner_only:
        probabilities = {
            "WORLD_SHIFT": 1.0 if label == "WORLD_SHIFT" else 0.0,
            "SENSOR_CORRUPTION": 1.0 if label == "SENSOR_CORRUPTION" else 0.0,
            "MISSING_LATENT_VARIABLE": (
                1.0 if label == "MISSING_LATENT_VARIABLE" else 0.0
            ),
        }
        result["beliefs"] = probabilities
        result["confidence"] = 1.0
    return result


def _beliefs(world: float, sensor: float, latent: float) -> dict[str, float]:
    return {
        "WORLD_SHIFT": world,
        "SENSOR_CORRUPTION": sensor,
        "MISSING_LATENT_VARIABLE": latent,
    }


def _decide_er1(prompt: str, planner_only: bool) -> dict[str, object]:
    if "trusted T=1" in prompt and "REPEAT_TRIAL -> X=1, primary O=1" in prompt:
        return _er1_diagnosis("NO_STRUCTURAL_CHANGE", planner_only)
    if "trusted T=1" in prompt and "REPEAT_TRIAL -> X=1, primary O=0" in prompt:
        return _er1_diagnosis("SENSOR_CORRUPTION", planner_only)
    if "trusted T=0" in prompt and "CHANGE_CONTEXT -> context=A, X=1, primary O=1" in prompt:
        return _er1_diagnosis("MISSING_LATENT_VARIABLE", planner_only)
    if "trusted T=0" in prompt and "CHANGE_CONTEXT -> context=A, X=1, primary O=0" in prompt:
        return _er1_diagnosis("WORLD_SHIFT", planner_only)
    if "trusted T=1" in prompt:
        return _er1_experiment("REPEAT_TRIAL", planner_only)
    if "trusted T=0" in prompt:
        return _er1_experiment("CHANGE_CONTEXT", planner_only)
    return _er1_experiment("USE_TRUSTED_SENSOR", planner_only)


def _er1_experiment(action: str, planner_only: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "decision": "RUN_EXPERIMENT",
        "action": action,
        "diagnosis": None,
        "reason_summary": "Collect another noisy observation before diagnosing.",
    }
    if not planner_only:
        result["beliefs"] = {
            "NO_STRUCTURAL_CHANGE": 0.25,
            "WORLD_SHIFT": 0.25,
            "SENSOR_CORRUPTION": 0.25,
            "MISSING_LATENT_VARIABLE": 0.25,
        }
        result["confidence"] = None
    return result


def _er1_diagnosis(label: str, planner_only: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "decision": "DIAGNOSE",
        "action": None,
        "diagnosis": label,
        "reason_summary": "The accumulated stochastic evidence supports this diagnosis.",
    }
    if not planner_only:
        result["beliefs"] = {
            hypothesis: 1.0 if hypothesis == label else 0.0
            for hypothesis in (
                "NO_STRUCTURAL_CHANGE",
                "WORLD_SHIFT",
                "SENSOR_CORRUPTION",
                "MISSING_LATENT_VARIABLE",
            )
        }
        result["confidence"] = 1.0
    return result
