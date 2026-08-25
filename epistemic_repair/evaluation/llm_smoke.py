"""Small, bounded smoke orchestration across modes and LLM conditions."""

from dataclasses import dataclass

from epistemic_repair.beliefs.state import HYPOTHESES
from epistemic_repair.environments.binary_machine import BinaryMachine
from epistemic_repair.evaluation.llm_metrics import (
    LLMConditionSummary,
    summarize_llm_results,
)
from epistemic_repair.evaluation.llm_runner import LLMEpisodeResult, LLMEpisodeRunner
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import LLMCondition
from epistemic_repair.policies.llm import (
    FullAutonomousLLMPolicy,
    PlannerOnlyLLMPolicy,
)


@dataclass(frozen=True, slots=True)
class LLMSmokeResult:
    """Separate raw episodes and summary for one experimental condition."""

    condition: LLMCondition
    episodes: tuple[LLMEpisodeResult, ...]
    summary: LLMConditionSummary


def run_llm_smoke(
    client: LLMClient,
    config: LLMConfig,
    *,
    conditions: tuple[LLMCondition, ...] = (
        LLMCondition.FULL_AUTONOMOUS,
        LLMCondition.PLANNER_ONLY,
    ),
    repetitions: int = 3,
    experiment_budget: int = 2,
    base_episode_seed: int = 0,
    failure_modes: tuple[FailureMode, ...] = HYPOTHESES,
) -> tuple[LLMSmokeResult, ...]:
    """Run a deliberately small smoke experiment without pooling conditions."""
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    smoke_results = []
    episode_index = 0
    for condition in conditions:
        policy = (
            FullAutonomousLLMPolicy(client, config)
            if condition is LLMCondition.FULL_AUTONOMOUS
            else PlannerOnlyLLMPolicy(client, config)
        )
        episodes = []
        for hypothesis in failure_modes:
            for _ in range(repetitions):
                runner = LLMEpisodeRunner(
                    condition=condition,
                    experiment_budget=experiment_budget,
                    episode_seed=base_episode_seed + episode_index,
                )
                episodes.append(runner.run(BinaryMachine(), hypothesis, policy))
                episode_index += 1
        episode_tuple = tuple(episodes)
        smoke_results.append(
            LLMSmokeResult(
                condition=condition,
                episodes=episode_tuple,
                summary=summarize_llm_results(episode_tuple),
            )
        )
    return tuple(smoke_results)
