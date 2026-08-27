"""Small, bounded ER-1 V2 LLM smoke orchestration."""

from dataclasses import dataclass

from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.environment import ER1V2BinaryMachine
from epistemic_repair.er1_v2.llm_policy import (
    ER1V2FullAutonomousLLMPolicy,
    ER1V2PlannerOnlyLLMPolicy,
)
from epistemic_repair.er1_v2.llm_metrics import (
    ER1V2LLMConditionSummary,
    summarize_er1_v2_llm_results,
)
from epistemic_repair.er1_v2.llm_runner import (
    ER1V2LLMEpisodeResult,
    ER1V2LLMEpisodeRunner,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.llm.base import LLMClient
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import LLMCondition


@dataclass(frozen=True, slots=True)
class ER1V2LLMSmokeResult:
    condition: LLMCondition
    episodes: tuple[ER1V2LLMEpisodeResult, ...]
    summary: ER1V2LLMConditionSummary


def run_er1_v2_llm_smoke(
    client: LLMClient,
    config: LLMConfig,
    *,
    conditions: tuple[LLMCondition, ...] = (
        LLMCondition.FULL_AUTONOMOUS,
        LLMCondition.PLANNER_ONLY,
    ),
    repetitions: int = 3,
    experiment_budget: int = 5,
    diagnosis_threshold: float = 0.90,
    base_episode_seed: int = 0,
    failure_modes: tuple[FailureMode, ...] = ER1_HYPOTHESES,
) -> tuple[ER1V2LLMSmokeResult, ...]:
    """Prepare/run V2 through a supplied client; tests use no-network clients."""
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    output = []
    episode_index = 0
    for condition in conditions:
        policy = (
            ER1V2FullAutonomousLLMPolicy(client, config)
            if condition is LLMCondition.FULL_AUTONOMOUS
            else ER1V2PlannerOnlyLLMPolicy(client, config)
        )
        episodes = []
        for hypothesis in failure_modes:
            if hypothesis not in ER1_HYPOTHESES:
                raise ValueError("failure_modes must contain only V2 hypotheses")
            for _ in range(repetitions):
                runner = ER1V2LLMEpisodeRunner(
                    condition=condition,
                    experiment_budget=experiment_budget,
                    diagnosis_threshold=diagnosis_threshold,
                    episode_seed=base_episode_seed + episode_index,
                )
                episodes.append(runner.run(ER1V2BinaryMachine(), hypothesis, policy))
                episode_index += 1
        episode_tuple = tuple(episodes)
        output.append(
            ER1V2LLMSmokeResult(
                condition=condition,
                episodes=episode_tuple,
                summary=summarize_er1_v2_llm_results(episode_tuple),
            )
        )
    return tuple(output)
