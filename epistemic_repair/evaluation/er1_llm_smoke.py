"""Small seeded smoke orchestration for ER-1 LLM conditions."""

from dataclasses import dataclass

from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.evaluation.er1_llm_metrics import (
    ER1LLMConditionSummary,
    summarize_er1_llm_results,
)
from epistemic_repair.evaluation.er1_llm_runner import (
    ER1LLMEpisodeResult,
    ER1LLMEpisodeRunner,
)
from epistemic_repair.llm.base import LLMClient
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import LLMCondition
from epistemic_repair.policies.llm import (
    ER1FullAutonomousLLMPolicy,
    ER1PlannerOnlyLLMPolicy,
)


@dataclass(frozen=True, slots=True)
class ER1LLMSmokeResult:
    condition: LLMCondition
    episodes: tuple[ER1LLMEpisodeResult, ...]
    summary: ER1LLMConditionSummary


def run_er1_llm_smoke(
    client: LLMClient,
    config: LLMConfig,
    *,
    conditions: tuple[LLMCondition, ...] = (
        LLMCondition.FULL_AUTONOMOUS,
        LLMCondition.PLANNER_ONLY,
    ),
    repetitions: int = 3,
    experiment_budget: int = 5,
    base_episode_seed: int = 0,
    failure_modes: tuple[FailureMode, ...] = ER1_HYPOTHESES,
) -> tuple[ER1LLMSmokeResult, ...]:
    """Run bounded ER-1 episodes while keeping conditions separate."""
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    results = []
    episode_index = 0
    for condition in conditions:
        policy = (
            ER1FullAutonomousLLMPolicy(client, config)
            if condition is LLMCondition.FULL_AUTONOMOUS
            else ER1PlannerOnlyLLMPolicy(client, config)
        )
        episodes = []
        for hypothesis in failure_modes:
            if hypothesis not in ER1_HYPOTHESES:
                raise ValueError("failure_modes must contain only ER-1 hypotheses")
            for _ in range(repetitions):
                runner = ER1LLMEpisodeRunner(
                    condition=condition,
                    experiment_budget=experiment_budget,
                    episode_seed=base_episode_seed + episode_index,
                )
                episodes.append(
                    runner.run(StochasticBinaryMachine(), hypothesis, policy)
                )
                episode_index += 1
        episode_tuple = tuple(episodes)
        results.append(
            ER1LLMSmokeResult(
                condition=condition,
                episodes=episode_tuple,
                summary=summarize_er1_llm_results(episode_tuple),
            )
        )
    return tuple(results)
