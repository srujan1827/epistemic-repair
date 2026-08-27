"""Seeded ER-1 evaluation across stochastic experiment budgets."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_DEFAULT_CONFIG, ER1_HYPOTHESES
from epistemic_repair.evaluation.stochastic_metrics import (
    StochasticEvaluationSummary,
    summarize_stochastic_results,
)
from epistemic_repair.evaluation.stochastic_runner import (
    StochasticDiagnosticEpisodeRunner,
)
from epistemic_repair.policies.stochastic import StochasticDiagnosticPolicy


@dataclass(frozen=True, slots=True)
class StochasticBudgetEvaluation:
    max_experiments: int
    summary: StochasticEvaluationSummary


def evaluate_stochastic_policy_budgets(
    policy_factory: Callable[[], StochasticDiagnosticPolicy],
    *,
    budgets: Iterable[int] = (1, 2, 3, 5, 8),
    diagnosis_threshold: float = ER1_DEFAULT_CONFIG.diagnosis_threshold,
    base_episode_seed: int = 0,
) -> tuple[StochasticBudgetEvaluation, ...]:
    """Evaluate a fresh ER-1 policy at each requested seeded budget."""
    evaluations = []
    for budget_index, budget in enumerate(budgets):
        policy = policy_factory()
        runner = StochasticDiagnosticEpisodeRunner(
            diagnosis_threshold=diagnosis_threshold,
            max_experiments=budget,
        )
        results = [
            runner.run(
                StochasticBinaryMachine(),
                hypothesis,
                policy,
                episode_seed=(
                    base_episode_seed
                    + budget_index * len(ER1_HYPOTHESES)
                    + hypothesis_index
                ),
            )
            for hypothesis_index, hypothesis in enumerate(ER1_HYPOTHESES)
        ]
        evaluations.append(
            StochasticBudgetEvaluation(
                max_experiments=budget,
                summary=summarize_stochastic_results(results),
            )
        )
    return tuple(evaluations)
