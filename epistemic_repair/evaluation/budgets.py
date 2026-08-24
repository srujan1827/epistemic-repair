"""Reproducible evaluation across configured experiment budgets."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from epistemic_repair.beliefs.state import HYPOTHESES
from epistemic_repair.environments.binary_machine import BinaryMachine
from epistemic_repair.evaluation.metrics import EvaluationSummary, summarize_results
from epistemic_repair.evaluation.runner import DiagnosticEpisodeRunner
from epistemic_repair.policies.diagnostic import DiagnosticPolicy


@dataclass(frozen=True, slots=True)
class BudgetEvaluation:
    """Aggregate metrics for one maximum experiment budget."""

    max_experiments: int
    summary: EvaluationSummary


def evaluate_policy_budgets(
    policy_factory: Callable[[], DiagnosticPolicy],
    *,
    budgets: Iterable[int] = (1, 2, 3, 5),
    diagnosis_threshold: float = 0.999,
) -> tuple[BudgetEvaluation, ...]:
    """Evaluate a freshly constructed policy across each requested budget."""
    evaluations = []
    for budget in budgets:
        policy = policy_factory()
        runner = DiagnosticEpisodeRunner(
            diagnosis_threshold=diagnosis_threshold,
            max_experiments=budget,
        )
        results = [
            runner.run(BinaryMachine(), hypothesis, policy)
            for hypothesis in HYPOTHESES
        ]
        evaluations.append(
            BudgetEvaluation(
                max_experiments=budget,
                summary=summarize_results(results),
            )
        )
    return tuple(evaluations)

