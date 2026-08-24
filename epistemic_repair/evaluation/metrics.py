"""Aggregate metrics for diagnostic policy evaluation."""

from dataclasses import dataclass
from typing import Sequence

from epistemic_repair.evaluation.runner import DiagnosticEpisodeResult


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate accuracy, efficiency, information, and regret metrics."""

    episode_count: int
    diagnosis_accuracy: float
    total_experiments: int
    mean_experiments: float
    success_rate_within_budget: float
    total_information_gain: float
    total_action_regret: float


def summarize_results(
    results: Sequence[DiagnosticEpisodeResult],
) -> EvaluationSummary:
    """Summarize a non-empty collection of diagnostic episode results."""
    if not results:
        raise ValueError("at least one episode result is required")

    count = len(results)
    correct_count = sum(result.diagnosis_correct for result in results)
    success_count = sum(result.success_within_budget for result in results)
    total_experiments = sum(result.experiments_used for result in results)
    return EvaluationSummary(
        episode_count=count,
        diagnosis_accuracy=correct_count / count,
        total_experiments=total_experiments,
        mean_experiments=total_experiments / count,
        success_rate_within_budget=success_count / count,
        total_information_gain=sum(
            result.cumulative_information_gain for result in results
        ),
        total_action_regret=sum(
            result.cumulative_action_regret for result in results
        ),
    )

