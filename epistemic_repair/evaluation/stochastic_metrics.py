"""Aggregate ER-1 accuracy, regret, and structural-overreaction metrics."""

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from epistemic_repair.evaluation.stochastic_runner import (
    StochasticDiagnosticEpisodeResult,
)
from epistemic_repair.failures.modes import FailureMode


@dataclass(frozen=True, slots=True)
class StochasticEvaluationSummary:
    episode_count: int
    diagnosis_accuracy: float
    success_within_budget: float
    mean_experiments: float
    mean_action_regret: float
    false_structural_diagnosis_rate: float
    missed_structural_failure_rate: float


def summarize_stochastic_results(
    results: Sequence[StochasticDiagnosticEpisodeResult],
) -> StochasticEvaluationSummary:
    """Summarize ER-1 results, including false and missed structural calls."""
    if not results:
        raise ValueError("at least one ER-1 result is required")
    no_change = [
        result
        for result in results
        if result.ground_truth is FailureMode.NO_STRUCTURAL_CHANGE
    ]
    structural = [
        result
        for result in results
        if result.ground_truth is not FailureMode.NO_STRUCTURAL_CHANGE
    ]
    return StochasticEvaluationSummary(
        episode_count=len(results),
        diagnosis_accuracy=mean(result.diagnosis_correct for result in results),
        success_within_budget=mean(
            result.success_within_budget for result in results
        ),
        mean_experiments=mean(result.experiments_used for result in results),
        mean_action_regret=mean(
            result.cumulative_action_regret for result in results
        ),
        false_structural_diagnosis_rate=(
            mean(
                result.predicted_diagnosis
                is not FailureMode.NO_STRUCTURAL_CHANGE
                for result in no_change
            )
            if no_change
            else 0.0
        ),
        missed_structural_failure_rate=(
            mean(
                result.predicted_diagnosis
                is FailureMode.NO_STRUCTURAL_CHANGE
                for result in structural
            )
            if structural
            else 0.0
        ),
    )
