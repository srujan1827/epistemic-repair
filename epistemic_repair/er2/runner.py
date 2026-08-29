"""ER-2 repair episodes and deterministic matrix/baseline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Sequence

from epistemic_repair.er2.evaluation import (
    ER2_HYPOTHESES,
    PostRepairMetrics,
    evaluate_repaired_state,
)
from epistemic_repair.er2.policies import (
    ER2RepairAgentView,
    ER2RepairPolicy,
    FixedRepairPolicy,
    RepairBaseline,
    baseline_policy,
)
from epistemic_repair.er2.state import RepairableAgentState, apply_repair
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


@dataclass(frozen=True, slots=True)
class ER2RepairEpisodeResult:
    """Evaluation result with ground truth attached outside the policy view."""

    true_hypothesis: FailureMode
    supplied_diagnosis: FailureMode
    applied_repair: RepairOperator
    correct_repair: RepairOperator
    repair_selection_correct: bool
    pre_repair_state: RepairableAgentState
    post_repair_state: RepairableAgentState
    metrics: PostRepairMetrics


class ER2RepairEpisodeRunner:
    """Apply one selected repair to a fresh healthy-assumption agent state."""

    def run(
        self,
        *,
        true_hypothesis: FailureMode,
        diagnosis: FailureMode,
        policy: ER2RepairPolicy,
    ) -> ER2RepairEpisodeResult:
        if true_hypothesis not in ER2_HYPOTHESES:
            raise ValueError("true_hypothesis must be an ER-2 hypothesis")
        view = ER2RepairAgentView(diagnosis=diagnosis)
        repair = policy.choose_repair(view)
        if not isinstance(repair, RepairOperator):
            raise TypeError("repair policy must return a RepairOperator")
        pre_state = RepairableAgentState()
        post_state = pre_state.clone()
        apply_repair(post_state, repair)
        metrics = evaluate_repaired_state(true_hypothesis, pre_state, post_state)
        correct = repair_for_failure(true_hypothesis)
        return ER2RepairEpisodeResult(
            true_hypothesis=true_hypothesis,
            supplied_diagnosis=diagnosis,
            applied_repair=repair,
            correct_repair=correct,
            repair_selection_correct=repair is correct,
            pre_repair_state=pre_state,
            post_repair_state=post_state,
            metrics=metrics,
        )


def build_wrong_repair_matrix() -> list[dict[str, Any]]:
    """Evaluate every true hypothesis against every repair (4×4)."""
    runner = ER2RepairEpisodeRunner()
    rows: list[dict[str, Any]] = []
    for hypothesis in ER2_HYPOTHESES:
        for repair in RepairOperator:
            result = runner.run(
                true_hypothesis=hypothesis,
                diagnosis=hypothesis,
                policy=FixedRepairPolicy(repair),
            )
            rows.append(episode_row(result))
    return rows


def evaluate_baselines(
    baselines: Sequence[RepairBaseline] = tuple(RepairBaseline),
) -> list[dict[str, Any]]:
    """Evaluate fixed and oracle repair selectors across all four hypotheses."""
    runner = ER2RepairEpisodeRunner()
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        policy = baseline_policy(baseline)
        episodes = [
            runner.run(
                true_hypothesis=hypothesis,
                diagnosis=hypothesis,
                policy=policy,
            )
            for hypothesis in ER2_HYPOTHESES
        ]
        output.append({
            "baseline": baseline.value,
            "episode_count": len(episodes),
            "repair_selection_accuracy": mean(
                episode.repair_selection_correct for episode in episodes
            ),
            "repair_success_rate": mean(
                episode.metrics.repair_success for episode in episodes
            ),
            "post_repair_accuracy": mean(
                episode.metrics.post_repair_accuracy for episode in episodes
            ),
            "affected_region_accuracy": mean(
                episode.metrics.affected_region_accuracy for episode in episodes
            ),
            "unaffected_region_accuracy": mean(
                episode.metrics.unaffected_region_accuracy for episode in episodes
            ),
            "collateral_damage": mean(
                episode.metrics.collateral_damage for episode in episodes
            ),
        })
    return output


def episode_row(result: ER2RepairEpisodeResult) -> dict[str, Any]:
    """Flatten one episode into an audit-friendly matrix row."""
    metrics = asdict(result.metrics)
    return {
        "true_hypothesis": result.true_hypothesis.value,
        "supplied_diagnosis": result.supplied_diagnosis.value,
        "applied_repair": result.applied_repair.value,
        "correct_repair": result.correct_repair.value,
        "repair_selection_correct": result.repair_selection_correct,
        **metrics,
        "pre_world_relation": result.pre_repair_state.world_relation.value,
        "post_world_relation": result.post_repair_state.world_relation.value,
        "pre_sensor_calibration": result.pre_repair_state.sensor_calibration.value,
        "post_sensor_calibration": result.post_repair_state.sensor_calibration.value,
        "pre_latent_structure": result.pre_repair_state.latent_structure.value,
        "post_latent_structure": result.post_repair_state.latent_structure.value,
    }
