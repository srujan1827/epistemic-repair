"""Run seeded ER-1 oracle episodes without any network or LLM provider."""

import argparse
from collections.abc import Sequence

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticLikelihoodModel,
)
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.evaluation.stochastic_metrics import (
    summarize_stochastic_results,
)
from epistemic_repair.evaluation.stochastic_runner import (
    StochasticDiagnosticEpisodeRunner,
)
from epistemic_repair.policies.stochastic import (
    StochasticOracleInformationGainPolicy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _format_beliefs(beliefs: StochasticHypothesisBeliefs) -> str:
    return ", ".join(
        f"{hypothesis.value}={beliefs.probability(hypothesis):.6f}"
        for hypothesis in ER1_HYPOTHESES
    )


def main() -> None:
    args = parse_args()
    model = StochasticLikelihoodModel()
    base = StochasticHypothesisBeliefs.base_prior()
    initial = model.conditioned_initial_beliefs()
    print("ER-1 seeded oracle demo (no network)")
    print(f"base prior: {_format_beliefs(base)}")
    print(f"conditioned on X=1,O=0: {_format_beliefs(initial)}")
    print(f"budget: {args.budget}; threshold: {args.threshold:.3f}")

    results = []
    for index, hypothesis in enumerate(ER1_HYPOTHESES):
        episode_seed = args.seed + index
        result = StochasticDiagnosticEpisodeRunner(
            diagnosis_threshold=args.threshold,
            max_experiments=args.budget,
            likelihood_model=model,
        ).run(
            StochasticBinaryMachine(),
            hypothesis,
            StochasticOracleInformationGainPolicy(),
            episode_seed=episode_seed,
        )
        results.append(result)
        print(f"\ntruth (evaluation only): {hypothesis.value}; seed={episode_seed}")
        print("initial observable: X=1 -> O=0")
        for step in result.trace:
            print(
                f"  step={step.step_number} action={step.chosen_action.value} "
                f"result={step.experiment_result} outcome={step.outcome.value} "
                f"posterior=[{_format_beliefs(step.posterior)}] "
                f"regret={step.action_regret:.6f}"
            )
        print(
            f"final={result.predicted_diagnosis.value} "
            f"experiments={result.experiments_used} "
            f"threshold_reached={result.reached_threshold} "
            f"correct={result.diagnosis_correct}"
        )

    summary = summarize_stochastic_results(results)
    print("\naggregate")
    print(
        f"accuracy={summary.diagnosis_accuracy:.3f} "
        f"mean_experiments={summary.mean_experiments:.3f} "
        f"mean_regret={summary.mean_action_regret:.6f} "
        f"false_structural={summary.false_structural_diagnosis_rate:.3f} "
        f"missed_structural={summary.missed_structural_failure_rate:.3f}"
    )


if __name__ == "__main__":
    main()
