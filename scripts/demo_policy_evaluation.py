"""Run oracle and random diagnostic policies on all deterministic failures."""

from epistemic_repair import (
    BinaryMachine,
    ChangeContextResult,
    DiagnosticEpisodeResult,
    DiagnosticEpisodeRunner,
    DiagnosticPolicy,
    FailureMode,
    HYPOTHESES,
    HypothesisBeliefs,
    OracleInformationGainPolicy,
    RandomDiagnosticPolicy,
    RepeatTrialResult,
    TrustedSensorResult,
    summarize_results,
)


_SHORT_NAMES = {
    FailureMode.WORLD_SHIFT: "W",
    FailureMode.SENSOR_CORRUPTION: "S",
    FailureMode.MISSING_LATENT_VARIABLE: "Z",
}


def format_beliefs(beliefs: HypothesisBeliefs) -> str:
    """Format hypothesis probabilities compactly for a trace."""
    return " ".join(
        f"{_SHORT_NAMES[hypothesis]}={beliefs.probability(hypothesis):.3f}"
        for hypothesis in HYPOTHESES
    )


def format_result(result: object) -> str:
    """Format only the evidence legitimately produced by an experiment."""
    if isinstance(result, TrustedSensorResult):
        return f"trusted Y={result.trusted_y}"
    if isinstance(result, ChangeContextResult):
        return f"context={result.context.value}, primary O={result.o}"
    if isinstance(result, RepeatTrialResult):
        return f"primary O={result.o}"
    raise TypeError(f"unexpected result type: {type(result).__name__}")


def print_episode(result: DiagnosticEpisodeResult) -> None:
    """Print a readable belief-and-experiment trace."""
    print(
        f"  Initial anomaly: X={result.initial_observation.x} "
        f"-> O={result.initial_observation.o}"
    )
    print(f"  Initial beliefs: {format_beliefs(HypothesisBeliefs.uniform())}")

    for step in result.trace:
        print(f"  Step {step.step_number}")
        print("    Available expected IG:")
        for action, score in step.information_gains.items():
            print(f"      {action.value} = {score:.6f} bits")
        print(f"    Chosen: {step.chosen_action.value}")
        print(f"    Observed: {format_result(step.experiment_result)}")
        print(f"    Posterior: {format_beliefs(step.posterior)}")
        print(f"    Action regret: {step.action_regret:.6f} bits")

    print(f"  Predicted diagnosis: {result.predicted_diagnosis.value}")
    print(f"  Ground truth (evaluation only): {result.ground_truth.value}")
    print(f"  Experiments used: {result.experiments_used}")
    print(
        "  Cumulative information gain: "
        f"{result.cumulative_information_gain:.6f} bits"
    )
    print(f"  Cumulative regret: {result.cumulative_action_regret:.6f} bits")


def run_policy(name: str, policy: DiagnosticPolicy) -> None:
    """Evaluate one policy across the complete hidden-mode set."""
    print(f"\n{name}")
    runner = DiagnosticEpisodeRunner(max_experiments=5)
    env = BinaryMachine()
    results = []

    for mode in HYPOTHESES:
        print(f"\nHidden scenario (evaluation only): {mode.value}")
        result = runner.run(env, mode, policy)
        results.append(result)
        print_episode(result)

    summary = summarize_results(results)
    print("\n  Aggregate")
    print(f"    Diagnosis accuracy: {summary.diagnosis_accuracy:.3f}")
    print(f"    Success within budget: {summary.success_rate_within_budget:.3f}")
    print(f"    Total experiments: {summary.total_experiments}")
    print(f"    Total action regret: {summary.total_action_regret:.6f} bits")


def main() -> None:
    """Run the normative oracle and a reproducible random baseline."""
    print("Policy evaluation for deterministic epistemic repair")
    run_policy("OracleInformationGainPolicy", OracleInformationGainPolicy())
    run_policy("RandomDiagnosticPolicy(seed=7)", RandomDiagnosticPolicy(seed=7))


if __name__ == "__main__":
    main()
