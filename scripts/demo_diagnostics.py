"""Show how active experiments distinguish the three hidden failure causes."""

from epistemic_repair import (
    BinaryMachine,
    ChangeContextResult,
    Context,
    DiagnosticAction,
    FailureMode,
    LatentInspectionResult,
    TrustedSensorResult,
)


def demonstrate_mode(env: BinaryMachine, mode: FailureMode) -> None:
    """Run the deterministic diagnostic path for one evaluation condition."""
    env.reset(mode)
    anomaly = env.step(1)

    print(f"\nHidden scenario (ground truth / evaluation only): {mode.value}")
    print(f"  Initial agent observation: X={anomaly.x} -> O={anomaly.o}")

    trusted = env.run_experiment(DiagnosticAction.USE_TRUSTED_SENSOR, x=1)
    assert isinstance(trusted, TrustedSensorResult)
    print(f"  USE_TRUSTED_SENSOR: trusted result={trusted.trusted_y}")

    if trusted.trusted_y == 1:
        print("  Evidence: the true world still gives Y=1; the primary sensor disagrees.")
        return

    print("  Evidence so far: WORLD_SHIFT or MISSING_LATENT_VARIABLE remains.")
    for context in (Context.A, Context.B):
        changed = env.run_experiment(
            DiagnosticAction.CHANGE_CONTEXT,
            x=1,
            context=context,
        )
        assert isinstance(changed, ChangeContextResult)
        print(f"  CHANGE_CONTEXT to {context.value}: X={changed.x} -> O={changed.o}")

    inspected = env.run_experiment(DiagnosticAction.INSPECT_LATENT_VARIABLE)
    assert isinstance(inspected, LatentInspectionResult)
    if inspected.available:
        print(f"  INSPECT_LATENT_VARIABLE: available=True, value={inspected.value}")
    else:
        print("  INSPECT_LATENT_VARIABLE: available=False")

    truth = env.get_ground_truth()
    print(
        "  Ground truth / evaluation only: "
        f"Y={truth.y}, Z={truth.z}, repair={truth.correct_repair.value}"
    )


def main() -> None:
    """Demonstrate active discrimination without implementing a diagnosis policy."""
    env = BinaryMachine()
    print("Active diagnostic experiments after the ambiguous X=1 -> O=0 anomaly")

    for mode in (
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        demonstrate_mode(env, mode)

    print("\nInterpretation")
    print("  trusted result=1 separates SENSOR_CORRUPTION.")
    print("  trusted result=0 leaves WORLD_SHIFT vs MISSING_LATENT_VARIABLE.")
    print("  context-invariant O=0 supports WORLD_SHIFT.")
    print("  context-dependent O (A:1, B:0) exposes the missing latent dependence.")


if __name__ == "__main__":
    main()
