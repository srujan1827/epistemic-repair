"""Demonstrate the BinaryMachine's observable ambiguity and hidden causes."""

from epistemic_repair import BinaryMachine, FailureMode


def run_episode(env: BinaryMachine, mode: FailureMode, x: int) -> None:
    """Run and print one deterministic episode."""
    env.reset(failure_mode=mode)
    observation = env.step(x)
    truth = env.get_ground_truth()

    print(f"\n{mode.value}")
    print(f"  agent-visible: X={observation.x} -> O={observation.o}")
    print("  ground truth / evaluation only:")
    print(f"    Y={truth.y}")
    print(f"    Z={truth.z if truth.z is not None else 'not used'}")
    print(f"    failure_mode={truth.failure_mode.value}")
    print(f"    correct_repair={truth.correct_repair.value}")


def main() -> None:
    """Run normal and failure episodes."""
    env = BinaryMachine()

    print("Normal behavior (two inputs)")
    env.reset()
    for x in (0, 1):
        observation = env.step(x)
        truth = env.get_ground_truth()
        print(
            f"  X={observation.x} -> O={observation.o} "
            f"(ground truth / evaluation only: Y={truth.y})"
        )

    print("\nFailure episodes: the visible anomaly is identical")
    for mode in (
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        run_episode(env, mode, x=1)

    print("\nSummary: all three failures produced agent-visible X=1 -> O=0.")
    print("Their hidden Y/cause and correct repair differ, so the anomaly alone is insufficient.")


if __name__ == "__main__":
    main()

