"""Seeded ER-1 V2 environment with transient trigger and persistent trials."""

from dataclasses import dataclass

from epistemic_repair.diagnostics.actions import Context, DiagnosticAction
from epistemic_repair.diagnostics.results import (
    StochasticBenchmarkExperimentResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.environments.stochastic_binary_machine import (
    StochasticBinaryMachine,
)
from epistemic_repair.er1_v2.config import (
    ER1_V2_DEFAULT_INVESTIGATION_CONFIG,
    ER1V2InvestigationConfig,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


@dataclass(frozen=True, slots=True)
class ER1V2GroundTruth:
    """Evaluation-only persistent state; trigger has no associated hidden Y."""

    failure_mode: FailureMode
    correct_repair: RepairOperator
    episode_seed: int
    x: int | None
    y: int | None
    o: int | None
    trusted_t: int | None
    context: Context


class ER1V2BinaryMachine(StochasticBinaryMachine):
    """Fresh persistent investigation trials under a private seeded RNG."""

    def __init__(
        self,
        config: ER1V2InvestigationConfig = (
            ER1_V2_DEFAULT_INVESTIGATION_CONFIG
        ),
    ) -> None:
        super().__init__()
        self.investigation_config = config

    def trigger_observation(self) -> Observation:
        """Construct X=1,O=0 without sampling or creating hidden trigger Y."""
        if self._initial_observation_issued:
            raise RuntimeError("transient trigger observation has already been issued")
        self._last_x = 1
        self._last_y = None
        self._last_o = 0
        self._last_t = None
        self._initial_observation_issued = True
        return Observation(x=1, o=0)

    def initial_anomaly(self, *, x: int = 1) -> Observation:
        """Compatibility spelling for the explicitly constructed trigger."""
        if x != 1:
            raise ValueError("ER-1 V2 trigger input must be X=1")
        return self.trigger_observation()

    def get_ground_truth(self) -> ER1V2GroundTruth:
        """Return hidden persistent state for evaluation only."""
        return ER1V2GroundTruth(
            failure_mode=self._failure_mode,
            correct_repair=repair_for_failure(self._failure_mode),
            episode_seed=self._episode_seed,
            x=self._last_x,
            y=self._last_y,
            o=self._last_o,
            trusted_t=self._last_t,
            context=self._context,
        )

    def _probability_y_one(self, x: int, context: Context) -> float:
        config = self.investigation_config
        if self._failure_mode is FailureMode.NO_STRUCTURAL_CHANGE:
            accuracy, matches_x = config.no_change_physical_accuracy, True
        elif self._failure_mode is FailureMode.WORLD_SHIFT:
            accuracy, matches_x = config.world_shift_physical_accuracy, False
        elif self._failure_mode is FailureMode.SENSOR_CORRUPTION:
            accuracy, matches_x = config.sensor_corruption_physical_accuracy, True
        else:
            accuracy = config.latent_physical_accuracy
            matches_x = context is Context.A
        preferred_y = x if matches_x else 1 - x
        return accuracy if preferred_y == 1 else 1.0 - accuracy

    def _sample_primary_observation(self, y: int) -> int:
        matches = (
            1.0
            - self.investigation_config.corrupted_sensor_inversion_accuracy
            if self._failure_mode is FailureMode.SENSOR_CORRUPTION
            else self.investigation_config.healthy_primary_sensor_accuracy
        )
        return self._sample_reliable_copy(y, matches)

    def _primary_probability(self, o: int, y: int) -> float:
        matches = (
            1.0
            - self.investigation_config.corrupted_sensor_inversion_accuracy
            if self._failure_mode is FailureMode.SENSOR_CORRUPTION
            else self.investigation_config.healthy_primary_sensor_accuracy
        )
        return matches if o == y else 1.0 - matches

    def _sample_reliable_copy(self, value: int, reliability: float) -> int:
        # Override only to make the V2 trusted reliability explicit at call sites.
        return value if self._rng.random() < reliability else 1 - value

    def run_experiment(
        self,
        action: DiagnosticAction,
        *,
        x: int,
        context: Context | None = None,
    ) -> StochasticBenchmarkExperimentResult:
        """Execute a fresh persistent trial and return only O or trusted T."""
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            if context is not None:
                raise ValueError("context is not valid for USE_TRUSTED_SENSOR")
            self._validate_bit(x, "x")
            y = self._sample_physical_output(x, self._context)
            trusted_t = self._sample_reliable_copy(
                y, self.investigation_config.trusted_sensor_accuracy
            )
            self._record(x=x, y=y, o=None, trusted_t=trusted_t)
            return StochasticTrustedSensorResult(x=x, trusted_t=trusted_t)
        return super().run_experiment(action, x=x, context=context)
