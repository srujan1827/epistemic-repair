"""Seeded stochastic binary world for ER-1 epistemic diagnosis."""

from dataclasses import dataclass
from random import Random

from epistemic_repair.diagnostics.actions import Context, DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticBenchmarkExperimentResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.environments.binary_machine import Observation
from epistemic_repair.er1.config import (
    ER1_DEFAULT_CONFIG,
    ER1_HYPOTHESES,
    ER1ProbabilityConfig,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


@dataclass(frozen=True, slots=True)
class StochasticGroundTruth:
    """Evaluation-only ER-1 state, including hidden physical output Y."""

    failure_mode: FailureMode
    correct_repair: RepairOperator
    episode_seed: int
    x: int | None
    y: int | None
    o: int | None
    trusted_t: int | None
    context: Context


class StochasticBinaryMachine:
    """Reproducibly stochastic ER-1 environment with private episode RNG."""

    def __init__(self, config: ER1ProbabilityConfig = ER1_DEFAULT_CONFIG) -> None:
        self.config = config
        self._failure_mode = FailureMode.NO_STRUCTURAL_CHANGE
        self._episode_seed = 0
        self._rng = Random(0)
        self._context = Context.B
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._last_o: int | None = None
        self._last_t: int | None = None
        self._initial_observation_issued = False

    def reset(
        self,
        failure_mode: FailureMode,
        *,
        episode_seed: int = 0,
    ) -> None:
        """Reset all state and reseed the private episode random generator."""
        if failure_mode not in ER1_HYPOTHESES:
            raise ValueError("failure_mode must be an ER-1 hypothesis")
        if type(episode_seed) is not int:
            raise ValueError("episode_seed must be an integer")
        self._failure_mode = failure_mode
        self._episode_seed = episode_seed
        self._rng = Random(episode_seed)
        self._context = Context.B
        self._last_x = None
        self._last_y = None
        self._last_o = None
        self._last_t = None
        self._initial_observation_issued = False

    def initial_anomaly(self, *, x: int = 1) -> Observation:
        """Return the benchmark-conditioned X=1, O=0 entry observation.

        Hidden Y is sampled from its correct conditional distribution given O=0;
        the visible anomaly itself is fixed because episode selection conditions on it.
        """
        self._validate_bit(x, "x")
        if self._initial_observation_issued:
            raise RuntimeError("initial anomaly has already been issued")
        probability_y_one = self._conditional_y_one_given_o_zero(x)
        y = self._sample_bit(probability_y_one)
        self._record(x=x, y=y, o=0, trusted_t=None)
        self._initial_observation_issued = True
        return Observation(x=x, o=0)

    def step(self, x: int) -> Observation:
        """Run an ordinary unconditioned stochastic primary-sensor trial."""
        self._validate_bit(x, "x")
        y = self._sample_physical_output(x, self._context)
        o = self._sample_primary_observation(y)
        self._record(x=x, y=y, o=o, trusted_t=None)
        return Observation(x=x, o=o)

    def run_experiment(
        self,
        action: DiagnosticAction,
        *,
        x: int,
        context: Context | None = None,
    ) -> StochasticBenchmarkExperimentResult:
        """Execute one ER-1 action and expose only O or trusted observation T."""
        if action not in (
            DiagnosticAction.REPEAT_TRIAL,
            DiagnosticAction.USE_TRUSTED_SENSOR,
            DiagnosticAction.CHANGE_CONTEXT,
        ):
            raise ValueError("action is not available in ER-1")
        self._validate_bit(x, "x")
        if action is DiagnosticAction.REPEAT_TRIAL:
            if context is not None:
                raise ValueError("context is not valid for REPEAT_TRIAL")
            observation = self.step(x)
            return RepeatTrialResult(x=x, o=observation.o)
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            if context is not None:
                raise ValueError("context is not valid for USE_TRUSTED_SENSOR")
            y = self._sample_physical_output(x, self._context)
            trusted_t = self._sample_reliable_copy(
                y, self.config.trusted_sensor_reliability
            )
            self._record(x=x, y=y, o=None, trusted_t=trusted_t)
            return StochasticTrustedSensorResult(x=x, trusted_t=trusted_t)
        if not isinstance(context, Context):
            raise TypeError("CHANGE_CONTEXT requires a target Context")
        if context is self._context:
            raise ValueError("CHANGE_CONTEXT must select the alternate context")
        self._context = context
        observation = self.step(x)
        return ChangeContextResult(context=context, x=x, o=observation.o)

    def get_ground_truth(self) -> StochasticGroundTruth:
        """Return evaluation-only hidden state after agent interaction."""
        return StochasticGroundTruth(
            failure_mode=self._failure_mode,
            correct_repair=repair_for_failure(self._failure_mode),
            episode_seed=self._episode_seed,
            x=self._last_x,
            y=self._last_y,
            o=self._last_o,
            trusted_t=self._last_t,
            context=self._context,
        )

    def _sample_physical_output(self, x: int, context: Context) -> int:
        return self._sample_bit(self._probability_y_one(x, context))

    def _probability_y_one(self, x: int, context: Context) -> float:
        preferred_matches_x = self._preferred_matches_x(context)
        probability_match = self.config.preferred_physical_probability
        preferred_y = x if preferred_matches_x else 1 - x
        return probability_match if preferred_y == 1 else 1.0 - probability_match

    def _preferred_matches_x(self, context: Context) -> bool:
        if self._failure_mode is FailureMode.WORLD_SHIFT:
            return False
        if self._failure_mode is FailureMode.MISSING_LATENT_VARIABLE:
            return context is Context.A
        return True

    def _sample_primary_observation(self, y: int) -> int:
        if self._failure_mode is FailureMode.SENSOR_CORRUPTION:
            probability_matches = (
                1.0 - self.config.corrupted_sensor_inversion_probability
            )
        else:
            probability_matches = self.config.primary_sensor_reliability
        return self._sample_reliable_copy(y, probability_matches)

    def _conditional_y_one_given_o_zero(self, x: int) -> float:
        p_y_one = self._probability_y_one(x, self._context)
        p_o_zero_given_y_one = self._primary_probability(0, 1)
        p_o_zero_given_y_zero = self._primary_probability(0, 0)
        numerator = p_y_one * p_o_zero_given_y_one
        denominator = numerator + (1.0 - p_y_one) * p_o_zero_given_y_zero
        return numerator / denominator

    def _primary_probability(self, o: int, y: int) -> float:
        if self._failure_mode is FailureMode.SENSOR_CORRUPTION:
            matches_probability = (
                1.0 - self.config.corrupted_sensor_inversion_probability
            )
        else:
            matches_probability = self.config.primary_sensor_reliability
        return matches_probability if o == y else 1.0 - matches_probability

    def _sample_reliable_copy(self, value: int, reliability: float) -> int:
        return value if self._rng.random() < reliability else 1 - value

    def _sample_bit(self, probability_one: float) -> int:
        return 1 if self._rng.random() < probability_one else 0

    def _record(
        self,
        *,
        x: int,
        y: int,
        o: int | None,
        trusted_t: int | None,
    ) -> None:
        self._last_x = x
        self._last_y = y
        self._last_o = o
        self._last_t = trusted_t

    @staticmethod
    def _validate_bit(value: int, name: str) -> None:
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{name} must be the integer 0 or 1")
