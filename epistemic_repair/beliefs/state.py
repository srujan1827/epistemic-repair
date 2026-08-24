"""Immutable probability distributions over benchmark hypotheses."""

from dataclasses import dataclass
from math import isclose, isfinite, log2
from typing import Mapping

from epistemic_repair.failures.modes import FailureMode


HYPOTHESES: tuple[FailureMode, ...] = (
    FailureMode.WORLD_SHIFT,
    FailureMode.SENSOR_CORRUPTION,
    FailureMode.MISSING_LATENT_VARIABLE,
)


@dataclass(frozen=True, slots=True)
class HypothesisBeliefs:
    """Normalized beliefs over the three deterministic benchmark hypotheses."""

    world_shift: float
    sensor_corruption: float
    missing_latent_variable: float

    def __post_init__(self) -> None:
        probabilities = self.values()
        if any(
            not self._is_valid_number(value) or value < 0.0
            for value in probabilities
        ):
            raise ValueError("belief probabilities must be finite and non-negative")
        if not isclose(sum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("belief probabilities must sum to 1")

    @classmethod
    def uniform(cls) -> "HypothesisBeliefs":
        """Return the V0 uniform prior."""
        probability = 1.0 / len(HYPOTHESES)
        return cls(probability, probability, probability)

    @classmethod
    def from_weights(
        cls, weights: Mapping[FailureMode, float]
    ) -> "HypothesisBeliefs":
        """Normalize non-negative weights into a belief distribution."""
        if set(weights) != set(HYPOTHESES):
            raise ValueError("weights must contain exactly the three hypotheses")
        if any(
            not cls._is_valid_number(value) or value < 0.0
            for value in weights.values()
        ):
            raise ValueError("weights must be finite and non-negative")
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("at least one hypothesis must have positive weight")
        return cls(
            world_shift=weights[FailureMode.WORLD_SHIFT] / total,
            sensor_corruption=weights[FailureMode.SENSOR_CORRUPTION] / total,
            missing_latent_variable=(
                weights[FailureMode.MISSING_LATENT_VARIABLE] / total
            ),
        )

    def probability(self, hypothesis: FailureMode) -> float:
        """Query the probability of a benchmark hypothesis."""
        if hypothesis is FailureMode.WORLD_SHIFT:
            return self.world_shift
        if hypothesis is FailureMode.SENSOR_CORRUPTION:
            return self.sensor_corruption
        if hypothesis is FailureMode.MISSING_LATENT_VARIABLE:
            return self.missing_latent_variable
        raise ValueError(f"{hypothesis!r} is not a benchmark hypothesis")

    def values(self) -> tuple[float, float, float]:
        """Return probabilities in the canonical hypothesis order."""
        return (
            self.world_shift,
            self.sensor_corruption,
            self.missing_latent_variable,
        )

    def as_dict(self) -> dict[FailureMode, float]:
        """Return a copy keyed by hypothesis."""
        return {
            hypothesis: self.probability(hypothesis) for hypothesis in HYPOTHESES
        }

    def entropy(self) -> float:
        """Return Shannon entropy in bits, treating zero terms as zero."""
        return -sum(
            probability * log2(probability)
            for probability in self.values()
            if probability > 0.0
        )

    def most_likely(self) -> FailureMode:
        """Return the maximum-probability hypothesis using canonical tie order."""
        return max(HYPOTHESES, key=self.probability)

    def confidence(self) -> float:
        """Return the largest posterior probability."""
        return max(self.values())

    @staticmethod
    def _is_valid_number(value: object) -> bool:
        """Accept ordinary numeric weights while excluding booleans and non-finite values."""
        return type(value) in (int, float) and isfinite(value)
