"""Immutable four-hypothesis belief distributions for stochastic ER-1."""

from dataclasses import dataclass
from math import isclose, isfinite, log2
from typing import Mapping

from epistemic_repair.er1.config import ER1_BASE_PRIOR, ER1_HYPOTHESES
from epistemic_repair.failures.modes import FailureMode


@dataclass(frozen=True, slots=True)
class StochasticHypothesisBeliefs:
    """Normalized beliefs over the four ER-1 hypotheses."""

    no_structural_change: float
    world_shift: float
    sensor_corruption: float
    missing_latent_variable: float

    def __post_init__(self) -> None:
        if any(
            not self._is_valid_number(value) or value < 0.0
            for value in self.values()
        ):
            raise ValueError("belief probabilities must be finite and non-negative")
        if not isclose(sum(self.values()), 1.0, abs_tol=1e-12):
            raise ValueError("belief probabilities must sum to 1")

    @classmethod
    def base_prior(cls) -> "StochasticHypothesisBeliefs":
        """Return the equal ER-1 prior before anomaly selection."""
        return cls.from_weights(ER1_BASE_PRIOR)

    @classmethod
    def from_weights(
        cls,
        weights: Mapping[FailureMode, float],
    ) -> "StochasticHypothesisBeliefs":
        """Normalize exact four-hypothesis non-negative weights."""
        if set(weights) != set(ER1_HYPOTHESES):
            raise ValueError("weights must contain exactly the four ER-1 hypotheses")
        if any(
            not cls._is_valid_number(value) or value < 0.0
            for value in weights.values()
        ):
            raise ValueError("weights must be finite and non-negative")
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("at least one hypothesis must have positive weight")
        return cls(
            no_structural_change=(
                weights[FailureMode.NO_STRUCTURAL_CHANGE] / total
            ),
            world_shift=weights[FailureMode.WORLD_SHIFT] / total,
            sensor_corruption=weights[FailureMode.SENSOR_CORRUPTION] / total,
            missing_latent_variable=(
                weights[FailureMode.MISSING_LATENT_VARIABLE] / total
            ),
        )

    def probability(self, hypothesis: FailureMode) -> float:
        """Return one ER-1 hypothesis probability."""
        if hypothesis is FailureMode.NO_STRUCTURAL_CHANGE:
            return self.no_structural_change
        if hypothesis is FailureMode.WORLD_SHIFT:
            return self.world_shift
        if hypothesis is FailureMode.SENSOR_CORRUPTION:
            return self.sensor_corruption
        if hypothesis is FailureMode.MISSING_LATENT_VARIABLE:
            return self.missing_latent_variable
        raise ValueError(f"{hypothesis!r} is not an ER-1 hypothesis")

    def values(self) -> tuple[float, float, float, float]:
        """Return probabilities in canonical ER-1 order."""
        return tuple(self.probability(hypothesis) for hypothesis in ER1_HYPOTHESES)

    def as_dict(self) -> dict[FailureMode, float]:
        """Return a copy keyed by ER-1 hypothesis."""
        return {
            hypothesis: self.probability(hypothesis)
            for hypothesis in ER1_HYPOTHESES
        }

    def entropy(self) -> float:
        """Return Shannon entropy in bits."""
        return -sum(
            probability * log2(probability)
            for probability in self.values()
            if probability > 0.0
        )

    def most_likely(self) -> FailureMode:
        """Return the maximum probability with canonical deterministic ties."""
        return max(ER1_HYPOTHESES, key=self.probability)

    def confidence(self) -> float:
        """Return the largest posterior probability."""
        return max(self.values())

    @staticmethod
    def _is_valid_number(value: object) -> bool:
        return type(value) in (int, float) and isfinite(value)
