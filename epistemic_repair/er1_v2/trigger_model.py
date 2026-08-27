"""One-time transient trigger conditioning for ER-1 V2."""

from dataclasses import dataclass
from typing import Mapping

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.er1.config import ER1_BASE_PRIOR, ER1_HYPOTHESES
from epistemic_repair.er1_v2.config import (
    ER1_V2_DEFAULT_TRIGGER_CONFIG,
    ER1V2TriggerConfig,
)
from epistemic_repair.failures.modes import FailureMode


@dataclass(frozen=True, slots=True)
class ER1V2TriggerEvent:
    """The fixed agent-visible event that caused investigation to begin."""

    x: int = 1
    o: int = 0

    def __post_init__(self) -> None:
        if self.x != 1 or self.o != 0:
            raise ValueError("ER-1 V2 trigger event must be X=1, O=0")


class TriggerLikelihoodModel:
    """Condition equal base beliefs once on the transient trigger anomaly."""

    event = ER1V2TriggerEvent()

    def __init__(
        self,
        config: ER1V2TriggerConfig = ER1_V2_DEFAULT_TRIGGER_CONFIG,
    ) -> None:
        self.config = config

    def likelihood(self, hypothesis: FailureMode) -> float:
        """Return P(A0 | H) for the fixed episode-entry trigger."""
        mapping = {
            FailureMode.NO_STRUCTURAL_CHANGE: self.config.no_structural_change,
            FailureMode.WORLD_SHIFT: self.config.world_shift,
            FailureMode.SENSOR_CORRUPTION: self.config.sensor_corruption,
            FailureMode.MISSING_LATENT_VARIABLE: (
                self.config.missing_latent_variable
            ),
        }
        try:
            return mapping[hypothesis]
        except KeyError as error:
            raise ValueError("hypothesis is not part of ER-1 V2") from error

    def conditioned_beliefs(
        self,
        base_prior: Mapping[FailureMode, float] = ER1_BASE_PRIOR,
    ) -> StochasticHypothesisBeliefs:
        """Return P(H|A0), applying trigger likelihoods exactly once."""
        if set(base_prior) != set(ER1_HYPOTHESES):
            raise ValueError("base_prior must contain exactly the ER-1 hypotheses")
        return StochasticHypothesisBeliefs.from_weights(
            {
                hypothesis: base_prior[hypothesis] * self.likelihood(hypothesis)
                for hypothesis in ER1_HYPOTHESES
            }
        )
