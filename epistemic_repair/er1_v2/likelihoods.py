"""Persistent investigation likelihoods for ER-1 V2."""

from typing import Mapping

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticLikelihoodModel,
)
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import Context
from epistemic_repair.er1.config import ER1_HYPOTHESES
from epistemic_repair.er1_v2.config import (
    ER1_V2_DEFAULT_INVESTIGATION_CONFIG,
    ER1V2InvestigationConfig,
)
from epistemic_repair.failures.modes import FailureMode


class ER1V2LikelihoodModel(StochasticLikelihoodModel):
    """Normative likelihoods for persistent post-trigger experiments only."""

    def __init__(
        self,
        config: ER1V2InvestigationConfig = (
            ER1_V2_DEFAULT_INVESTIGATION_CONFIG
        ),
    ) -> None:
        # The inherited implementation uses only trusted reliability from this
        # object; V2 overrides every physical and primary-sensor probability.
        super().__init__()
        self.investigation_config = config

    def initial_anomaly_likelihood(self, hypothesis: FailureMode) -> float:
        """Reject ambiguous use of persistent dynamics for trigger likelihoods."""
        raise RuntimeError(
            "ER-1 V2 trigger likelihoods belong to TriggerLikelihoodModel"
        )

    def conditioned_initial_beliefs(
        self,
        base_prior: Mapping[FailureMode, float] | None = None,
    ) -> StochasticHypothesisBeliefs:
        """Reject trigger conditioning through the investigation model."""
        raise RuntimeError(
            "ER-1 V2 initial beliefs belong to TriggerLikelihoodModel"
        )

    def probability_y(
        self,
        y: int,
        hypothesis: FailureMode,
        context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return persistent P(Y=y | H,X,context)."""
        self._validate_inputs(hypothesis, context, x)
        self._validate_bit(y, "y")
        accuracy, matches_x = self._physical_spec(hypothesis, context)
        preferred_y = x if matches_x else 1 - x
        return accuracy if y == preferred_y else 1.0 - accuracy

    def probability_o_given_y(
        self,
        o: int,
        y: int,
        hypothesis: FailureMode,
    ) -> float:
        """Return persistent primary-sensor P(O=o | Y,H)."""
        self._validate_bit(o, "o")
        self._validate_bit(y, "y")
        if hypothesis not in ER1_HYPOTHESES:
            raise ValueError("hypothesis is not part of ER-1 V2")
        matches = (
            1.0
            - self.investigation_config.corrupted_sensor_inversion_accuracy
            if hypothesis is FailureMode.SENSOR_CORRUPTION
            else self.investigation_config.healthy_primary_sensor_accuracy
        )
        return matches if o == y else 1.0 - matches

    def probability_trusted_observation(
        self,
        trusted_t: int,
        hypothesis: FailureMode,
        context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return persistent P(T=t | H,X,context), marginalizing Y."""
        self._validate_bit(trusted_t, "trusted_t")
        reliability = self.investigation_config.trusted_sensor_accuracy
        return sum(
            self.probability_y(y, hypothesis, context, x=x)
            * (reliability if trusted_t == y else 1.0 - reliability)
            for y in (0, 1)
        )

    def _physical_spec(
        self,
        hypothesis: FailureMode,
        context: Context,
    ) -> tuple[float, bool]:
        config = self.investigation_config
        if hypothesis is FailureMode.NO_STRUCTURAL_CHANGE:
            return config.no_change_physical_accuracy, True
        if hypothesis is FailureMode.WORLD_SHIFT:
            return config.world_shift_physical_accuracy, False
        if hypothesis is FailureMode.SENSOR_CORRUPTION:
            return config.sensor_corruption_physical_accuracy, True
        return config.latent_physical_accuracy, context is Context.A
