"""Analytic ER-1 likelihoods, Bayesian updates, and information gain."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from epistemic_repair.beliefs.likelihoods import ImpossibleObservationError
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    RepeatTrialResult,
    StochasticBenchmarkExperimentResult,
    StochasticTrustedSensorResult,
)
from epistemic_repair.er1.config import (
    ER1_BASE_PRIOR,
    ER1_DEFAULT_CONFIG,
    ER1_HYPOTHESES,
    ER1ProbabilityConfig,
)
from epistemic_repair.failures.modes import FailureMode


class StochasticOutcomeSignal(str, Enum):
    """Agent-visible evidence channel for ER-1."""

    PRIMARY_O = "PRIMARY_O"
    TRUSTED_T = "TRUSTED_T"


@dataclass(frozen=True, slots=True)
class StochasticExperimentOutcome:
    """Binary ER-1 evidence consumed by the normative likelihood model."""

    signal: StochasticOutcomeSignal
    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value not in (0, 1):
            raise ValueError("outcome value must be the integer 0 or 1")


class StochasticLikelihoodModel:
    """Exact ER-1 generative likelihoods for X=1 and contexts A/B."""

    input_x = 1
    initial_context = Context.B
    initial_observation_value = 0

    def __init__(self, config: ER1ProbabilityConfig = ER1_DEFAULT_CONFIG) -> None:
        self.config = config

    @staticmethod
    def target_context(current_context: Context) -> Context:
        """Return the alternate controlled context."""
        if not isinstance(current_context, Context):
            raise TypeError("current_context must be a Context")
        return Context.A if current_context is Context.B else Context.B

    def probability_y(
        self,
        y: int,
        hypothesis: FailureMode,
        context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return P(Y=y | H, X, context)."""
        self._validate_inputs(hypothesis, context, x)
        self._validate_bit(y, "y")
        preferred_matches_x = self._preferred_matches_x(hypothesis, context)
        preferred_y = x if preferred_matches_x else 1 - x
        preferred = self.config.preferred_physical_probability
        return preferred if y == preferred_y else 1.0 - preferred

    def probability_primary_observation(
        self,
        o: int,
        hypothesis: FailureMode,
        context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return P(O=o | H, X, context), marginalizing hidden Y."""
        self._validate_bit(o, "o")
        return sum(
            self.probability_y(y, hypothesis, context, x=x)
            * self.probability_o_given_y(o, y, hypothesis)
            for y in (0, 1)
        )

    def probability_o_given_y(
        self,
        o: int,
        y: int,
        hypothesis: FailureMode,
    ) -> float:
        """Return the primary sensor channel probability P(O=o | Y,H)."""
        self._validate_bit(o, "o")
        self._validate_bit(y, "y")
        if hypothesis not in ER1_HYPOTHESES:
            raise ValueError("hypothesis is not part of ER-1")
        if hypothesis is FailureMode.SENSOR_CORRUPTION:
            matches = 1.0 - self.config.corrupted_sensor_inversion_probability
        else:
            matches = self.config.primary_sensor_reliability
        return matches if o == y else 1.0 - matches

    def probability_trusted_observation(
        self,
        trusted_t: int,
        hypothesis: FailureMode,
        context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return P(T=t | H, X, context), marginalizing hidden Y."""
        self._validate_bit(trusted_t, "trusted_t")
        reliability = self.config.trusted_sensor_reliability
        return sum(
            self.probability_y(y, hypothesis, context, x=x)
            * (reliability if trusted_t == y else 1.0 - reliability)
            for y in (0, 1)
        )

    def initial_anomaly_likelihood(self, hypothesis: FailureMode) -> float:
        """Return P(X=1,O=0 entry anomaly | H); X is benchmark-controlled."""
        return self.probability_primary_observation(
            self.initial_observation_value,
            hypothesis,
            self.initial_context,
            x=self.input_x,
        )

    def conditioned_initial_beliefs(
        self,
        base_prior: Mapping[FailureMode, float] = ER1_BASE_PRIOR,
    ) -> StochasticHypothesisBeliefs:
        """Condition a base prior on benchmark selection event X=1,O=0."""
        if set(base_prior) != set(ER1_HYPOTHESES):
            raise ValueError("base_prior must contain exactly the ER-1 hypotheses")
        weights = {
            hypothesis: base_prior[hypothesis]
            * self.initial_anomaly_likelihood(hypothesis)
            for hypothesis in ER1_HYPOTHESES
        }
        return StochasticHypothesisBeliefs.from_weights(weights)

    def likelihood(
        self,
        outcome: StochasticExperimentOutcome,
        hypothesis: FailureMode,
        action: DiagnosticAction,
        current_context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return P(outcome | H, action, current context)."""
        self._validate_action(action)
        self._validate_inputs(hypothesis, current_context, x)
        effective_context = (
            self.target_context(current_context)
            if action is DiagnosticAction.CHANGE_CONTEXT
            else current_context
        )
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            if outcome.signal is not StochasticOutcomeSignal.TRUSTED_T:
                return 0.0
            return self.probability_trusted_observation(
                outcome.value, hypothesis, effective_context, x=x
            )
        if outcome.signal is not StochasticOutcomeSignal.PRIMARY_O:
            return 0.0
        return self.probability_primary_observation(
            outcome.value, hypothesis, effective_context, x=x
        )

    def predictive_distribution(
        self,
        beliefs: StochasticHypothesisBeliefs,
        action: DiagnosticAction,
        current_context: Context,
        *,
        x: int = 1,
    ) -> dict[StochasticExperimentOutcome, float]:
        """Marginalize ER-1 outcome probabilities over current beliefs."""
        signal = (
            StochasticOutcomeSignal.TRUSTED_T
            if action is DiagnosticAction.USE_TRUSTED_SENSOR
            else StochasticOutcomeSignal.PRIMARY_O
        )
        return {
            StochasticExperimentOutcome(signal, value): sum(
                beliefs.probability(hypothesis)
                * self.likelihood(
                    StochasticExperimentOutcome(signal, value),
                    hypothesis,
                    action,
                    current_context,
                    x=x,
                )
                for hypothesis in ER1_HYPOTHESES
            )
            for value in (0, 1)
        }

    def update(
        self,
        prior: StochasticHypothesisBeliefs,
        action: DiagnosticAction,
        outcome: StochasticExperimentOutcome,
        current_context: Context,
        *,
        x: int = 1,
    ) -> StochasticHypothesisBeliefs:
        """Apply a soft Bayes update using ER-1 likelihoods."""
        weights = {
            hypothesis: prior.probability(hypothesis)
            * self.likelihood(
                outcome,
                hypothesis,
                action,
                current_context,
                x=x,
            )
            for hypothesis in ER1_HYPOTHESES
        }
        if sum(weights.values()) <= 0.0:
            raise ImpossibleObservationError(
                "observation has zero probability under all ER-1 hypotheses"
            )
        return StochasticHypothesisBeliefs.from_weights(weights)

    def outcome_from_result(
        self,
        action: DiagnosticAction,
        result: StochasticBenchmarkExperimentResult,
        current_context: Context,
    ) -> StochasticExperimentOutcome:
        """Convert a typed ER-1 result into normative observable evidence."""
        self._validate_action(action)
        if action is DiagnosticAction.REPEAT_TRIAL:
            if not isinstance(result, RepeatTrialResult):
                raise TypeError("REPEAT_TRIAL requires RepeatTrialResult")
            return StochasticExperimentOutcome(
                StochasticOutcomeSignal.PRIMARY_O, result.o
            )
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            if not isinstance(result, StochasticTrustedSensorResult):
                raise TypeError(
                    "USE_TRUSTED_SENSOR requires StochasticTrustedSensorResult"
                )
            return StochasticExperimentOutcome(
                StochasticOutcomeSignal.TRUSTED_T, result.trusted_t
            )
        if not isinstance(result, ChangeContextResult):
            raise TypeError("CHANGE_CONTEXT requires ChangeContextResult")
        if result.context is not self.target_context(current_context):
            raise ValueError("context result does not match modeled intervention")
        return StochasticExperimentOutcome(
            StochasticOutcomeSignal.PRIMARY_O, result.o
        )

    @staticmethod
    def _preferred_matches_x(
        hypothesis: FailureMode,
        context: Context,
    ) -> bool:
        if hypothesis is FailureMode.WORLD_SHIFT:
            return False
        if hypothesis is FailureMode.MISSING_LATENT_VARIABLE:
            return context is Context.A
        return True

    @staticmethod
    def _validate_action(action: DiagnosticAction) -> None:
        if action not in BENCHMARK_ACTIONS:
            raise ValueError("action is not available to ER-1 benchmark policies")

    @classmethod
    def _validate_inputs(
        cls,
        hypothesis: FailureMode,
        context: Context,
        x: int,
    ) -> None:
        if hypothesis not in ER1_HYPOTHESES:
            raise ValueError("hypothesis is not part of ER-1")
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        cls._validate_bit(x, "x")

    @staticmethod
    def _validate_bit(value: int, name: str) -> None:
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{name} must be the integer 0 or 1")


@dataclass(frozen=True, slots=True)
class StochasticActionInformationGains:
    """Expected information gain for each ER-1 benchmark action."""

    repeat_trial: float
    use_trusted_sensor: float
    change_context: float

    def for_action(self, action: DiagnosticAction) -> float:
        if action is DiagnosticAction.REPEAT_TRIAL:
            return self.repeat_trial
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            return self.use_trusted_sensor
        if action is DiagnosticAction.CHANGE_CONTEXT:
            return self.change_context
        raise ValueError("action is not available to ER-1")

    def items(self) -> tuple[tuple[DiagnosticAction, float], ...]:
        return tuple((action, self.for_action(action)) for action in BENCHMARK_ACTIONS)

    def best_value(self) -> float:
        return max(value for _, value in self.items())


def stochastic_expected_information_gain(
    beliefs: StochasticHypothesisBeliefs,
    action: DiagnosticAction,
    likelihood_model: StochasticLikelihoodModel,
    current_context: Context,
    *,
    x: int = 1,
) -> float:
    """Compute ER-1 expected entropy reduction over all stochastic outcomes."""
    expected_entropy = 0.0
    for outcome, probability in likelihood_model.predictive_distribution(
        beliefs, action, current_context, x=x
    ).items():
        if probability <= 0.0:
            continue
        posterior = likelihood_model.update(
            beliefs, action, outcome, current_context, x=x
        )
        expected_entropy += probability * posterior.entropy()
    gain = beliefs.entropy() - expected_entropy
    return 0.0 if abs(gain) < 1e-12 else gain


def stochastic_information_gains(
    beliefs: StochasticHypothesisBeliefs,
    likelihood_model: StochasticLikelihoodModel,
    current_context: Context,
    *,
    x: int = 1,
) -> StochasticActionInformationGains:
    """Return expected information gain for all ER-1 actions."""
    return StochasticActionInformationGains(
        repeat_trial=stochastic_expected_information_gain(
            beliefs,
            DiagnosticAction.REPEAT_TRIAL,
            likelihood_model,
            current_context,
            x=x,
        ),
        use_trusted_sensor=stochastic_expected_information_gain(
            beliefs,
            DiagnosticAction.USE_TRUSTED_SENSOR,
            likelihood_model,
            current_context,
            x=x,
        ),
        change_context=stochastic_expected_information_gain(
            beliefs,
            DiagnosticAction.CHANGE_CONTEXT,
            likelihood_model,
            current_context,
            x=x,
        ),
    )
