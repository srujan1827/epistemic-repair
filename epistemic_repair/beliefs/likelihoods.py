"""Central deterministic outcome model and Bayesian information calculations."""

from dataclasses import dataclass
from enum import Enum

from epistemic_repair.beliefs.state import HYPOTHESES, HypothesisBeliefs
from epistemic_repair.diagnostics.actions import (
    BENCHMARK_ACTIONS,
    Context,
    DiagnosticAction,
)
from epistemic_repair.diagnostics.results import (
    BenchmarkExperimentResult,
    ChangeContextResult,
    RepeatTrialResult,
    TrustedSensorResult,
)
from epistemic_repair.failures.modes import FailureMode


class OutcomeSignal(str, Enum):
    """Observable channel used by a benchmark experiment outcome."""

    PRIMARY_O = "PRIMARY_O"
    TRUSTED_Y = "TRUSTED_Y"


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Minimal evidence used by the deterministic likelihood model."""

    signal: OutcomeSignal
    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value not in (0, 1):
            raise ValueError("outcome value must be the integer 0 or 1")


class ImpossibleObservationError(ValueError):
    """Raised when an observation has zero probability under every hypothesis."""


class DeterministicLikelihoodModel:
    """Normative V0 outcome model for X=1 and observable context state."""

    input_x = 1

    @staticmethod
    def target_context(current_context: Context) -> Context:
        """Return the context selected by the benchmark change action."""
        if not isinstance(current_context, Context):
            raise TypeError("current_context must be a Context")
        return Context.A if current_context is Context.B else Context.B

    def predict_outcome(
        self,
        action: DiagnosticAction,
        hypothesis: FailureMode,
        current_context: Context,
        *,
        x: int = 1,
    ) -> ExperimentOutcome:
        """Predict the deterministic result of an action under one hypothesis."""
        self._validate_action(action)
        self._validate_bit(x)
        if not isinstance(current_context, Context):
            raise TypeError("current_context must be a Context")
        if hypothesis not in HYPOTHESES:
            raise ValueError("hypothesis is not part of the benchmark")

        effective_context = (
            self.target_context(current_context)
            if action is DiagnosticAction.CHANGE_CONTEXT
            else current_context
        )
        y = self._physical_output(hypothesis, effective_context, x)

        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            return ExperimentOutcome(OutcomeSignal.TRUSTED_Y, y)

        o = 1 - y if hypothesis is FailureMode.SENSOR_CORRUPTION else y
        return ExperimentOutcome(OutcomeSignal.PRIMARY_O, o)

    def outcome_from_result(
        self,
        action: DiagnosticAction,
        result: BenchmarkExperimentResult,
        current_context: Context,
    ) -> ExperimentOutcome:
        """Convert a typed environment result into likelihood evidence."""
        self._validate_action(action)
        if not isinstance(current_context, Context):
            raise TypeError("current_context must be a Context")
        self._validate_bit(result.x)

        if action is DiagnosticAction.REPEAT_TRIAL:
            if not isinstance(result, RepeatTrialResult):
                raise TypeError("REPEAT_TRIAL requires RepeatTrialResult")
            return ExperimentOutcome(OutcomeSignal.PRIMARY_O, result.o)

        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            if not isinstance(result, TrustedSensorResult):
                raise TypeError("USE_TRUSTED_SENSOR requires TrustedSensorResult")
            return ExperimentOutcome(OutcomeSignal.TRUSTED_Y, result.trusted_y)

        if not isinstance(result, ChangeContextResult):
            raise TypeError("CHANGE_CONTEXT requires ChangeContextResult")
        expected_context = self.target_context(current_context)
        if result.context is not expected_context:
            raise ValueError("context result does not match the modeled intervention")
        return ExperimentOutcome(OutcomeSignal.PRIMARY_O, result.o)

    def likelihood(
        self,
        outcome: ExperimentOutcome,
        hypothesis: FailureMode,
        action: DiagnosticAction,
        current_context: Context,
        *,
        x: int = 1,
    ) -> float:
        """Return P(outcome | hypothesis, action, context)."""
        predicted = self.predict_outcome(
            action, hypothesis, current_context, x=x
        )
        return 1.0 if predicted == outcome else 0.0

    def predictive_distribution(
        self,
        beliefs: HypothesisBeliefs,
        action: DiagnosticAction,
        current_context: Context,
        *,
        x: int = 1,
    ) -> dict[ExperimentOutcome, float]:
        """Return outcome probabilities marginalized over current beliefs."""
        distribution: dict[ExperimentOutcome, float] = {}
        for hypothesis in HYPOTHESES:
            outcome = self.predict_outcome(
                action, hypothesis, current_context, x=x
            )
            distribution[outcome] = (
                distribution.get(outcome, 0.0) + beliefs.probability(hypothesis)
            )
        return distribution

    def update(
        self,
        prior: HypothesisBeliefs,
        action: DiagnosticAction,
        outcome: ExperimentOutcome,
        current_context: Context,
        *,
        x: int = 1,
    ) -> HypothesisBeliefs:
        """Apply exact Bayes updating from the deterministic likelihoods."""
        weights = {
            hypothesis: prior.probability(hypothesis)
            * self.likelihood(
                outcome, hypothesis, action, current_context, x=x
            )
            for hypothesis in HYPOTHESES
        }
        if sum(weights.values()) <= 0.0:
            raise ImpossibleObservationError(
                "observation has zero probability under all current hypotheses"
            )
        return HypothesisBeliefs.from_weights(weights)

    @staticmethod
    def _physical_output(
        hypothesis: FailureMode, context: Context, x: int
    ) -> int:
        """Return Y for a binary input under a benchmark hypothesis/context."""
        if hypothesis is FailureMode.WORLD_SHIFT:
            return 1 - x
        if hypothesis is FailureMode.SENSOR_CORRUPTION:
            return x
        return x if context is Context.A else 1 - x

    @staticmethod
    def _validate_action(action: DiagnosticAction) -> None:
        if action not in BENCHMARK_ACTIONS:
            raise ValueError(f"{action!r} is not available to benchmark policies")

    @staticmethod
    def _validate_bit(x: int) -> None:
        if type(x) is not int or x not in (0, 1):
            raise ValueError("x must be the integer 0 or 1")


def expected_information_gain(
    beliefs: HypothesisBeliefs,
    action: DiagnosticAction,
    likelihood_model: DeterministicLikelihoodModel,
    current_context: Context,
    *,
    x: int = 1,
) -> float:
    """Compute expected entropy reduction before observing an outcome."""
    expected_posterior_entropy = 0.0
    for outcome, predictive_probability in likelihood_model.predictive_distribution(
        beliefs, action, current_context, x=x
    ).items():
        if predictive_probability <= 0.0:
            continue
        posterior = likelihood_model.update(
            beliefs, action, outcome, current_context, x=x
        )
        expected_posterior_entropy += (
            predictive_probability * posterior.entropy()
        )

    gain = beliefs.entropy() - expected_posterior_entropy
    return 0.0 if abs(gain) < 1e-12 else gain


@dataclass(frozen=True, slots=True)
class ActionInformationGains:
    """Expected information gain for the exact benchmark action set."""

    repeat_trial: float
    use_trusted_sensor: float
    change_context: float

    def for_action(self, action: DiagnosticAction) -> float:
        """Return the score for one benchmark action."""
        if action is DiagnosticAction.REPEAT_TRIAL:
            return self.repeat_trial
        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            return self.use_trusted_sensor
        if action is DiagnosticAction.CHANGE_CONTEXT:
            return self.change_context
        raise ValueError(f"{action!r} is not available to benchmark policies")

    def items(self) -> tuple[tuple[DiagnosticAction, float], ...]:
        """Return scores in deterministic benchmark order."""
        return tuple((action, self.for_action(action)) for action in BENCHMARK_ACTIONS)

    def best_value(self) -> float:
        """Return the maximum available expected information gain."""
        return max(score for _, score in self.items())


def information_gains(
    beliefs: HypothesisBeliefs,
    likelihood_model: DeterministicLikelihoodModel,
    current_context: Context,
    *,
    x: int = 1,
) -> ActionInformationGains:
    """Calculate expected information gain for every benchmark action."""
    return ActionInformationGains(
        repeat_trial=expected_information_gain(
            beliefs,
            DiagnosticAction.REPEAT_TRIAL,
            likelihood_model,
            current_context,
            x=x,
        ),
        use_trusted_sensor=expected_information_gain(
            beliefs,
            DiagnosticAction.USE_TRUSTED_SENSOR,
            likelihood_model,
            current_context,
            x=x,
        ),
        change_context=expected_information_gain(
            beliefs,
            DiagnosticAction.CHANGE_CONTEXT,
            likelihood_model,
            current_context,
            x=x,
        ),
    )
