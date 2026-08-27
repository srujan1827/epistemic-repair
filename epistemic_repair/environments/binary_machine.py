"""Deterministic binary world with deliberately ambiguous failures."""

from dataclasses import dataclass

from epistemic_repair.diagnostics.actions import Context, DiagnosticAction
from epistemic_repair.diagnostics.results import (
    ChangeContextResult,
    ExperimentResult,
    LatentInspectionResult,
    RepeatTrialResult,
    TrustedSensorResult,
)
from epistemic_repair.failures.modes import FailureMode
from epistemic_repair.repair.operators import RepairOperator, repair_for_failure


_LATENT_BY_CONTEXT: dict[Context, int] = {
    Context.A: 0,
    Context.B: 1,
}


@dataclass(frozen=True, slots=True)
class Observation:
    """Agent-visible result of an action; hidden state is intentionally absent."""

    x: int
    o: int


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Evaluation-only snapshot of the current episode and latest transition."""

    failure_mode: FailureMode
    correct_repair: RepairOperator
    x: int | None
    y: int | None
    o: int | None
    z: int | None
    context: Context


class BinaryMachine:
    """A deterministic binary input/output environment.

    ``step`` exposes only X and sensor observation O. ``get_ground_truth`` is a
    separate evaluation interface and must not be supplied to an agent.
    """

    def __init__(self) -> None:
        self._failure_mode = FailureMode.NORMAL
        self._context = Context.B
        self._z: int | None = None
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._last_o: int | None = None

    def reset(
        self,
        failure_mode: FailureMode = FailureMode.NORMAL,
        *,
        latent_z: int | None = None,
    ) -> None:
        """Start a fresh episode and clear all transition state.

        ``latent_z`` is valid only for ``MISSING_LATENT_VARIABLE``. It defaults
        to 1 for that failure episode, while callers may set it to 0 to produce
        the pre-anomaly behavior Y=X.
        """
        if not isinstance(failure_mode, FailureMode):
            raise TypeError("failure_mode must be a FailureMode")
        if failure_mode is FailureMode.NO_STRUCTURAL_CHANGE:
            raise ValueError(
                "NO_STRUCTURAL_CHANGE is an ER-1-only stochastic hypothesis"
            )

        self._context = Context.B
        if failure_mode is FailureMode.MISSING_LATENT_VARIABLE:
            if latent_z is None:
                latent_z = 1
            self._validate_bit(latent_z, name="latent_z")
            self._context = Context.A if latent_z == 0 else Context.B
            self._z = self._latent_for_context(self._context)
        else:
            if latent_z is not None:
                raise ValueError(
                    "latent_z is only valid for MISSING_LATENT_VARIABLE"
                )
            self._z = None

        self._failure_mode = failure_mode
        self._last_x = None
        self._last_y = None
        self._last_o = None

    def step(self, x: int) -> Observation:
        """Apply binary input X and return only the agent-visible observation."""
        self._validate_bit(x, name="x")
        _, o = self._run_transition(x)
        return Observation(x=x, o=o)

    def run_experiment(
        self,
        action: DiagnosticAction,
        *,
        x: int | None = None,
        context: Context | None = None,
    ) -> ExperimentResult:
        """Run one explicit diagnostic action and return only its valid evidence.

        ``REPEAT_TRIAL`` and ``USE_TRUSTED_SENSOR`` require ``x``.
        ``CHANGE_CONTEXT`` requires both ``x`` and a target ``context``.
        ``INSPECT_LATENT_VARIABLE`` takes neither argument.
        """
        if not isinstance(action, DiagnosticAction):
            raise TypeError("action must be a DiagnosticAction")

        if action is DiagnosticAction.INSPECT_LATENT_VARIABLE:
            self._reject_unexpected_argument(x, name="x")
            self._reject_unexpected_argument(context, name="context")
            if self._failure_mode is FailureMode.MISSING_LATENT_VARIABLE:
                assert self._z is not None
                return LatentInspectionResult(available=True, value=self._z)
            return LatentInspectionResult(available=False)

        if x is None:
            raise ValueError(f"x is required for {action.value}")
        self._validate_bit(x, name="x")

        if action is DiagnosticAction.REPEAT_TRIAL:
            self._reject_unexpected_argument(context, name="context")
            observation = self.step(x)
            return RepeatTrialResult(x=observation.x, o=observation.o)

        if action is DiagnosticAction.USE_TRUSTED_SENSOR:
            self._reject_unexpected_argument(context, name="context")
            y, _ = self._run_transition(x)
            return TrustedSensorResult(x=x, trusted_y=y)

        if context is None:
            raise ValueError("context is required for CHANGE_CONTEXT")
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        self._context = context
        self._sync_latent_to_context()
        observation = self.step(x)
        return ChangeContextResult(
            context=context,
            x=observation.x,
            o=observation.o,
        )

    def _run_transition(self, x: int) -> tuple[int, int]:
        """Compute and record one physical and primary-sensor transition."""
        if self._failure_mode is FailureMode.WORLD_SHIFT:
            y = 1 - x
        elif self._failure_mode is FailureMode.MISSING_LATENT_VARIABLE:
            # reset guarantees that Z is present for this condition.
            assert self._z is not None
            y = x if self._z == 0 else 1 - x
        else:
            y = x

        o = 1 - y if self._failure_mode is FailureMode.SENSOR_CORRUPTION else y

        self._last_x = x
        self._last_y = y
        self._last_o = o
        return y, o

    def get_ground_truth(self) -> GroundTruth:
        """Return hidden state for evaluation and debugging only."""
        return GroundTruth(
            failure_mode=self._failure_mode,
            correct_repair=repair_for_failure(self._failure_mode),
            x=self._last_x,
            y=self._last_y,
            o=self._last_o,
            z=self._z,
            context=self._context,
        )

    def _sync_latent_to_context(self) -> None:
        """Update Z from the controlled context only when Z is causally present."""
        if self._failure_mode is FailureMode.MISSING_LATENT_VARIABLE:
            self._z = self._latent_for_context(self._context)
        else:
            self._z = None

    @staticmethod
    def _latent_for_context(context: Context) -> int:
        """Apply the explicit deterministic context-to-latent mechanism."""
        return _LATENT_BY_CONTEXT[context]

    @staticmethod
    def _reject_unexpected_argument(value: object, *, name: str) -> None:
        """Reject arguments that are not meaningful for a selected action."""
        if value is not None:
            raise ValueError(f"{name} is not valid for this diagnostic action")

    @staticmethod
    def _validate_bit(value: int, *, name: str) -> None:
        """Reject values outside the exact integer bit domain."""
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{name} must be the integer 0 or 1")
