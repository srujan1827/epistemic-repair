"""LLM investigator policies using only restricted benchmark-agent views."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from epistemic_repair.beliefs.state import HypothesisBeliefs
from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMTransientError,
)
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.sanitize import sanitize_text
from epistemic_repair.llm.schemas import (
    FullAutonomousDecision,
    LLMDecision,
    PlannerOnlyDecision,
    StructuredResponseError,
    er1_full_autonomous_json_schema,
    er1_planner_only_json_schema,
    full_autonomous_json_schema,
    parse_er1_full_autonomous_response,
    parse_er1_planner_only_response,
    parse_full_autonomous_response,
    parse_planner_only_response,
    planner_only_json_schema,
)
from epistemic_repair.policies.views import BenchmarkAgentView
from epistemic_repair.policies.stochastic_views import StochasticBenchmarkAgentView
from epistemic_repair.prompts.binary_er1 import (
    BINARY_ER1_PROMPT_VERSION,
    build_er1_full_autonomous_prompt,
    build_er1_planner_only_prompt,
)
from epistemic_repair.prompts.binary_v0 import (
    BINARY_V0_PROMPT_VERSION,
    build_full_autonomous_prompt,
    build_planner_only_prompt,
)


class LLMAttemptStatus(str, Enum):
    """Auditable outcome of one provider request attempt."""

    VALID = "VALID"
    INVALID_FORMAT = "INVALID_FORMAT"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True, slots=True)
class LLMAttemptRecord:
    """Sanitized record of one bounded provider attempt."""

    attempt_number: int
    status: LLMAttemptStatus
    raw_output: str | None
    provider_request_id: str | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class LLMPolicyResult:
    """A parsed decision or explicit failure plus its complete attempt history."""

    prompt: str
    prompt_version: str
    decision: LLMDecision | None
    attempts: tuple[LLMAttemptRecord, ...]

    @property
    def retry_count(self) -> int:
        """Return provider calls beyond the initial attempt."""
        return max(0, len(self.attempts) - 1)

    @property
    def succeeded(self) -> bool:
        """Return whether a valid structured decision was produced."""
        return self.decision is not None


class FullAutonomousLLMPolicy:
    """LLM policy responsible for beliefs, planning, stopping, and diagnosis."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config

    def decide(self, view: BenchmarkAgentView) -> LLMPolicyResult:
        """Request one autonomous decision from exactly a restricted view."""
        if type(view) is not BenchmarkAgentView:
            raise TypeError("full-autonomous policy requires BenchmarkAgentView")
        prompt = build_full_autonomous_prompt(view)
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=prompt,
            schema=full_autonomous_json_schema(),
            parser=parse_full_autonomous_response,
            prompt_version=BINARY_V0_PROMPT_VERSION,
        )


class PlannerOnlyLLMPolicy:
    """LLM policy selecting experiments from safe history and supplied beliefs."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config

    def decide(
        self,
        view: BenchmarkAgentView,
        normative_beliefs: HypothesisBeliefs,
    ) -> LLMPolicyResult:
        """Request one planning decision without exposing likelihood machinery."""
        if type(view) is not BenchmarkAgentView:
            raise TypeError("planner-only policy requires BenchmarkAgentView")
        prompt = build_planner_only_prompt(view, normative_beliefs)
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=prompt,
            schema=planner_only_json_schema(),
            parser=parse_planner_only_response,
            prompt_version=BINARY_V0_PROMPT_VERSION,
        )


class ER1FullAutonomousLLMPolicy:
    """ER-1 LLM policy responsible for beliefs, planning, and stopping."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config

    def decide(self, view: StochasticBenchmarkAgentView) -> LLMPolicyResult:
        if type(view) is not StochasticBenchmarkAgentView:
            raise TypeError(
                "ER-1 autonomous policy requires StochasticBenchmarkAgentView"
            )
        prompt = build_er1_full_autonomous_prompt(view)
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=prompt,
            schema=er1_full_autonomous_json_schema(),
            parser=parse_er1_full_autonomous_response,
            prompt_version=BINARY_ER1_PROMPT_VERSION,
        )


class ER1PlannerOnlyLLMPolicy:
    """ER-1 planner policy using supplied four-way normative beliefs."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config

    def decide(
        self,
        view: StochasticBenchmarkAgentView,
        normative_beliefs: StochasticHypothesisBeliefs,
    ) -> LLMPolicyResult:
        if type(view) is not StochasticBenchmarkAgentView:
            raise TypeError(
                "ER-1 planner policy requires StochasticBenchmarkAgentView"
            )
        prompt = build_er1_planner_only_prompt(view, normative_beliefs)
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=prompt,
            schema=er1_planner_only_json_schema(),
            parser=parse_er1_planner_only_response,
            prompt_version=BINARY_ER1_PROMPT_VERSION,
        )


def _generate_with_retries(
    *,
    client: LLMClient,
    config: LLMConfig,
    prompt: str,
    schema: dict[str, object],
    parser: Callable[[str], LLMDecision],
    prompt_version: str,
) -> LLMPolicyResult:
    attempts: list[LLMAttemptRecord] = []
    request = LLMRequest(prompt=prompt, response_schema=schema)

    for attempt_number in range(1, config.max_retries + 2):
        try:
            response = client.generate(request)
            sanitized_output = sanitize_text(response.text)
            try:
                decision = parser(response.text)
            except StructuredResponseError as error:
                attempts.append(
                    LLMAttemptRecord(
                        attempt_number=attempt_number,
                        status=LLMAttemptStatus.INVALID_FORMAT,
                        raw_output=sanitized_output,
                        provider_request_id=response.provider_request_id,
                        error_type=type(error).__name__,
                        error_message=sanitize_text(str(error)),
                    )
                )
                if attempt_number <= config.max_retries:
                    continue
                return LLMPolicyResult(
                    prompt=prompt,
                    prompt_version=prompt_version,
                    decision=None,
                    attempts=tuple(attempts),
                )

            attempts.append(
                LLMAttemptRecord(
                    attempt_number=attempt_number,
                    status=LLMAttemptStatus.VALID,
                    raw_output=sanitized_output,
                    provider_request_id=response.provider_request_id,
                    error_type=None,
                    error_message=None,
                )
            )
            return LLMPolicyResult(
                prompt=prompt,
                prompt_version=prompt_version,
                decision=decision,
                attempts=tuple(attempts),
            )
        except LLMTransientError as error:
            attempts.append(
                _error_attempt(attempt_number, LLMAttemptStatus.TRANSIENT_ERROR, error)
            )
            if attempt_number <= config.max_retries:
                continue
        except LLMError as error:
            attempts.append(
                _error_attempt(attempt_number, LLMAttemptStatus.PROVIDER_ERROR, error)
            )

        return LLMPolicyResult(
            prompt=prompt,
            prompt_version=prompt_version,
            decision=None,
            attempts=tuple(attempts),
        )

    raise AssertionError("bounded retry loop must return")


def _error_attempt(
    attempt_number: int,
    status: LLMAttemptStatus,
    error: Exception,
) -> LLMAttemptRecord:
    return LLMAttemptRecord(
        attempt_number=attempt_number,
        status=status,
        raw_output=None,
        provider_request_id=None,
        error_type=type(error).__name__,
        error_message=sanitize_text(str(error)),
    )
