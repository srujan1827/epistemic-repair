"""Provider-neutral bounded LLM policy for ER-2 repair selection."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_repair.er2.llm_prompts import (
    ER2_REPAIR_PROMPT_VERSION,
    ER2LLMPromptView,
    build_repair_selection_prompt,
)
from epistemic_repair.er2.llm_schemas import (
    ER2RepairSelectionDecision,
    ER2StructuredResponseError,
    er2_repair_selection_json_schema,
    parse_er2_repair_selection_response,
)
from epistemic_repair.llm.base import LLMClient, LLMError, LLMRequest, LLMTransientError
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.sanitize import sanitize_text
from epistemic_repair.policies.llm import LLMAttemptRecord, LLMAttemptStatus


@dataclass(frozen=True, slots=True)
class ER2LLMPolicyResult:
    """A repair decision or explicit failure with sanitized attempt history."""

    prompt: str
    prompt_version: str
    decision: ER2RepairSelectionDecision | None
    attempts: tuple[LLMAttemptRecord, ...]

    @property
    def succeeded(self) -> bool:
        return self.decision is not None

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)


class ER2CausalRepairLLMPolicy:
    """Ask an LLM to choose one neutral repair option from a safe prompt view."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config

    def decide(self, view: ER2LLMPromptView) -> ER2LLMPolicyResult:
        if type(view) is not ER2LLMPromptView:
            raise TypeError("ER-2 LLM policy requires ER2LLMPromptView")
        prompt = build_repair_selection_prompt(view)
        request = LLMRequest(
            prompt=prompt,
            response_schema=er2_repair_selection_json_schema(),
        )
        attempts: list[LLMAttemptRecord] = []
        for attempt_number in range(1, self.config.max_retries + 2):
            try:
                response = self._client.generate(request)
                raw_output = sanitize_text(response.text)
                try:
                    decision = parse_er2_repair_selection_response(response.text)
                except ER2StructuredResponseError as error:
                    attempts.append(
                        LLMAttemptRecord(
                            attempt_number=attempt_number,
                            status=LLMAttemptStatus.INVALID_FORMAT,
                            raw_output=raw_output,
                            provider_request_id=response.provider_request_id,
                            error_type=type(error).__name__,
                            error_message=sanitize_text(str(error)),
                        )
                    )
                    if attempt_number <= self.config.max_retries:
                        continue
                    return self._result(prompt, None, attempts)
                attempts.append(
                    LLMAttemptRecord(
                        attempt_number=attempt_number,
                        status=LLMAttemptStatus.VALID,
                        raw_output=raw_output,
                        provider_request_id=response.provider_request_id,
                        error_type=None,
                        error_message=None,
                    )
                )
                return self._result(prompt, decision, attempts)
            except LLMTransientError as error:
                attempts.append(_error_attempt(attempt_number, LLMAttemptStatus.TRANSIENT_ERROR, error))
                if attempt_number <= self.config.max_retries:
                    continue
            except LLMError as error:
                attempts.append(_error_attempt(attempt_number, LLMAttemptStatus.PROVIDER_ERROR, error))
            return self._result(prompt, None, attempts)
        raise AssertionError("bounded retry loop must return")

    @staticmethod
    def _result(
        prompt: str,
        decision: ER2RepairSelectionDecision | None,
        attempts: list[LLMAttemptRecord],
    ) -> ER2LLMPolicyResult:
        return ER2LLMPolicyResult(
            prompt=prompt,
            prompt_version=ER2_REPAIR_PROMPT_VERSION,
            decision=decision,
            attempts=tuple(attempts),
        )


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
