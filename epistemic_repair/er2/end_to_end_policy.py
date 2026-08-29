"""Provider-neutral repair policy for the end-to-end ER-1 V2 -> ER-2 study."""

from __future__ import annotations

from epistemic_repair.er2.end_to_end_prompts import (
    END_TO_END_REPAIR_PROMPT_VERSION,
    EndToEndRepairPromptView,
    build_end_to_end_repair_prompt,
)
from epistemic_repair.er2.llm_policy import ER2LLMPolicyResult
from epistemic_repair.er2.llm_schemas import (
    ER2StructuredResponseError,
    er2_repair_selection_json_schema,
    parse_er2_repair_selection_response,
)
from epistemic_repair.llm.base import LLMClient, LLMError, LLMRequest, LLMTransientError
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.sanitize import sanitize_text
from epistemic_repair.policies.llm import LLMAttemptRecord, LLMAttemptStatus


class EndToEndRepairLLMPolicy:
    """Choose a neutral repair from ER-1 evidence without a supplied diagnosis."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config

    def decide(self, view: EndToEndRepairPromptView) -> ER2LLMPolicyResult:
        if type(view) is not EndToEndRepairPromptView:
            raise TypeError("policy requires exactly EndToEndRepairPromptView")
        prompt = build_end_to_end_repair_prompt(view)
        request = LLMRequest(prompt, er2_repair_selection_json_schema())
        attempts: list[LLMAttemptRecord] = []
        for attempt_number in range(1, self.config.max_retries + 2):
            try:
                response = self._client.generate(request)
                raw_output = sanitize_text(response.text)
                try:
                    decision = parse_er2_repair_selection_response(response.text)
                except ER2StructuredResponseError as error:
                    attempts.append(_attempt(
                        attempt_number,
                        LLMAttemptStatus.INVALID_FORMAT,
                        raw_output=raw_output,
                        request_id=response.provider_request_id,
                        error=error,
                    ))
                    if attempt_number <= self.config.max_retries:
                        continue
                    return _result(prompt, None, attempts)
                attempts.append(_attempt(
                    attempt_number,
                    LLMAttemptStatus.VALID,
                    raw_output=raw_output,
                    request_id=response.provider_request_id,
                ))
                return _result(prompt, decision, attempts)
            except LLMTransientError as error:
                attempts.append(_attempt(
                    attempt_number, LLMAttemptStatus.TRANSIENT_ERROR, error=error
                ))
                if attempt_number <= self.config.max_retries:
                    continue
            except LLMError as error:
                attempts.append(_attempt(
                    attempt_number, LLMAttemptStatus.PROVIDER_ERROR, error=error
                ))
            return _result(prompt, None, attempts)
        raise AssertionError("bounded retry loop must return")


def _result(prompt, decision, attempts) -> ER2LLMPolicyResult:
    return ER2LLMPolicyResult(
        prompt=prompt,
        prompt_version=END_TO_END_REPAIR_PROMPT_VERSION,
        decision=decision,
        attempts=tuple(attempts),
    )


def _attempt(
    number: int,
    status: LLMAttemptStatus,
    *,
    raw_output: str | None = None,
    request_id: str | None = None,
    error: Exception | None = None,
) -> LLMAttemptRecord:
    return LLMAttemptRecord(
        attempt_number=number,
        status=status,
        raw_output=raw_output,
        provider_request_id=request_id,
        error_type=type(error).__name__ if error else None,
        error_message=sanitize_text(str(error)) if error else None,
    )
