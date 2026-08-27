"""Provider-neutral LLM interfaces and the single Gemini provider adapter."""

from epistemic_repair.llm.base import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMFormatError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMTransientError,
)
from epistemic_repair.llm.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_ID,
    LLMConfig,
    load_dotenv_if_present,
    require_gemini_api_key,
)
from epistemic_repair.llm.gemini import GeminiLLMClient
from epistemic_repair.llm.mock import DeterministicMockLLMClient
from epistemic_repair.llm.schemas import (
    DecisionType,
    ER1FullAutonomousDecision,
    ER1LLMDecision,
    ER1PlannerOnlyDecision,
    FullAutonomousDecision,
    LLMCondition,
    LLMDecision,
    PlannerOnlyDecision,
    StructuredResponseError,
    er1_full_autonomous_json_schema,
    er1_planner_only_json_schema,
    full_autonomous_json_schema,
    parse_full_autonomous_response,
    parse_er1_full_autonomous_response,
    parse_er1_planner_only_response,
    parse_planner_only_response,
    planner_only_json_schema,
)


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create the configured named provider without fallback or auto-routing."""
    if config.provider.lower() == "gemini":
        return GeminiLLMClient(config)
    raise LLMConfigurationError(
        f"Unsupported provider {config.provider!r}; only 'gemini' is implemented"
    )


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL_ID",
    "DecisionType",
    "DeterministicMockLLMClient",
    "ER1FullAutonomousDecision",
    "ER1LLMDecision",
    "ER1PlannerOnlyDecision",
    "FullAutonomousDecision",
    "GeminiLLMClient",
    "LLMClient",
    "LLMCondition",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMDecision",
    "LLMError",
    "LLMFormatError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMTransientError",
    "PlannerOnlyDecision",
    "StructuredResponseError",
    "create_llm_client",
    "er1_full_autonomous_json_schema",
    "er1_planner_only_json_schema",
    "full_autonomous_json_schema",
    "load_dotenv_if_present",
    "parse_full_autonomous_response",
    "parse_er1_full_autonomous_response",
    "parse_er1_planner_only_response",
    "parse_planner_only_response",
    "planner_only_json_schema",
    "require_gemini_api_key",
]
