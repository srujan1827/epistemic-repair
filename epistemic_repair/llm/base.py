"""Provider-neutral LLM request, response, client, and error interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A text-only request with a strict provider-independent JSON schema."""

    prompt: str
    response_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider response needed for parsing and reproducibility auditing."""

    text: str
    provider_request_id: str | None = None


class LLMClient(ABC):
    """Minimal provider-neutral synchronous generation interface."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one structured response or raise a typed LLM error."""


class LLMError(RuntimeError):
    """Base class for provider and generation failures."""


class LLMConfigurationError(LLMError):
    """Missing or invalid provider configuration."""


class LLMTransientError(LLMError):
    """A provider failure that may succeed on a bounded retry."""


class LLMTimeoutError(LLMTransientError):
    """The provider request exceeded its configured timeout."""


class LLMRateLimitError(LLMTransientError):
    """The provider rejected a request due to rate limiting."""


class LLMProviderError(LLMError):
    """A non-transient provider error such as an unavailable model ID."""


class LLMFormatError(LLMTransientError):
    """An empty or malformed structured provider response."""

