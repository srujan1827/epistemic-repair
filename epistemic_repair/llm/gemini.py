"""Gemini adapter isolated behind the generic LLM client interface."""

from typing import Any

from epistemic_repair.llm.base import (
    LLMClient,
    LLMConfigurationError,
    LLMFormatError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMTransientError,
)
from epistemic_repair.llm.config import LLMConfig, require_gemini_api_key


class GeminiLLMClient(LLMClient):
    """Official Google Gen AI SDK adapter with no tools or provider fallback."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_key: str | None = None,
        dotenv_path: str = ".env",
        sdk_client: Any | None = None,
    ) -> None:
        if config.provider.lower() != "gemini":
            raise LLMConfigurationError("GeminiLLMClient requires provider='gemini'")
        self.config = config
        self._api_key = api_key or require_gemini_api_key(dotenv_path=dotenv_path)
        if sdk_client is not None:
            self._client = sdk_client
            return

        try:
            from google import genai
        except ImportError as error:
            raise LLMConfigurationError(
                "Gemini support requires the optional dependency. "
                "Install with: python -m pip install -e '.[gemini]'"
            ) from error

        http_options: dict[str, Any] = {
            # Keep SDK retries disabled; benchmark retries are bounded and traced.
            "retry_options": {"attempts": 1},
        }
        if config.request_timeout_seconds is not None:
            http_options["timeout"] = int(config.request_timeout_seconds * 1000)
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=http_options,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate strict JSON without tools, routing, or sampling parameters."""
        generation_config = {
            "response_mime_type": "application/json",
            "response_json_schema": dict(request.response_schema),
            "max_output_tokens": self.config.max_output_tokens,
            "thinking_config": {
                "thinking_level": self.config.thinking_level,
                "include_thoughts": False,
            },
        }
        try:
            response = self._client.models.generate_content(
                model=self.config.model_id,
                contents=request.prompt,
                config=generation_config,
            )
        except Exception as error:  # Provider classes are lazy/optional.
            raise self._classify_provider_error(error) from error

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMFormatError("Gemini returned an empty structured response")
        request_id = getattr(response, "response_id", None)
        if request_id is not None and not isinstance(request_id, str):
            request_id = str(request_id)
        return LLMResponse(text=text, provider_request_id=request_id)

    @staticmethod
    def _classify_provider_error(error: Exception) -> Exception:
        """Map SDK errors into stable provider-neutral failure categories."""
        code = getattr(error, "code", None)
        name = type(error).__name__.lower()
        message = str(error) or type(error).__name__
        if code == 429:
            return LLMRateLimitError(f"Gemini rate limit: {message}")
        if "timeout" in name or isinstance(error, TimeoutError):
            return LLMTimeoutError(f"Gemini request timed out: {message}")
        if code in {408, 500, 502, 503, 504} or any(
            marker in name for marker in ("connecterror", "servererror")
        ):
            return LLMTransientError(f"Transient Gemini API failure: {message}")
        if code in {400, 401, 403, 404}:
            return LLMProviderError(
                "Gemini request was rejected. Check GEMINI_API_KEY, model_id, "
                f"and provider availability: {message}"
            )
        return LLMProviderError(f"Gemini API failure: {message}")

