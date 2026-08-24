"""Validated provider-neutral model configuration and local environment loading."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import MutableMapping

from epistemic_repair.llm.base import LLMConfigurationError


DEFAULT_MODEL_ID = "gemini-3.7-flash"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Reproducibility and call-control configuration for one model family."""

    provider: str = "gemini"
    model_id: str = DEFAULT_MODEL_ID
    thinking_level: str = "medium"
    max_output_tokens: int = 512
    request_timeout_seconds: float | None = 60.0
    max_retries: int = 1
    max_decision_calls: int = 4

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.thinking_level.strip():
            raise ValueError("thinking_level must be non-empty")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.request_timeout_seconds is not None and (
            not isinstance(self.request_timeout_seconds, (int, float))
            or isinstance(self.request_timeout_seconds, bool)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive or None")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if type(self.max_decision_calls) is not int or self.max_decision_calls <= 0:
            raise ValueError("max_decision_calls must be a positive integer")


def load_dotenv_if_present(
    path: str | Path = ".env",
    *,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Load simple KEY=VALUE entries without overriding existing environment values."""
    target = Path(path)
    if not target.is_file():
        return False
    destination = os.environ if environ is None else environ
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            destination.setdefault(key, value)
    return True


def require_gemini_api_key(
    *,
    dotenv_path: str | Path = ".env",
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Load optional local configuration and return the required secret."""
    destination = os.environ if environ is None else environ
    load_dotenv_if_present(dotenv_path, environ=destination)
    api_key = destination.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is required for the Gemini provider. "
            "Set it in the shell or a gitignored local .env file."
        )
    return api_key

