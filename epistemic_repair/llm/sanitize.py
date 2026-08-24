"""Redact configured API secrets from auditable text fields."""

import os
from typing import Mapping


def sanitize_text(text: str, *, environ: Mapping[str, str] | None = None) -> str:
    """Replace known non-empty API key values without exposing them."""
    source = os.environ if environ is None else environ
    sanitized = text
    for name, value in source.items():
        if name.endswith("_API_KEY") and value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    return sanitized

