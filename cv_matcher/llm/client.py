"""
Anthropic Claude client with prompt caching.

Usage:
    client = LLMClient()                          # reads ANTHROPIC_API_KEY from env
    text   = client.complete(system, user)        # cached system prompt
    data   = client.complete_json(system, user)   # parses JSON response
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Model to use — Sonnet is the best balance of quality and cost for this use case
_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1024


class LLMClient:
    """
    Thin wrapper around the Anthropic SDK.
    - System prompt is sent with cache_control so it is cached across calls.
    - Falls back gracefully if the API key is missing or a call fails.
    """

    def __init__(self, api_key: str | None = None, model: str = _MODEL) -> None:
        self._model = model
        self._client = None
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — LLM features will be disabled. "
                "Set the environment variable to enable narrative generation."
            )
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        except ImportError:
            logger.warning("anthropic package not installed. Run: pip install anthropic")

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system: str, user: str, max_tokens: int = _MAX_TOKENS) -> str:
        """
        Call the API with prompt caching on the system prompt.
        Returns the text response, or an empty string on failure.
        """
        if not self.available:
            return ""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""

    def complete_json(
        self, system: str, user: str, max_tokens: int = _MAX_TOKENS
    ) -> dict[str, Any] | list[Any] | None:
        """
        Like complete() but parses and returns the JSON payload.
        Returns None on parse failure.
        """
        raw = self.complete(system, user, max_tokens=max_tokens)
        if not raw:
            return None
        # Strip accidental markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed (%s). Raw response:\n%s", exc, raw[:500])
            return None


# Module-level singleton — created lazily so import doesn't fail without a key
_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
