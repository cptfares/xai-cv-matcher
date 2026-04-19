"""
OpenAI GPT client wrapper.

Usage:
    client = LLMClient()                          # reads OPENAI_API_KEY from env
    text   = client.complete(system, user)        # text response
    data   = client.complete_json(system, user)   # parses JSON response
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o"
_MAX_TOKENS = 1024


class LLMClient:
    """
    Thin wrapper around the OpenAI SDK.
    Falls back gracefully if the API key is missing or a call fails.
    """

    def __init__(self, api_key: str | None = None, model: str = _MODEL) -> None:
        self._model = model
        self._client = None
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            logger.warning(
                "OPENAI_API_KEY not set — LLM features will be disabled. "
                "Set the environment variable to enable narrative generation."
            )
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system: str, user: str, max_tokens: int = _MAX_TOKENS) -> str:
        """
        Call the API. Returns the text response, or empty string on failure.
        """
        if not self.available:
            return ""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content.strip()
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
        if not self.available:
            return None
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s", exc)
            return None
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None


# Module-level singleton
_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
