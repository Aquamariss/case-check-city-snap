"""Concrete LLM provider implementations used by the gateway service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import httpx

from .exceptions import LLMProviderError

# ✅ Base OpenAI API endpoint
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
# ✅ Modern chat completion endpoint
_OPENAI_RESPONSES_ENDPOINT = "/chat/completions"
# ✅ Real modern model (replace if you want a different one)
_OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"

logger = logging.getLogger(__name__)


def _normalize_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    if not isinstance(messages, Sequence):  # pragma: no cover
        raise LLMProviderError("`messages` must be a sequence of dict objects")

    normalized: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LLMProviderError(f"Message at index {index} is not a mapping")

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise LLMProviderError(f"Message at index {index} is missing a valid role")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError(f"Message at index {index} is missing text content")

        normalized.append({"role": role, "content": content})
    return normalized


class OpenAILLMProvider:
    """Async client for OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI provider requires a non-empty API key")

        self._api_key = api_key.strip()
        self._base_url = (base_url or _OPENAI_DEFAULT_BASE_URL).rstrip("/")
        self._model = model or _OPENAI_DEFAULT_MODEL
        self._timeout = timeout

    async def generate(self, *, messages: Sequence[Dict[str, str]]) -> str:
        payload = {
            "model": self._model,
            "messages": _normalize_messages(messages),
            # ✅ Force JSON output, so parser won't fail
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                response = await client.post(_OPENAI_RESPONSES_ENDPOINT, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            req = getattr(exc, 'request', None)
            where = f" during {req.method} {req.url}" if req else ""
            raise LLMProviderError(f"OpenAI HTTP error{where}: {exc!r}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError(f"OpenAI returned invalid JSON: {exc}") from exc

        # ✅ Extract output safely
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise LLMProviderError(f"OpenAI returned unexpected payload structure: {data}")
