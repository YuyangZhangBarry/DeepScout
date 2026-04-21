import json
import re
from typing import Any

import httpx

from backend.app.config import Settings, get_settings


def completion_message_text(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat completion response."""
    try:
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content")
        return (content or "").strip()
    except (TypeError, AttributeError):
        return ""


def parse_json_object_from_content(raw: str) -> dict[str, Any]:
    """Parse JSON object from model output; strips optional ```json fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    return json.loads(text)


class DeepseekClient:
    """OpenAI-compatible chat client for DeepSeek."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        base = self._settings.deepseek_base_url.rstrip("/")
        self._url = f"{base}/v1/chat/completions"

    @property
    def model(self) -> str:
        return self._settings.deepseek_model

    def _headers(self) -> dict[str, str]:
        key = self._settings.deepseek_api_key
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
