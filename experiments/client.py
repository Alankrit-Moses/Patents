from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import EndpointConfig


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any]


class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible chat/completions APIs."""

    def __init__(self, config: EndpointConfig):
        self.config = config

    @property
    def url(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_output_tokens if max_tokens is None else max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.resolved_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                text = raw["choices"][0]["message"]["content"]
                return LLMResponse(text=text, raw=raw, usage=raw.get("usage", {}))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM API HTTP {error.code}: {detail}")
                if error.code not in {429, 500, 502, 503, 504} or attempt == 1:
                    raise last_error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt == 1:
                    raise
            time.sleep(1.0)
        assert last_error is not None
        raise last_error
