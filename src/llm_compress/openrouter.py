from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any


class LLMError(RuntimeError):
    """Raised when the OpenRouter request cannot produce a usable response."""


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None
    model: str = "openrouter/auto"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 120
    temperature: float = 0.1
    app_title: str = "llm-compress"
    site_url: str = "https://github.com/"
    retries: int = 2

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
    ) -> "OpenRouterConfig":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            model=model or os.environ.get("OPENROUTER_MODEL", "openrouter/auto"),
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
            ),
            timeout=timeout or int(os.environ.get("OPENROUTER_TIMEOUT", "120")),
            temperature=(
                temperature
                if temperature is not None
                else float(os.environ.get("OPENROUTER_TEMPERATURE", "0.1"))
            ),
            app_title=os.environ.get("OPENROUTER_APP_TITLE", "llm-compress"),
            site_url=os.environ.get("OPENROUTER_SITE_URL", "https://github.com/"),
            retries=int(os.environ.get("OPENROUTER_RETRIES", "2")),
        )

    def with_overrides(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> "OpenRouterConfig":
        return replace(
            self,
            model=model or self.model,
            temperature=self.temperature if temperature is None else temperature,
        )


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    @property
    def model(self) -> str:
        return self.config.model

    def with_overrides(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> "OpenRouterClient":
        return OpenRouterClient(
            self.config.with_overrides(model=model, temperature=temperature)
        )

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_completion_tokens: int | None = None,
    ) -> str:
        if not self.config.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
        }
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.app_title,
        }

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            request = urllib.request.Request(
                self.config.base_url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                return self._extract_content(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = LLMError(f"OpenRouter HTTP {exc.code}: {detail[:1000]}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, LLMError) as exc:
                last_error = exc

            if attempt < self.config.retries:
                time.sleep(min(2**attempt, 8))

        raise LLMError(str(last_error) if last_error else "OpenRouter request failed")

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"OpenRouter response missing message content: {data}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("OpenRouter response content was empty")
        return content
