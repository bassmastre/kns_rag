from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class LLMAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float | None = 0.0
    max_tokens: int | None = 512
    timeout_seconds: float = 180.0
    seed: int | None = None
    max_retries: int = 4


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


def chat_config_from_mapping(
    mapping: dict[str, Any],
    *,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
    seed: int | None = None,
    max_retries: int | None = None,
) -> ChatConfig:
    resolved_provider = str(provider or mapping.get("provider") or mapping.get("mode") or "").strip()
    resolved_base_url = str(base_url or mapping.get("base_url") or "").strip()
    resolved_model = str(model or mapping.get("name") or mapping.get("model") or "").strip()
    resolved_key_env = api_key_env if api_key_env is not None else mapping.get("api_key_env")

    if not resolved_provider:
        raise ValueError("LLM provider is missing; set llm.<role>.provider or pass --provider")
    if not resolved_base_url:
        raise ValueError("LLM base_url is missing; set llm.<role>.base_url or pass --base-url")
    if not resolved_model:
        raise ValueError("LLM model is missing; set llm.<role>.name or pass --model")

    return ChatConfig(
        provider=resolved_provider,
        base_url=resolved_base_url,
        model=resolved_model,
        api_key_env=str(resolved_key_env).strip() if resolved_key_env else None,
        temperature=temperature if temperature is not None else mapping.get("temperature", 0.0),
        max_tokens=max_tokens if max_tokens is not None else mapping.get("max_tokens", 512),
        timeout_seconds=float(
            timeout_seconds if timeout_seconds is not None else mapping.get("timeout_seconds", 180)
        ),
        seed=seed if seed is not None else mapping.get("seed"),
        max_retries=int(max_retries if max_retries is not None else mapping.get("max_retries", 4)),
    )


def _endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith(suffix) else base + suffix


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _api_key(config: ChatConfig) -> str:
    if config.api_key_env:
        value = os.environ.get(config.api_key_env)
        if value:
            return value
    if config.provider == "openai_compatible" and _is_local_url(config.base_url):
        return "ollama"
    env_hint = config.api_key_env or "the configured API key environment variable"
    raise ValueError(f"missing API key in {env_hint}")


def _retry_delay(exc: Exception, attempt: int) -> float:
    if isinstance(exc, HTTPError):
        value = exc.headers.get("Retry-After") if exc.headers else None
        if value:
            try:
                return min(float(value), 60.0)
            except ValueError:
                pass
    return min(2.0**attempt, 30.0)


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}

    for attempt in range(max_retries + 1):
        request = Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise LLMAPIError(f"unexpected non-object response from {url}")
            return data
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in RETRYABLE_STATUS_CODES
            if attempt >= max_retries or not retryable:
                raise LLMAPIError(f"HTTP {exc.code} from {url}: {error_body[:1000]}") from exc
            time.sleep(_retry_delay(exc, attempt))
        except (URLError, socket.timeout, TimeoutError) as exc:
            if attempt >= max_retries:
                raise LLMAPIError(f"request failed for {url}: {exc}") from exc
            time.sleep(_retry_delay(exc, attempt))
        except json.JSONDecodeError as exc:
            raise LLMAPIError(f"invalid JSON response from {url}: {exc}") from exc

    raise AssertionError("unreachable")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return ""


class ChatClient:
    def __init__(self, config: ChatConfig):
        self.config = config

    def complete(self, prompt: str) -> ChatResponse:
        provider = self.config.provider
        if provider == "openai_compatible":
            return self._complete_openai_compatible(prompt)
        if provider == "anthropic":
            return self._complete_anthropic(prompt)
        raise ValueError(f"unsupported LLM provider: {provider}")

    def _complete_openai_compatible(self, prompt: str) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        if self.config.seed is not None:
            payload["seed"] = self.config.seed

        data = _post_json(
            url=_endpoint(self.config.base_url, "/chat/completions"),
            payload=payload,
            headers={"Authorization": f"Bearer {_api_key(self.config)}"},
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        try:
            text = _content_text(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAPIError(f"missing chat completion content: {data}") from exc
        if not text:
            raise LLMAPIError("empty chat completion content")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatResponse(text=text, usage=usage, raw=data)

    def _complete_anthropic(self, prompt: str) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens or 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        data = _post_json(
            url=_endpoint(self.config.base_url, "/messages"),
            payload=payload,
            headers={
                "x-api-key": _api_key(self.config),
                "anthropic-version": "2023-06-01",
            },
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        text = _content_text(data.get("content"))
        if not text:
            raise LLMAPIError(f"missing Anthropic message content: {data}")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatResponse(text=text, usage=usage, raw=data)
