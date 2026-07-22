from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Message = dict[str, str]


class ChatBackend(Protocol):
    model_name: str

    def generate(self, messages: list[Message]) -> str:
        """Generate one assistant response for a chat message sequence."""


@dataclass
class OpenAICompatibleBackend:
    """Minimal client for OpenAI-compatible chat-completions servers.

    This backend works with local servers such as vLLM/LM Studio as well as
    compatible hosted endpoints. It intentionally uses the standard library so
    API mode does not require an additional Python SDK.
    """

    model_name: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    max_new_tokens: int = 256
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    retries: int = 2
    seed: int | None = 42
    reasoning_effort: str | None = None

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def generate(self, messages: list[Message]) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        if self.seed is not None:
            payload["seed"] = self.seed

        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = Request(self._endpoint(), data=body, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"chat endpoint returned no choices: {data}")
                message = choices[0].get("message", {})
                content = message.get("content")
                if not content or not str(content).strip():
                    reasoning = message.get("reasoning") or message.get("reasoning_content")
                    if reasoning and str(reasoning).strip():
                        raise RuntimeError(
                            f"chat endpoint returned only reasoning, no content "
                            f"(reasoning consumed the token budget): {str(reasoning)[:200]}"
                        )
                    raise RuntimeError(f"chat endpoint returned empty content: {data}")
                return str(content).strip()
            except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"chat request failed after {self.retries + 1} attempts") from last_error


class TransformersBackend:
    """Local Hugging Face causal-LM chat backend loaded once per process."""

    def __init__(
        self,
        *,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 42,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
        max_input_tokens: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise ImportError(
                "transformers mode requires torch, transformers, and accelerate. "
                "Install with: pip install -e '.[generation]'"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        set_seed(seed)

        dtype: Any = torch_dtype
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype, None)
            if dtype is None:
                raise ValueError(f"unsupported torch_dtype: {torch_dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if device_map:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _render(self, messages: list[Message]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for message in messages:
            parts.append(f"{message['role'].upper()}:\n{message['content']}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def generate(self, messages: list[Message]) -> str:
        rendered = self._render(messages)
        tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.max_input_tokens is not None:
            tokenizer_kwargs.update(
                {
                    "truncation": True,
                    "max_length": self.max_input_tokens,
                }
            )
        encoded = self.tokenizer(rendered, **tokenizer_kwargs)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with self._torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        generated = output[0, input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def create_chat_backend(settings: dict[str, Any], *, role: str) -> ChatBackend:
    """Create a generator or judge backend from config.yaml settings."""
    mode = str(settings.get("mode") or "").strip().lower()
    model_name = str(settings.get("name") or "").strip()
    if not mode:
        raise ValueError(f"llm.{role}.mode is required")
    if not model_name:
        raise ValueError(f"llm.{role}.name is required")

    common = {
        "model_name": model_name,
        "max_new_tokens": int(settings.get("max_new_tokens") or 256),
        "temperature": float(settings.get("temperature") or 0.0),
    }
    if mode == "openai_compatible":
        base_url = str(settings.get("base_url") or "").strip()
        if not base_url:
            raise ValueError(f"llm.{role}.base_url is required for openai_compatible mode")
        return OpenAICompatibleBackend(
            **common,
            base_url=base_url,
            api_key_env=str(settings.get("api_key_env") or "OPENAI_API_KEY"),
            timeout_seconds=float(settings.get("timeout_seconds") or 120.0),
            retries=(
                int(settings["retries"])
                if settings.get("retries") is not None
                else 2
            ),
            seed=int(settings["seed"]) if settings.get("seed") is not None else None,
            reasoning_effort=(
                str(settings["reasoning_effort"])
                if settings.get("reasoning_effort") is not None
                else None
            ),
        )
    if mode == "transformers":
        return TransformersBackend(
            **common,
            seed=int(settings.get("seed") or 42),
            device_map=settings.get("device_map", "auto"),
            torch_dtype=str(settings.get("torch_dtype") or "auto"),
            max_input_tokens=(
                int(settings["max_input_tokens"])
                if settings.get("max_input_tokens") is not None
                else None
            ),
            trust_remote_code=bool(settings.get("trust_remote_code", False)),
        )
    raise ValueError(
        f"unsupported llm.{role}.mode={mode!r}; expected 'transformers' or 'openai_compatible'"
    )
