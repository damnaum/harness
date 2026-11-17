"""Provider abstraction.  Every provider implements ``complete`` and returns a
uniform ``ModelResponse``.  Real providers use stdlib urllib (no SDK deps);
the mock provider is deterministic and powers offline mode + tests.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..errors import ModelError, ModelUnavailableError, RateLimitError
from ..types import Message, ModelInfo, Role, ToolCall


@dataclass
class ModelResponse:
    content: str
    model_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic (good enough for budgeting/costing)."""
    return max(1, len(text) // 4)


def messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(m.content) + 4 for m in messages)


class BaseProvider:
    """Interface every provider must implement."""

    provider_name = "base"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def complete(
        self,
        model: ModelInfo,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    def health(self) -> bool:
        return True

    # -- shared HTTP helper ------------------------------------------------
    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429:
                raise RateLimitError(f"rate limited by {self.provider_name}: {body}") from exc
            if exc.code in (401, 403):
                raise ModelUnavailableError(
                    f"auth failed for {self.provider_name}: {body}", status=exc.code
                ) from exc
            raise ModelError(f"{self.provider_name} HTTP {exc.code}: {body}", status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise ModelUnavailableError(f"{self.provider_name} unreachable: {exc.reason}") from exc

    def _post_json_temperature_safe(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """POST, retrying once without ``temperature`` if the model rejects it.

        Reasoning-tuned models differ on whether they accept a temperature and
        the set changes per release. Rather than hard-coding which models do,
        we send the intent and fall back only when the provider says no — so a
        supported temperature is never silently dropped.
        """
        try:
            return self._post_json(url, payload, headers)
        except ModelError as exc:
            message = str(exc).lower()
            unsupported = "temperature" in message and (
                "unsupported" in message or "not supported" in message or "invalid" in message
            )
            if not unsupported:
                raise
            retry: dict[str, Any] = {}
            for key, value in payload.items():
                if key == "temperature":
                    continue
                if key == "generationConfig" and isinstance(value, dict):
                    value = {k: v for k, v in value.items() if k != "temperature"}
                retry[key] = value
            return self._post_json(url, retry, headers)

    def _price(self, model: ModelInfo, tokens_in: int, tokens_out: int) -> float:
        return model.cost_estimate(tokens_in, tokens_out)


class OpenAICompatibleProvider(BaseProvider):
    """Works for OpenAI, xAI, DeepSeek, Qwen (DashScope), Moonshot, Zhipu,
    Mistral, Meta — anything exposing the OpenAI chat-completions shape."""

    provider_name = "openai-compatible"

    DEFAULT_BASES = {
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "moonshot": "https://api.moonshot.ai/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "mistral": "https://api.mistral.ai/v1",
        "meta": "https://api.llama.com/v1",
    }

    def __init__(self, provider_key: str = "openai", **kwargs):
        super().__init__(**kwargs)
        self.provider_key = provider_key
        self.provider_name = provider_key
        if not self.base_url:
            self.base_url = self.DEFAULT_BASES.get(provider_key, "https://api.openai.com/v1")

    @staticmethod
    def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Serialise to the chat-completions shape.

        Tool use requires three message kinds providers will reject if we get
        them wrong: assistant turns carrying ``tool_calls``, tool results keyed
        by ``tool_call_id``, and plain turns. Getting this wrong is the single
        most common cause of 400s in a ReAct loop against a real API.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.TOOL:
                out.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content,
                })
            elif m.role == Role.ASSISTANT and m.metadata.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for i, tc in enumerate(m.metadata["tool_calls"])
                    ],
                })
            else:
                out.append({"role": m.role.value, "content": m.content})
        return out

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        start = time.time()
        payload: dict[str, Any] = {
            "model": model.id.split("/", 1)[-1],
            "messages": self._wire_messages(messages),
        }
        payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {"type": "object", "properties": {}})}}
                for t in tools
            ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            raw = self._post_json_temperature_safe(
                f"{self.base_url.rstrip('/')}/chat/completions", payload, headers
            )
        except ModelError as exc:
            # Reasoning models (o-series, some gpt-5 variants) reject ``max_tokens``
            # and require ``max_completion_tokens``. Only retry when the provider
            # actually says so; otherwise the error must surface.
            message = str(exc).lower()
            if "max_tokens" not in message or "max_completion_tokens" not in message:
                raise
            retry_payload = {
                ("max_completion_tokens" if k == "max_tokens" else k): v
                for k, v in payload.items()
            }
            raw = self._post_json_temperature_safe(
                f"{self.base_url.rstrip('/')}/chat/completions", retry_payload, headers
            )
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
        usage = raw.get("usage", {})
        tokens_in = int(usage.get("prompt_tokens", messages_tokens(messages)))
        tokens_out = int(usage.get("completion_tokens", estimate_tokens(msg.get("content") or "")))
        return ModelResponse(
            content=msg.get("content") or "",
            model_id=model.id,
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._price(model, tokens_in, tokens_out),
            latency_ms=(time.time() - start) * 1000,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=raw,
        )


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or "https://api.anthropic.com/v1"

    @staticmethod
    def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Anthropic requires alternating user/assistant turns with content
        blocks; tool results are ``tool_result`` blocks inside a user turn."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                continue
            if m.role == Role.TOOL:
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
                # Merge consecutive tool results into a single user turn.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == Role.ASSISTANT and m.metadata.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for i, tc in enumerate(m.metadata["tool_calls"]):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or f"toolu_{i}",
                        "name": tc.get("name", ""),
                        "input": tc.get("arguments", {}),
                    })
                out.append({"role": "assistant", "content": blocks})
            else:
                role = "user" if m.role != Role.ASSISTANT else "assistant"
                # Merge consecutive plain turns: some gateways reject a
                # non-alternating sequence even though the API itself tolerates it.
                if out and out[-1]["role"] == role and isinstance(out[-1]["content"], str):
                    out[-1]["content"] = f"{out[-1]['content']}\n\n{m.content}".strip()
                else:
                    out.append({"role": role, "content": m.content})
        return out or [{"role": "user", "content": "Hello"}]

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        start = time.time()
        system = "\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        user_msgs = self._wire_messages(messages)
        payload: dict[str, Any] = {
            "model": model.id.split("/", 1)[-1],
            "messages": user_msgs,
            # Anthropic requires max_tokens. A hardcoded 4096 truncated long code
            # answers for models that allow far more; use the model's own limit.
            "max_tokens": max_tokens or min(getattr(model, "max_output", 8192), 8192),
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("parameters", {"type": "object", "properties": {}})}
                for t in tools
            ]
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        raw = self._post_json_temperature_safe(f"{self.base_url.rstrip('/')}/messages", payload, headers)
        content_parts = raw.get("content", [])
        text = "\n".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
        tool_calls = [
            ToolCall(id=p.get("id", ""), name=p.get("name", ""), arguments=p.get("input", {}))
            for p in content_parts if p.get("type") == "tool_use"
        ]
        usage = raw.get("usage", {})
        tokens_in = int(usage.get("input_tokens", messages_tokens(messages)))
        tokens_out = int(usage.get("output_tokens", estimate_tokens(text)))
        return ModelResponse(
            content=text, model_id=model.id, tool_calls=tool_calls,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=self._price(model, tokens_in, tokens_out),
            latency_ms=(time.time() - start) * 1000,
            raw=raw,
        )


class GoogleProvider(BaseProvider):
    provider_name = "google"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta"

    @staticmethod
    def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Gemini wants role ``user``/``model`` turns with typed parts.

        Tool calls become ``functionCall`` parts, results become
        ``functionResponse`` parts, and the system prompt goes in
        ``systemInstruction`` (not as a turn, where it would be ignored).
        """
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                continue
            if m.role == Role.TOOL:
                part = {
                    "functionResponse": {
                        "name": m.name or "tool",
                        "response": {"content": m.content},
                    }
                }
                # Gemini expects every functionResponse for one model turn in a
                # single user turn; one turn per result is rejected or misread.
                if contents and contents[-1]["role"] == "user" and any(
                    "functionResponse" in p for p in contents[-1]["parts"]
                ):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            elif m.role == Role.ASSISTANT and m.metadata.get("tool_calls"):
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.metadata["tool_calls"]:
                    parts.append({"functionCall": {
                        "name": tc.get("name", ""),
                        "args": tc.get("arguments", {}),
                    }})
                contents.append({"role": "model", "parts": parts})
            else:
                role = "model" if m.role == Role.ASSISTANT else "user"
                if contents and contents[-1]["role"] == role and all(
                    "text" in p for p in contents[-1]["parts"]
                ):
                    # Plain consecutive turns of the same role are merged too.
                    contents[-1]["parts"].append({"text": m.content})
                else:
                    contents.append({"role": role, "parts": [{"text": m.content}]})
        return contents or [{"role": "user", "parts": [{"text": "Hello"}]}]

    @staticmethod
    def _wire_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations = []
        for t in tools:
            schema = t.get("parameters") or {"type": "object", "properties": {}}
            # Gemini rejects a few JSON-schema keywords; keep the safe subset.
            declarations.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {k: v for k, v in schema.items() if k in ("type", "properties", "required")},
            })
        return [{"functionDeclarations": declarations}]

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        start = time.time()
        model_name = model.id.split("/", 1)[-1]
        system_text = "\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        payload: dict[str, Any] = {
            "contents": self._wire_messages(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens or min(getattr(model, "max_output", 8192), 16_384),
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        if tools:
            payload["tools"] = self._wire_tools(tools)

        url = f"{self.base_url.rstrip('/')}/models/{model_name}:generateContent"
        if self.api_key:
            url += f"?key={self.api_key}"
        raw = self._post_json_temperature_safe(url, payload, {"Content-Type": "application/json"})

        candidates = raw.get("candidates") or [{}]
        parts = (candidates[0].get("content") or {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if "text" in p)
        tool_calls = [
            ToolCall(
                id=p.get("functionCall", {}).get("id") or f"fc_{i}",
                name=p["functionCall"].get("name", ""),
                arguments=p["functionCall"].get("args", {}) or {},
            )
            for i, p in enumerate(parts) if "functionCall" in p
        ]
        usage = raw.get("usageMetadata", {})
        tokens_in = int(usage.get("promptTokenCount", messages_tokens(messages)))
        tokens_out = int(usage.get("candidatesTokenCount", estimate_tokens(text)))
        return ModelResponse(
            content=text, model_id=model.id, tool_calls=tool_calls,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=self._price(model, tokens_in, tokens_out),
            latency_ms=(time.time() - start) * 1000,
            finish_reason=(candidates[0].get("finishReason") or "stop").lower(),
            raw=raw,
        )


class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or "http://localhost:11434"

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        start = time.time()
        payload = {
            "model": model.id.split("/", 1)[-1],
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        raw = self._post_json(f"{self.base_url.rstrip('/')}/api/chat", payload, {"Content-Type": "application/json"})
        text = (raw.get("message") or {}).get("content", "")
        tokens_in = int(raw.get("prompt_eval_count", messages_tokens(messages)))
        tokens_out = int(raw.get("eval_count", estimate_tokens(text)))
        return ModelResponse(
            content=text, model_id=model.id, tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=0.0, latency_ms=(time.time() - start) * 1000, raw=raw,
        )


def build_provider(model: ModelInfo, api_keys: dict[str, str] | None = None, timeout: float = 120.0) -> BaseProvider:
    """Factory: model.provider -> provider instance with the right key."""
    from .providers.mock import MockProvider

    keys = api_keys or {}
    provider = model.provider
    if provider == "mock":
        return MockProvider()
    if provider == "anthropic":
        return AnthropicProvider(api_key=keys.get("anthropic"), timeout=timeout)
    if provider == "google":
        return GoogleProvider(api_key=keys.get("google"), timeout=timeout)
    if provider == "ollama":
        return OllamaProvider(timeout=timeout)
    return OpenAICompatibleProvider(provider_key=provider, api_key=keys.get(provider), timeout=timeout)
