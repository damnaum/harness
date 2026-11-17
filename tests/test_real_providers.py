"""Provider contract tests — no network.

Real providers are the highest-risk code in a harness: a wrong request shape or
a mis-parsed response is invisible offline and expensive in production. These
tests stub the HTTP layer and assert both directions of the contract:

* the payload we send (tool calls echoed, system prompt preserved, tools declared)
* the ModelResponse we build (text, tool calls, usage, cost)

Error mapping (429 -> RateLimitError, 401 -> ModelUnavailableError) is covered
too, because callers branch on those types.
"""

from __future__ import annotations

import json
import unittest

from apex.errors import ModelError, ModelUnavailableError, RateLimitError
from apex.models.base import (
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    build_provider,
)
from apex.models.registry import get_model
from apex.types import Message, Role, ToolCall


class StubTransportMixin:
    """Captures the payload and returns a canned response."""

    def craft(self, provider, response):
        captured = {}

        def fake_post(url, payload, headers):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            if isinstance(response, Exception):
                raise response
            return response

        provider._post_json = fake_post  # type: ignore[method-assign]
        return captured


def tool_history() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are APEX."),
        Message(role=Role.USER, content="read notes.txt"),
        Message(
            role=Role.ASSISTANT,
            content="Looking.",
            metadata={"tool_calls": [{"id": "call_1", "name": "read_file", "arguments": {"path": "notes.txt"}}]},
        ),
        Message(role=Role.TOOL, content="hello", name="read_file", tool_call_id="call_1"),
    ]


TOOL_SPEC = [{
    "name": "read_file",
    "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}]


class TestOpenAIProvider(StubTransportMixin, unittest.TestCase):
    def setUp(self):
        self.provider = OpenAICompatibleProvider(provider_key="openai", api_key="sk-test")
        self.model = get_model("openai/gpt-5.1")

    def test_request_shape(self):
        response = {
            "choices": [{"message": {"content": "hi", "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
            ]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
        captured = self.craft(self.provider, response)
        result = self.provider.complete(self.model, tool_history(), tools=TOOL_SPEC)

        payload = captured["payload"]
        self.assertEqual(payload["model"], "gpt-5.1")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")
        self.assertIn("temperature", payload)

        roles = [m["role"] for m in payload["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["function"]["arguments"], '{"path": "notes.txt"}')
        self.assertEqual(payload["messages"][3]["tool_call_id"], "call_1")
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")

        self.assertEqual(result.content, "hi")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.txt"})
        self.assertEqual(result.tokens_in, 11)
        self.assertEqual(result.tokens_out, 7)
        self.assertGreater(result.cost_usd, 0)

    def test_malformed_tool_arguments_degrade_to_empty(self):
        response = {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": "not json"}}
        ]}}]}
        self.craft(self.provider, response)
        result = self.provider.complete(self.model, tool_history())
        self.assertEqual(result.tool_calls[0].arguments, {})

    def test_temperature_retried_without_when_rejected(self):
        # o3-class models answer 400 "unsupported parameter: temperature".
        calls = []

        def fake_post(url, payload, headers):
            calls.append(dict(payload))
            if "temperature" in payload:
                raise ModelError("openai HTTP 400: Unsupported parameter: 'temperature'")
            return {"choices": [{"message": {"content": "ok"}}]}

        self.provider._post_json = fake_post
        result = self.provider.complete(get_model("openai/o3"), tool_history())
        self.assertEqual(result.content, "ok")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("temperature", calls[1])

    def test_unrelated_400_is_not_retried(self):
        calls = []

        def fake_post(url, payload, headers):
            calls.append(payload)
            raise ModelError("openai HTTP 400: bad request body")

        self.provider._post_json = fake_post
        with self.assertRaises(ModelError):
            self.provider.complete(self.model, tool_history())
        self.assertEqual(len(calls), 1)

    def test_missing_usage_falls_back_to_estimates(self):
        self.craft(self.provider, {"choices": [{"message": {"content": "hello there"}}]})
        result = self.provider.complete(self.model, [Message(role=Role.USER, content="hi")])
        self.assertGreater(result.tokens_in, 0)
        self.assertGreater(result.tokens_out, 0)

    def test_alternate_bases(self):
        for key, host in (("deepseek", "api.deepseek.com"), ("qwen", "dashscope.aliyuncs.com"),
                          ("xai", "api.x.ai"), ("mistral", "api.mistral.ai")):
            provider = OpenAICompatibleProvider(provider_key=key, api_key="k")
            self.assertIn(host, provider.base_url)


class TestAnthropicProvider(StubTransportMixin, unittest.TestCase):
    def setUp(self):
        self.provider = AnthropicProvider(api_key="ak-test")
        self.model = get_model("anthropic/claude-sonnet-4.5")

    def test_request_shape_and_system_prompt(self):
        response = {
            "content": [
                {"type": "text", "text": "Here you go."},
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {"path": "a.txt"}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }
        captured = self.craft(self.provider, response)
        result = self.provider.complete(self.model, tool_history(), tools=TOOL_SPEC)

        payload = captured["payload"]
        self.assertEqual(payload["model"], "claude-sonnet-4.5")
        self.assertEqual(payload["system"], "You are APEX.")
        # Derived from the model's own output limit, not a hardcoded constant.
        self.assertGreater(payload["max_tokens"], 4096)
        self.assertLessEqual(payload["max_tokens"], 8192)
        self.assertEqual(captured["headers"]["x-api-key"], "ak-test")
        self.assertEqual(captured["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["tools"][0]["input_schema"]["required"], ["path"])

        roles = [m["role"] for m in payload["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "tool_use")
        self.assertEqual(payload["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(payload["messages"][2]["content"][0]["tool_use_id"], "call_1")

        self.assertEqual(result.content, "Here you go.")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.txt"})

    def test_consecutive_tool_results_merge_into_one_turn(self):
        messages = tool_history() + [Message(role=Role.TOOL, content="second", name="x", tool_call_id="call_2")]
        captured = self.craft(self.provider, {"content": [{"type": "text", "text": "ok"}]})
        self.provider.complete(self.model, messages)
        wire = captured["payload"]["messages"]
        last = wire[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(len(last["content"]), 2, "both tool results must share one user turn")
        self.assertEqual({b["tool_use_id"] for b in last["content"]}, {"call_1", "call_2"})

    def test_empty_history_gets_placeholder(self):
        self.craft(self.provider, {"content": [{"type": "text", "text": "ok"}]})
        result = self.provider.complete(self.model, [])
        self.assertEqual(result.content, "ok")


class TestGoogleProvider(StubTransportMixin, unittest.TestCase):
    def setUp(self):
        self.provider = GoogleProvider(api_key="gk-test")
        self.model = get_model("google/gemini-3-pro")

    def test_request_shape(self):
        response = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "Sure."},
                    {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}},
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 12},
        }
        captured = self.craft(self.provider, response)
        result = self.provider.complete(self.model, tool_history(), tools=TOOL_SPEC, temperature=0.3)

        payload = captured["payload"]
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "You are APEX.")
        self.assertEqual(payload["generationConfig"]["temperature"], 0.3)
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"][0]["functionDeclarations"][0]["name"], "read_file")
        self.assertIn("?key=gk-test", captured["url"])

        roles = [c["role"] for c in payload["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])
        self.assertTrue(any("functionCall" in part for part in payload["contents"][1]["parts"]))
        self.assertTrue(any("functionResponse" in part for part in payload["contents"][2]["parts"]))

        self.assertEqual(result.content, "Sure.")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tokens_in, 30)
        self.assertEqual(result.finish_reason, "stop")

    def test_schema_keywords_filtered(self):
        risky = [{
            "name": "t", "description": "d",
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}},
                           "required": ["a"], "$schema": "http://json-schema.org/draft-07/schema#",
                           "additionalProperties": False},
        }]
        self.craft(self.provider, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
        self.provider.complete(self.model, [Message(role=Role.USER, content="x")], tools=risky,
                               temperature=0.1)

    def test_no_api_key_means_no_query_param(self):
        provider = GoogleProvider(api_key=None)
        captured = self.craft(provider, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
        provider.complete(self.model, [Message(role=Role.USER, content="x")])
        self.assertNotIn("?key=", captured["url"])


class TestErrorMapping(StubTransportMixin, unittest.TestCase):
    def setUp(self):
        self.provider = OpenAICompatibleProvider(provider_key="openai", api_key="sk")
        self.model = get_model("openai/gpt-5.1")

    def test_rate_limit(self):
        self.craft(self.provider, RateLimitError("429"))
        with self.assertRaises(RateLimitError):
            self.provider.complete(self.model, tool_history())

    def test_unavailable(self):
        self.craft(self.provider, ModelUnavailableError("down"))
        with self.assertRaises(ModelUnavailableError):
            self.provider.complete(self.model, tool_history())

    def test_generic_model_error(self):
        self.craft(self.provider, ModelError("boom"))
        with self.assertRaises(ModelError):
            self.provider.complete(self.model, tool_history())


class TestProviderFactory(unittest.TestCase):
    def test_factory_routes_by_provider(self):
        models = {m.provider for m in (get_model("openai/gpt-5.2"), get_model("anthropic/claude-opus-4.5"),
                                       get_model("google/gemini-3-pro"), get_model("mock/apex-mock"))}
        self.assertEqual(models, {"openai", "anthropic", "google", "mock"})

    def test_mock_provider_needs_no_key(self):
        provider = build_provider(get_model("mock/apex-mock"), {})
        self.assertEqual(provider.provider_name, "mock")

    def test_ollama_provider(self):
        provider = build_provider(get_model("ollama/llama3.1:8b"), {"ollama": ""})
        self.assertIsInstance(provider, OllamaProvider)

    def test_openai_compatible_provider_for_qwen(self):
        provider = build_provider(get_model("qwen/qwen3-coder"), {"qwen": "key"})
        self.assertEqual(provider.provider_name, "qwen")


if __name__ == "__main__":
    unittest.main()
