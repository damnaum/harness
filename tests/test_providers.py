"""Tests for the offline provider — the deterministic reasoner.

These are behavioural tests: they assert that the mock actually *solves* the
problem classes the eval suite relies on, not that it returns a fixed string.
"""

from __future__ import annotations

import unittest

from apex.models.providers.mock import (
    MockProvider,
    extract_options,
    mock_answer,
    solve_word_problem,
    synthesise_code,
    unwrap_task,
)
from apex.models.registry import get_model
from apex.types import Message, Role, ToolCall


def complete(content: str, system: str = "", tools=None, extra: list[Message] | None = None):
    provider = MockProvider()
    model = get_model("mock/apex-mock")
    messages = []
    if system:
        messages.append(Message(role=Role.SYSTEM, content=system))
    messages.append(Message(role=Role.USER, content=content))
    messages.extend(extra or [])
    return provider.complete(model, messages, tools=tools)


class TestUnwrap(unittest.TestCase):
    def test_strips_task_prefix(self):
        self.assertEqual(unwrap_task("Task: add two numbers"), "add two numbers")

    def test_strips_strategy_instructions(self):
        prompt = "Task: Write a Python function to add two numbers\n\nThink step by step, then state FINAL: <answer>"
        self.assertEqual(unwrap_task(prompt), "Write a Python function to add two numbers")

    def test_strips_injected_context(self):
        prompt = "Task: what is 2+2\n\nRelevant past context:\n- Fact: the sky is blue"
        self.assertEqual(unwrap_task(prompt), "what is 2+2")

    def test_empty_prompt_survives(self):
        self.assertEqual(unwrap_task("   "), "")


class TestWordProblems(unittest.TestCase):
    def assert_solves(self, text: str, expected: str):
        solved = solve_word_problem(text)
        self.assertIsNotNone(solved, f"no solution for: {text}")
        answer, reasoning = solved
        self.assertIn(expected, answer, f"{text} -> {answer}")
        self.assertTrue(reasoning)

    def test_speed(self):
        self.assert_solves("A train travels 60 km in 1.5 hours, what is its speed?", "40")

    def test_distance(self):
        self.assert_solves("How far does a car travel at 80 km/h for 2 hours?", "160")

    def test_percentage(self):
        self.assert_solves("What is 15% of 240?", "36")

    def test_discount(self):
        self.assert_solves("An item priced at 200 with a 25% discount", "150")

    def test_average(self):
        self.assert_solves("What is the average of 4, 8, 12 and 16?", "10")

    def test_simple_arithmetic(self):
        self.assert_solves("Calculate 12 * 12", "144")

    def test_temperature(self):
        self.assert_solves("Convert 100 celsius to fahrenheit", "212")

    def test_factorial(self):
        self.assert_solves("What is the factorial of 6", "720")

    def test_unrecognised_returns_none(self):
        self.assertIsNone(solve_word_problem("Write a haiku about the sea"))


class TestCodeSynthesis(unittest.TestCase):
    def test_fibonacci_is_real_code(self):
        code = synthesise_code("Write a Python function that returns the nth Fibonacci number")
        self.assertIn("def fibonacci", code)
        namespace: dict = {}
        exec(code.split("```python")[1].split("```")[0], namespace)  # noqa: S102 - test fixture
        self.assertEqual(namespace["fibonacci"](10), 55)
        self.assertEqual(namespace["fibonacci"](0), 0)

    def test_add_function_is_correct(self):
        code = synthesise_code("Write a Python function to add two numbers")
        namespace: dict = {}
        exec(code.split("```python")[1].split("```")[0], namespace)  # noqa: S102
        self.assertEqual(namespace["add"](2, 3), 5)

    def test_prime_check(self):
        code = synthesise_code("Write a function to check if a number is prime")
        namespace: dict = {}
        exec(code.split("```python")[1].split("```")[0], namespace)  # noqa: S102
        fn = namespace["is_prime"]
        self.assertTrue(fn(29))
        self.assertFalse(fn(28))
        self.assertFalse(fn(1))

    def test_named_function_is_honoured(self):
        code = synthesise_code("Write a python function called tally that adds two numbers")
        self.assertIn("def tally", code)

    def test_generic_intent_gets_scaffold(self):
        code = synthesise_code("Write a function to compute the shard key")
        self.assertIn("def ", code)
        self.assertIn("TODO", code)

    def test_non_code_request_returns_none(self):
        self.assertIsNone(synthesise_code("Summarise the meeting notes"))


class TestOptions(unittest.TestCase):
    def test_extracts_three_options(self):
        opts = extract_options("Compare LangGraph, AutoGen and CrewAI for building agents")
        self.assertEqual([o.lower() for o in opts], ["langgraph", "autogen", "crewai"])

    def test_versus_form(self):
        opts = extract_options("Compare Redis versus Postgres")
        self.assertEqual(len(opts), 2)

    def test_comparison_table_uses_real_names(self):
        out = mock_answer("Compare LangGraph and CrewAI for a production system")
        self.assertIn("LangGraph", out)
        self.assertIn("CrewAI", out)
        self.assertIn("Orchestration", out)


class TestResponses(unittest.TestCase):
    def test_arithmetic_response_has_final(self):
        resp = complete("Calculate 12 * 12")
        self.assertIn("144", resp.content)
        self.assertIn("FINAL:", resp.content)

    def test_tool_call_emitted_when_tools_offered(self):
        tools = [{"name": "read_file", "description": "read", "parameters": {}}]
        resp = complete("Read the file notes.txt", tools=tools)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "read_file")
        self.assertEqual(resp.tool_calls[0].arguments.get("path"), "notes.txt")

    def test_no_tool_call_when_no_tools(self):
        resp = complete("Read the file notes.txt", tools=None)
        self.assertEqual(resp.tool_calls, [])

    def test_grounded_answer_after_observation(self):
        tools = [{"name": "read_file", "description": "read", "parameters": {}}]
        history = [
            Message(role=Role.ASSISTANT, content="Action: read_file"),
            Message(role=Role.TOOL, name="read_file", content="def add(a, b):\n    return a + b", tool_call_id="1"),
        ]
        resp = complete("Read the file calc.py", tools=tools, extra=history)
        self.assertEqual(resp.tool_calls, [], "must not repeat the same tool call")
        self.assertIn("add", resp.content)
        self.assertIn("read_file", resp.content)

    def test_memory_question_uses_context(self):
        prompt = (
            "Task: What do you know about our staging deploy?\n\n"
            "Relevant past context:\n- Fact: staging uses blue-green rollout"
        )
        resp = complete(prompt)
        self.assertIn("blue-green", resp.content)

    def test_role_prompts_shape_output(self):
        critic = complete("review this plan", system="You are the APEX Critic — a strict reviewer.")
        self.assertIn("Review", critic.content)
        analyst = complete("identify the risks", system="You are the APEX Analyst.")
        self.assertIn("Analysis", analyst.content)

    def test_deterministic(self):
        a = complete("Explain how a hash map works").content
        b = complete("Explain how a hash map works").content
        self.assertEqual(a, b)

    def test_token_accounting(self):
        resp = complete("Write a short note about testing")
        self.assertGreater(resp.tokens_in + resp.tokens_out, 0)
        self.assertEqual(resp.cost_usd, 0.0)

    def test_long_prompt_is_handled(self):
        resp = complete("Explain caching " + ("in great detail " * 200))
        self.assertTrue(resp.content)


class TestWireFormatContracts(unittest.TestCase):
    """The three wire formats have non-obvious multi-turn requirements.

    Getting these wrong produces a 400 in the middle of a ReAct loop, which is
    the worst possible time to discover a serialisation bug.
    """

    def _two_tool_turn(self):
        from apex.types import Message, Role

        return [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="do two things"),
            Message(role=Role.ASSISTANT, content="", metadata={"tool_calls": [
                {"id": "c1", "name": "read_file", "arguments": {"path": "a"}},
                {"id": "c2", "name": "read_file", "arguments": {"path": "b"}},
            ]}),
            Message(role=Role.TOOL, content="A", name="read_file", tool_call_id="c1"),
            Message(role=Role.TOOL, content="B", name="read_file", tool_call_id="c2"),
        ]

    def _capture(self, provider, model_id, messages):
        captured = {}

        def fake(url, payload, headers):
            captured["payload"] = payload
            captured["url"] = url
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "content": [{"type": "text", "text": "ok"}],
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usage": {},
            }

        provider._post_json = fake
        from apex.models.registry import get_model

        provider.complete(get_model(model_id), messages)
        return captured["payload"]

    def test_openai_tool_messages_carry_ids(self):
        from apex.models.base import OpenAICompatibleProvider

        payload = self._capture(OpenAICompatibleProvider(api_key="k"), "openai/gpt-5.2", self._two_tool_turn())
        roles = [m["role"] for m in payload["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(payload["messages"][3]["tool_call_id"], "c1")
        self.assertEqual(payload["messages"][4]["tool_call_id"], "c2")
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["type"], "function")

    def test_anthropic_merges_parallel_tool_results(self):
        from apex.models.base import AnthropicProvider

        payload = self._capture(AnthropicProvider(api_key="k"), "anthropic/claude-opus-4.5", self._two_tool_turn())
        self.assertEqual(payload["system"], "sys")
        self.assertNotIn("system", [m["role"] for m in payload["messages"]])
        tool_turns = [
            m for m in payload["messages"]
            if m["role"] == "user" and isinstance(m["content"], list)
            and m["content"] and m["content"][0].get("type") == "tool_result"
        ]
        self.assertEqual(len(tool_turns), 1, "both tool_result blocks belong to one user turn")
        self.assertEqual(len(tool_turns[0]["content"]), 2)
        self.assertEqual(tool_turns[0]["content"][0]["tool_use_id"], "c1")
        self.assertEqual(tool_turns[0]["content"][1]["tool_use_id"], "c2")
        assistants = [m for m in payload["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(assistants[0]["content"]), 2)

    def test_anthropic_merges_consecutive_plain_turns(self):
        from apex.models.base import AnthropicProvider
        from apex.types import Message, Role

        payload = self._capture(
            AnthropicProvider(api_key="k"), "anthropic/claude-opus-4.5",
            [Message(role=Role.USER, content="one"), Message(role=Role.USER, content="two")],
        )
        self.assertEqual(len(payload["messages"]), 1)
        self.assertIn("two", payload["messages"][0]["content"])

    def test_anthropic_max_tokens_follows_the_model(self):
        from apex.models.base import AnthropicProvider

        payload = self._capture(AnthropicProvider(api_key="k"), "anthropic/claude-opus-4.5", self._two_tool_turn())
        self.assertGreater(payload["max_tokens"], 4096)

    def test_google_merges_function_responses_into_one_turn(self):
        from apex.models.base import GoogleProvider

        payload = self._capture(GoogleProvider(api_key="k"), "google/gemini-3-pro", self._two_tool_turn())
        contents = payload["contents"]
        self.assertEqual([c["role"] for c in contents], ["user", "model", "user"])
        responses = [p for p in contents[2]["parts"] if "functionResponse" in p]
        self.assertEqual(len(responses), 2)
        self.assertIn("systemInstruction", payload)
        self.assertIn("maxOutputTokens", payload["generationConfig"])

    def test_google_system_prompt_is_not_a_turn(self):
        from apex.models.base import GoogleProvider

        payload = self._capture(GoogleProvider(api_key="k"), "google/gemini-3-pro", self._two_tool_turn())
        for entry in payload["contents"]:
            for part in entry["parts"]:
                self.assertNotEqual(part.get("text"), "sys")

    def test_openai_retries_with_max_completion_tokens(self):
        from apex.models.base import OpenAICompatibleProvider
        from apex.models.registry import get_model

        provider = OpenAICompatibleProvider(api_key="k")
        seen = []

        def fake(url, payload, headers):
            seen.append(dict(payload))
            if "max_tokens" in payload:
                from apex.errors import ModelError

                raise ModelError("400: Unsupported parameter: 'max_tokens' ... use 'max_completion_tokens'")
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

        provider._post_json = fake
        provider.complete(get_model("openai/o3"), [__import__("apex.types", fromlist=["Message"]).Message(
            role=__import__("apex.types", fromlist=["Role"]).Role.USER, content="hi")], max_tokens=64)
        self.assertEqual(len(seen), 2)
        self.assertIn("max_tokens", seen[0])
        self.assertIn("max_completion_tokens", seen[1])

    def test_unrelated_model_error_is_not_retried(self):
        from apex.errors import ModelError
        from apex.models.base import OpenAICompatibleProvider
        from apex.models.registry import get_model
        from apex.types import Message, Role

        provider = OpenAICompatibleProvider(api_key="k")
        calls = {"n": 0}

        def fake(*args, **kwargs):
            calls["n"] += 1
            raise ModelError("400: context length exceeded")

        provider._post_json = fake
        with self.assertRaises(ModelError):
            provider.complete(get_model("openai/gpt-5.2"), [Message(role=Role.USER, content="hi")])
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
