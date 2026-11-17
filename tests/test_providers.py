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


if __name__ == "__main__":
    unittest.main()
