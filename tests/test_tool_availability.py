"""Tools must actually be reachable, and unreachable tools must be loud.

The regression these tests lock down: the harness used a keyword heuristic to
decide whether to send tool schemas to the model. "Read notes.txt and tell me
the release" matched no keyword ("file" was absent), so the model received an
empty tool list and answered from imagination — a silent, confidently wrong
result. Two independent fixes are asserted here:

* tool schemas are offered on every request (``agents.always_offer_tools``),
  opt-out still works;
* the heuristic that picks the react strategy recognises concrete filenames.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apex.harness as harness_module
from apex.config import ApexConfig
from apex.harness import ApexHarness
from apex.models.base import ModelResponse
from apex.types import Message, Role, ToolCall


class RecordingProvider:
    """Wraps the real (mock) provider and records what the model was offered."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, *args, **kwargs):
        model = args[0]
        messages = args[1]
        tools = args[2] if len(args) > 2 else kwargs.get("tools")
        self.calls.append(
            {
                "model": model.id,
                "tools": [self._name(t) for t in (tools or [])],
                "messages": [(m.role.value, m.content) for m in messages],
            }
        )
        return self.inner.complete(*args, **kwargs)

    @staticmethod
    def _name(spec):
        if isinstance(spec, dict):
            return spec.get("name") or (spec.get("function") or {}).get("name")
        return getattr(spec, "name", None)


class LoopForeverProvider:
    """Always asks for another tool call — the agent must still terminate."""

    def __init__(self):
        self.calls = 0

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        self.calls += 1
        return ModelResponse(
            content="",
            tool_calls=[ToolCall(id=f"c{self.calls}", name="list_dir", arguments={"path": "."})],
            model_id=model.id,
            tokens_in=5,
            tokens_out=2,
            cost_usd=0.0,
        )


class FailingToolProvider:
    """Calls read_file on a path that does not exist, then answers."""

    def __init__(self):
        self.calls = 0

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        self.calls += 1
        has_observation = any(m.role == Role.TOOL for m in messages)
        if not has_observation:
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "missing.txt"})],
                model_id=model.id,
                tokens_in=5,
                tokens_out=2,
                cost_usd=0.0,
            )
        observation = next(m.content for m in messages if m.role == Role.TOOL)
        return ModelResponse(
            content=f"I could not read the file. Tool said: {observation}",
            model_id=model.id,
            tokens_in=5,
            tokens_out=8,
            cost_usd=0.0,
        )


class HarnessCase(unittest.TestCase):
    _real_build = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)
        (self.workspace / "notes.txt").write_text("release=2026.11\nstatus=green\n")
        # build_provider is called per model call, so patching must be undone even
        # if a test fails midway — otherwise every later test inherits the stub.
        type(self)._real_build = harness_module.build_provider
        self.addCleanup(self._restore_build)

    def _restore_build(self):
        if type(self)._real_build is not None:
            harness_module.build_provider = type(self)._real_build

    def patch_provider(self, factory):
        harness_module.build_provider = factory

    def harness(self, **agent_kwargs) -> ApexHarness:
        config = ApexConfig(profile="offline", workspace=str(self.workspace))
        for key, value in agent_kwargs.items():
            setattr(config.agents, key, value)
        return ApexHarness(config)

    def recording_harness(self, **agent_kwargs):
        harness = self.harness(**agent_kwargs)
        real_build = type(self)._real_build
        recorders: list[RecordingProvider] = []

        def build(*args, **kwargs):
            recorder = RecordingProvider(real_build(*args, **kwargs))
            recorders.append(recorder)
            return recorder

        self.patch_provider(build)
        return harness, recorders


class TestSchemasAreOffered(HarnessCase):
    def test_plain_question_still_sees_the_tools(self):
        harness, recorders = self.recording_harness()
        harness.run("What is 2 + 2?")
        calls = [c for r in recorders for c in r.calls]
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["tools"]), len(harness.tools.names()))
        self.assertIn("read_file", calls[0]["tools"])

    def test_opt_out_really_disables_schemas(self):
        harness, recorders = self.recording_harness(always_offer_tools=False)
        harness.run("What is 2 + 2?")
        calls = [c for r in recorders for c in r.calls]
        self.assertTrue(calls)
        self.assertEqual(calls[0]["tools"], [])

    def test_opt_out_keeps_tools_for_tool_shaped_tasks(self):
        harness, recorders = self.recording_harness(always_offer_tools=False)
        harness.run("Read notes.txt and tell me the release")
        calls = [c for r in recorders for c in r.calls]
        self.assertTrue(calls)
        self.assertIn("read_file", calls[0]["tools"])

    def test_env_var_can_opt_out(self):
        import os

        os.environ["APEX_TOOLS_ALWAYS_OFFER"] = "false"
        try:
            self.assertFalse(ApexConfig.from_env().agents.always_offer_tools)
        finally:
            del os.environ["APEX_TOOLS_ALWAYS_OFFER"]
        os.environ["APEX_TOOLS_ALWAYS_OFFER"] = "true"
        try:
            self.assertTrue(ApexConfig.from_env().agents.always_offer_tools)
        finally:
            del os.environ["APEX_TOOLS_ALWAYS_OFFER"]


class TestToolIntentDetection(HarnessCase):
    def test_filename_makes_a_task_tool_shaped(self):
        harness = self.harness()
        from apex.models.router import classify_task

        for task in (
            "Read notes.txt and tell me the release",
            "Append a line to log.json",
            "Summarise data.csv by column",
            "What does deploy.yaml contain?",
            "check /etc/hosts.md",
        ):
            self.assertTrue(
                harness._needs_tools(classify_task(task), task), f"missed: {task}"
            )

    def test_prose_tasks_do_not_need_tools(self):
        harness = self.harness()
        from apex.models.router import classify_task

        for task in ("Explain recursion simply", "Write a haiku about rain"):
            self.assertFalse(
                harness._needs_tools(classify_task(task), task), f"false positive: {task}"
            )

    def test_filename_task_takes_the_react_path(self):
        harness = self.harness()
        result = harness.run("Read notes.txt and tell me the release")
        self.assertEqual(result.strategy.value, "react")
        self.assertTrue(result.success)


class TestGroundedAnswers(HarnessCase):
    def test_answer_contains_the_file_content(self):
        result = self.harness().run("Read notes.txt and tell me the release")
        self.assertTrue(result.success)
        self.assertIn("2026.11", result.output)
        self.assertIn("green", result.output)

    def test_missing_file_is_reported_not_invented(self):
        harness = self.harness()
        self.patch_provider(lambda *a, **k: FailingToolProvider())
        result = harness.run("Read missing.txt and tell me the release")
        self.assertIn("could not read", result.output.lower())
        self.assertIn("missing.txt", result.output)
        self.assertNotIn("2026.11", result.output)

    def test_runaway_tool_loop_terminates(self):
        harness = self.harness(max_steps=4)
        provider = LoopForeverProvider()
        self.patch_provider(lambda *a, **k: provider)
        result = harness.run("List the directory")
        self.assertGreaterEqual(provider.calls, 2, "the loop ended before any tool ran")
        self.assertLessEqual(provider.calls, 8, "agent looped past its step budget")
        self.assertIsInstance(result.output, str)


class TestToolSpecShape(unittest.TestCase):
    def test_specs_are_serialisable_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = ApexHarness(ApexConfig(profile="offline", workspace=tmp))
        specs = harness.tools.specs()
        self.assertTrue(specs)
        blob = json.dumps(specs)
        self.assertLess(len(blob), 20_000, "tool schemas should stay small")
        names = [s.get("name") or s.get("function", {}).get("name") for s in specs]
        self.assertEqual(len(names), len(set(names)), "duplicate tool names offered")
        for spec in specs:
            self.assertIsInstance(spec, dict)
            self.assertTrue(spec.get("description") or spec.get("function", {}).get("description"))


class TestApprovalPolicyIsVisible(HarnessCase):
    """A denied privileged tool must reach the operator, not just the model."""

    def prod_harness(self) -> ApexHarness:
        config = ApexConfig(profile="prod", workspace=str(self.workspace))
        config.models.default_model = "mock/apex-mock"
        return ApexHarness(config)

    def test_denied_tool_warns_and_counts(self):
        harness = self.prod_harness()
        self.assertFalse(harness.config.safety.auto_approve_tools)
        result = harness.run("Run the shell command ls and report")
        denial = [w for w in result.warnings if "denied by policy" in w]
        self.assertEqual(len(denial), 1, f"warnings were {result.warnings}")
        self.assertIn("bash", denial[0])
        self.assertEqual(harness.metrics.snapshot()["counters"].get("events.tool_denied"), 1)
        self.assertIn("denied", result.output.lower())

    def test_denial_warns_once_per_run(self):
        harness = self.prod_harness()
        result = harness.run("Run the shell command ls and report")
        denials = [w for w in result.warnings if "denied by policy" in w]
        self.assertEqual(len(denials), len(set(denials)))

    def test_approval_handler_is_honoured(self):
        harness = self.prod_harness()
        seen = []

        def approve(name, arguments):
            seen.append(name)
            return True

        harness.tools.approval_handler = approve
        harness.run("Run the shell command echo hi and report")
        self.assertIn("bash", seen)
        self.assertFalse([w for w in harness.run("Run the shell command echo hi").warnings
                          if "denied by policy" in w])

    def test_handler_that_refuses_keeps_the_warning(self):
        harness = self.prod_harness()
        harness.tools.approval_handler = lambda name, arguments: False
        result = harness.run("Run the shell command ls and report")
        self.assertTrue([w for w in result.warnings if "denied by policy" in w])

    def test_non_policy_tool_failure_is_counted_separately(self):
        harness = self.prod_harness()
        harness.tools.approval_handler = lambda name, arguments: True
        result = harness.run("Read missing.txt and tell me the release")
        counters = harness.metrics.snapshot()["counters"]
        self.assertGreaterEqual(counters.get("events.tool_failed", 0), 1)
        self.assertIsNone(counters.get("events.tool_denied"))
        self.assertFalse([w for w in result.warnings if "denied by policy" in w])


class GreedyProvider:
    """Emits a fresh 20 KB observation every round, like a noisy log reader."""

    def __init__(self, rounds: int = 14):
        self.rounds = rounds
        self.prompts: list[list[tuple[str, str]]] = []

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        self.prompts.append([(m.role.value, m.content) for m in messages])
        n = len(self.prompts)
        if n <= self.rounds:
            return ModelResponse(
                content="",
                model_id=model.id,
                tool_calls=[ToolCall(id=f"c{n}", name="read_file",
                                     arguments={"path": "notes.txt", "line": n})],
                tokens_in=10,
                tokens_out=5,
            )
        return ModelResponse(content="FINAL: done", model_id=model.id, tokens_in=10, tokens_out=5)


class TestLoopCompaction(HarnessCase):
    """The ReAct loop must compact, not grow until the provider rejects it."""

    def setUp(self):
        super().setUp()
        (self.workspace / "notes.txt").write_text("x" * 20_000)

    def run_greedy(self, rounds=14, max_tokens=None):
        harness = self.harness()
        if max_tokens:
            harness.config.context.max_tokens = max_tokens
            harness.context.budget.max_tokens = max_tokens
        provider = GreedyProvider(rounds=rounds)
        self.patch_provider(lambda *a, **k: provider)
        result = harness.run("Read notes.txt and tell me the release")
        return harness, provider, result

    def test_prompt_stays_within_the_budget(self):
        harness, provider, _ = self.run_greedy()
        from apex.models.base import messages_tokens

        usable = harness.context.budget.usable
        sizes = [
            sum(len(content) for _, content in prompt) // 4 for prompt in provider.prompts
        ]
        self.assertGreater(max(sizes), usable // 2, "the test never reached the budget")
        for index, prompt in enumerate(provider.prompts):
            self.assertLessEqual(
                sizes[index], usable + 200,
                f"turn {index} was {sizes[index]} tokens against a {usable} budget",
            )

    def test_compaction_keeps_the_task_in_view(self):
        _, provider, result = self.run_greedy()
        task = "Read notes.txt and tell me the release"
        for index, prompt in enumerate(provider.prompts):
            self.assertTrue(
                any(task in content for _, content in prompt),
                f"turn {index} lost the task after compaction",
            )
        self.assertTrue(result.success)

    def test_compaction_never_dangles_a_tool_turn(self):
        _, provider, _ = self.run_greedy()
        for index, prompt in enumerate(provider.prompts):
            body = [role for role, _ in prompt if role != "system"]
            self.assertNotEqual(body[0], "tool", f"turn {index} starts with a tool observation")
            declared = 0
            for role, _ in prompt:
                if role == "assistant":
                    declared += 1
                elif role == "tool":
                    self.assertGreater(declared, 0, f"turn {index} has an orphan tool message")

    def test_small_conversation_is_not_rewritten(self):
        harness = self.harness()
        from apex.types import Message, Role

        messages = [Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="q")]
        self.assertEqual(harness.context.fit_messages(messages), messages)


if __name__ == "__main__":
    unittest.main()
