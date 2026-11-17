import unittest

from apex.memory.store import Episode, MemoryStore, TfidfIndex
from apex.reasoning.critic import heuristic_score
from apex.reasoning.selector import select_strategy
from apex.reasoning.strategies import ReasoningStrategy, extract_final, majority_vote
from apex.types import Complexity, Task, TaskType


class TestStrategies(unittest.TestCase):
    def test_extract_final(self):
        self.assertEqual(extract_final("blah FINAL: 42"), "42")
        self.assertEqual(extract_final("no marker"), "no marker")

    def test_majority_vote(self):
        self.assertEqual(majority_vote(["FINAL: 42", "FINAL: 41", "FINAL: 42"]), "42")

    def test_selector_code(self):
        task = Task(description="Write code", task_type=TaskType.CODE, complexity=Complexity.MODERATE)
        strat, _ = select_strategy(task)
        self.assertEqual(strat, ReasoningStrategy.REFLEXION)

    def test_selector_tools(self):
        task = Task(description="Read file x", task_type=TaskType.GENERAL, complexity=Complexity.SIMPLE)
        strat, _ = select_strategy(task, needs_tools=True)
        self.assertEqual(strat, ReasoningStrategy.REACT)

    def test_selector_trivial(self):
        task = Task(description="Hi", task_type=TaskType.GENERAL, complexity=Complexity.TRIVIAL)
        strat, _ = select_strategy(task)
        self.assertEqual(strat, ReasoningStrategy.DIRECT)

    def test_selector_hard_reasoning(self):
        task = Task(description="Prove it", task_type=TaskType.REASONING, complexity=Complexity.EXPERT)
        strat, _ = select_strategy(task)
        self.assertEqual(strat, ReasoningStrategy.SELF_CONSISTENCY)


class TestCritic(unittest.TestCase):
    def test_good_answer_scores_high(self):
        crit = heuristic_score("## Answer\n\n1. First step\n2. Second step\n\nFINAL: done. " * 3, "do the thing")
        self.assertGreater(crit.score, 0.5)

    def test_tiny_answer_scores_low(self):
        crit = heuristic_score("ok", "do the thing")
        self.assertLess(crit.score, 0.6)


class TestMemory(unittest.TestCase):
    def test_tfidf_recall(self):
        idx = TfidfIndex()
        idx.add("the cat sat on the mat")
        idx.add("quantum computing uses qubits")
        idx.add("dogs are loyal animals")
        hits = idx.search("quantum qubits", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("quantum", idx.docs[hits[0][0]])

    def test_episodic_recall(self):
        mem = MemoryStore()
        mem.remember_episode(Episode(task="fix login bug", output="patched auth"))
        mem.remember_episode(Episode(task="cook pasta", output="boil water"))
        hits = mem.recall_episodes("login bug fix", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("login", hits[0].task)

    def test_semantic_recall(self):
        mem = MemoryStore()
        mem.remember_fact("User prefers dark mode")
        mem.remember_fact("Project deadline is Friday")
        hits = mem.recall_facts("dark mode preference", top_k=1)
        self.assertIn("dark mode", hits[0].text)

    def test_procedural_best(self):
        mem = MemoryStore()
        mem.remember_procedure("code", "cot", "mock/apex-mock", 0.5)
        mem.remember_procedure("code", "reflexion", "openai/gpt-5.2", 0.9)
        best = mem.best_procedure("code")
        self.assertEqual(best["strategy"], "reflexion")

    def test_episodic_limit(self):
        mem = MemoryStore(episodic_limit=5)
        for i in range(10):
            mem.remember_episode(Episode(task=f"t{i}", output="o"))
        self.assertLessEqual(len(mem.episodic), 5)


if __name__ == "__main__":
    unittest.main()


class TestContextManager(unittest.TestCase):
    """The context manager is wired into the ReAct loop; these pin its policy."""

    def setUp(self):
        from apex.context.manager import ContextBudget, ContextManager

        self.ctx = ContextManager(ContextBudget(max_tokens=1000, reserve_for_output=500))

    def _msgs(self, n: int, size: int = 400):
        """A *valid* conversation: every tool observation answers a declared call.

        Tool messages that no assistant turn requested are dropped by the
        manager (providers reject them), so a helper that emitted bare tool
        messages would quietly stop testing eviction at all.
        """
        from apex.types import Message, Role

        msgs = [Message(role=Role.SYSTEM, content="system prompt")]
        for i in range(n):
            msgs.append(Message(
                role=Role.ASSISTANT, content="",
                metadata={"tool_calls": [{"id": str(i), "name": f"t{i}", "arguments": {}}]},
            ))
            msgs.append(Message(role=Role.TOOL, content="x" * size, name=f"t{i}", tool_call_id=str(i)))
        msgs.append(Message(role=Role.USER, content="latest question"))
        return msgs

    def test_under_budget_is_untouched(self):
        from apex.types import Message, Role

        msgs = [Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="u")]
        self.assertEqual(len(self.ctx.fit_messages(msgs)), 2)

    def test_system_prompt_never_dropped(self):
        fitted = self.ctx.fit_messages(self._msgs(20))
        from apex.types import Role

        self.assertTrue(any(m.role == Role.SYSTEM for m in fitted))

    def test_newest_message_kept(self):
        fitted = self.ctx.fit_messages(self._msgs(20))
        self.assertEqual(fitted[-1].content, "latest question")

    def test_oversized_tool_output_summarised(self):
        from apex.types import Message, Role

        big = "First sentence. Middle detail. " + ("filler " * 10_000) + "Last conclusion."
        msgs = [Message(role=Role.SYSTEM, content="s"),
                Message(role=Role.ASSISTANT, content="",
                        metadata={"tool_calls": [{"id": "1", "name": "t", "arguments": {}}]}),
                Message(role=Role.TOOL, content=big, name="t", tool_call_id="1"),
                Message(role=Role.USER, content="q")]
        fitted = self.ctx.fit_messages(msgs)
        tool_msg = next(m for m in fitted if m.role == Role.TOOL)
        self.assertLess(len(tool_msg.content), len(big))
        self.assertIn("First sentence", tool_msg.content)

    def test_result_fits_budget(self):
        from apex.context.manager import ContextBudget, ContextManager
        from apex.models.base import messages_tokens

        ctx = ContextManager(ContextBudget(max_tokens=800, reserve_for_output=400))
        fitted = ctx.fit_messages(self._msgs(30))
        self.assertLessEqual(messages_tokens(fitted), ctx.budget.usable + 50)

    def test_empty_input(self):
        self.assertEqual(self.ctx.fit_messages([]), [])

    def test_single_oversized_message_truncated(self):
        from apex.types import Message, Role

        fitted = self.ctx.fit_messages([Message(role=Role.USER, content="y" * 100_000)])
        self.assertLess(len(fitted[0].content), 100_000)


class TestToolTurnIntegrity(unittest.TestCase):
    """Trimming must never produce a conversation the providers reject."""

    def setUp(self):
        from apex.context.manager import ContextBudget, ContextManager

        self.ctx = ContextManager(ContextBudget(max_tokens=900, reserve_for_output=400))

    def test_orphan_tool_message_dropped(self):
        from apex.types import Message, Role

        msgs = [
            Message(role=Role.SYSTEM, content="s"),
            Message(role=Role.TOOL, content="orphan", name="t", tool_call_id="gone"),
            Message(role=Role.USER, content="q"),
        ]
        fitted = self.ctx.fit_messages(msgs)
        self.assertFalse([m for m in fitted if m.role == Role.TOOL])
        self.assertEqual([m.role for m in fitted], [Role.SYSTEM, Role.USER])

    def test_matching_pair_survives(self):
        from apex.types import Message, Role

        msgs = [
            Message(role=Role.SYSTEM, content="s"),
            Message(role=Role.ASSISTANT, content="",
                    metadata={"tool_calls": [{"id": "a1", "name": "read_file", "arguments": {}}]}),
            Message(role=Role.TOOL, content="kept", name="read_file", tool_call_id="a1"),
        ]
        fitted = self.ctx.fit_messages(msgs)
        self.assertEqual([m.role for m in fitted], [Role.SYSTEM, Role.ASSISTANT, Role.TOOL])

    def test_eviction_never_leaves_a_dangling_tool_turn(self):
        from apex.types import Message, Role

        msgs = [Message(role=Role.SYSTEM, content="s")]
        for i in range(40):
            msgs.append(Message(
                role=Role.ASSISTANT, content="",
                metadata={"tool_calls": [{"id": f"c{i}", "name": "read_file", "arguments": {}}]},
            ))
            msgs.append(Message(role=Role.TOOL, content="y" * 800, name="read_file", tool_call_id=f"c{i}"))
        fitted = self.ctx.fit_messages(msgs)

        declared: set[str] = set()
        for message in fitted:
            if message.role == Role.ASSISTANT:
                declared.update(c["id"] for c in (message.metadata or {}).get("tool_calls", []))
            elif message.role == Role.TOOL:
                self.assertIn(message.tool_call_id, declared,
                              "tool observation survived without its assistant turn")
        self.assertNotEqual(fitted[1].role, Role.TOOL, "conversation cannot start with a tool turn")

    def test_trimmed_conversation_survives_anthropic_wire_format(self):
        from apex.models.base import AnthropicProvider
        from apex.types import Message, Role

        msgs = [Message(role=Role.SYSTEM, content="s")]
        for i in range(40):
            msgs.append(Message(
                role=Role.ASSISTANT, content="",
                metadata={"tool_calls": [{"id": f"c{i}", "name": "read_file", "arguments": {"path": "x"}}]},
            ))
            msgs.append(Message(role=Role.TOOL, content="y" * 800, name="read_file", tool_call_id=f"c{i}"))
        fitted = self.ctx.fit_messages(msgs)
        wire = AnthropicProvider(api_key="k")._wire_messages(fitted)
        tool_use_ids = {
            block["id"]
            for turn in wire
            for block in turn["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        for turn in wire:
            for block in turn["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    self.assertIn(block["tool_use_id"], tool_use_ids,
                                  "Anthropic would reject this unmatched tool_result with a 400")


class TestLoopContextWiring(unittest.TestCase):
    def test_loop_uses_context_manager(self):
        from apex.context.manager import ContextBudget, ContextManager
        from apex.execution.loop import ReActLoop
        from apex.models.base import ModelResponse
        from apex.types import ToolCall, ToolResult

        ctx = ContextManager(ContextBudget(max_tokens=600, reserve_for_output=300))
        seen_sizes = []

        def model_fn(messages, tools):
            seen_sizes.append(len(messages))
            if len(seen_sizes) < 4:
                return ModelResponse(
                    content="loop", model_id="f", tokens_in=1, tokens_out=1, cost_usd=0.0,
                    latency_ms=0.0, tool_calls=[ToolCall(name=f"t{len(seen_sizes)}", arguments={})],
                )
            return ModelResponse(content="done", model_id="f", tokens_in=1, tokens_out=1,
                                 cost_usd=0.0, latency_ms=0.0)

        loop = ReActLoop(
            model_fn=model_fn,
            tool_executor=lambda c: ToolResult(tool_call_id=c.id, tool_name=c.name,
                                               output="z" * 5000, success=True),
            max_steps=10,
            context_manager=ctx,
        )
        self.assertEqual(loop.run("system", "user")["output"], "done")
        # Without compaction the message count would keep climbing.
        self.assertLessEqual(max(seen_sizes), 8)
