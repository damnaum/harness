import unittest

from apex.execution.loop import CircuitBreaker, ReActLoop, RetryPolicy
from apex.planning.dag import heuristic_plan, parse_model_plan, topological_order
from apex.types import Task, TaskType, ToolCall, ToolResult


def _fake_model_factory(script):
    """script: list of (content, tool_calls) responses."""
    state = {"i": 0}

    def fn(messages, tools):
        from apex.models.base import ModelResponse

        content, calls = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        return ModelResponse(content=content, model_id="fake", tool_calls=calls, tokens_in=10, tokens_out=5)

    return fn


class TestReActLoop(unittest.TestCase):
    def test_direct_answer(self):
        loop = ReActLoop(_fake_model_factory([("hello", [])]), lambda c: ToolResult(c.id, c.name, "x"))
        out = loop.run("sys", "hi")
        self.assertEqual(out["output"], "hello")
        self.assertEqual(out["steps"], 0)

    def test_tool_then_answer(self):
        script = [("using tool", [ToolCall(name="calculator", arguments={"expression": "2+2"})]), ("answer is 4", [])]
        loop = ReActLoop(_fake_model_factory(script), lambda c: ToolResult(c.id, c.name, "4"))
        out = loop.run("sys", "calc")
        self.assertEqual(out["output"], "answer is 4")
        self.assertEqual(len(out["tool_calls"]), 1)

    def test_loop_detection(self):
        call = ToolCall(name="calculator", arguments={"expression": "1+1"})
        script = [("again", [call])] * 5
        loop = ReActLoop(_fake_model_factory(script), lambda c: ToolResult(c.id, c.name, "2"))
        from apex.errors import LoopDetectedError

        with self.assertRaises(LoopDetectedError):
            loop.run("sys", "hi")

    def test_max_steps(self):
        # Different args each time so loop detection doesn't fire first.
        script = [(f"step {i}", [ToolCall(name="t", arguments={"i": i})]) for i in range(10)]
        loop = ReActLoop(_fake_model_factory(script), lambda c: ToolResult(c.id, c.name, "x"), max_steps=3)
        from apex.errors import MaxStepsExceededError

        with self.assertRaises(MaxStepsExceededError):
            loop.run("sys", "hi")


class TestRetryAndBreaker(unittest.TestCase):
    def test_retry_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("boom")
            return "ok"

        self.assertEqual(RetryPolicy(max_retries=3, base_delay=0.01).run(flaky), "ok")

    def test_retry_exhausts(self):
        def always():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            RetryPolicy(max_retries=2, base_delay=0.01).run(always)

    def test_circuit_breaker_opens(self):
        breaker = CircuitBreaker(failure_threshold=2, cooldown=60)

        def fail():
            raise ValueError("x")

        for _ in range(2):
            with self.assertRaises(ValueError):
                breaker.call(fail)
        from apex.errors import CircuitOpenError

        with self.assertRaises(CircuitOpenError):
            breaker.call(lambda: "never")


class TestPlanning(unittest.TestCase):
    def test_heuristic_plan_steps(self):
        task = Task(description="Research X. Then compare options. Finally summarise findings.", task_type=TaskType.RESEARCH)
        plan = heuristic_plan(task)
        self.assertGreaterEqual(len(plan.steps), 3)
        # Sequential dependencies.
        self.assertEqual(plan.steps[1].depends_on, [plan.steps[0].id])

    def test_topological_order(self):
        task = Task(description="A. B. C.", task_type=TaskType.GENERAL)
        plan = heuristic_plan(task)
        order = topological_order(plan)
        self.assertEqual(len(order), len(plan.steps))
        self.assertEqual(order[0].id, plan.steps[0].id)

    def test_parse_model_plan(self):
        task = Task(description="do thing", task_type=TaskType.GENERAL)
        text = '{"steps": [{"id": "a", "title": "First", "description": "do first", "agent": "coder", "depends_on": []}, {"id": "b", "title": "Second", "description": "do second", "agent": "tester", "depends_on": ["a"]}]}'
        plan = parse_model_plan(text, task)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[1].depends_on, [plan.steps[0].id])

    def test_parse_model_plan_fallback(self):
        task = Task(description="do thing", task_type=TaskType.GENERAL)
        plan = parse_model_plan("not json at all", task)
        self.assertGreaterEqual(len(plan.steps), 1)




class TestLoopRobustness(unittest.TestCase):
    """The loop must survive misbehaving tools and models."""

    def _resp(self, content="", calls=None):
        from apex.models.base import ModelResponse

        return ModelResponse(
            content=content, model_id="fake", tool_calls=calls or [],
            tokens_in=1, tokens_out=1, cost_usd=0.0, latency_ms=0.0, finish_reason="stop",
        )

    def test_exception_in_executor_becomes_observation(self):
        from apex.execution.loop import ReActLoop
        from apex.types import ToolCall

        calls = [ToolCall(name="bad", arguments={"x": 1})]
        responses = [self._resp("thinking", calls), self._resp("final answer")]

        def executor(_call):
            raise RuntimeError("executor exploded")

        loop = ReActLoop(
            model_fn=lambda messages, tools: responses.pop(0),
            tool_executor=executor,
            max_steps=3,
        )
        out = loop.run("system", "user")
        self.assertEqual(out["output"], "final answer")

    def test_unknown_tool_result_is_fed_back(self):
        from apex.execution.loop import ReActLoop
        from apex.types import ToolCall

        responses = [
            self._resp("try", [ToolCall(name="ghost", arguments={})]),
            self._resp("recovered"),
        ]
        loop = ReActLoop(
            model_fn=lambda messages, tools: responses.pop(0),
            tool_executor=lambda call: __import__("apex.types", fromlist=["ToolResult"]).ToolResult(
                tool_call_id=call.id, tool_name=call.name, output="", success=False, error="unknown tool"
            ),
            max_steps=3,
        )
        self.assertEqual(loop.run("s", "u")["output"], "recovered")

    def test_repeated_identical_call_is_detected(self):
        from apex.errors import LoopDetectedError
        from apex.execution.loop import ReActLoop
        from apex.types import ToolCall, ToolResult

        call = ToolCall(name="same", arguments={"a": 1})
        loop = ReActLoop(
            model_fn=lambda messages, tools: self._resp("again", [call]),
            tool_executor=lambda c: ToolResult(tool_call_id=c.id, tool_name=c.name, output="ok", success=True),
            max_steps=5,
        )
        with self.assertRaises(LoopDetectedError):
            loop.run("s", "u")


if __name__ == "__main__":
    unittest.main()
