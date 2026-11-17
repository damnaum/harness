"""Per-run spend cap and failure accounting.

A long agent loop on a frontier model can cost more than the task is worth, and
a failed run still costs money. Both facts need to be true in the accounting:
the run must stop at the cap, and the failed result must report what it spent.
"""

from __future__ import annotations

import tempfile
import unittest

from apex.config import ApexConfig
from apex.errors import BudgetExceededError
from apex.harness import ApexHarness
from apex.models.base import ModelResponse
from apex.types import ReasoningStrategy, ToolCall


def _tool_call_response(model_id: str, cost: float) -> ModelResponse:
    return ModelResponse(
        content="",
        model_id=model_id,
        tool_calls=[ToolCall(name="calculator", arguments={"expression": "2+2"})],
        tokens_in=10,
        tokens_out=5,
        cost_usd=cost,
    )


class BudgetTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.harness = ApexHarness(ApexConfig(profile="offline", workspace=self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def stub_provider(self, cost: float, calls: dict):
        def fake(decision, messages, tools=None, temperature=0.7):
            calls["n"] += 1
            return _tool_call_response(decision.model.id, cost), decision.model

        self.harness._complete_with_fallback = fake


class TestBudgetEnforcement(BudgetTestBase):
    def test_stops_at_cap(self):
        self.harness.config.models.max_cost_usd = 0.2
        calls = {"n": 0}
        self.stub_provider(0.25, calls)
        result = self.harness.run("Calculate 2 + 2", strategy=ReasoningStrategy.REACT)
        self.assertFalse(result.success)
        self.assertEqual(calls["n"], 1, "must not make a second call after the cap")
        self.assertTrue(any("budget exceeded" in w for w in result.warnings))

    def test_failed_run_reports_spend(self):
        self.harness.config.models.max_cost_usd = 0.2
        calls = {"n": 0}
        self.stub_provider(0.25, calls)
        result = self.harness.run("Calculate 2 + 2", strategy=ReasoningStrategy.REACT)
        self.assertAlmostEqual(result.total_cost_usd, 0.25)
        self.assertEqual(result.total_tokens_in, 10)
        self.assertEqual(result.total_tokens_out, 5)
        self.assertTrue(result.model_used)

    def test_zero_budget_means_unlimited(self):
        self.harness.config.models.max_cost_usd = 0.0
        calls = {"n": 0}

        # First call asks for a tool, second answers for real.
        def fake(decision, messages, tools=None, temperature=0.7):
            calls["n"] += 1
            if calls["n"] == 1:
                return _tool_call_response(decision.model.id, 5.0), decision.model
            return ModelResponse(
                content="FINAL: 4", model_id=decision.model.id,
                tokens_in=5, tokens_out=2, cost_usd=5.0,
            ), decision.model

        self.harness._complete_with_fallback = fake
        result = self.harness.run("Calculate 2 + 2", strategy=ReasoningStrategy.REACT)
        self.assertTrue(result.success)
        self.assertEqual(calls["n"], 2)
        self.assertAlmostEqual(result.total_cost_usd, 10.0)

    def test_cap_not_hit_when_under_budget(self):
        self.harness.config.models.max_cost_usd = 10.0
        calls = {"n": 0}

        def fake(decision, messages, tools=None, temperature=0.7):
            calls["n"] += 1
            return ModelResponse(
                content="FINAL: 4", model_id=decision.model.id,
                tokens_in=5, tokens_out=2, cost_usd=0.01,
            ), decision.model

        self.harness._complete_with_fallback = fake
        result = self.harness.run("Calculate 2 + 2", strategy=ReasoningStrategy.REACT)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.total_cost_usd, 0.01)

    def test_budget_event_emitted(self):
        self.harness.config.models.max_cost_usd = 0.1
        calls = {"n": 0}
        self.stub_provider(0.2, calls)
        seen = []
        self.harness.events.subscribe("*", lambda event: seen.append(event.type))
        self.harness.run("Calculate 2 + 2", strategy=ReasoningStrategy.REACT)
        self.assertIn("budget_exceeded", seen)


class TestBudgetConfig(unittest.TestCase):
    def test_env_var(self):
        import os

        os.environ["APEX_MAX_COST_USD"] = "1.25"
        try:
            cfg = ApexConfig.from_env()
            self.assertAlmostEqual(cfg.models.max_cost_usd, 1.25)
        finally:
            del os.environ["APEX_MAX_COST_USD"]

    def test_default_is_unlimited(self):
        cfg = ApexConfig(profile="offline")
        self.assertEqual(cfg.models.max_cost_usd, 0.0)

    def test_error_is_typed(self):
        self.assertEqual(BudgetExceededError.code, "budget_exceeded")
        from apex.errors import ApexError

        self.assertTrue(issubclass(BudgetExceededError, ApexError))


if __name__ == "__main__":
    unittest.main()
