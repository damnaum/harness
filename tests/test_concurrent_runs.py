"""Concurrent runs on one harness instance.

The HTTP server shares a single harness across up to APEX_MAX_CONCURRENCY
requests, so any per-run state stored on the instance leaks between requests:
a warning from request A appears in B's result, or B's spend is attributed to A.
Per-run state is thread-local; these tests hold that invariant.
"""

from __future__ import annotations

import tempfile
import threading
import unittest

from apex.config import ApexConfig
from apex.harness import ApexHarness


class TestConcurrentRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.harness = ApexHarness(ApexConfig(profile="offline", workspace=self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _run_in_threads(self, fn, count: int) -> dict:
        results: dict = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(count)

        def worker(index: int):
            try:
                barrier.wait(timeout=10)
                results[index] = fn(index)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [])
        return results

    def test_warnings_do_not_cross_between_runs(self):
        results = self._run_in_threads(
            lambda i: self.harness.run("What is 2 + 2?", model=f"nope/model-{i}"), 6
        )
        for index, result in results.items():
            mine = [w for w in result.warnings if "unknown model" in w]
            self.assertEqual(len(mine), 1, index)
            self.assertIn(f"nope/model-{index}", mine[0])

    def test_memory_stays_consistent_under_concurrent_runs(self):
        self._run_in_threads(lambda i: self.harness.run(f"What is {i} + {i}?"), 8)
        episodes = self.harness.memory.stats()["episodes"]
        self.assertGreaterEqual(episodes, 8)
        self.assertLessEqual(episodes, self.harness.memory.episodic_limit)
        # The index and the list must agree after the dust settles.
        hits = self.harness.memory.recall_episodes("what is the sum", top_k=3)
        for episode in hits:
            self.assertIn(episode, self.harness.memory.episodic)

    def test_failed_runs_report_their_own_spend(self):
        from apex.models.base import ModelResponse
        from apex.types import ReasoningStrategy, ToolCall

        # Patch once (patching per-thread on a shared instance would race) and
        # read the thread's identity from thread-local state instead.
        per_thread = threading.local()

        def fake(decision, messages, tools=None, temperature=0.7):
            index = getattr(per_thread, "index", 0)
            return ModelResponse(
                content="", model_id=decision.model.id,
                tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+1"})],
                tokens_in=10 + index, tokens_out=5, cost_usd=0.25,
            ), decision.model

        self.harness._complete_with_fallback = fake
        self.harness.config.models.max_cost_usd = 0.2

        def work(index: int):
            per_thread.index = index
            return self.harness.run("Calculate 1 + 1", strategy=ReasoningStrategy.REACT)

        results = self._run_in_threads(work, 4)
        for index, result in results.items():
            self.assertFalse(result.success, index)
            # Each run spent exactly one call's worth — not another run's total.
            self.assertAlmostEqual(result.total_cost_usd, 0.25, places=6, msg=index)
            self.assertEqual(result.total_tokens_in, 10 + index, index)


if __name__ == "__main__":
    unittest.main()
