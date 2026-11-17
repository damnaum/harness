"""Reasoning-strategy behaviour: voting, best-of-N selection, ReAct grounding.

These pin the semantics that separate the strategies — a Tree-of-Thoughts run
must *select* the best candidate, not majority-vote it, and self-consistency
must treat "40", "40.0" and "The answer is 40." as the same answer.
"""

from __future__ import annotations

import unittest

from apex.config import ApexConfig
from apex.harness import ApexHarness
from apex.reasoning.strategies import (
    SPECS,
    best_of_n,
    extract_final,
    majority_vote,
    normalise_answer,
)
from apex.types import ReasoningStrategy


class TestNormalisation(unittest.TestCase):
    def test_number_formatting_unified(self):
        self.assertEqual(normalise_answer("FINAL: 40.0"), normalise_answer("FINAL: 40"))

    def test_filler_removed(self):
        self.assertEqual(normalise_answer("The answer is 40."), "40")

    def test_case_and_punctuation(self):
        self.assertEqual(normalise_answer("FINAL: PARIS!"), "paris")

    def test_units_preserved(self):
        self.assertIn("km/h", normalise_answer("FINAL: 40 km/h"))

    def test_empty(self):
        self.assertEqual(normalise_answer(""), "")


class TestVoting(unittest.TestCase):
    def test_majority_wins(self):
        self.assertEqual(extract_final(majority_vote(["FINAL: 40", "FINAL: 42", "FINAL: 40.0"])), "40")

    def test_equivalent_answers_agree(self):
        vote = majority_vote(["FINAL: 40", "The answer is 40.0.", "FINAL: 7"])
        self.assertIn("40", vote)

    def test_tie_is_deterministic(self):
        self.assertEqual(majority_vote(["FINAL: a", "FINAL: b"]), majority_vote(["FINAL: a", "FINAL: b"]))

    def test_empty_input(self):
        self.assertEqual(majority_vote([]), "")
        self.assertEqual(majority_vote(["", "  "]), "")

    def test_original_text_returned(self):
        vote = majority_vote(["FINAL: 40 km/h", "FINAL: 40 km/h"])
        self.assertIn("km/h", vote)


class TestBestOfN(unittest.TestCase):
    def test_prefers_structured_complete_answer(self):
        weak = "ok"
        strong = "## Answer\n\n1. First step\n2. Second step\n\nFINAL: 40 km/h"
        self.assertIn("FINAL: 40", best_of_n([weak, strong], "solve the speed"))

    def test_empty_candidates(self):
        self.assertEqual(best_of_n([], "x"), "")

    def test_single_candidate_returned(self):
        self.assertEqual(best_of_n(["only"], "x"), "only")

    def test_prefers_task_relevant_answer(self):
        off_topic = "## Notes\n\n1. Something\n2. Else\n\nFINAL: unrelated"
        on_topic = "## Answer\n\n1. Compute 60/1.5\n2. Result 40\n\nFINAL: 40"
        self.assertIn("40", best_of_n([off_topic, on_topic], "compute the speed from 60 km in 1.5 hours"))


class TestSpecs(unittest.TestCase):
    def test_every_strategy_has_a_spec(self):
        for strategy in ReasoningStrategy:
            self.assertIn(strategy, SPECS)
            self.assertTrue(SPECS[strategy].description)

    def test_sampling_strategies_sample(self):
        self.assertGreater(SPECS[ReasoningStrategy.SELF_CONSISTENCY].samples, 1)
        self.assertGreater(SPECS[ReasoningStrategy.TOT].samples, 1)

    def test_react_is_single_sample(self):
        self.assertEqual(SPECS[ReasoningStrategy.REACT].samples, 1)

    def test_reflexion_has_reflection_budget(self):
        self.assertGreater(SPECS[ReasoningStrategy.REFLEXION].max_reflections, 0)


class TestHarnessStrategyExecution(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.apex = ApexHarness(ApexConfig(profile="offline", workspace=self.tmp))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_explicit_strategy_is_honoured(self):
        result = self.apex.run("Prove that sqrt(2) is irrational", strategy=ReasoningStrategy.DIRECT)
        self.assertEqual(result.strategy, ReasoningStrategy.DIRECT)

    def test_strategy_reason_recorded(self):
        result = self.apex.run("What is 2 + 2?")
        self.assertIn("strategy_reason", result.metadata)

    def test_final_answer_extracted_into_metadata(self):
        result = self.apex.run("What is 15% of 240?")
        self.assertIn("36", result.metadata["final_answer"])

    def test_tool_round_trip_grounds_the_answer(self):
        from pathlib import Path

        Path(self.tmp, "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        result = self.apex.run("Read the file calc.py and tell me what it does")
        self.assertTrue(result.success)
        self.assertIn("def add", result.output)
        self.assertIn("read_file", result.output)

    def test_missing_file_is_reported_not_invented(self):
        result = self.apex.run("Read the file missing_file_xyz.py and tell me what it does")
        self.assertTrue(result.success)
        self.assertIn("ERROR", result.output)

    def test_self_consistency_samples(self):
        result = self.apex.run("What is 6 * 7?", strategy=ReasoningStrategy.SELF_CONSISTENCY)
        self.assertIn("42", result.output)

    def test_tree_of_thoughts_returns_answer(self):
        result = self.apex.run("Compare caching strategies", strategy=ReasoningStrategy.TOT)
        self.assertTrue(result.output)

    def test_swarm_path_has_plan(self):
        result = self.apex.run(
            "Plan a phased migration with milestones, then propose a rollback strategy",
            strategy=ReasoningStrategy.PLAN_EXECUTE,
        )
        self.assertIsNotNone(result.plan)
        self.assertTrue(result.plan.steps)
        self.assertGreaterEqual(len(result.agent_results), 1)

    def test_metrics_accumulate(self):
        before = self.apex.metrics.snapshot()["counters"].get("runs.total", 0)
        self.apex.run("What is 1 + 1?")
        after = self.apex.metrics.snapshot()["counters"].get("runs.total", 0)
        self.assertEqual(after, before + 1)

    def test_trace_spans_recorded(self):
        self.apex.run("What is 1 + 1?")
        self.assertTrue(self.apex.tracer.export())


class TestSamplingSafety(unittest.TestCase):
    """Sampling must never repeat a side effect.

    Self-consistency/ToT/LATS re-run the whole tool loop N times. With read-only
    tools that is mere cost; ``write_file``/``bash``/``python_exec`` would perform
    the same write or command N times. The harness must refuse to sample then.
    """

    def setUp(self):
        import tempfile

        from apex.config import ApexConfig
        from apex.harness import ApexHarness

        self.tmp = tempfile.TemporaryDirectory()
        self.harness = ApexHarness(ApexConfig(profile="offline", workspace=self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_self_consistency_disabled_with_write_tools(self):
        from apex.types import ReasoningStrategy

        result = self.harness.run(
            "Write a Python function that adds two numbers",
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
        )
        self.assertTrue(result.success)
        self.assertTrue(
            any("sampling disabled" in w for w in result.warnings),
            f"expected a sampling warning, got {result.warnings}",
        )

    def test_warning_recorded_in_event_stream(self):
        from apex.types import ReasoningStrategy

        seen = []
        self.harness.events.subscribe("*", lambda event: seen.append(event.type))
        self.harness.run(
            "Write a Python function that reverses a string",
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
        )
        self.assertIn("sampling_disabled", seen)

    def test_no_warning_when_sampling_is_safe(self):
        from apex.types import ReasoningStrategy

        # Strip the side-effecting tools; sampling should then proceed normally.
        for name in ("bash", "write_file", "python_exec"):
            self.harness.tools.tools.pop(name, None)
        result = self.harness.run(
            "Prove that the square root of 2 is irrational",
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
        )
        self.assertFalse(any("sampling disabled" in w for w in result.warnings))

    def test_side_effecting_tool_detection(self):
        self.assertEqual(
            self.harness._side_effecting_tools(),
            {"bash", "python_exec", "write_file"},
        )


if __name__ == "__main__":
    unittest.main()
