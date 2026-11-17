"""Evaluation suite tests — scorers, code extraction and sandboxed execution.

The exec scorer is the only part of the eval whose failure mode is *running
untrusted code*, so it gets the most attention here: timeouts, syntax errors,
missing functions and probe failures all have to be reported, not raised.
"""

from __future__ import annotations

import unittest

from apex.eval.benchmark import (
    DEFAULT_SUITE,
    EvalTask,
    extract_code,
    run_benchmark,
    run_probes,
    score_output,
)


GOOD_FIB = """
Here is the implementation:

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
"""

BROKEN_FIB = """
```python
def fibonacci(n):
    return n
```
"""


class TestExtractCode(unittest.TestCase):
    def test_fenced_block(self):
        self.assertIn("def fibonacci", extract_code(GOOD_FIB))
        self.assertNotIn("```", extract_code(GOOD_FIB))

    def test_python_tag_variant(self):
        self.assertIn("return 1", extract_code("```py\nreturn 1\n```"))

    def test_no_fence_returns_text(self):
        self.assertEqual(extract_code("def f(): pass"), "def f(): pass")

    def test_empty(self):
        self.assertEqual(extract_code(""), "")


class TestRunProbes(unittest.TestCase):
    def test_passing_probes(self):
        ok, score, detail = run_probes(extract_code(GOOD_FIB), ("fibonacci(10) == 55",))
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        self.assertIn("1/1", detail)

    def test_failing_probe(self):
        ok, score, detail = run_probes(extract_code(BROKEN_FIB), ("fibonacci(10) == 55",))
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        self.assertIn("failed", detail)

    def test_partial_pass_scores_fractionally(self):
        ok, score, _ = run_probes(extract_code(GOOD_FIB), ("fibonacci(10) == 55", "fibonacci(3) == 999"))
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)

    def test_syntax_error_reported(self):
        ok, score, detail = run_probes("def broken(:\n    pass", ("True",))
        self.assertFalse(ok)
        self.assertIn("execution failed", detail)

    def test_infinite_loop_times_out(self):
        ok, score, detail = run_probes("while True:\n    pass", ("True",), timeout=1.0)
        self.assertFalse(ok)
        self.assertIn("timed out", detail)

    def test_missing_name_is_a_failed_probe_not_a_crash(self):
        ok, _score, detail = run_probes("x = 1", ("undefined_function() == 1",))
        self.assertFalse(ok)
        self.assertIn("failed", detail)

    def test_empty_code(self):
        ok, _score, detail = run_probes("", ("True",))
        self.assertFalse(ok)
        self.assertIn("no code", detail)

    def test_no_probes(self):
        ok, _score, detail = run_probes("x = 1", ())
        self.assertFalse(ok)

    def test_environment_has_no_secrets(self):
        import os

        os.environ["APEX_EVAL_SECRET_KEY"] = "sk-must-not-leak"
        try:
            code = "import os\nleaked = any('SECRET' in k for k in os.environ)"
            ok, _score, _detail = run_probes(code, ("leaked is False",))
        finally:
            del os.environ["APEX_EVAL_SECRET_KEY"]
        self.assertTrue(ok)


class TestScorers(unittest.TestCase):
    def test_exact(self):
        task = EvalTask(id="t", prompt="p", scorer="exact", expected="42")
        self.assertTrue(score_output(task, " 42 ")[0])
        self.assertFalse(score_output(task, "42 and more")[0])

    def test_contains(self):
        task = EvalTask(id="t", prompt="p", scorer="contains", expected="40")
        self.assertTrue(score_output(task, "the answer is 40 km/h")[0])

    def test_regex(self):
        task = EvalTask(id="t", prompt="p", scorer="regex", pattern=r"def\s+\w+\(")
        self.assertTrue(score_output(task, "def add(a, b):")[0])
        self.assertFalse(score_output(task, "no code here")[0])

    def test_quality(self):
        task = EvalTask(id="t", prompt="explain caching", scorer="quality", min_score=0.5)
        good = "## Caching\n\n1. Store results\n2. Reuse them\n\nFINAL: caching saves work."
        self.assertTrue(score_output(task, good)[0])

    def test_exec_scorer_uses_fences(self):
        task = EvalTask(id="t", prompt="p", scorer="exec", probes=("fibonacci(10) == 55",))
        self.assertTrue(score_output(task, GOOD_FIB)[0])
        self.assertFalse(score_output(task, BROKEN_FIB)[0])


class TestBenchmark(unittest.TestCase):
    def test_report_shape(self):
        def fake_run(prompt: str):
            class Result:
                output = "FINAL: 20"
                total_tokens_in = 5
                total_tokens_out = 2
                total_cost_usd = 0.001

            return Result()

        suite = [EvalTask(id="only", prompt="p", scorer="contains", expected="20")]
        report = run_benchmark(fake_run, suite=suite)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertEqual(report["total_tokens"], 7)
        self.assertAlmostEqual(report["total_cost"], 0.001)

    def test_exception_in_run_is_a_failure_not_a_crash(self):
        def exploding_run(prompt: str):
            raise RuntimeError("model down")

        report = run_benchmark(exploding_run, suite=[EvalTask(id="x", prompt="p")])
        self.assertEqual(report["passed"], 0)
        self.assertIn("error", report["results"][0]["detail"])

    def test_default_suite_includes_exec_tasks(self):
        scorers = {task.scorer for task in DEFAULT_SUITE}
        self.assertIn("exec", scorers)
        self.assertGreaterEqual(len(DEFAULT_SUITE), 11)


if __name__ == "__main__":
    unittest.main()
