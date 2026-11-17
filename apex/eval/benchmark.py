"""Evaluation harness — benchmark tasks + scorers + report.

Scorers: exact match, contains, regex, length bounds, heuristic quality.
The benchmark runner executes a HarnessResult-producing callable over a task
suite and reports pass rate, mean score, tokens and cost.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EvalTask:
    id: str
    prompt: str
    scorer: str = "contains"  # exact | contains | regex | quality | exec
    expected: str = ""
    pattern: str = ""
    min_score: float = 0.6
    # For scorer="exec": the code written by the model is run, then these
    # probe expressions are evaluated against the namespace it defines.
    probes: tuple[str, ...] = ()
    timeout: float = 10.0


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    score: float
    output: str = ""
    tokens: int = 0
    cost: float = 0.0
    duration_ms: float = 0.0
    detail: str = ""


DEFAULT_SUITE: list[EvalTask] = [
    EvalTask(id="math-add", prompt="Calculate 12 + 8", scorer="contains", expected="20"),
    EvalTask(id="math-mul", prompt="Calculate 7 * 6", scorer="contains", expected="42"),
    EvalTask(id="plan-basic", prompt="Make a plan to learn Python in 4 weeks", scorer="quality"),
    EvalTask(id="explain-basic", prompt="Explain what an API is", scorer="quality"),
    EvalTask(id="code-basic", prompt="Write a Python function to add two numbers", scorer="contains", expected="def"),
    EvalTask(id="compare-basic", prompt="Compare two options for a small project", scorer="quality"),
    EvalTask(id="reason-basic", prompt="Solve: if a train travels 60 km in 1.5 hours, what is its speed?", scorer="contains", expected="40"),
    EvalTask(id="write-basic", prompt="Write a short professional email asking for a meeting", scorer="quality"),
    # Execution-verified tasks: passing these means the code actually ran and
    # produced the right values, not that the text merely contained "def".
    EvalTask(
        id="exec-fibonacci",
        prompt="Write a Python function called fibonacci that returns the nth Fibonacci number (0-indexed)",
        scorer="exec",
        probes=("fibonacci(0) == 0", "fibonacci(1) == 1", "fibonacci(10) == 55"),
    ),
    EvalTask(
        id="exec-add",
        prompt="Write a Python function called add that adds two numbers",
        scorer="exec",
        probes=("add(2, 3) == 5", "add(-1, 1) == 0"),
    ),
    EvalTask(
        id="exec-prime",
        prompt="Write a Python function called is_prime that returns True for prime numbers",
        scorer="exec",
        probes=("is_prime(29) is True", "is_prime(28) is False", "is_prime(1) is False"),
    ),
]


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to the raw text."""
    match = _CODE_BLOCK.search(text or "")
    return match.group(1) if match else (text or "")


def run_probes(code: str, probes: tuple[str, ...], timeout: float = 10.0) -> tuple[bool, float, str]:
    """Execute generated code in a subprocess and evaluate probe expressions.

    Runs in a temporary directory with a restricted environment and a hard
    timeout: generated code is untrusted, and an eval suite that can hang or
    leak credentials is worse than no eval suite.
    """
    if not code.strip():
        return False, 0.0, "no code produced"
    if not probes:
        return False, 0.0, "no probes defined"

    harness_code = (
        code
        + "\n\nimport json as _json, sys as _sys\n"
        + "_results = {}\n"
        + "".join(
            f"try:\n    _results[{json.dumps(probe)}] = bool({probe})\n"
            f"except Exception as _exc:\n    _results[{json.dumps(probe)}] = False\n"
            for probe in probes
        )
        + "_sys.stdout.write('__APEX_PROBES__' + _json.dumps(_results))\n"
    )

    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory(prefix="apex-eval-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", harness_code],
                cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return False, 0.0, f"code timed out after {timeout}s"
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__APEX_PROBES__"):
            payload = line[len("__APEX_PROBES__"):]
    if payload is None:
        stderr = (proc.stderr or "").strip().splitlines()
        reason = stderr[-1][:160] if stderr else "code produced no output"
        return False, 0.0, f"execution failed: {reason}"
    try:
        results = json.loads(payload)
    except json.JSONDecodeError:
        return False, 0.0, "unreadable probe results"
    passed = sum(1 for value in results.values() if value)
    total = len(results) or 1
    ok = passed == total
    detail = f"probes {passed}/{total} passed" + (
        "" if ok else f": failed {[k for k, v in results.items() if not v]}"
    )
    return ok, passed / total, detail


def score_output(task: EvalTask, output: str) -> tuple[bool, float, str]:
    if task.scorer == "exact":
        ok = output.strip().lower() == task.expected.strip().lower()
        return ok, 1.0 if ok else 0.0, "exact match" if ok else "no exact match"
    if task.scorer == "contains":
        ok = task.expected.lower() in output.lower()
        return ok, 1.0 if ok else 0.0, f"contains '{task.expected}'" if ok else f"missing '{task.expected}'"
    if task.scorer == "regex":
        ok = re.search(task.pattern, output, re.I) is not None
        return ok, 1.0 if ok else 0.0, "regex matched" if ok else "regex not matched"
    if task.scorer == "exec":
        ok, score, detail = run_probes(extract_code(output), task.probes, task.timeout)
        return ok, score, detail
    # quality scorer: reuse the heuristic critic.
    from ..reasoning.critic import heuristic_score

    crit = heuristic_score(output, task.prompt)
    return crit.score >= task.min_score, crit.score, f"quality={crit.score:.2f} {crit.reasons}"


def run_benchmark(run_fn: Callable[[str], object], suite: list[EvalTask] | None = None) -> dict:
    suite = suite or DEFAULT_SUITE
    results: list[EvalResult] = []
    for task in suite:
        start = time.time()
        try:
            result = run_fn(task.prompt)
            output = getattr(result, "output", str(result))
            tokens = getattr(result, "total_tokens_in", 0) + getattr(result, "total_tokens_out", 0)
            cost = getattr(result, "total_cost_usd", 0.0)
        except Exception as exc:  # noqa: BLE001
            results.append(EvalResult(task_id=task.id, passed=False, score=0.0, detail=f"error: {exc}"))
            continue
        passed, score, detail = score_output(task, output)
        results.append(
            EvalResult(
                task_id=task.id, passed=passed, score=score, output=output[:500],
                tokens=tokens, cost=cost, duration_ms=(time.time() - start) * 1000, detail=detail,
            )
        )
    passed_n = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed_n,
        "pass_rate": passed_n / len(results) if results else 0.0,
        "mean_score": sum(r.score for r in results) / len(results) if results else 0.0,
        "total_tokens": sum(r.tokens for r in results),
        "total_cost": sum(r.cost for r in results),
        "results": [
            {"task_id": r.task_id, "passed": r.passed, "score": r.score, "detail": r.detail,
             "tokens": r.tokens, "cost": r.cost, "duration_ms": r.duration_ms}
            for r in results
        ],
    }
