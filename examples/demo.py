#!/usr/bin/env python3
"""End-to-end APEX demo — runs entirely offline, no API keys, no network.

Exercises: routing, classification, reasoning-strategy selection, tool use with
observations, durable memory, the agent swarm with a critic, tracing, metrics,
the eval suite and the CLI-facing diagnostics.

Run:  python examples/demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running straight from a checkout: python examples/demo.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.config import ApexConfig  # noqa: E402
from apex.harness import ApexHarness  # noqa: E402
from apex.models.router import classify_task  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="apex-demo-"))
    (workspace / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    harness = ApexHarness(ApexConfig(profile="offline", workspace=str(workspace)))
    print(f"workspace: {workspace}")

    # ------------------------------------------------------------------ routing
    banner("1. Routing — every model scored, the best one chosen")
    for task in (
        "What is 17 * 23?",
        "Write a Python function that reverses a string",
        "Research the latest advances in battery technology",
        "Plan a phased migration of a monolith to microservices",
    ):
        decision = harness.router.route(classify_task(task))
        fb = ", ".join(m.id for m in decision.fallbacks)
        print(f"  {task[:52]:54s} -> {decision.model.id:24s} fallbacks: {fb}")

    # ------------------------------------------------------------- simple runs
    banner("2. Reasoning strategies chosen automatically")
    for task in (
        "What is 15% of 240?",
        "Prove that the square root of 2 is irrational",
        "Summarise the trade-offs between caching and recomputation",
    ):
        result = harness.run(task)
        print(f"  {result.strategy.value:18s} {result.model_used:16s} {task[:44]}")
        print(f"    -> {result.output.splitlines()[0][:100]}")

    # ------------------------------------------------------------- tool use
    banner("3. Tool use — the loop observes, then answers from the observation")
    result = harness.run("Read the file calc.py and tell me what it does")
    print("\n".join(f"    {line}" for line in result.output.splitlines()[:10]))
    print(f"    grounded in tool output: {'read_file' in result.output}")

    # ------------------------------------------------------------------ memory
    banner("4. Durable memory across runs")
    harness.remember("Staging deploys use blue-green rollout behind a feature flag")
    result = harness.run("What do you know about our staging deploys?")
    print("\n".join(f"    {line}" for line in result.output.splitlines()[:6]))
    print(f"    memory: {harness.memory.stats()}")

    # ------------------------------------------------------------------- swarm
    banner("5. Swarm — planner, specialists, critic, synthesis")
    result = harness.run(
        "Plan a migration of our monolith to microservices: identify the risks, "
        "then propose a phased roadmap with milestones and a rollback strategy"
    )
    assert result.plan is not None
    for step in result.plan.steps:
        print(f"    [{step.status.value:8s}] {step.agent:11s} {step.title[:56]}")
    print(f"\n    critic score: {result.metadata.get('critic_score', 0):.2f}")
    print(f"    agents run:   {len(result.agent_results)}")

    # --------------------------------------------------------------- evaluation
    banner("6. Eval suite")
    from apex.eval.benchmark import run_benchmark

    report = run_benchmark(harness.run)
    print(f"    pass_rate={report['pass_rate']:.0%}  mean_score={report['mean_score']:.2f}  "
          f"tokens={report['total_tokens']}  cost=${report['total_cost']:.4f}")
    for row in report["results"]:
        print(f"      {'PASS' if row['passed'] else 'FAIL'} {row['task_id']:14s} score={row['score']:.2f}")

    # ------------------------------------------------------------ observability
    banner("7. Observability")
    print(f"    metrics: {harness.metrics.snapshot()['counters']}")
    print(f"    spans:   {len(harness.tracer.export())}")
    doctor = harness.doctor()
    tools_n = len(doctor["tools"]) if isinstance(doctor["tools"], list) else doctor["tools"]
    providers_n = len(doctor["providers"]) if isinstance(doctor["providers"], list) else doctor["providers"]
    problems = doctor["config_problems"]
    problems_n = len(problems) if isinstance(problems, list) else problems
    print(f"    doctor:  models={doctor['models_registered']} tools={tools_n} "
          f"providers={providers_n} config_problems={problems_n}")
    print("\nDone. Everything above ran offline, deterministic and free.")


if __name__ == "__main__":
    main()
