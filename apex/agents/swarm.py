"""Swarm orchestration — runs a Plan's DAG with the specialist agents.

Execution semantics:
- Steps run in topological order; independent steps may run in parallel
  (thread pool) when ``parallel=True``.
- Each step's output is fed into the shared AgentContext for later steps.
- A failed step marks dependents as skipped but never aborts the whole plan.
- The critic agent reviews the merged output at the end (if enabled).
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..planning.dag import topological_order
from ..types import AgentResult, Plan, PlanStep, StepStatus
from .base import Agent, AgentContext


class Swarm:
    def __init__(self, agents: dict[str, Agent], parallel: bool = False, max_workers: int = 4):
        self.agents = agents
        self.parallel = parallel
        self.max_workers = max_workers

    def _agent_for(self, step: PlanStep) -> Agent:
        return self.agents.get(step.agent) or self.agents.get("executor") or next(iter(self.agents.values()))

    def _run_step(self, step: PlanStep, context: AgentContext) -> AgentResult:
        step.status = StepStatus.RUNNING
        agent = self._agent_for(step)
        result = agent.run(step.description, context)
        if result.success:
            step.status = StepStatus.DONE
            step.output = result.output
            context.prior_outputs[step.agent] = result.output
        else:
            step.status = StepStatus.FAILED
            step.error = result.metadata.get("error", "unknown")
        return result

    def run_plan(self, plan: Plan, task_description: str, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> list[AgentResult]:
        on_event = on_event or (lambda *a: None)
        context = AgentContext(task_description=task_description)
        results: list[AgentResult] = []
        order = topological_order(plan)
        by_id = {s.id: s for s in plan.steps}

        if self.parallel:
            # Group into waves of ready steps.
            remaining = list(order)
            while remaining:
                ready = [s for s in remaining if all(by_id[d].status == StepStatus.DONE for d in s.depends_on)]
                if not ready:
                    break
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {pool.submit(self._run_step, s, context): s for s in ready}
                    for fut in futures:
                        results.append(fut.result())
                        on_event("step_done", {"step": fut.result().agent})
                remaining = [s for s in remaining if s not in ready]
        else:
            for step in order:
                if any(by_id[d].status in (StepStatus.FAILED, StepStatus.SKIPPED) for d in step.depends_on):
                    step.status = StepStatus.SKIPPED
                    continue
                results.append(self._run_step(step, context))
                on_event("step_done", {"step": step.id, "agent": step.agent, "status": step.status.value})

        return results

    def synthesise(self, plan: Plan, task_description: str) -> str:
        """Merge step outputs into one readable answer.

        Deterministic on purpose: an LLM merge step would add latency, cost and
        a new failure mode to every run. We instead de-duplicate repeated
        blocks, drop the verification gate from the body (it is a review, not
        a deliverable) and present it as a closing quality section.
        """
        body: list[str] = []
        review: list[str] = []
        seen: set[str] = set()
        done = 0

        for idx, step in enumerate(plan.steps, start=1):
            if step.status != StepStatus.DONE or not step.output:
                if step.status == StepStatus.FAILED:
                    review.append(f"- Step {idx} ({step.agent}) failed: {step.error}")
                continue
            done += 1
            output = step.output.strip()
            is_review = step.agent == "critic" or "verify" in step.title.lower()
            if is_review:
                review.append(output)
                continue

            # De-duplicate paragraph-level repeats between steps (common when
            # agents share context and restate it).
            kept = []
            for block in output.split("\n\n"):
                key = " ".join(block.split()).lower()[:200]
                if key and key not in seen:
                    seen.add(key)
                    kept.append(block.strip())
            if not kept:
                continue
            title = re.sub(r"^\s*\d+[.)]\s*", "", step.title)
            body.append(f"### Step {idx} — {title} _({step.agent})_")
            body.append("\n\n".join(kept))
            body.append("")

        parts = [f"**Task:** {task_description}", ""]
        if body:
            parts.extend(body)
        else:
            parts.append("_No step produced output._")
        if review:
            parts.append("### Quality review")
            parts.extend(review)
        parts.append(f"\n_Plan executed: {done}/{len(plan.steps)} steps completed._")
        return "\n".join(parts).strip()


def run_with_critic(swarm: Swarm, plan: Plan, task_description: str, threshold: float = 0.6, on_event=None) -> tuple[str, list[AgentResult], float]:
    """Run the plan, then have the critic review the synthesis. Returns
    (final_output, agent_results, critic_score)."""
    from ..reasoning.critic import heuristic_score

    results = swarm.run_plan(plan, task_description, on_event=on_event)
    synthesis = swarm.synthesise(plan, task_description)

    critic = swarm.agents.get("critic")
    score = heuristic_score(synthesis, task_description).score
    plan_already_reviewed = any(
        s.agent == "critic" and s.status == StepStatus.DONE for s in plan.steps
    )
    if critic is not None and not plan_already_reviewed:
        review = critic.run(
            f"Review this output for the task \"{task_description}\". "
            f"Score it 0-1 and list concrete fixes if needed.\n\n{synthesis[:4000]}"
        )
        results.append(review)
        if review.success:
            # Blend model review with heuristic (model review mentions a score).
            import re

            m = re.search(r"(0\.\d+|1\.0|1)", review.output)
            if m:
                try:
                    model_score = float(m.group(1))
                    score = (score + model_score) / 2
                except ValueError:
                    pass
    return synthesis, results, score
