"""Planner + swarm tests: decomposition, routing, DAG order, synthesis, context.

These cover the parts where multi-agent systems usually fail quietly: steps that
are silently dropped, agents that receive unbounded context, and syntheses that
dump raw step output instead of an answer.
"""

from __future__ import annotations

import unittest

from apex.agents.base import AgentContext
from apex.agents.swarm import Swarm
from apex.models.router import classify_task
from apex.planning.dag import (
    assign_agent,
    decompose,
    heuristic_plan,
    parse_model_plan,
    topological_order,
)
from apex.types import AgentResult, Plan, PlanStep, StepStatus, TaskType


class TestDecompose(unittest.TestCase):
    def test_sequencing_words(self):
        parts = decompose("Research the market, then draft the plan, finally review the budget")
        self.assertEqual(len(parts), 3)

    def test_enumerated_list(self):
        parts = decompose("1. Research the market 2. Draft the plan 3. Review the budget")
        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[0].startswith("Research"))

    def test_colon_introduced_subtasks(self):
        parts = decompose("Migrate the monolith: identify the risks, then propose a roadmap")
        self.assertGreaterEqual(len(parts), 2)

    def test_single_step_returns_one(self):
        self.assertEqual(decompose("What is 2 + 2?"), ["What is 2 + 2?"])

    def test_empty(self):
        self.assertEqual(decompose("   "), [])

    def test_duplicates_removed(self):
        parts = decompose("Add tests. Add tests.")
        self.assertEqual(len(parts), 1)

    def test_no_mid_word_truncation(self):
        for clause in decompose("Propose a phased roadmap with milestones and a rollback strategy"):
            self.assertNotIn("…", clause)


class TestAgentAssignment(unittest.TestCase):
    def test_researcher(self):
        self.assertEqual(assign_agent("Research the latest battery advances"), "researcher")

    def test_coder(self):
        self.assertEqual(assign_agent("Implement the retry helper in Python"), "coder")

    def test_tester_plural(self):
        self.assertEqual(assign_agent("Add unit tests for the parser"), "tester")

    def test_analyst(self):
        self.assertEqual(assign_agent("Analyse the cost of the migration"), "analyst")

    def test_planner(self):
        self.assertEqual(assign_agent("Propose a phased roadmap with milestones"), "planner")

    def test_default(self):
        self.assertEqual(assign_agent("Say hello"), "executor")


class TestPlan(unittest.TestCase):
    def test_plan_steps_are_ordered_and_chained(self):
        plan = heuristic_plan(classify_task("Compile the report, then review the numbers"))
        self.assertGreaterEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].depends_on, [])
        for prev, cur in zip(plan.steps, plan.steps[1:]):
            self.assertIn(prev.id, cur.depends_on)

    def test_plan_ends_with_verification_gate(self):
        plan = heuristic_plan(classify_task("Plan a phased migration with milestones and a rollback plan"))
        self.assertEqual(plan.steps[-1].agent, "critic")

    def test_titles_are_numbered_and_capped(self):
        plan = heuristic_plan(classify_task("Summarise " + "very long detail " * 30))
        self.assertTrue(plan.steps[0].title[0].isdigit())
        self.assertLessEqual(len(plan.steps[0].title), 90)

    def test_topological_order_respects_dependencies(self):
        plan = heuristic_plan(classify_task("Do A then do B then do C"))
        order = topological_order(plan)
        self.assertEqual(len(order), len(plan.steps))
        indices = {s.id: i for i, s in enumerate(order)}
        for step in plan.steps:
            for dep in step.depends_on:
                self.assertLess(indices[dep], indices[step.id])

    def test_cycle_is_detected(self):
        a, b = PlanStep(title="a", description="a"), PlanStep(title="b", description="b")
        a.depends_on = [b.id]
        b.depends_on = [a.id]
        with self.assertRaises(ValueError):
            topological_order(Plan(task_id="t", steps=[a, b]))

    def test_model_plan_json_is_parsed(self):
        raw = (
            '{"steps": ['
            '{"id": "s1", "title": "write code", "description": "write the parser", "agent": "coder", "depends_on": []},'
            '{"id": "s2", "title": "test it", "description": "run the tests", "agent": "tester", "depends_on": ["s1"]}'
            "]}"
        )
        plan = parse_model_plan(raw, classify_task("write and test a parser"))
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[1].depends_on, [plan.steps[0].id])

    def test_model_plan_garbage_falls_back(self):
        plan = parse_model_plan("not json at all", classify_task("do the thing"))
        self.assertTrue(plan.steps)

    def test_unknown_dependency_is_dropped(self):
        raw = '{"steps": [{"id": "s1", "title": "x", "description": "x", "depends_on": ["nope"]}]}'
        plan = parse_model_plan(raw, classify_task("x"))
        self.assertEqual(plan.steps[0].depends_on, [])


class FakeAgent:
    """Minimal agent double: returns text, or fails on demand."""

    def __init__(self, name: str, output: str = "done", fail: bool = False):
        self.name = name
        self.output = output
        self.fail = fail
        self.calls: list[str] = []

    def run(self, task: str, context=None) -> AgentResult:
        self.calls.append(task)
        if self.fail:
            return AgentResult(agent=self.name, output="", success=False, model_used="fake",
                               metadata={"error": "boom"})
        return AgentResult(agent=self.name, output=self.output, success=True, model_used="fake")


class TestSwarm(unittest.TestCase):
    def _plan(self) -> Plan:
        s1 = PlanStep(title="1. write", description="write the code", agent="coder")
        s2 = PlanStep(title="2. test", description="test the code", agent="tester", depends_on=[s1.id])
        return Plan(task_id="t", steps=[s1, s2])

    def test_steps_run_in_order(self):
        agents = {"coder": FakeAgent("coder", "code here"), "tester": FakeAgent("tester", "tests pass")}
        plan = self._plan()
        results = Swarm(agents).run_plan(plan, "write and test")
        self.assertEqual([r.agent for r in results], ["coder", "tester"])
        self.assertEqual(plan.steps[0].status, StepStatus.DONE)

    def test_failed_dependency_skips_dependents(self):
        agents = {"coder": FakeAgent("coder", fail=True), "tester": FakeAgent("tester")}
        plan = self._plan()
        Swarm(agents).run_plan(plan, "write and test")
        self.assertEqual(plan.steps[0].status, StepStatus.FAILED)
        self.assertEqual(plan.steps[1].status, StepStatus.SKIPPED)
        self.assertEqual(agents["tester"].calls, [])

    def test_parallel_wave_execution(self):
        agents = {"coder": FakeAgent("coder", "a"), "tester": FakeAgent("tester", "b")}
        s1 = PlanStep(title="a", description="a", agent="coder")
        s2 = PlanStep(title="b", description="b", agent="tester")  # independent
        plan = Plan(task_id="t", steps=[s1, s2])
        results = Swarm(agents, parallel=True).run_plan(plan, "two independent steps")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))

    def test_unknown_agent_falls_back_to_executor(self):
        agents = {"executor": FakeAgent("executor", "fallback")}
        plan = Plan(task_id="t", steps=[PlanStep(title="x", description="x", agent="nonexistent")])
        results = Swarm(agents).run_plan(plan, "x")
        self.assertEqual(results[0].output, "fallback")

    def test_synthesis_deduplicates_and_labels(self):
        agents = {"coder": FakeAgent("coder", "shared paragraph"), "tester": FakeAgent("tester", "shared paragraph")}
        plan = self._plan()
        swarm = Swarm(agents)
        swarm.run_plan(plan, "write and test")
        text = swarm.synthesise(plan, "write and test")
        self.assertEqual(text.count("shared paragraph"), 1)
        self.assertIn("(coder)", text)
        self.assertIn("Plan executed: 2/2", text)

    def test_synthesis_reports_failures(self):
        agents = {"coder": FakeAgent("coder", fail=True), "tester": FakeAgent("tester")}
        plan = self._plan()
        swarm = Swarm(agents)
        swarm.run_plan(plan, "write and test")
        text = swarm.synthesise(plan, "write and test")
        self.assertIn("failed: boom", text)


class TestContextBudget(unittest.TestCase):
    def test_total_render_is_bounded(self):
        ctx = AgentContext(prior_outputs={f"agent{i}": "x" * 5000 for i in range(12)})
        rendered = ctx.render()
        self.assertLessEqual(len(rendered), AgentContext.CONTEXT_CHAR_BUDGET + 200)

    def test_newest_output_included(self):
        ctx = AgentContext(prior_outputs={"old": "ancient", "new": "fresh finding"})
        self.assertIn("fresh finding", ctx.render())

    def test_empty_context_renders_empty(self):
        self.assertEqual(AgentContext().render(), "")


class TestClassifierIntegration(unittest.TestCase):
    def test_research_is_not_code(self):
        task = classify_task("Research the latest advances in battery technology")
        self.assertEqual(task.task_type, TaskType.RESEARCH)

    def test_code_wins_over_data_nouns(self):
        task = classify_task("Write a Python function to parse CSV files")
        self.assertEqual(task.task_type, TaskType.CODE)

    def test_arithmetic_is_reasoning(self):
        self.assertEqual(classify_task("What is 17 * 23?").task_type, TaskType.REASONING)

    def test_debugging_needs_code_context(self):
        self.assertEqual(classify_task("Debug this Python traceback").task_type, TaskType.DEBUGGING)

    def test_multi_step_bumps_complexity(self):
        single = classify_task("Summarise the report")
        multi = classify_task("Summarise the report, then review the budget, and then send it")
        order = ["trivial", "simple", "moderate", "complex", "expert"]
        self.assertGreater(order.index(multi.complexity.value), order.index(single.complexity.value))


if __name__ == "__main__":
    unittest.main()
