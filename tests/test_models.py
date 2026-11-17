import unittest

from apex.models.registry import MODELS, best_for, get_model, list_models, providers
from apex.models.router import ModelRouter, classify_task
from apex.types import Complexity, TaskType


class TestRegistry(unittest.TestCase):
    def test_registry_nonempty(self):
        self.assertGreaterEqual(len(MODELS), 20)

    def test_ids_unique(self):
        ids = [m.id for m in MODELS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_model(self):
        m = get_model("mock/apex-mock")
        self.assertIsNotNone(m)
        self.assertEqual(m.provider, "mock")
        self.assertIsNone(get_model("nope/nothing"))

    def test_providers(self):
        self.assertIn("openai", providers())
        self.assertIn("anthropic", providers())
        self.assertIn("mock", providers())

    def test_list_by_provider(self):
        self.assertTrue(all(m.provider == "openai" for m in list_models("openai")))

    def test_best_for_code(self):
        top = best_for("code", top_n=3)
        self.assertEqual(len(top), 3)
        self.assertGreaterEqual(top[0].coding, top[-1].coding)

    def test_cost_estimate(self):
        m = get_model("openai/gpt-5.2")
        cost = m.cost_estimate(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, m.input_price_per_1m + m.output_price_per_1m)

    def test_mock_is_free(self):
        m = get_model("mock/apex-mock")
        self.assertEqual(m.cost_estimate(1000, 1000), 0.0)


class TestRouter(unittest.TestCase):
    def test_route_returns_decision(self):
        task = classify_task("Write a Python function to parse CSV files")
        decision = ModelRouter(mode="balanced").route(task)
        self.assertIsNotNone(decision.model)
        self.assertGreater(len(decision.fallbacks), 0)
        self.assertTrue(decision.reason)

    def test_route_preferred(self):
        task = classify_task("Explain APIs")
        decision = ModelRouter().route(task, preferred="mock/apex-mock")
        self.assertEqual(decision.model.id, "mock/apex-mock")

    def test_route_exclude(self):
        task = classify_task("Explain APIs")
        decision = ModelRouter(exclude=["mock/apex-mock"]).route(task)
        self.assertNotEqual(decision.model.id, "mock/apex-mock")

    def test_quality_mode_prefers_frontier(self):
        task = classify_task("Design a distributed system architecture with failure handling and consistency guarantees")
        q = ModelRouter(mode="quality").route(task)
        self.assertGreaterEqual(q.model.reasoning, 8)

    def test_cheapest_mode_prefers_cheap(self):
        task = classify_task("Explain what an API is")
        c = ModelRouter(mode="cheapest").route(task)
        self.assertLessEqual(c.model.input_price_per_1m, 1.0)


class TestClassify(unittest.TestCase):
    def test_code(self):
        t = classify_task("Write a Python function to add two numbers")
        self.assertEqual(t.task_type, TaskType.CODE)

    def test_debugging(self):
        t = classify_task("Debug this Python bug where the function crashes")
        self.assertEqual(t.task_type, TaskType.DEBUGGING)

    def test_research(self):
        t = classify_task("Research the latest advances in battery technology")
        self.assertEqual(t.task_type, TaskType.RESEARCH)

    def test_reasoning(self):
        t = classify_task("Solve this logic puzzle and prove the answer")
        self.assertEqual(t.task_type, TaskType.REASONING)

    def test_complexity_scales(self):
        short = classify_task("What is an API?")
        long = classify_task(" ".join(["word"] * 200))
        self.assertEqual(short.complexity, Complexity.TRIVIAL)
        self.assertEqual(long.complexity, Complexity.EXPERT)


class TestScopeClassification(unittest.TestCase):
    """Complexity must follow the number of deliverables, not just word count.

    A 22-word request to design, implement, benchmark and compare two rate
    limiters used to be classified SIMPLE — so it got no plan, no swarm and a
    cheap model. Word count is a poor proxy for scope.
    """

    CASES = [
        ("Design and implement a rate limiter, compare token bucket vs sliding "
         "window, write code, run a benchmark, recommend one", Complexity.EXPERT),
        ("Analyse why our p99 latency doubled after the migration and propose fixes",
         Complexity.MODERATE),
        ("Refactor the payment module into smaller services and migrate the database schema",
         Complexity.MODERATE),
        ("Compare Postgres and Cassandra for our workload and recommend one",
         Complexity.MODERATE),
        ("Build a REST API with auth, tests and documentation", Complexity.MODERATE),
        ("Write a function to reverse a string", Complexity.TRIVIAL),
        ("What is 2 + 2?", Complexity.TRIVIAL),
        ("Summarise this article in three bullet points", Complexity.TRIVIAL),
        ("Design, implement, test, deploy, document, benchmark and review a new auth service",
         Complexity.EXPERT),
    ]

    def test_scope_floor(self):
        for text, expected in self.CASES:
            with self.subTest(task=text[:40]):
                self.assertEqual(classify_task(text).complexity, expected)

    def test_floor_never_downgrades(self):
        long_prose = " ".join(["word"] * 200)
        self.assertEqual(classify_task(long_prose).complexity, Complexity.EXPERT)
        verbose_one_verb = ("Explain in great detail, over many paragraphs and with several "
                            "examples, what a hash table is and why it matters " + "context " * 40)
        self.assertEqual(classify_task(verbose_one_verb).complexity, Complexity.EXPERT)

    def test_multi_part_task_gets_a_plan(self):
        import tempfile

        from apex.config import ApexConfig
        from apex.harness import ApexHarness

        with tempfile.TemporaryDirectory() as tmp:
            harness = ApexHarness(ApexConfig(profile="offline", workspace=tmp))
            result = harness.run(
                "Design and implement a rate limiter, compare token bucket vs sliding "
                "window, write code, run a benchmark, recommend one"
            )
        self.assertIsNotNone(result.plan, "a multi-deliverable task must be planned")
        self.assertTrue(result.agent_results, "a planned task must run through the swarm")

    def test_trivial_task_stays_single_agent(self):
        import tempfile

        from apex.config import ApexConfig
        from apex.harness import ApexHarness

        with tempfile.TemporaryDirectory() as tmp:
            harness = ApexHarness(ApexConfig(profile="offline", workspace=tmp))
            result = harness.run("What is 2 + 2?")
        self.assertIsNone(result.plan, "a trivial question must not pay for a swarm")


class TestRegistryIntegrity(unittest.TestCase):
    """Guards the invariants other subsystems assume about the registry."""

    def test_ids_unique(self):
        from apex.models.registry import MODELS

        ids = [m.id for m in MODELS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scores_in_range(self):
        from apex.models.registry import MODELS

        for model in MODELS:
            for field in ("reasoning", "coding", "tools", "research",
                          "creativity", "speed", "long_context", "reliability"):
                value = getattr(model, field)
                self.assertGreaterEqual(value, 0, model.id)
                self.assertLessEqual(value, 10, model.id)

    def test_prices_non_negative_and_ordered(self):
        from apex.models.registry import MODELS

        for model in MODELS:
            self.assertGreaterEqual(model.input_price_per_1m, 0.0, model.id)
            self.assertGreaterEqual(model.output_price_per_1m, 0.0, model.id)

    def test_context_window_sane(self):
        from apex.models.registry import MODELS

        for model in MODELS:
            self.assertGreaterEqual(model.context_window, 8_000, model.id)
            self.assertGreater(model.max_output, 0, model.id)
            self.assertLessEqual(model.max_output, model.context_window, model.id)

    def test_every_task_type_has_a_capable_model(self):
        from apex.models.registry import MODELS
        from apex.types import TaskType

        for task_type in TaskType:
            best = max(m.capability(task_type) for m in MODELS)
            self.assertGreaterEqual(best, 8, f"no strong model for {task_type}")

    def test_mock_model_is_always_routable(self):
        from apex.models.registry import get_model

        self.assertIsNotNone(get_model("mock/apex-mock"))

    def test_all_providers_implement_interface(self):
        import inspect

        from apex.models import base
        from apex.models.providers import mock

        found = [
            obj for _, obj in inspect.getmembers(base, inspect.isclass)
            if issubclass(obj, base.BaseProvider) and obj is not base.BaseProvider
        ] + [mock.MockProvider]
        self.assertGreaterEqual(len(found), 5)
        for cls in found:
            self.assertTrue(callable(getattr(cls, "complete", None)), cls.__name__)

    def test_docs_prices_match_registry(self):
        """docs/MODELS.md must not disagree with the registry it documents."""
        import pathlib
        import re

        from apex.models.registry import MODELS

        doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "MODELS.md"
        text = doc.read_text()
        documented = {
            (round(float(a), 2), round(float(b), 2))
            for a, b in re.findall(r"(\d+\.\d+)\s*→\s*(\d+\.\d+)", text)
        }
        priced = [m for m in MODELS if m.input_price_per_1m > 0]
        self.assertGreaterEqual(len(priced), 15)

        for model in priced:
            pair = (round(model.input_price_per_1m, 2), round(model.output_price_per_1m, 2))
            self.assertIn(pair, documented, f"{model.id} price missing or stale in docs/MODELS.md")

    def test_docs_cover_every_provider(self):
        import pathlib

        from apex.models.registry import MODELS

        path = pathlib.Path(__file__).resolve().parents[1] / "docs" / "MODELS.md"
        text = path.read_text().lower()
        for provider in {m.provider for m in MODELS}:
            self.assertIn(provider, text, f"provider {provider} missing from docs/MODELS.md")


if __name__ == "__main__":
    unittest.main()
