"""Configuration layering and SDK facade tests.

Configuration precedence is easy to get subtly wrong (a profile silently
overwriting an explicit file value is a classic), so it is pinned here.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from apex.config import ApexConfig
from apex.sdk import Apex, ApexError


class TestConfigLayering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "apex.json"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, data: dict) -> str:
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return str(self.path)

    def test_file_overrides_profile_defaults(self):
        path = self.write({"profile": "quality", "agents": {"max_steps": 77}})
        cfg = ApexConfig.load(path)
        self.assertEqual(cfg.profile, "quality")
        self.assertEqual(cfg.agents.max_steps, 77, "explicit file value must beat the profile")

    def test_profile_supplies_defaults(self):
        path = self.write({"profile": "quality", "agents": {}})
        cfg = ApexConfig.load(path)
        self.assertEqual(cfg.models.router_mode, "quality")
        self.assertEqual(cfg.agents.max_steps, 40)

    def test_env_beats_file(self):
        path = self.write({"agents": {"max_steps": 77}})
        os.environ["APEX_MAX_STEPS"] = "11"
        try:
            self.assertEqual(ApexConfig.load(path).agents.max_steps, 11)
        finally:
            del os.environ["APEX_MAX_STEPS"]

    def test_env_profile(self):
        os.environ["APEX_PROFILE"] = "offline"
        try:
            cfg = ApexConfig.load()
            self.assertEqual(cfg.profile, "offline")
            self.assertEqual(cfg.models.default_model, "mock/apex-mock")
        finally:
            del os.environ["APEX_PROFILE"]

    def test_env_booleans(self):
        os.environ["APEX_MEMORY_PERSIST"] = "false"
        try:
            self.assertFalse(ApexConfig.load().memory.persist)
        finally:
            del os.environ["APEX_MEMORY_PERSIST"]

    def test_yaml_config_supported(self):
        path = Path(self.tmp) / "apex.yaml"
        path.write_text(
            "profile: fast\nagents:\n  max_steps: 12\nmodels:\n  router_mode: cheapest\n",
            encoding="utf-8",
        )
        cfg = ApexConfig.load(str(path))
        self.assertEqual(cfg.agents.max_steps, 12)
        self.assertEqual(cfg.models.router_mode, "cheapest")

    def test_missing_file_raises(self):
        from apex.errors import ConfigError

        with self.assertRaises(ConfigError):
            ApexConfig.load(str(Path(self.tmp) / "absent.json"))

    def test_validation_reports_all_problems(self):
        cfg = ApexConfig(profile="offline")
        cfg.profile = "nope"
        cfg.models.router_mode = "turbo"
        cfg.agents.max_steps = 0
        cfg.context.reserve_for_output = 10**9
        problems = cfg.validate()
        self.assertGreaterEqual(len(problems), 4)
        self.assertTrue(any("profile" in p for p in problems))
        self.assertTrue(any("router_mode" in p for p in problems))
        self.assertTrue(any("max_steps" in p for p in problems))

    def test_harness_creates_missing_workspace(self):
        from apex.harness import ApexHarness

        target = Path(self.tmp) / "fresh" / "sandbox"
        ApexHarness(ApexConfig(profile="offline", workspace=str(target)))
        self.assertTrue(target.exists())

    def test_roundtrip_to_dict(self):
        cfg = ApexConfig(profile="offline")
        restored = ApexConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.agents.max_steps, cfg.agents.max_steps)
        self.assertEqual(restored.profile, "offline")


class TestSdk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.apex = Apex(profile="offline", workspace=self.tmp)

    def tearDown(self):
        import shutil

        self.apex.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run(self):
        result = self.apex.run("What is 8 * 8?")
        self.assertTrue(result.success)
        self.assertIn("64", result.output)

    def test_run_rejects_empty_task(self):
        with self.assertRaises(ApexError):
            self.apex.run("   ")

    def test_strategy_by_name(self):
        result = self.apex.run("Summarise the trade-offs", strategy="direct")
        self.assertEqual(result.strategy.value, "direct")

    def test_bad_strategy_message_lists_options(self):
        with self.assertRaises(ApexError) as ctx:
            self.apex.run("x", strategy="telepathy")
        self.assertIn("direct", str(ctx.exception))

    def test_remember_and_recall(self):
        self.apex.remember("the release train leaves on Fridays")
        self.assertIn("Fridays", " ".join(self.apex.recall("when does the release train leave")))

    def test_remember_rejects_empty(self):
        with self.assertRaises(ApexError):
            self.apex.remember("  ")

    def test_route_includes_classification(self):
        payload = self.apex.route("Prove that sqrt(2) is irrational")
        self.assertEqual(payload["classification"]["task_type"], "reasoning")
        self.assertEqual(payload["model"]["id"], "mock/apex-mock")

    def test_models_listing(self):
        self.assertGreater(len(self.apex.models()), 20)

    def test_doctor_reports_state(self):
        doctor = self.apex.doctor()
        self.assertTrue(doctor["degraded_offline"])
        self.assertIn("mock", doctor["available_providers"])
        self.assertEqual(doctor["config_problems"], [])

    def test_context_manager_closes_and_persists(self):
        with Apex(profile="offline", workspace=self.tmp) as apex:
            apex.remember("persist this fact")
        text = (Path(self.tmp) / ".apex" / "memory.json").read_text()
        self.assertIn("persist this fact", text)

    def test_eval_via_sdk(self):
        from apex.eval.benchmark import DEFAULT_SUITE

        report = self.apex.eval()
        self.assertEqual(report["total"], len(DEFAULT_SUITE))
        self.assertGreaterEqual(report["pass_rate"], 0.8)


class TestDegradation(unittest.TestCase):
    def test_no_keys_routes_to_offline_provider(self):
        from unittest import mock

        from apex.harness import ApexHarness

        keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY",
                "XAI_API_KEY", "QWEN_API_KEY", "MISTRAL_API_KEY", "MOONSHOT_API_KEY",
                "ZHIPU_API_KEY", "OLLAMA_HOST")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with mock.patch.dict(os.environ, clean, clear=True):
            tmp = tempfile.mkdtemp()
            try:
                from apex.models.router import classify_task

                h = ApexHarness(ApexConfig(workspace=tmp))
                self.assertTrue(h.degraded)
                decision = h.router.route(classify_task("Prove a theorem"))
                self.assertEqual(decision.model.provider, "mock")
                result = h.run("What is 3 * 3?")
                self.assertTrue(result.success)
                self.assertTrue(any("offline" in w for w in result.warnings))
            finally:
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)

    def test_key_enables_provider(self):
        from apex.harness import ApexHarness

        tmp = tempfile.mkdtemp()
        try:
            h = ApexHarness(ApexConfig(workspace=tmp), api_keys={"openai": "sk-test"})
            self.assertIn("openai", h.available_providers)
            self.assertFalse(h.degraded)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
