"""Config-file parsing and application.

The config file is the layer operators touch at 3am. Its failure modes matter
more than its features:

* a silently misparsed list (``require_approval_for`` becoming a dict) means a
  security setting is loaded but never applied;
* a comment not stripped means ``auto_approve_tools: false   # deny`` becomes the
  truthy string ``'false   # deny'`` — the exact opposite of what was written;
* a quoted number means ``validate()`` crashes instead of reporting the problem.

All three were real. These tests exist to keep them fixed.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from apex.config import ApexConfig, _parse_simple_yaml, _read_config_file
from apex.errors import ConfigError


class TestYamlSubset(unittest.TestCase):
    def test_nested_map(self):
        data = _parse_simple_yaml("models:\n  router_mode: quality\n  max_retries: 5\n")
        self.assertEqual(data, {"models": {"router_mode": "quality", "max_retries": 5}})

    def test_block_list(self):
        data = _parse_simple_yaml("safety:\n  require_approval_for:\n    - bash\n    - write_file\n")
        self.assertEqual(data["safety"]["require_approval_for"], ["bash", "write_file"])

    def test_inline_list(self):
        data = _parse_simple_yaml("models:\n  fallback_chain: [a/b, c/d]\n")
        self.assertEqual(data["models"]["fallback_chain"], ["a/b", "c/d"])

    def test_inline_comment_is_stripped(self):
        data = _parse_simple_yaml("safety:\n  auto_approve_tools: false   # deny by default\n")
        self.assertIs(data["safety"]["auto_approve_tools"], False)

    def test_hash_inside_quotes_is_kept(self):
        data = _parse_simple_yaml('workspace: "/tmp/a#b"\n')
        self.assertEqual(data["workspace"], "/tmp/a#b")

    def test_urls_survive_colons(self):
        data = _parse_simple_yaml("models:\n  base_url: https://api.example.com/v1\n")
        self.assertEqual(data["models"]["base_url"], "https://api.example.com/v1")

    def test_booleans_and_null(self):
        data = _parse_simple_yaml(
            "a: true\nb: no\nc: on\nd: off\ne: null\nf: 1.5\ng: 7\n"
        )
        self.assertEqual(
            (data["a"], data["b"], data["c"], data["d"], data["e"], data["f"], data["g"]),
            (True, False, True, False, None, 1.5, 7),
        )

    def test_comments_and_blank_lines_ignored(self):
        data = _parse_simple_yaml("# header\n\nprofile: offline\n\n# footer\n")
        self.assertEqual(data, {"profile": "offline"})

    def test_empty_document(self):
        self.assertEqual(_parse_simple_yaml(""), {})

    def test_missing_colon_is_an_error(self):
        with self.assertRaises(ConfigError):
            _parse_simple_yaml("profile offline\n")

    def test_odd_indentation_is_an_error(self):
        with self.assertRaises(ConfigError):
            _parse_simple_yaml("models:\n   router_mode: quality\n")

    def test_indented_top_level_is_an_error(self):
        with self.assertRaises(ConfigError):
            _parse_simple_yaml("  profile: offline\n")

    def test_list_item_where_map_expected_is_an_error(self):
        with self.assertRaises(ConfigError):
            _parse_simple_yaml("profile: offline\n- stray\n")

    def test_nested_mapping_in_list_is_rejected_explicitly(self):
        with self.assertRaises(ConfigError):
            _parse_simple_yaml("safety:\n  require_approval_for:\n    - name: bash\n")


class TestConfigApplication(unittest.TestCase):
    def _load(self, text: str) -> ApexConfig:
        return ApexConfig.from_dict(_parse_simple_yaml(text))

    def test_quoted_number_becomes_int(self):
        cfg = self._load('agents:\n  max_steps: "25"\n')
        self.assertEqual(cfg.agents.max_steps, 25)
        self.assertIsInstance(cfg.agents.max_steps, int)
        self.assertEqual([p for p in cfg.validate() if "max_steps" in p], [])

    def test_string_boolean_is_coerced(self):
        cfg = self._load("safety:\n  auto_approve_tools: 'false'\n")
        self.assertIs(cfg.safety.auto_approve_tools, False)

    def test_wrong_boolean_is_reported(self):
        cfg = self._load("safety:\n  auto_approve_tools: maybe\n")
        self.assertTrue(any("auto_approve_tools" in p for p in cfg.validate()))

    def test_unknown_setting_is_reported(self):
        cfg = self._load("safety:\n  auto_approve_tool: false\n")
        self.assertTrue(any("unknown setting" in p and "auto_approve_tool" in p for p in cfg.validate()))

class TestConfigApplicationMore(unittest.TestCase):
    def test_unknown_top_level_key_is_reported(self):
        cfg = ApexConfig.from_dict(_parse_simple_yaml("model:\n  router_mode: quality\n"))
        self.assertTrue(any("unknown config key 'model'" in p for p in cfg.validate()))

    def test_float_cap(self):
        cfg = ApexConfig.from_dict(_parse_simple_yaml("models:\n  max_cost_usd: 2\n"))
        self.assertEqual(cfg.models.max_cost_usd, 2.0)
        self.assertIsInstance(cfg.models.max_cost_usd, float)

    def test_list_field_from_comma_string(self):
        cfg = ApexConfig.from_dict(_parse_simple_yaml("safety:\n  require_approval_for: bash, write_file\n"))
        self.assertEqual(cfg.safety.require_approval_for, ["bash", "write_file"])

    def test_section_must_be_mapping(self):
        cfg = ApexConfig.from_dict(_parse_simple_yaml("safety: true\n"))
        self.assertTrue(any("safety" in p and "mapping" in p for p in cfg.validate()))

    def test_profile_from_file_applies_then_explicit_wins(self):
        cfg = ApexConfig.from_dict(_parse_simple_yaml("profile: prod\nmodels:\n  router_mode: fast\n"))
        self.assertIs(cfg.safety.auto_approve_tools, False)  # from prod
        self.assertEqual(cfg.models.router_mode, "fast")     # explicit beats profile

    def test_good_file_has_no_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ApexConfig.from_dict(
                _parse_simple_yaml(
                    f"profile: prod\nworkspace: {tmp}\n"
                    "safety:\n  require_approval_for: [bash, write_file]\n"
                    "models:\n  router_mode: quality\n  max_cost_usd: 1.5\n"
                )
            )
            self.assertEqual(cfg.validate(), [])


class TestStrictness(unittest.TestCase):
    def test_strict_harness_refuses_bad_config(self):
        from apex.harness import ApexHarness

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ApexConfig.from_dict(_parse_simple_yaml("agents:\n  max_steps: lots\n"))
            cfg.workspace = tmp
            with self.assertRaises(ValueError):
                ApexHarness(cfg)

    def test_non_strict_harness_reports_instead_of_raising(self):
        from apex.harness import ApexHarness

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ApexConfig.from_dict(_parse_simple_yaml("agents:\n  max_steps: lots\n"))
            cfg.workspace = tmp
            harness = ApexHarness(cfg, strict=False)
            self.assertTrue(harness.config_problems)


class TestConfigFileOnDisk(unittest.TestCase):
    def test_yaml_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "apex.yaml"
            path.write_text(
                "profile: offline\n"
                "safety:\n"
                "  auto_approve_tools: false   # comment survives\n"
                "  require_approval_for:\n"
                "    - bash\n",
                encoding="utf-8",
            )
            data = _read_config_file(path)
            self.assertIs(data["safety"]["auto_approve_tools"], False)
            self.assertEqual(data["safety"]["require_approval_for"], ["bash"])

    def test_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "apex.json"
            path.write_text('{"profile": "offline", "agents": {"max_steps": 9}}', encoding="utf-8")
            cfg = ApexConfig.from_file(path)
            self.assertEqual(cfg.agents.max_steps, 9)

    def test_missing_file_raises_config_error(self):
        with self.assertRaises(ConfigError):
            _read_config_file("/nonexistent/apex.yaml")

    def test_non_mapping_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "apex.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ConfigError):
                _read_config_file(path)

    def test_doctor_survives_a_bad_config(self):
        from apex.harness import ApexHarness

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ApexConfig.from_dict(_parse_simple_yaml("agents:\n  max_steps: lots\n"))
            cfg.workspace = tmp
            report = ApexHarness(cfg, strict=False).doctor()   # must not raise
            self.assertTrue(any("max_steps" in p for p in report["config_problems"]))


if __name__ == "__main__":
    unittest.main()
