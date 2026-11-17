"""Documentation has to stay true to the code.

Docs drift silently, and a production README that advertises the wrong test count
or a feature that was removed is worse than no README: it erodes trust in every
other claim. These tests bind the documents to the code they describe.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

try:  # `python -m unittest discover -s tests`
    from toml_compat import loads as _toml_loads
except ModuleNotFoundError:  # `python -m unittest tests.test_docs_consistency`
    from tests.toml_compat import loads as _toml_loads


def _pyproject() -> dict:
    return _toml_loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _suite_size() -> int:
    import unittest as ut

    def count(suite):
        return sum(count(item) for item in suite) if isinstance(suite, ut.TestSuite) else 1

    return count(ut.defaultTestLoader.discover(str(ROOT / "tests")))


class TestReadme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_test_count_matches_the_suite(self):
        claimed = {int(m) for m in re.findall(r"(\d{3,4}) tests", self.readme)}
        self.assertTrue(claimed, "README no longer states a test count")
        self.assertEqual(
            claimed, {_suite_size()},
            f"README claims {sorted(claimed)} tests; the suite collects {_suite_size()}",
        )

    def test_no_stale_feature_counts(self):
        stale = ["8-task", "eight tasks", "189 tests", "216 tests", "239 tests", "267 tests",
                 "303 tests", "312 tests", "426 tests", "457 tests", "492 tests"]
        for phrase in stale:
            self.assertNotIn(phrase, self.readme, f"stale phrase in README: {phrase}")

    def test_model_count_matches_registry(self):
        from apex.models.registry import MODELS

        claimed = {int(m) for m in re.findall(r"(\d+) models", self.readme)}
        if claimed:
            self.assertEqual(claimed, {len(MODELS)})

    def test_every_doc_link_resolves(self):
        for target in re.findall(r"\]\((docs/[^)]+)\)", self.readme):
            self.assertTrue((ROOT / target).is_file(), f"broken link: {target}")


class TestDocs(unittest.TestCase):
    def test_architecture_lists_every_module(self):
        architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        modules = sorted(
            p.name for p in (ROOT / "apex").iterdir()
            if p.is_dir() and not p.name.startswith("__")
        )
        for module in modules:
            self.assertIn(f"apex/{module}", architecture, f"{module}/ missing from the module map")

    def test_strategy_doc_matches_the_specs(self):
        from apex.reasoning.strategies import SPECS

        raw = (ROOT / "docs" / "STRATEGY.md").read_text(encoding="utf-8").lower()
        # Normalise separators: the doc writes "chain-of-thought", the enum is
        # CHAIN_OF_THOUGHT.
        doc = re.sub(r"[^a-z0-9]", "", raw)
        for strategy in SPECS:
            canonical = re.sub(r"[^a-z0-9]", "", strategy.value.lower())
            self.assertIn(canonical, doc, f"{strategy.value} missing from docs/STRATEGY.md")

    def test_changelog_version_matches_code(self):
        import apex

        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"[{apex.__version__}]", changelog)

    def test_security_doc_lists_the_defended_surface(self):
        security = (ROOT / "docs" / "SECURITY.md").read_text(encoding="utf-8").lower()
        for topic in ("injection", "ssrf", "approval", "sandbox", "redact"):
            self.assertIn(topic, security, topic)

    def test_operations_documents_every_env_var(self):
        import inspect

        from apex import config

        source = inspect.getsource(config)
        documented = (ROOT / "docs" / "OPERATIONS.md").read_text(encoding="utf-8")
        env_vars = set(re.findall(r'"(MAX_STEPS|PLAN_STEPS|WORKSPACE|MODEL|ROUTER_MODE|'
                                   r'CHECKPOINT_DIR|SESSION_ID|LOG_LEVEL|TRACE|MEMORY_PERSIST|'
                                   r'MAX_COST_USD)"', source))
        missing = [name for name in env_vars if f"APEX_{name}" not in documented]
        self.assertEqual(missing, [], f"undocumented env vars: {missing}")

    def test_api_doc_covers_every_route(self):
        import inspect

        from apex.server import api

        doc = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
        source = inspect.getsource(api)
        routes = set(re.findall(r'path == "(/[^"]+)"', source))
        missing = [route for route in routes if route not in doc]
        self.assertEqual(missing, [], f"undocumented routes: {missing}")

    def test_license_file_matches_metadata(self):
        project = _pyproject()["project"]
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT", str(project["license"]))
        self.assertIn("MIT", license_text)


if __name__ == "__main__":
    unittest.main()
