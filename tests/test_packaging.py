"""Packaging integrity.

A pyproject.toml that parses as TOML can still be wrong: a stray section header
once absorbed ``dependencies`` into ``[project.urls]``, which broke
``pip install .`` for every user while every in-repo test kept passing (the
tests import from the working tree, not from an installed wheel).
"""

from __future__ import annotations

import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPyproject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        cls.project = cls.data["project"]

    def test_dependencies_is_a_list_of_strings(self):
        deps = self.project["dependencies"]
        self.assertIsInstance(deps, list, "dependencies must be a top-level project list")
        self.assertEqual(deps, [], "the core declares zero hard dependencies")
        for dep in deps:
            self.assertIsInstance(dep, str)

    def test_urls_are_all_strings(self):
        for key, value in (self.project.get("urls") or {}).items():
            self.assertIsInstance(value, str, f"project.urls.{key} must be a string")
            self.assertTrue(value.startswith("http"), key)

    def test_metadata_contract(self):
        self.assertEqual(self.project["name"], "apex-harness")
        self.assertTrue(self.project["version"])
        self.assertEqual(self.project["requires-python"], ">=3.10")
        self.assertIn("MIT", str(self.project["license"]))

    def test_readme_exists(self):
        self.assertTrue((ROOT / self.project["readme"]).is_file())

    def test_entry_point_is_importable(self):
        script = self.project["scripts"]["apex"]
        module_name, func_name = script.split(":")
        module = __import__(module_name, fromlist=[func_name])
        self.assertTrue(callable(getattr(module, func_name)))

    def test_version_is_single_sourced(self):
        import apex

        self.assertEqual(self.project["version"], apex.__version__)

    def test_packages_find_includes_apex(self):
        include = self.data["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertIn("apex*", include)

    def test_no_optional_dependency_is_required(self):
        optional = self.project.get("optional-dependencies", {})
        for name, deps in optional.items():
            self.assertIsInstance(deps, list, name)

    def test_py_typed_marker_present(self):
        self.assertTrue((ROOT / "apex" / "py.typed").is_file(), "ship py.typed for type checkers")


class TestArtifacts(unittest.TestCase):
    def test_dockerfile_is_present_and_non_root(self):
        dockerfile = ROOT / "Dockerfile"
        self.assertTrue(dockerfile.is_file())
        text = dockerfile.read_text(encoding="utf-8")
        self.assertIn("USER apex", text)
        self.assertNotIn("USER root", text)
        self.assertIn("HEALTHCHECK", text)
        self.assertIn("ENTRYPOINT [\"apex\"]", text)

    def test_dockerignore_excludes_state(self):
        text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for pattern in (".git", ".apex", "__pycache__"):
            self.assertIn(pattern, text)

    def test_license_and_docs(self):
        self.assertTrue((ROOT / "LICENSE").is_file())
        for doc in ("SECURITY.md", "CHANGELOG.md", "docs/SECURITY.md", "docs/OPERATIONS.md"):
            self.assertTrue((ROOT / doc).is_file(), doc)


if __name__ == "__main__":
    unittest.main()
