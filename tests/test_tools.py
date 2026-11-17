import tempfile
import unittest
from pathlib import Path

from apex.tools.builtin import CalculatorTool, ListDirTool, ReadFileTool, SearchFilesTool, WriteFileTool
from apex.tools.registry import ToolRegistry, default_registry
from apex.types import ToolCall


class TestBuiltinTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        Path(self.root, "hello.txt").write_text("hello world\nsecond line")

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_file(self):
        out = ReadFileTool(self.root).execute({"path": "hello.txt"})
        self.assertIn("hello world", out)

    def test_read_file_escape_blocked(self):
        with self.assertRaises(PermissionError):
            ReadFileTool(self.root).execute({"path": "../secret.txt"})

    def test_write_and_read(self):
        WriteFileTool(self.root).execute({"path": "sub/new.txt", "content": "data"})
        self.assertEqual(ReadFileTool(self.root).execute({"path": "sub/new.txt"}), "data")

    def test_list_dir(self):
        out = ListDirTool(self.root).execute({"path": "."})
        self.assertIn("hello.txt", out)

    def test_search_files(self):
        out = SearchFilesTool(self.root).execute({"pattern": "hello"})
        self.assertIn("hello.txt", out)

    def test_calculator(self):
        tool = CalculatorTool()
        self.assertIn("20", tool.execute({"expression": "12+8"}))
        self.assertIn("42", tool.execute({"expression": "7*6"}))
        self.assertIn("12", tool.execute({"expression": "sqrt(144)"}))

    def test_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            CalculatorTool().execute({"expression": "__import__('os').system('id')"})


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = default_registry(sandbox_root=self.tmp.name, auto_approve=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_tools_loaded(self):
        for name in ("read_file", "write_file", "list_dir", "calculator", "bash", "git_status"):
            self.assertIn(name, self.reg.names())

    def test_unknown_tool_lookup_raises(self):
        from apex.errors import ToolNotFoundError

        with self.assertRaises(ToolNotFoundError):
            self.reg.get("nope")

    def test_unknown_tool_execute_returns_failed_result(self):
        # Models hallucinate tool names; the loop must get an observation back,
        # not an exception that kills the run.
        result = self.reg.execute(ToolCall(name="nope", arguments={}))
        self.assertFalse(result.success)
        self.assertIn("unknown tool", result.error)

    def test_execute_calculator(self):
        result = self.reg.execute(ToolCall(name="calculator", arguments={"expression": "2*3"}))
        self.assertTrue(result.success)
        self.assertIn("6", result.output)

    def test_sandbox_escape(self):
        from apex.errors import ToolPermissionError

        with self.assertRaises(ToolPermissionError):
            self.reg.resolve_in_sandbox("/etc/passwd")

    def test_output_truncation(self):
        self.reg.max_output_chars = 10
        result = self.reg.execute(ToolCall(name="list_dir", arguments={"path": "."}))
        self.assertLessEqual(len(result.output), 30)




class TestToolTimeout(unittest.TestCase):
    """A hung tool must not hang the agent loop."""

    def test_timeout_returns_error_result(self):
        import time as _time

        from apex.tools.base import Tool

        class SlowTool(Tool):
            name = "slow"
            description = "sleeps"
            parameters = {"type": "object", "properties": {}}

            def execute(self, arguments):
                _time.sleep(2)
                return "never"

        result = SlowTool().run("id", {}, timeout=0.1)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)
        self.assertLess(result.duration_ms, 1500)

    def test_fast_tool_succeeds(self):
        from apex.tools.base import Tool

        class EchoTool(Tool):
            name = "echo"
            description = "echoes"
            parameters = {"type": "object", "properties": {}}

            def execute(self, arguments):
                return f"echo:{arguments.get('v')}"

        result = EchoTool().run("id", {"v": 7}, timeout=5)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "echo:7")

    def test_non_string_output_is_coerced(self):
        from apex.tools.base import Tool

        class DictTool(Tool):
            name = "dictout"
            description = "returns a dict"
            parameters = {"type": "object", "properties": {}}

            def execute(self, arguments):
                return {"a": 1}

        result = DictTool().run("id", {})
        self.assertTrue(result.success)
        self.assertIsInstance(result.output, str)
        self.assertIn("'a'", result.output)

    def test_exception_becomes_failed_result(self):
        from apex.tools.base import Tool

        class BoomTool(Tool):
            name = "boom"
            description = "raises"
            parameters = {"type": "object", "properties": {}}

            def execute(self, arguments):
                raise RuntimeError("kaboom")

        result = BoomTool().run("id", {})
        self.assertFalse(result.success)
        self.assertIn("kaboom", result.error)


class TestToolEdgeCases(unittest.TestCase):
    """Tools face hostile input from models, not from careful humans."""

    def setUp(self):
        import tempfile as _tempfile

        from apex.config import ApexConfig
        from apex.harness import ApexHarness

        self.tmp = _tempfile.TemporaryDirectory()
        self.harness = ApexHarness(ApexConfig(profile="balanced", workspace=self.tmp.name))
        self.harness.tools.approval_handler = lambda name, args: True
        self.root = __import__("pathlib").Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, name, **arguments):
        from apex.types import ToolCall

        return self.harness.tools.execute(ToolCall(name=name, arguments=arguments))

    def test_binary_file_is_refused_with_guidance(self):
        (self.root / "blob.bin").write_bytes(b"\x00\x01\xff" * 50)
        result = self.call("read_file", path="blob.bin")
        self.assertFalse(result.success)
        self.assertIn("binary", result.error)
        self.assertIn("python_exec", result.error)

    def test_latin1_text_still_reads(self):
        (self.root / "latin.txt").write_bytes(b"caf\xe9 na\xefve")
        result = self.call("read_file", path="latin.txt")
        self.assertTrue(result.success)
        self.assertIn("caf", result.output)

    def test_large_file_is_truncated_not_loadable_whole(self):
        (self.root / "big.txt").write_text("x" * 500_000)
        result = self.call("read_file", path="big.txt")
        self.assertTrue(result.success)
        self.assertIn("truncated", result.output)
        self.assertLess(len(result.output), 400_000)

    def test_directory_is_not_a_file(self):
        (self.root / "a_dir").mkdir()
        result = self.call("read_file", path="a_dir")
        self.assertFalse(result.success)
        self.assertIn("not a file", result.error)

    def test_missing_required_arguments_are_named(self):
        for name, args in (("read_file", {}), ("write_file", {"path": "x"}),
                           ("search_files", {}), ("bash", {}), ("python_exec", {})):
            result = self.call(name, **args)
            self.assertFalse(result.success, name)
            self.assertIn("missing required argument", result.error, name)

    def test_write_file_rejects_non_string_content(self):
        result = self.call("write_file", path="a.txt", content=5)
        self.assertFalse(result.success)
        self.assertIn("string", result.error)

    def test_invalid_regex_is_explained(self):
        result = self.call("search_files", pattern="[unclosed")
        self.assertFalse(result.success)
        self.assertIn("invalid regular expression", result.error)

    def test_search_skips_git_and_binary_suffixes(self):
        (self.root / "code.py").write_text("needle = 1\n")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("needle\n")
        (self.root / "blob.bin").write_bytes(b"needle\x00")
        result = self.call("search_files", pattern="needle")
        self.assertTrue(result.success)
        self.assertIn("code.py", result.output)
        self.assertNotIn(".git", result.output)
        self.assertNotIn("blob.bin", result.output)

    def test_bash_success_reports_exit_zero(self):
        result = self.call("bash", command="echo hi")
        self.assertTrue(result.success)
        self.assertIn("exit=0", result.output)
        self.assertIn("hi", result.output)

    def test_bash_non_zero_exit_is_a_failure_with_output(self):
        result = self.call("bash", command="echo boom >&2; exit 2")
        self.assertFalse(result.success)
        self.assertIn("exit=2", result.error)
        self.assertIn("boom", result.error)

    def test_python_exec_traceback_is_a_failure(self):
        result = self.call("python_exec", code="1/0")
        self.assertFalse(result.success)
        self.assertIn("ZeroDivisionError", result.error)

    def test_python_exec_stdout_on_success(self):
        result = self.call("python_exec", code="print(6*7)")
        self.assertTrue(result.success)
        self.assertIn("42", result.output)

    def test_calculator_rejects_code_execution(self):
        for expression in ("__import__('os').system('id')", "open('/etc/passwd').read()"):
            result = self.call("calculator", expression=expression)
            self.assertFalse(result.success, expression)


if __name__ == "__main__":
    unittest.main()
