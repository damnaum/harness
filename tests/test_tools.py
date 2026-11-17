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


if __name__ == "__main__":
    unittest.main()
