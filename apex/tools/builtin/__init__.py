"""Builtin tools — filesystem, shell, python, calculator, web, git, search.

All filesystem tools are confined to the sandbox root.  Shell/python tools are
marked dangerous (approval-gated) and run with timeouts + restricted env.
"""

from __future__ import annotations

import ast
import operator
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from ...errors import ToolError
from ..base import Tool
from ..sandbox import (
    check_command,
    check_url,
    guard_power,
    read_limited,
    restricted_env,
    safe_path,
)


# ---------------------------------------------------------------- helpers
def _required(arguments: dict[str, Any], name: str) -> Any:
    """Fetch a required argument with a message a model can act on.

    ``arguments["path"]`` on a missing key raises ``KeyError('path')``, which the
    loop renders as ``ERROR: 'path'`` — technically correct, useless to a model.
    """
    if name not in arguments:
        raise ValueError(f"missing required argument '{name}'")
    return arguments[name]


def _text_of(path: Path, limit: int = 200_000) -> str:
    """Read a file as text, refusing binaries and capping the size."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        head = handle.read(8192)
    if b"\x00" in head:
        raise ValueError(
            f"binary file ({size} bytes) — read_file handles text only; "
            "use python_exec or bash if you need the bytes"
        )
    try:
        head.decode("utf-8")
        encoding_ok = True
    except UnicodeDecodeError:
        encoding_ok = False
    if not encoding_ok and size > 0:
        # Not UTF-8 in the first block: latin-1 cannot fail, but say so.
        text = path.read_bytes()[:limit].decode("latin-1")
        return text + ("\n...[truncated]" if size > limit else "")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated: {len(text) - limit} more chars]"
    return text


# ---------------------------------------------------------------- filesystem
class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file inside the workspace. Returns its content."
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]}

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def execute(self, arguments):
        raw_path = _required(arguments, "path")
        path = safe_path(self.root, raw_path)
        if not path.exists():
            raise FileNotFoundError(f"no such file: {raw_path}")
        if not path.is_file():
            raise ValueError(f"not a file: {raw_path}")
        return _text_of(path)


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write text content to a file inside the workspace (creates parent dirs)."
    dangerous = True
    parameters = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def execute(self, arguments):
        raw_path = _required(arguments, "path")
        content = _required(arguments, "content")
        if not isinstance(content, str):
            raise ValueError("'content' must be a string")
        path = safe_path(self.root, raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {raw_path}"


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories in a workspace directory."
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path (default '.')"}}}

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def execute(self, arguments):
        path = safe_path(self.root, arguments.get("path", "."))
        if not path.exists():
            raise FileNotFoundError(f"no such directory: {path}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return "\n".join(entries) or "(empty)"


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Grep-like search for a pattern across workspace text files."
    parameters = {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Subdirectory to search (default '.')"}}, "required": ["pattern"]}

    def __init__(self, root: str):
        self.root = Path(root).resolve()

    MAX_FILE_BYTES = 2_000_000  # never load a log/artifact dump into memory
    MAX_FILES = 5_000

    def execute(self, arguments):
        pattern = _required(arguments, "pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("'pattern' must be a non-empty string")
        if len(pattern) > 500:
            raise ValueError("pattern too long (max 500 chars)")
        base = safe_path(self.root, arguments.get("path", "."))
        if not base.is_dir():
            raise FileNotFoundError(f"no such directory: {arguments.get('path', '.')}")
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        hits = []
        scanned = 0
        for f in base.rglob("*"):
            if scanned >= self.MAX_FILES:
                hits.append(f"...[stopped after {self.MAX_FILES} files]")
                break
            if not f.is_file() or ".git" in f.parts:
                continue
            if f.suffix not in (".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".js", ".ts"):
                continue
            try:
                if f.stat().st_size > self.MAX_FILE_BYTES:
                    continue
                scanned += 1
                with open(f, encoding="utf-8", errors="replace") as handle:
                    for i, line in enumerate(handle, 1):
                        if rx.search(line):
                            hits.append(f"{f.relative_to(self.root)}:{i}: {line.strip()[:160]}")
                            if len(hits) >= 50:
                                return "\n".join(hits) + "\n...[more]"
            except OSError:
                continue
        return "\n".join(hits) if hits else "(no matches)"


# ---------------------------------------------------------------- shell / python


class BashTool(Tool):
    name = "bash"
    description = "Run a shell command in the workspace with a timeout. Dangerous: approval-gated."
    dangerous = True
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    def __init__(self, root: str, timeout: float = 30.0):
        self.root = root
        self.timeout = timeout

    def execute(self, arguments):
        cmd = _required(arguments, "command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("command must be a non-empty string")
        check_command(cmd)
        proc = subprocess.run(
            cmd, shell=True, cwd=self.root, capture_output=True, text=True,
            timeout=self.timeout, env=restricted_env(),
        )
        out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
        out = out.strip()
        if proc.returncode != 0:
            # A non-zero exit is a failure the model has to react to. Reporting it
            # as success made agents retry the same broken command.
            raise ToolError(f"exit={proc.returncode}\n{out}" if out else f"exit={proc.returncode}")
        return f"exit=0\n{out}" if out else "exit=0"


class PythonExecTool(Tool):
    name = "python_exec"
    description = "Execute a Python snippet and return stdout. Dangerous: approval-gated."
    dangerous = True
    parameters = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}

    def __init__(self, root: str, timeout: float = 30.0):
        self.root = root
        self.timeout = timeout

    def execute(self, arguments):
        code = _required(arguments, "code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        proc = subprocess.run(
            ["python3", "-c", code], cwd=self.root, capture_output=True, text=True,
            timeout=self.timeout, env=restricted_env(),
        )
        out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
        out = out.strip()
        if proc.returncode != 0:
            raise ToolError(f"exit={proc.returncode}\n{out}" if out else f"exit={proc.returncode}")
        return f"exit=0\n{out}" if out else "exit=0"


# ---------------------------------------------------------------- calculator (safe AST eval)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_FUNCS = {
    "sqrt": lambda x: x ** 0.5 if x >= 0 else (_ for _ in ()).throw(ValueError("sqrt of negative")),
    "abs": abs, "min": min, "max": max, "round": round,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            return guard_power(left, right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        result = _OPS[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > 1e100:
            raise ValueError("result out of range")
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Safely evaluate a math expression like '12*8+5' or 'sqrt(144)'. No code execution."
    parameters = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}

    def execute(self, arguments):
        expr = str(_required(arguments, "expression")).strip()
        result = _eval(ast.parse(expr, mode="eval"))
        return f"{expr} = {result}"


# ---------------------------------------------------------------- web / git / search
class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return its text content (truncated)."
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    def execute(self, arguments):
        url = check_url(_required(arguments, "url"))
        req = urllib.request.Request(url, headers={"User-Agent": "APEX-harness/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = read_limited(resp, 1_000_000).decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web (via DuckDuckGo HTML endpoint) and return top results."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def execute(self, arguments):
        import urllib.parse

        q = urllib.parse.quote(str(_required(arguments, "query")))
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "APEX-harness/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = read_limited(resp, 2_000_000).decode("utf-8", errors="replace")
        results = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html)
        lines = []
        for i, title in enumerate(results[:5]):
            title = re.sub(r"<[^>]+>", "", title)
            snip = re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else ""
            lines.append(f"{i + 1}. {title} — {snip.strip()}")
        return "\n".join(lines) if lines else "(no results)"


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show git status + recent commits in the workspace."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, root: str):
        self.root = root

    def execute(self, arguments):
        def run(args):
            p = subprocess.run(["git"] + args, cwd=self.root, capture_output=True, text=True, timeout=10)
            return p.stdout.strip() if p.returncode == 0 else f"(git {' '.join(args)} failed: {p.stderr.strip()[:200]})"

        return f"--- status ---\n{run(['status', '--short'])}\n--- log ---\n{run(['log', '--oneline', '-5'])}"


def all_tools(root: str, command_timeout: float = 30.0) -> list[Tool]:
    return [
        ReadFileTool(root),
        WriteFileTool(root),
        ListDirTool(root),
        SearchFilesTool(root),
        BashTool(root, command_timeout),
        PythonExecTool(root, command_timeout),
        CalculatorTool(),
        WebFetchTool(),
        WebSearchTool(),
        GitStatusTool(root),
    ]
