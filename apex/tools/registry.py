"""Tool registry — discovery, permission checks, sandboxed execution.

Every tool call flows through here:
  1. lookup (unknown tool -> ToolNotFoundError)
  2. permission check (dangerous tools need approval unless auto-approved)
  3. sandbox path/command validation
  4. timed execution
  5. output truncation (context protection)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import ToolNotFoundError, ToolPermissionError
from ..types import ToolCall, ToolResult
from .base import Tool


class ToolRegistry:
    def __init__(
        self,
        sandbox_root: str = ".",
        require_approval_for: list[str] | None = None,
        auto_approve: bool = False,
        max_output_chars: int = 20_000,
        command_timeout: float = 30.0,
    ):
        self.tools: dict[str, Tool] = {}
        self.sandbox_root = Path(sandbox_root).resolve()
        self.require_approval_for = set(require_approval_for or [])
        self.auto_approve = auto_approve
        self.max_output_chars = max_output_chars
        self.command_timeout = command_timeout
        self.approval_handler: Optional[Callable[[str, dict[str, Any]], bool]] = None

    # -- registration ------------------------------------------------------
    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ToolNotFoundError(f"unknown tool: {name}. Available: {sorted(self.tools)}")
        return self.tools[name]

    def names(self) -> list[str]:
        return sorted(self.tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    # -- sandbox ------------------------------------------------------------
    def resolve_in_sandbox(self, path: str) -> Path:
        """Resolve a user path and guarantee it stays inside the sandbox root."""
        candidate = (self.sandbox_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            candidate.relative_to(self.sandbox_root)
        except ValueError:
            raise ToolPermissionError(f"path escapes sandbox: {path}") from None
        return candidate

    # -- execution -----------------------------------------------------------
    def execute(self, call: ToolCall) -> ToolResult:
        try:
            tool = self.get(call.name)
        except ToolNotFoundError as exc:
            # A hallucinated tool name must not abort the run: report it as a
            # failed observation so the model can choose another action.
            return ToolResult(
                tool_call_id=call.id, tool_name=call.name, output="",
                success=False, error=str(exc),
            )
        needs_approval = call.name in self.require_approval_for or tool.dangerous
        if needs_approval and not self.auto_approve:
            # No handler means no approval: deny rather than silently allowing
            # a privileged tool. The previous default was "approved = True",
            # which made the gate decorative.
            approved = (
                self.approval_handler(call.name, call.arguments)
                if self.approval_handler is not None
                else False
            )
            if not approved:
                return ToolResult(
                    tool_call_id=call.id, tool_name=call.name, output="",
                    success=False, error=f"tool '{call.name}' requires approval and was denied",
                )
        result = tool.run(call.id, call.arguments, timeout=self.command_timeout)
        if len(result.output) > self.max_output_chars:
            result.output = result.output[: self.max_output_chars] + "\n...[truncated]"
        return result

    def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        return [self.execute(c) for c in calls]


def default_registry(sandbox_root: str = ".", **kwargs) -> ToolRegistry:
    """Build a registry pre-loaded with all builtin tools."""
    from .builtin import all_tools

    reg = ToolRegistry(sandbox_root=sandbox_root, **kwargs)
    reg.register_all(all_tools(sandbox_root))
    return reg
