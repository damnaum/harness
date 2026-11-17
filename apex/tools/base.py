"""Tool abstraction.  A Tool has a JSON-schema-ish spec (for the model) and an
``execute`` method (for the harness).  All tools are timeout-guarded and
permission-checked by the registry before execution."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from ..types import ToolResult


class Tool(ABC):
    name: str = "tool"
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    dangerous: bool = False  # requires approval / sandbox

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        """Run the tool and return its textual output. Raise on failure."""

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def run(self, tool_call_id: str, arguments: dict[str, Any], timeout: float = 30.0) -> ToolResult:
        """Execute with a hard wall-clock bound.

        Python cannot kill a thread, so a timed-out call is abandoned and the
        loop continues with an error result — the alternative (blocking the
        agent forever on a hung subprocess) is strictly worse.
        """
        start = time.time()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{self.name}")
        try:
            future = pool.submit(self.execute, arguments)
            output = future.result(timeout=timeout)
            if not isinstance(output, str):
                output = str(output)
            return ToolResult(
                tool_call_id=tool_call_id, tool_name=self.name, output=output,
                success=True, duration_ms=(time.time() - start) * 1000,
            )
        except FutureTimeout:
            return ToolResult(
                tool_call_id=tool_call_id, tool_name=self.name, output="",
                success=False, error=f"tool '{self.name}' timed out after {timeout}s",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - tools report, never crash the loop
            return ToolResult(
                tool_call_id=tool_call_id, tool_name=self.name, output="",
                success=False, error=str(exc), duration_ms=(time.time() - start) * 1000,
            )
        finally:
            pool.shutdown(wait=False)
