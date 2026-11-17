"""ReAct execution loop + retry/backoff + circuit breaker + checkpointing.

The loop:  model -> (tool calls?) -> execute tools -> feed observations back
-> repeat, until the model answers without tool calls, the step budget is
exhausted, a loop is detected, or the circuit breaker opens.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import CircuitOpenError, LoopDetectedError, MaxStepsExceededError
from ..types import Message, Role, ToolCall


class RetryPolicy:
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 10.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def delay(self, attempt: int) -> float:
        return min(self.max_delay, self.base_delay * (2**attempt)) + random.uniform(0, 0.25)

    def run(self, fn: Callable, *args, **kwargs):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.delay(attempt))
        raise last_exc


class CircuitBreaker:
    """Opens after N consecutive failures; half-opens after a cooldown."""

    def __init__(self, failure_threshold: int = 5, cooldown: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at: Optional[float] = None

    def call(self, fn: Callable, *args, **kwargs):
        if self.opened_at is not None:
            if time.time() - self.opened_at < self.cooldown:
                raise CircuitOpenError("circuit breaker is open")
            self.opened_at = None  # half-open: allow one trial
            self.failures = 0
        try:
            result = fn(*args, **kwargs)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.time()
            raise


class Checkpointer:
    """JSON checkpointing of loop state so long runs can resume."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, state: dict[str, Any]) -> Path:
        path = self.directory / f"{run_id}.json"
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return path

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.directory / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))


def _fingerprint(call: ToolCall) -> str:
    raw = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class ReActLoop:
    """Drives one agent's think/act/observe cycle.

    ``model_fn(messages, tools) -> ModelResponse`` is injected so the loop is
    provider-agnostic and unit-testable with a fake model function.
    """

    def __init__(
        self,
        model_fn: Callable,
        tool_executor: Callable[[ToolCall], Any],
        max_steps: int = 25,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        context_manager: Any = None,
        guardrails: Any = None,
    ):
        self.model_fn = model_fn
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.on_event = on_event or (lambda *a: None)
        self.context_manager = context_manager
        self.guardrails = guardrails
        self.seen: list[str] = []

    def run(self, system: str, user: str, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        messages = [Message(role=Role.SYSTEM, content=system), Message(role=Role.USER, content=user)]
        tool_calls_made: list[ToolCall] = []
        total_in = total_out = 0
        total_cost = 0.0
        final_text = ""

        for step in range(self.max_steps):
            # Compact before every call: tool observations accumulate fast.
            view = self.context_manager.fit_messages(messages) if self.context_manager else messages
            resp = self.model_fn(view, tools)
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            total_cost += resp.cost_usd
            self.on_event("model_response", {"step": step, "model": resp.model_id, "tokens_out": resp.tokens_out})

            if not resp.tool_calls:
                final_text = resp.content
                break

            # Record the assistant turn *with* its tool calls: providers that
            # support native tool calling require the call to be echoed back
            # before the matching result, or they reject the request.
            messages.append(Message(
                role=Role.ASSISTANT,
                content=resp.content or "",
                metadata={"tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
                ]},
            ))
            for call in resp.tool_calls:
                fp = _fingerprint(call)
                if self.seen.count(fp) >= 2:
                    raise LoopDetectedError(f"repeated identical tool call: {call.name} {call.arguments}")
                self.seen.append(fp)
                tool_calls_made.append(call)
                self.on_event("tool_call", {"step": step, "tool": call.name, "args": call.arguments})
                try:
                    result = self.tool_executor(call)
                    obs = result.output if result.success else f"ERROR: {result.error}"
                    ok = bool(result.success)
                except Exception as exc:  # noqa: BLE001 - a tool must never kill the loop
                    obs = f"ERROR: tool '{call.name}' failed: {exc}"
                    ok = False

                # Indirect prompt injection: tool output is attacker-controlled
                # data. Flag instructions hidden in it before the model sees it.
                if self.guardrails is not None and isinstance(obs, str):
                    verdict = self.guardrails.check_tool_output(obs, call.name)
                    if not verdict.allowed:
                        obs = verdict.redacted_text
                        self.on_event("injection_in_tool_output", {"tool": call.name, "reasons": verdict.reasons})
                messages.append(
                    Message(role=Role.TOOL, content=obs, name=call.name, tool_call_id=call.id)
                )
                self.on_event("tool_result", {"step": step, "tool": call.name, "success": ok})
        else:
            raise MaxStepsExceededError(f"exceeded {self.max_steps} steps without a final answer")

        return {
            "output": final_text,
            "tool_calls": tool_calls_made,
            "tokens_in": total_in,
            "tokens_out": total_out,
            "cost_usd": total_cost,
            "steps": len(tool_calls_made),
            "messages": messages,
        }
