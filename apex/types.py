"""APEX core types — single source of truth for every layer of the harness.

All modules import from here.  Types are plain dataclasses (stdlib only) so
they serialise cleanly to JSON for tracing, checkpointing and the API server.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskType(str, Enum):
    GENERAL = "general"
    CODE = "code"
    REASONING = "reasoning"
    RESEARCH = "research"
    DATA = "data"
    CREATIVE = "creative"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    WRITING = "writing"


class Complexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReasoningStrategy(str, Enum):
    DIRECT = "direct"
    COT = "chain_of_thought"
    REACT = "react"
    TOT = "tree_of_thoughts"
    REFLEXION = "reflexion"
    SELF_CONSISTENCY = "self_consistency"
    PLAN_EXECUTE = "plan_and_execute"
    LATS = "lats"


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        d = dict(d)
        d["role"] = Role(d.get("role", "user"))
        return cls(**d)


@dataclass
class ToolCall:
    id: str = field(default_factory=lambda: _uid("call_"))
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelInfo:
    """One row of the 2026 model registry (see apex.models.registry)."""

    id: str
    provider: str
    display_name: str
    context_window: int
    max_output: int
    input_price_per_1m: float  # USD
    output_price_per_1m: float  # USD
    reasoning: int = 5  # 0-10 capability scores
    coding: int = 5
    tools: int = 5
    research: int = 5
    creativity: int = 5
    speed: int = 5  # higher = faster
    long_context: int = 5
    reliability: int = 5
    strengths: tuple[str, ...] = ()
    best_for: tuple[str, ...] = ()
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_reasoning_mode: bool = False

    def capability(self, task_type: TaskType) -> int:
        mapping = {
            TaskType.CODE: self.coding,
            TaskType.DEBUGGING: self.coding,
            TaskType.REASONING: self.reasoning,
            TaskType.PLANNING: self.reasoning,
            TaskType.RESEARCH: self.research,
            TaskType.DATA: self.reasoning,
            TaskType.CREATIVE: self.creativity,
            TaskType.WRITING: self.creativity,
            TaskType.GENERAL: max(self.reasoning, self.coding, self.research),
        }
        return mapping.get(task_type, 5)

    def cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_price_per_1m
            + output_tokens / 1_000_000 * self.output_price_per_1m
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strengths"] = list(self.strengths)
        d["best_for"] = list(self.best_for)
        return d


@dataclass
class RoutingDecision:
    model: ModelInfo
    fallbacks: list[ModelInfo] = field(default_factory=list)
    score: float = 0.0
    reason: str = ""
    estimated_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "fallbacks": [m.to_dict() for m in self.fallbacks],
            "score": self.score,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost,
        }


@dataclass
class Task:
    id: str = field(default_factory=lambda: _uid("task_"))
    description: str = ""
    task_type: TaskType = TaskType.GENERAL
    complexity: Complexity = Complexity.SIMPLE
    context: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        d["complexity"] = self.complexity.value
        return d


@dataclass
class PlanStep:
    id: str = field(default_factory=lambda: _uid("step_"))
    title: str = ""
    description: str = ""
    agent: str = "executor"
    depends_on: list[str] = field(default_factory=list)
    tools_hint: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    output: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Plan:
    id: str = field(default_factory=lambda: _uid("plan_"))
    task_id: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    strategy: ReasoningStrategy = ReasoningStrategy.PLAN_EXECUTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "strategy": self.strategy.value,
            "steps": [s.to_dict() for s in self.steps],
        }

    def ready_steps(self) -> list[PlanStep]:
        done = {s.id for s in self.steps if s.status == StepStatus.DONE}
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING and all(d in done for d in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)


@dataclass
class AgentResult:
    agent: str
    output: str
    success: bool = True
    steps_taken: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tool_calls"] = [t.to_dict() for t in self.tool_calls]
        return d


@dataclass
class HarnessResult:
    task_id: str
    output: str
    success: bool
    model_used: str
    strategy: ReasoningStrategy
    plan: Optional[Plan] = None
    agent_results: list[AgentResult] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    trace_id: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "output": self.output,
            "success": self.success,
            "model_used": self.model_used,
            "strategy": self.strategy.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost_usd": self.total_cost_usd,
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
