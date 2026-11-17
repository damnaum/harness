"""APEX — Advanced Peak-level Execution eXperience.

A production-grade agentic harness: multi-model routing over the 2026 frontier,
multi-agent swarm orchestration, 8 reasoning strategies, sandboxed tools,
episodic/semantic/procedural memory, guardrails, tracing and evals — all on
the Python standard library with zero hard dependencies.

Quick start:
    from apex import Apex
    result = Apex(profile="offline").run("Explain what an API is")
    print(result.output)
"""

from .config import ApexConfig
from .harness import ApexHarness
from .sdk import Apex
from .types import HarnessResult, ReasoningStrategy, TaskType

__version__ = "1.0.0"
__all__ = ["Apex", "ApexHarness", "ApexConfig", "HarnessResult", "ReasoningStrategy", "TaskType"]
