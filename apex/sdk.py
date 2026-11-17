"""Programmatic SDK — the simplest way to embed APEX in an application.

    from apex import Apex

    apex = Apex(profile="balanced")
    result = apex.run("Write a Python function to parse CSV files")
    print(result.output, result.total_cost_usd)

    # Pin a strategy or model when you know better than the router:
    apex.run("Prove sqrt(2) is irrational", strategy="self_consistency")
    apex.run("Refactor this module", model="anthropic/claude-sonnet-4.5")

    # Durable memory:
    apex.remember("Our staging cluster is in eu-west-1")

    # Evaluate a change before shipping it:
    report = apex.eval()

The SDK is a thin, stable facade over :class:`apex.harness.ApexHarness`. Use the
harness directly when you need spans, custom agents or a custom tool registry.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .config import ApexConfig
from .harness import ApexHarness
from .types import HarnessResult, ReasoningStrategy


class ApexError(ValueError):
    """Raised for invalid SDK-level arguments (bad strategy name, etc.)."""


class Apex:
    def __init__(
        self,
        profile: str = "balanced",
        config: ApexConfig | None = None,
        api_keys: dict[str, str] | None = None,
        workspace: str = ".",
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        if config is None:
            config = ApexConfig(profile=profile, workspace=workspace)
        self.harness = ApexHarness(config=config, api_keys=api_keys, on_event=on_event)

    # -- lifecycle ------------------------------------------------------
    @classmethod
    def from_env(cls, config_path: str | None = None) -> "Apex":
        """Build from ``APEX_*`` environment variables and an optional config file."""
        config = ApexConfig.load(config_path) if config_path else ApexConfig.from_env()
        return cls(config=config)

    def __enter__(self) -> "Apex":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Flush durable state. Safe to call twice."""
        self.harness._persist_memory()

    # -- core -----------------------------------------------------------
    def run(
        self,
        task: str,
        model: str | None = None,
        strategy: str | ReasoningStrategy | None = None,
    ) -> HarnessResult:
        if not isinstance(task, str) or not task.strip():
            raise ApexError("task must be a non-empty string")
        return self.harness.run(task, model=model, strategy=self._strategy(strategy))

    def _strategy(self, strategy: str | ReasoningStrategy | None) -> ReasoningStrategy | None:
        if strategy is None or isinstance(strategy, ReasoningStrategy):
            return strategy
        try:
            return ReasoningStrategy(str(strategy))
        except ValueError:
            allowed = [s.value for s in ReasoningStrategy]
            raise ApexError(f"unknown strategy '{strategy}'; allowed: {allowed}") from None

    # -- memory ---------------------------------------------------------
    def remember(self, fact: str, source: str = "user") -> None:
        if not isinstance(fact, str) or not fact.strip():
            raise ApexError("fact must be a non-empty string")
        self.harness.remember(fact.strip(), source)

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        facts = self.harness.memory.recall_facts(query, top_k=top_k)
        return [f.text for f in facts]

    # -- introspection --------------------------------------------------
    def models(self) -> list[dict[str, Any]]:
        from .models.registry import list_models

        return [m.to_dict() for m in list_models()]

    def route(self, task: str) -> dict[str, Any]:
        from .models.router import classify_task

        classified = classify_task(task)
        decision = self.harness.router.route(classified)
        payload = decision.to_dict()
        payload["classification"] = {
            "task_type": classified.task_type.value,
            "complexity": classified.complexity.value,
        }
        return payload

    def doctor(self) -> dict[str, Any]:
        return self.harness.doctor()

    # -- operations -----------------------------------------------------
    def eval(self, suite: list | None = None) -> dict[str, Any]:
        from .eval.benchmark import run_benchmark

        return run_benchmark(self.harness.run, suite=suite)


__all__ = ["Apex", "ApexError"]
