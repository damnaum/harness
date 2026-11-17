"""APEX Harness — the orchestrator (the brain).

Pipeline for every task:
  1. Safety check on input (injection / PII).
  2. Classify task (type + complexity) and recall memory.
  3. Route to the best model (+ fallback chain) for the task.
  4. Select the reasoning strategy.
  5. Plan (DAG) for multi-step tasks; single-agent ReAct otherwise.
  6. Execute with tools, retries, circuit breaker, checkpointing.
  7. Critic review; retry once with feedback if below threshold.
  8. Safety check on output, persist episode + procedure to memory.
  9. Return a fully-traced HarnessResult (tokens, cost, trace, warnings).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .agents.base import make_agents
from .agents.swarm import Swarm, run_with_critic
from .config import ApexConfig
from .context.manager import ContextBudget, ContextManager
from .errors import ApexError
from .execution.loop import Checkpointer, CircuitBreaker, RetryPolicy
from .memory.store import Episode, MemoryStore
from .models.base import build_provider
from .models.router import ModelRouter, classify_task
from .observability import EventBus, Logger, Metrics, Tracer
from .planning.dag import heuristic_plan, parse_model_plan
from .reasoning.critic import heuristic_score
from .reasoning.selector import select_strategy
from .reasoning.strategies import ReasoningStrategy, extract_final, get_spec
from .safety.guardrails import Guardrails
from .tools.registry import default_registry
from .types import HarnessResult, Role, Task


def _one_line(text: str, limit: int) -> str:
    """Collapse to a single line so injected context stays parseable."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _api_keys_from_env() -> dict[str, str]:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "meta": "META_API_KEY",
    }
    return {k: os.environ[v] for k, v in mapping.items() if os.environ.get(v)}


class ApexHarness:
    def __init__(
        self,
        config: ApexConfig | None = None,
        api_keys: dict[str, str] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.config = config or ApexConfig()
        # The workspace is a sandbox root: create it rather than refusing to
        # start on a fresh checkout.
        Path(self.config.workspace).mkdir(parents=True, exist_ok=True)
        problems = self.config.validate()
        if problems:
            raise ValueError(f"invalid config: {problems}")
        self.api_keys = api_keys if api_keys is not None else _api_keys_from_env()
        self.logger = Logger(self.config.observability.log_level)
        self.metrics = Metrics()
        self.tracer = Tracer(self.config.observability.trace_enabled)
        self.events = EventBus()
        self._user_on_event = on_event
        self.events.subscribe("*", lambda e: self._user_on_event(e.type, e.payload) if self._user_on_event else None)
        self.memory = MemoryStore(
            episodic_limit=self.config.memory.episodic_limit,
            semantic_limit=self.config.memory.semantic_limit,
        )
        self.memory_path = Path(self.config.workspace) / self.config.memory.path
        if self.config.memory.enabled and self.config.memory.persist:
            self.memory.load(self.memory_path)
        self.guardrails = Guardrails(
            block_injection=self.config.safety.block_injection,
            redact_pii=self.config.safety.redact_pii,
            max_output_chars=self.config.safety.max_output_chars,
        )
        self.tools = default_registry(
            sandbox_root=self.config.workspace,
            require_approval_for=self.config.safety.require_approval_for,
            auto_approve=self.config.safety.auto_approve_tools,
            max_output_chars=20_000,
            command_timeout=self.config.safety.command_timeout,
        )
        self.retry = RetryPolicy(max_retries=self.config.models.max_retries, base_delay=self.config.models.retry_base_delay)
        self.breaker = CircuitBreaker()
        self.checkpointer = Checkpointer(self.config.checkpoint_dir)
        self.available_providers = self._resolve_available_providers()
        self.degraded = self.available_providers == {"mock"}
        self.router = ModelRouter(
            mode=self.config.models.router_mode,
            available_providers=self.available_providers,
        )
        self.context = ContextManager(
            ContextBudget(max_tokens=self.config.context.max_tokens, reserve_for_output=self.config.context.reserve_for_output)
        )

    def _resolve_available_providers(self) -> set[str]:
        """Providers we can actually call: mock always, others need a key.

        Ollama is local and has no key, so it is only considered available when
        the operator opts in (``APEX_ENABLE_OLLAMA=1`` or ``OLLAMA_HOST`` set) —
        otherwise every routed task would pay a connection timeout.
        """
        available = {"mock"}
        for provider, key in (self.api_keys or {}).items():
            if key:
                available.add(provider)
        if os.environ.get("OLLAMA_HOST") or os.environ.get("APEX_ENABLE_OLLAMA", "").lower() in ("1", "true", "yes"):
            available.add("ollama")
        return available

    # ------------------------------------------------------------------
    def _emit(self, event_type: str, **payload: Any) -> None:
        self.events.emit(event_type, **payload)
        self.metrics.incr(f"events.{event_type}")

    def _complete_with_fallback(self, decision, messages, tools=None, temperature=0.7):
        """Try the routed model, then each fallback on failure."""
        chain = [decision.model] + list(decision.fallbacks)
        last_exc: Exception | None = None
        for model in chain:
            provider = build_provider(model, self.api_keys, timeout=self.config.models.timeout_seconds)
            try:
                return self.breaker.call(
                    lambda: self.retry.run(provider.complete, model, messages, tools, temperature)
                ), model
            except ApexError as exc:
                last_exc = exc
                self.logger.warn("model failed, trying fallback", model=model.id, error=str(exc))
                self._emit("model_fallback", model=model.id, error=str(exc))
                continue
        raise last_exc or RuntimeError("all models in the chain failed")

    # ------------------------------------------------------------------
    def run(self, task_description: str, model: str | None = None, strategy: ReasoningStrategy | None = None) -> HarnessResult:
        start = time.time()
        trace_id = self.tracer.new_trace()
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        warnings: list[str] = []

        with self.tracer.span("harness.run", task=task_description[:120]):
            # 1. safety on input
            if self.config.safety.enabled:
                verdict = self.guardrails.check_input(task_description)
                if not verdict.allowed:
                    self._emit("safety_block", reasons=verdict.reasons)
                    return HarnessResult(
                        task_id=task_id, output="", success=False, model_used="",
                        strategy=ReasoningStrategy.DIRECT, trace_id=trace_id,
                        warnings=[f"blocked: {verdict.reasons}"],
                        duration_ms=(time.time() - start) * 1000,
                    )
                task_description = verdict.redacted_text

            # 2. classify + memory recall
            task = classify_task(task_description)
            task.id = task_id
            memory_hits: list[str] = []
            if self.config.memory.enabled:
                # Facts first: durable, user-stated and highest signal.
                for fact in self.memory.recall_facts(task_description, top_k=3):
                    memory_hits.append(f"Fact: {_one_line(fact.text, 240)}")
                for ep in self.memory.recall_episodes(task_description, top_k=2):
                    memory_hits.append(
                        f"Past task: {_one_line(ep.task, 120)} -> outcome: {_one_line(ep.output, 180)}"
                    )
            self._emit("task_classified", task_type=task.task_type.value, complexity=task.complexity.value)

            # 3. route model (explicit arg > configured default > auto-route)
            preferred = model or (self.config.models.default_model or None)
            if preferred:
                from .models.registry import get_model

                if get_model(preferred) is None:
                    # Silently serving a different model than the caller asked for
                    # is worse than a visible warning: cost and quality assumptions
                    # both change.
                    warnings.append(
                        f"unknown model '{preferred}' — fell back to routing; "
                        f"use `apex models` for valid ids"
                    )
            decision = self.router.route(task, preferred=preferred)
            if self.degraded:
                warnings.append(
                    "no provider credentials configured — serving from the offline "
                    "provider; set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY "
                    "for frontier models"
                )
            self._emit("model_routed", model=decision.model.id, reason=decision.reason)
            self.logger.info("routed", model=decision.model.id, task_type=task.task_type.value)

            # 4. strategy
            needs_tools = task.task_type.value in ("code", "debugging", "research", "data") or any(
                k in task_description.lower() for k in ("file", "run", "execute", "search", "calculate")
            )
            if strategy is not None:
                chosen_strategy, strategy_reason = strategy, "explicitly requested by caller"
            else:
                chosen_strategy, strategy_reason = select_strategy(task, needs_tools=needs_tools)
            spec = get_spec(chosen_strategy)
            self._emit("strategy_selected", strategy=chosen_strategy.value, reason=strategy_reason)

            # 5+6. plan & execute
            agent_results = []
            plan = None
            output = ""
            tokens_in = tokens_out = 0
            total_cost = 0.0
            model_used = decision.model.id
            critic_score = 0.0

            try:
                if chosen_strategy == ReasoningStrategy.PLAN_EXECUTE or task.complexity.value in ("complex", "expert"):
                    plan = heuristic_plan(task, max_steps=self.config.agents.max_plan_steps)
                    provider = build_provider(decision.model, self.api_keys, timeout=self.config.models.timeout_seconds)
                    agents = make_agents(
                        decision.model, provider, self.tools,
                        max_steps=self.config.agents.max_steps,
                        on_event=lambda t, p: self._emit(t, **p),
                        context_manager=self.context,
                        guardrails=self.guardrails,
                    )
                    swarm = Swarm(agents, parallel=False)
                    output, agent_results, critic_score = run_with_critic(
                        swarm, plan, task_description,
                        threshold=self.config.agents.critic_threshold,
                        on_event=lambda t, p: self._emit(t, **p),
                    )
                    for r in agent_results:
                        tokens_in += r.tokens_in
                        tokens_out += r.tokens_out
                        total_cost += r.cost_usd
                else:
                    output, tokens_in, tokens_out, total_cost, model_used = self._run_single_agent(
                        decision, spec, task_description, memory_hits, needs_tools
                    )
                    critic_score = heuristic_score(output, task_description).score

            except Exception as exc:  # noqa: BLE001 - harness never crashes the caller
                self.logger.error("harness run failed", error=str(exc))
                self._emit("run_failed", error=str(exc))
                return HarnessResult(
                    task_id=task_id, output="", success=False, model_used=model_used,
                    strategy=chosen_strategy, plan=plan, agent_results=agent_results,
                    total_tokens_in=tokens_in, total_tokens_out=tokens_out,
                    total_cost_usd=total_cost, duration_ms=(time.time() - start) * 1000,
                    trace_id=trace_id, warnings=[str(exc)],
                )

            # 7. output safety
            if self.config.safety.enabled:
                verdict = self.guardrails.check_output(output)
                output = verdict.redacted_text
                warnings.extend(verdict.reasons)

            # 8. memory
            if self.config.memory.enabled and output:
                self.memory.remember_episode(Episode(
                    task=task_description, output=output, task_type=task.task_type.value,
                    strategy=chosen_strategy.value, model=model_used, score=critic_score,
                ))
                self.memory.remember_procedure(task.task_type.value, chosen_strategy.value, model_used, critic_score)
                self._persist_memory()

            # 9. checkpoint
            try:
                self.checkpointer.save(task_id, {
                    "task": task_description, "output": output[:2000], "model": model_used,
                    "strategy": chosen_strategy.value, "critic_score": critic_score,
                    "tokens_in": tokens_in, "tokens_out": tokens_out, "cost": total_cost,
                })
            except Exception:  # noqa: BLE001 - checkpointing is best-effort
                pass

            duration_ms = (time.time() - start) * 1000
            self.metrics.incr("runs.total")
            self.metrics.incr("runs.by_model", labels={"model": model_used})
            self.metrics.incr("runs.by_strategy", labels={"strategy": chosen_strategy.value})
            self.metrics.observe("runs.duration_ms", duration_ms)
            self.metrics.observe("runs.duration_ms", duration_ms, labels={"model": model_used})
            self.metrics.incr("tokens.in", tokens_in)
            self.metrics.incr("tokens.out", tokens_out)
            self.metrics.incr("cost.usd", total_cost)
            metric_key = model_used.replace("/", "_")
            self.metrics.incr(f"cost.by_model.{metric_key}", total_cost)
            if not output:
                self.metrics.incr("runs.failed")
            self._emit("run_completed", task_id=task_id, model=model_used, duration_ms=duration_ms)

            return HarnessResult(
                task_id=task_id, output=output, success=bool(output),
                model_used=model_used, strategy=chosen_strategy, plan=plan,
                agent_results=agent_results, total_tokens_in=tokens_in,
                total_tokens_out=tokens_out, total_cost_usd=total_cost,
                duration_ms=duration_ms, trace_id=trace_id, warnings=warnings,
                metadata={
                    "critic_score": critic_score,
                    "final_answer": extract_final(output),
                    "task_type": task.task_type.value,
                    "complexity": task.complexity.value,
                    "routing": decision.reason,
                    "strategy_reason": strategy_reason,
                    "available_providers": sorted(self.available_providers),
                },
            )

    # ------------------------------------------------------------------
    def _run_single_agent(self, decision, spec, task_description, memory_hits, needs_tools):
        """Execute one agent through the selected reasoning strategy.

        Responsibilities kept in one place on purpose:
        * multi-round tool loop with loop detection (not a single round trip);
        * sampling policy per strategy — majority vote for self-consistency,
          best-of-N for ToT/LATS, single pass otherwise;
        * Reflexion revision loop bounded by the strategy's reflection budget;
        * correct assistant/tool message shapes so native tool calling works.
        """
        from .reasoning.strategies import best_of_n, extract_final, majority_vote
        from .types import Message

        provider = build_provider(
            decision.model, self.api_keys, timeout=self.config.models.timeout_seconds
        )
        system = "You are APEX, an elite agentic assistant. " + spec.system_suffix
        user = spec.user_template.format(task=task_description)
        if memory_hits:
            user += "\n\nRelevant past context:\n" + "\n".join(f"- {h}" for h in memory_hits)

        base_messages = self.context.fit_messages([
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=user),
        ])
        tools = self.tools.specs() if needs_tools else None

        tokens_in = tokens_out = 0
        cost = 0.0
        model_used = decision.model.id

        def call(messages, temperature):
            nonlocal tokens_in, tokens_out, cost, model_used
            resp, used = self._complete_with_fallback(decision, messages, tools=tools, temperature=temperature)
            tokens_in += resp.tokens_in
            tokens_out += resp.tokens_out
            cost += resp.cost_usd
            model_used = used.id
            return resp

        def solve_once() -> str:
            """One attempt: model <-> tool rounds until a final answer."""
            messages = list(base_messages)
            seen: dict[str, int] = {}
            for _ in range(self.config.agents.max_steps):
                resp = call(messages, spec.temperature)
                if not resp.tool_calls:
                    return resp.content or ""
                messages.append(Message(
                    role=Role.ASSISTANT, content=resp.content or "",
                    metadata={"tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
                    ]},
                ))
                for call_req in resp.tool_calls:
                    fingerprint = f"{call_req.name}:{sorted(call_req.arguments.items())}"
                    seen[fingerprint] = seen.get(fingerprint, 0) + 1
                    if seen[fingerprint] > 2:
                        return resp.content or "stopped: repeated identical tool call"
                    result = self.tools.execute(call_req)
                    observation = result.output if result.success else f"ERROR: {result.error}"
                    if isinstance(observation, str):
                        verdict = self.guardrails.check_tool_output(observation, call_req.name)
                        if not verdict.allowed:
                            observation = verdict.redacted_text
                            self._emit("injection_in_tool_output",
                                       tool=call_req.name, reasons=verdict.reasons)
                    messages.append(Message(
                        role=Role.TOOL, content=observation,
                        name=call_req.name, tool_call_id=call_req.id,
                    ))
            return "stopped: step budget exhausted before a final answer"

        samples = spec.samples if spec.samples > 1 else 1
        answers = [solve_once() for _ in range(samples)]

        if samples > 1 and spec.strategy == ReasoningStrategy.SELF_CONSISTENCY:
            output = majority_vote(answers)
        elif samples > 1:
            # Tree-of-thoughts and LATS: pick the best candidate, do not vote —
            # there is no ground truth to vote against, only quality to rank.
            output = best_of_n(answers, task_description)
        else:
            output = answers[0] if answers else ""

        # Reflexion: bounded critique -> revise loop (spec.max_reflections).
        if spec.strategy == ReasoningStrategy.REFLEXION and spec.max_reflections > 0:
            for round_index in range(spec.max_reflections):
                score = heuristic_score(output, task_description).score
                if score >= self.config.agents.critic_threshold:
                    break
                critique = heuristic_score(output, task_description)
                feedback = (
                    f"Your previous answer scored {score:.2f}/1.0. "
                    f"Weaknesses: {', '.join(critique.suggestions) or 'incomplete or unclear'}. "
                    "Revise it and end with a clear FINAL answer."
                )
                messages = list(base_messages) + [
                    Message(role=Role.ASSISTANT, content=output),
                    Message(role=Role.USER, content=feedback),
                ]
                revised = call(messages, spec.temperature)
                output = revised.content or output

        # Keep the full answer (reasoning is auditable); expose the extracted
        # final in metadata for programmatic use.
        return output.strip(), tokens_in, tokens_out, cost, model_used

    # ------------------------------------------------------------------
    def _persist_memory(self) -> None:
        """Best-effort durable memory write; never fails a run."""
        if not (self.config.memory.enabled and self.config.memory.persist):
            return
        try:
            self.memory.save(self.memory_path)
        except OSError as exc:  # read-only FS, permissions, full disk
            self.logger.warn("memory persistence failed", error=str(exc))

    def remember(self, fact: str, source: str = "user") -> None:
        self.memory.remember_fact(fact, source)
        self._persist_memory()

    def doctor(self) -> dict[str, Any]:
        """Self-diagnostics: config, tools, models, memory, providers."""
        from .models.registry import MODELS, providers

        return {
            "config": self.config.to_dict(),
            "config_problems": self.config.validate(),
            "tools": self.tools.names(),
            "models_registered": len(MODELS),
            "providers": providers(),
            "memory": self.memory.stats(),
            "metrics": self.metrics.snapshot(),
            "api_keys_configured": sorted(k for k, v in self.api_keys.items() if v),
            "available_providers": sorted(self.available_providers),
            "degraded_offline": self.degraded,
            "memory_path": str(self.memory_path) if self.config.memory.persist else None,
            "routable_models": sum(
                1 for m in MODELS if m.provider in self.available_providers
            ),
        }
