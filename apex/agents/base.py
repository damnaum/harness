"""Agent base class + shared agent context.

An Agent wraps: a model (via provider), a system prompt, optional tools and
the ReAct loop.  Specialised agents (planner/coder/...) differ only in prompt
+ tool subset + strategy — composition over inheritance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..errors import BudgetExceededError
from ..execution.loop import ReActLoop
from ..types import AgentResult, Message, ModelInfo, Role


@dataclass
class AgentContext:
    """Shared state passed between agents in a swarm run."""

    task_description: str = ""
    prior_outputs: dict[str, str] = field(default_factory=dict)
    memory_hits: list[str] = field(default_factory=list)

    CONTEXT_CHAR_BUDGET = 6000
    PER_OUTPUT_HEAD = 900
    PER_OUTPUT_TAIL = 200

    @staticmethod
    def _brief(text: str, head: int, tail: int) -> str:
        """Head+tail compaction: keeps the conclusion and the key claims."""
        text = text.strip()
        if len(text) <= head + tail:
            return text
        return f"{text[:head].rstrip()}\n…[trimmed]…\n{text[-tail:].lstrip()}"

    def render(self) -> str:
        """Render shared context under a hard character budget.

        Multi-agent runs otherwise blow the prompt budget linearly with the
        number of steps: every agent would receive every earlier output in
        full. We compact each output to head+tail and cap the total.
        """
        parts: list[str] = []
        used = 0

        if self.memory_hits:
            mem = "Relevant past context:\n" + "\n".join(f"- {h}" for h in self.memory_hits[:5])
            mem = self._brief(mem, 800, 200)
            parts.append(mem)
            used += len(mem)

        # Newest outputs first: they are the most relevant to the current step.
        for agent, out in reversed(list(self.prior_outputs.items())):
            block = f"[{agent} output]\n{self._brief(out, self.PER_OUTPUT_HEAD, self.PER_OUTPUT_TAIL)}"
            if used + len(block) > self.CONTEXT_CHAR_BUDGET:
                remaining = self.CONTEXT_CHAR_BUDGET - used
                if remaining < 200:
                    break
                block = block[:remaining]
            parts.append(block)
            used += len(block)

        if not parts:
            return ""
        header = (
            "Context from earlier steps. Use it, do not repeat it verbatim, "
            "and do not re-derive facts already established here."
        )
        return header + "\n\n" + "\n\n".join(parts)


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: ModelInfo,
        provider,
        tool_registry=None,
        tools: list[str] | None = None,  # None = all tools
        max_steps: int = 25,
        temperature: float = 0.7,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        context_manager: Any = None,
        guardrails: Any = None,
        model_fn: Callable[[list[Message], Any], Any] | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.provider = provider
        self.tool_registry = tool_registry
        self.tools = tools
        self.max_steps = max_steps
        self.temperature = temperature
        self.on_event = on_event or (lambda *a: None)
        self.context_manager = context_manager
        self.guardrails = guardrails
        # Injected so a swarm agent uses the same retry/breaker/fallback/budget
        # path as single-agent runs instead of calling the provider directly.
        self._injected_model_fn = model_fn

    def tool_specs(self) -> list[dict[str, Any]] | None:
        if self.tool_registry is None:
            return None
        if self.tools is None:
            return self.tool_registry.specs()
        return [self.tool_registry.get(n).spec() for n in self.tools if n in self.tool_registry.tools]

    def _model_fn(self, messages: list[Message], tools) -> Any:
        if self._injected_model_fn is not None:
            return self._injected_model_fn(messages, tools)
        return self.provider.complete(self.model, messages, tools=tools, temperature=self.temperature)

    def run(self, task: str, context: AgentContext | None = None) -> AgentResult:
        start = time.time()
        user_text = task
        if context and context.render():
            user_text = f"{task}\n\n{context.render()}"
        loop = ReActLoop(
            model_fn=self._model_fn,
            tool_executor=self.tool_registry.execute if self.tool_registry else (lambda c: (_ for _ in ()).throw(RuntimeError("no tools"))),
            max_steps=self.max_steps,
            on_event=self.on_event,
            context_manager=self.context_manager,
            guardrails=self.guardrails,
        )
        try:
            result = loop.run(self.system_prompt, user_text, tools=self.tool_specs())
            return AgentResult(
                agent=self.name,
                output=result["output"],
                success=True,
                steps_taken=result["steps"],
                tool_calls=result["tool_calls"],
                model_used=self.model.id,
                tokens_in=result["tokens_in"],
                tokens_out=result["tokens_out"],
                cost_usd=result["cost_usd"],
                duration_ms=(time.time() - start) * 1000,
            )
        except BudgetExceededError:
            # Spend limits are fatal: continuing would spend money the operator
            # explicitly capped. Propagate so the run stops and reports the spend.
            raise
        except Exception as exc:  # noqa: BLE001 - agents report failure, swarm continues
            return AgentResult(
                agent=self.name, output="", success=False,
                model_used=self.model.id, duration_ms=(time.time() - start) * 1000,
                metadata={"error": str(exc)},
            )


# ------------------------------------------------------------------ factory
def make_agents(model, provider, tool_registry, max_steps=25, on_event=None,
                context_manager=None, guardrails=None, model_fn=None) -> dict[str, Agent]:
    """Build the full APEX swarm with role-tuned prompts and tool subsets."""
    from .prompts import AGENT_PROMPTS

    tool_map = {
        "supervisor": None,
        "planner": None,
        "executor": None,
        "researcher": ["web_search", "web_fetch", "read_file", "list_dir", "search_files"],
        "coder": ["read_file", "write_file", "list_dir", "search_files", "bash", "python_exec"],
        "tester": ["bash", "python_exec", "read_file", "list_dir", "search_files"],
        "writer": ["read_file", "write_file", "list_dir"],
        "analyst": ["calculator", "python_exec", "read_file", "list_dir", "search_files"],
        "critic": ["read_file", "list_dir", "search_files"],
    }
    agents = {}
    for name, prompt in AGENT_PROMPTS.items():
        agents[name] = Agent(
            name=name, system_prompt=prompt, model=model, provider=provider,
            context_manager=context_manager,
            guardrails=guardrails,
            tool_registry=tool_registry, tools=tool_map.get(name),
            max_steps=max_steps, on_event=on_event,
            model_fn=model_fn,
        )
    return agents
