"""Context manager — token budgeting, priority eviction, extractive summary.

Keeps the working set inside the model's context window:
- system prompt (never evicted)
- recent turns (kept, newest first)
- tool results (truncated, then summarised when over budget)
- memory recall (lowest priority, evicted first)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.base import estimate_tokens
from ..types import Message, Role


@dataclass
class ContextBudget:
    max_tokens: int = 32_000
    reserve_for_output: int = 8_000

    @property
    def usable(self) -> int:
        return max(1000, self.max_tokens - self.reserve_for_output)


def summarise_extractively(text: str, max_chars: int = 600) -> str:
    """Extractive summary: first sentence + sentences with keywords + last sentence."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return text[:max_chars]
    if len(sentences) <= 2:
        return text[:max_chars]
    picked = [sentences[0]]
    for s in sentences[1:-1]:
        if any(k in s.lower() for k in ("important", "result", "key", "conclusion", "error", "final")):
            picked.append(s)
    picked.append(sentences[-1])
    out = ". ".join(picked)
    return out[:max_chars] + ("..." if len(out) > max_chars else "")


class ContextManager:
    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()
        self.messages: list[Message] = []

    def add(self, role: Role, content: str, **meta) -> Message:
        msg = Message(role=role, content=content, metadata=meta)
        self.messages.append(msg)
        return msg

    def tokens_used(self) -> int:
        return sum(estimate_tokens(m.content) + 4 for m in self.messages)

    def fit(self) -> list[Message]:
        """Return a budget-fitting view; evict oldest non-system messages first,
        summarising tool outputs that are too large."""
        usable = self.budget.usable
        system = [m for m in self.messages if m.role == Role.SYSTEM]
        rest = [m for m in self.messages if m.role != Role.SYSTEM]
        # Summarise oversized tool results in place (copy).
        fitted_rest = []
        for m in rest:
            if m.role == Role.TOOL and len(m.content) > 2000:
                m = Message(role=m.role, content=summarise_extractively(m.content, 1500), name=m.name, tool_call_id=m.tool_call_id)
            fitted_rest.append(m)
        total = sum(estimate_tokens(m.content) + 4 for m in system + fitted_rest)
        while total > usable and fitted_rest:
            evicted = fitted_rest.pop(0)
            total -= estimate_tokens(evicted.content) + 4
        return system + fitted_rest

    # -- stateless fitting ---------------------------------------------------
    def fit_messages(self, messages: list[Message]) -> list[Message]:
        """Return a budget-fitting copy of an arbitrary message list.

        Used by the ReAct loop: tool observations can be megabytes, and a loop
        that never compacts will eventually exceed the model window (a hard
        provider-side failure). Policy, in priority order:

        1. system messages are never dropped;
        2. the most recent messages are kept verbatim;
        3. oversized tool observations are summarised;
        4. oldest non-system messages are evicted.
        """
        if not messages:
            return []

        usable = self.budget.usable
        system = [m for m in messages if m.role == Role.SYSTEM]
        rest = [m for m in messages if m.role != Role.SYSTEM]

        def cost(m: Message) -> int:
            return estimate_tokens(m.content) + 4

        total = sum(cost(m) for m in system + rest)
        if total <= usable:
            return list(messages)

        # 2. Summarise oversized tool observations first: they are the bulk and
        #    the conclusion matters more than the raw dump.
        compacted: list[Message] = []
        for m in rest:
            if m.role == Role.TOOL and len(m.content) > 1200:
                m = Message(
                    role=m.role, content=summarise_extractively(m.content, 800),
                    name=m.name, tool_call_id=m.tool_call_id,
                )
            compacted.append(m)

        # 3. Keep the newest messages that fit; never split the tail.
        kept: list[Message] = []
        running = sum(cost(m) for m in system)
        for m in reversed(compacted):
            c = cost(m)
            if running + c > usable and kept:
                break
            kept.append(m)
            running += c
        kept.reverse()

        # 4. If the newest message alone exceeds the budget, truncate it.
        if kept and running > usable and len(kept) == 1:
            head = max(200, usable * 3)
            trimmed = kept[0]
            kept = [Message(role=trimmed.role, content=trimmed.content[:head] + "…[truncated]",
                            name=trimmed.name, tool_call_id=trimmed.tool_call_id)]

        return system + kept

    def clear(self) -> None:
        self.messages.clear()
