"""DAG planner — decomposes a task into a dependency graph of steps.

Two modes:
- Heuristic planner (offline/deterministic): splits on connectives, assigns
  agents by keyword, links steps sequentially with sensible dependencies.
- Model planner: asks the model for a JSON plan and validates it into a DAG.
"""

from __future__ import annotations

import json
import re

from ..types import Plan, PlanStep, ReasoningStrategy, StepStatus, Task


_AGENT_KEYWORDS = {
    "researcher": ("research", "find", "search", "compare", "explain", "latest", "study"),
    "coder": ("code", "function", "implement", "python", "script", "program", "api"),
    "tester": ("test", "verify", "check", "validate", "debug", "fix", "bug"),
    "writer": ("write", "essay", "blog", "email", "story", "document", "summary"),
    "analyst": ("data", "calculate", "analyse", "analyze", "statistics", "chart"),
}


# Agent routing lexicon with weights. Higher weight wins; ties fall back to
# the declaration order, which keeps assignment deterministic.
_AGENT_WEIGHTS: dict[str, dict[str, float]] = {
    "researcher": {
        "research": 1.4, "find": 0.9, "search": 1.0, "compare": 0.9,
        "explain": 0.7, "latest": 1.0, "study": 1.0, "survey": 1.1,
        "state of the art": 1.3, "sources": 1.0, "investigate": 1.2,
    },
    "coder": {
        "code": 1.3, "function": 1.1, "implement": 1.3, "python": 1.2,
        "script": 1.1, "program": 1.0, "api": 1.0, "refactor": 1.3,
        "module": 1.0, "class": 0.9, "endpoint": 1.1, "library": 1.0,
    },
    "tester": {
        "test": 1.3, "verify": 1.2, "check": 0.9, "validate": 1.2,
        "debug": 1.4, "fix": 1.1, "bug": 1.3, "coverage": 1.2,
        "regression": 1.2, "assert": 1.0,
    },
    "writer": {
        "write": 1.1, "essay": 1.3, "blog": 1.2, "email": 1.2,
        "story": 1.3, "document": 1.2, "documentation": 1.2, "summary": 0.9,
        "draft": 1.0, "copy": 0.8, "readme": 1.2,
    },
    "analyst": {
        "data": 1.0, "calculate": 1.2, "analyse": 1.2, "analyze": 1.2,
        "statistics": 1.2, "chart": 1.1, "risk": 0.9, "risks": 0.9,
        "trade-off": 1.0, "cost": 1.0, "estimate": 1.0, "metric": 1.1,
        "evaluate": 1.0, "assessment": 1.1, "review": 0.9, "budget": 1.1,
        "analyse": 1.2, "tradeoff": 1.0, "forecast": 1.2,
    },
    "planner": {
        "plan": 1.3, "roadmap": 1.4, "milestone": 1.2, "phase": 1.1,
        "phased": 1.2, "schedule": 1.2, "strategy": 1.2, "architecture": 1.1,
        "design": 0.9, "rollout": 1.2, "migrate": 1.0, "migration": 1.0,
        "breakdown": 1.1, "break down": 1.1, "steps": 0.8,
    },
}


def assign_agent(text: str) -> str:
    """Score-based agent routing (word-boundary aware, deterministic)."""
    from ..models.router import keyword_score

    scores = {agent: keyword_score(text, lex) for agent, lex in _AGENT_WEIGHTS.items()}
    best = max(scores, key=lambda a: (scores[a], -list(_AGENT_WEIGHTS).index(a)))
    return best if scores[best] > 0 else "executor"


# Clause separators, in priority order. We split on these to recover the
# individual actions hidden inside one long sentence.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[.;]\s+|:\s+|\s*(?:,\s*)?(?:and\s+then|then|after\s+that|next|finally|lastly|also)\s+)",
    re.IGNORECASE,
)
_ENUM_SPLIT_RE = re.compile(r"(?:^|\s)(?:\d+[.)]|[-*•])\s+", re.MULTILINE)
_CONJUNCTION_SPLIT_RE = re.compile(r"\s+and\s+(?=(?:identify|propose|write|create|build|design|review|test|verify|analyse|analyze|explain|compare|list|draft|implement|optimise|optimize|evaluate|research|prepare|deploy|migrate|define|plan|estimate|document|validate|check|run|measure|monitor|fix|refactor)\b)", re.IGNORECASE)

_MAX_TITLE = 72


def _title_for(clause: str) -> str:
    """Human-readable step title: imperative, capitalised, word-boundary cut."""
    text = " ".join(clause.split())
    text = text.strip(" .;,:")
    if not text:
        return "step"
    text = text[0].upper() + text[1:]
    if len(text) > _MAX_TITLE:
        cut = text[:_MAX_TITLE].rsplit(" ", 1)[0]
        text = (cut or text[:_MAX_TITLE]).rstrip(" ,;") + "…"
    return text


def decompose(description: str) -> list[str]:
    """Split a task description into ordered, self-contained action clauses.

    Handles enumerated lists, colon-introduced sub-lists, explicit sequencing
    words ("then", "finally") and verb-led conjunctions ("... and propose ...").
    Single-step tasks come back as a one-element list.
    """
    raw = description.strip()
    if not raw:
        return []

    # 1. Enumerated / bulleted lists win when present.
    if len(_ENUM_SPLIT_RE.findall(raw)) >= 2:
        chunks = [c.strip(" .;,") for c in _ENUM_SPLIT_RE.split(raw)]
    else:
        # 2. Split on explicit sequencing and clause punctuation.
        chunks = [c.strip(" .;,") for c in _CLAUSE_SPLIT_RE.split(raw)]

    # 3. Sub-split long chunks on verb-led "and".
    out: list[str] = []
    for chunk in chunks:
        if len(chunk.split()) > 18:
            out.extend(c.strip(" .;,") for c in _CONJUNCTION_SPLIT_RE.split(chunk))
        else:
            out.append(chunk)

    # 4. Normalise, drop fragments and de-duplicate while preserving order.
    cleaned: list[str] = []
    seen: set[str] = set()
    for clause in out:
        clause = " ".join(clause.split())
        if len(clause) < 4:
            continue
        key = clause.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(clause)
    return cleaned or [raw]


def heuristic_plan(task: Task, max_steps: int = 12) -> Plan:
    """Deterministic planner: clause decomposition + sequential dependencies."""
    clauses = decompose(task.description)[: max(1, max_steps - 1)]

    steps: list[PlanStep] = []
    prev_id: str | None = None
    for i, clause in enumerate(clauses, start=1):
        step = PlanStep(
            title=f"{i}. {_title_for(clause)}",
            description=clause,
            agent=assign_agent(clause),
            depends_on=[prev_id] if prev_id else [],
            tools_hint=_tools_hint(clause),
        )
        steps.append(step)
        prev_id = step.id

    # Non-trivial plans end in an explicit verification gate.
    if task.complexity.value in ("simple", "moderate", "complex", "expert") and len(steps) < max_steps:
        steps.append(
            PlanStep(
                title="Verify the result against the original goal",
                description=(
                    "Review the accumulated output for correctness, completeness and clarity; "
                    "call out any gap against the original request and state the residual risk."
                ),
                agent="critic",
                depends_on=[steps[-1].id] if steps else [],
            )
        )
    return Plan(task_id=task.id, steps=steps, strategy=ReasoningStrategy.PLAN_EXECUTE)


def _tools_hint(text: str) -> list[str]:
    low = text.lower()
    hints = []
    if any(k in low for k in ("file", "read", "directory", "folder")):
        hints += ["read_file", "list_dir"]
    if any(k in low for k in ("calculate", "compute", "math")):
        hints.append("calculator")
    if any(k in low for k in ("search", "find online", "latest")):
        hints.append("web_search")
    if any(k in low for k in ("run", "execute", "command")):
        hints.append("bash")
    if "git" in low:
        hints.append("git_status")
    return hints


def parse_model_plan(text: str, task: Task, max_steps: int = 12) -> Plan:
    """Parse a JSON plan from the model; fall back to heuristic on failure."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            raw_steps = data.get("steps", [])[:max_steps]
            steps = []
            id_map: dict[str, str] = {}
            for raw in raw_steps:
                step = PlanStep(
                    title=str(raw.get("title", "step"))[:80],
                    description=str(raw.get("description", raw.get("title", ""))),
                    agent=str(raw.get("agent", "executor")),
                    depends_on=[],
                )
                id_map[str(raw.get("id", step.id))] = step.id
                steps.append(step)
            # Wire dependencies by original ids.
            for raw, step in zip(raw_steps, steps):
                for dep in raw.get("depends_on", []) or []:
                    if dep in id_map:
                        step.depends_on.append(id_map[dep])
            if steps:
                # Validate acyclicity roughly: every dep must exist.
                valid = {s.id for s in steps}
                for s in steps:
                    s.depends_on = [d for d in s.depends_on if d in valid and d != s.id]
                return Plan(task_id=task.id, steps=steps, strategy=ReasoningStrategy.PLAN_EXECUTE)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return heuristic_plan(task, max_steps)


def topological_order(plan: Plan) -> list[PlanStep]:
    """Kahn's algorithm — execution order respecting dependencies."""
    indegree = {s.id: len(s.depends_on) for s in plan.steps}
    order: list[PlanStep] = []
    queue = [s for s in plan.steps if indegree[s.id] == 0]
    by_id = {s.id: s for s in plan.steps}
    while queue:
        node = queue.pop(0)
        order.append(node)
        for s in plan.steps:
            if node.id in s.depends_on:
                indegree[s.id] -= 1
                if indegree[s.id] == 0:
                    queue.append(s)
    if len(order) != len(plan.steps):
        raise ValueError("plan contains a cycle")
    return order
