"""Intelligent model router.

Scores every registry model for a task:
    score = capability_match*Wc + quality*Wq + reliability*Wr
            - cost_penalty*Wcost - latency_penalty*Wlat
Policies (router_mode): balanced | quality | fast | cheapest.
Returns the winner plus an ordered fallback chain (same provider first,
then next-best) so a provider outage degrades gracefully.
"""

from __future__ import annotations

import math
import re

from ..types import Complexity, ModelInfo, RoutingDecision, Task, TaskType
from .registry import MODELS

_COMPLEXITY_QUALITY = {
    Complexity.TRIVIAL: 0.3,
    Complexity.SIMPLE: 0.5,
    Complexity.MODERATE: 0.7,
    Complexity.COMPLEX: 0.9,
    Complexity.EXPERT: 1.0,
}


class ModelRouter:
    def __init__(
        self,
        mode: str = "balanced",
        exclude: list[str] | None = None,
        available_providers: set[str] | None = None,
    ):
        self.mode = mode
        self.exclude = set(exclude or [])
        # None = no filtering (used by tests/previews); a set restricts routing
        # to providers that actually have credentials, so a keyless install
        # degrades to the offline provider instead of failing every task.
        self.available_providers = available_providers

    def _weights(self) -> tuple[float, float, float, float]:
        # (capability, quality, cost, latency)
        return {
            "balanced": (0.45, 0.25, 0.15, 0.15),
            "quality": (0.55, 0.35, 0.05, 0.05),
            "fast": (0.35, 0.15, 0.15, 0.35),
            "cheapest": (0.35, 0.10, 0.45, 0.10),
        }.get(self.mode, (0.45, 0.25, 0.15, 0.15))

    def score(self, model: ModelInfo, task: Task, est_input: int = 2000, est_output: int = 1000) -> float:
        wc, wq, wcost, wlat = self._weights()
        capability = model.capability(task.task_type) / 10.0
        quality = _COMPLEXITY_QUALITY.get(task.complexity, 0.5) * (
            (model.reasoning + model.coding + model.reliability) / 30.0
        )
        cost = model.cost_estimate(est_input, est_output)
        # log-scale cost penalty: $0 -> 0, $0.01 -> ~0.3, $0.10 -> ~0.6, $1 -> ~0.9
        cost_penalty = min(1.0, math.log10(cost * 1000 + 1) / 3.0) if cost > 0 else 0.0
        latency_penalty = (10 - model.speed) / 10.0
        reliability_bonus = model.reliability / 10.0 * 0.1
        return (
            capability * wc
            + quality * wq
            - cost_penalty * wcost
            - latency_penalty * wlat
            + reliability_bonus
        )

    def route(
        self,
        task: Task,
        preferred: str | None = None,
        est_input: int = 2000,
        est_output: int = 1000,
        top_k_fallbacks: int = 3,
    ) -> RoutingDecision:
        candidates = [
            m for m in MODELS
            if m.id not in self.exclude
            and (self.available_providers is None or m.provider in self.available_providers)
        ]
        if not candidates:
            raise ValueError("no models available (all excluded)")

        if preferred:
            from .registry import get_model

            chosen = get_model(preferred)
            if chosen is not None and chosen.id not in self.exclude:
                ranked = sorted(
                    [m for m in candidates if m.id != chosen.id],
                    key=lambda m: self.score(m, task, est_input, est_output),
                    reverse=True,
                )
                return RoutingDecision(
                    model=chosen,
                    fallbacks=ranked[:top_k_fallbacks],
                    score=self.score(chosen, task, est_input, est_output),
                    reason=f"explicit preference: {preferred}",
                    estimated_cost=chosen.cost_estimate(est_input, est_output),
                )

        ranked = sorted(candidates, key=lambda m: self.score(m, task, est_input, est_output), reverse=True)
        winner = ranked[0]
        # Fallbacks: prefer same-provider alternates first (shared keys/region),
        # then global next-best.
        same_provider = [m for m in ranked[1:] if m.provider == winner.provider]
        others = [m for m in ranked[1:] if m.provider != winner.provider]
        fallbacks = (same_provider + others)[:top_k_fallbacks]
        return RoutingDecision(
            model=winner,
            fallbacks=fallbacks,
            score=self.score(winner, task, est_input, est_output),
            reason=(
                f"mode={self.mode} task={task.task_type.value} complexity={task.complexity.value} "
                f"capability={winner.capability(task.task_type)}/10 "
                f"est_cost=${winner.cost_estimate(est_input, est_output):.4f}"
            ),
            estimated_cost=winner.cost_estimate(est_input, est_output),
        )


_WORD_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _word_re(word: str) -> "re.Pattern[str]":
    """Word-boundary regex with simple plural tolerance ("test" ~ "tests")."""
    if word not in _WORD_RE_CACHE:
        pattern = r"(?<![\w-])" + re.escape(word) + r"(?:s|es)?(?![\w-])"
        _WORD_RE_CACHE[word] = re.compile(pattern)
    return _WORD_RE_CACHE[word]


def keyword_score(text: str, keywords: dict[str, float]) -> float:
    """Weighted keyword hit score with word-boundary semantics.

    Multi-word phrases match as substrings; single words must be whole tokens
    (so "latest" never matches "test", and "api" never matches "capital").
    """
    haystack = text.lower()
    score = 0.0
    for kw, weight in keywords.items():
        if " " in kw:
            if kw in haystack:
                score += weight
        elif _word_re(kw).search(haystack):
            score += weight
    return score


# Task-type keyword lexicon. Weights encode signal strength: an explicit
# "refactor" is a stronger code signal than a bare "function".
_TASK_LEXICON: dict[TaskType, dict[str, float]] = {
    TaskType.CODE: {
        "code": 1.0, "function": 0.8, "python": 0.9, "javascript": 0.9,
        "typescript": 0.9, "class": 0.6, "script": 0.8, "program": 0.7,
        "algorithm": 0.8, "refactor": 1.2, "sql": 0.9, "compile": 0.9,
        "implement": 1.0, "library": 0.7, "module": 0.6, "sdk": 0.8,
        "api": 0.7, "endpoint": 0.9, "regex": 0.9, "dockerfile": 1.0,
    },
    TaskType.DEBUGGING: {
        "bug": 1.2, "debug": 1.3, "error": 0.9, "traceback": 1.3,
        "exception": 1.0, "broken": 1.2, "fix": 0.8, "crash": 1.2,
        "stack trace": 1.3, "not working": 1.1, "fails": 1.0, "regression": 1.1,
        "segfault": 1.3, "memory leak": 1.2,
    },
    TaskType.RESEARCH: {
        "research": 1.3, "latest": 1.0, "news": 1.1, "study": 0.9,
        "survey": 1.0, "landscape": 0.9, "compare": 0.9, "comparison": 0.9,
        "state of the art": 1.2, "advances": 1.0, "trends": 1.0,
        "find out": 0.8, "look up": 0.9, "sources": 0.9, "cite": 0.8,
        "what is": 0.6, "how does": 0.6, "explain": 0.5,
    },
    TaskType.REASONING: {
        "prove": 1.3, "theorem": 1.2, "logic": 1.1, "math": 1.1,
        "calculate": 1.1, "solve": 1.0, "derive": 1.1, "deduce": 1.2,
        "puzzle": 1.2, "paradox": 1.1, "why": 0.5, "reason": 0.8,
        "optimise": 0.9, "optimize": 0.9, "strategy": 0.7, "proof": 1.2,
        "probability": 1.0, "complexity analysis": 1.1,
    },
    TaskType.WRITING: {
        "write": 1.0, "story": 1.1, "poem": 1.3, "essay": 1.2,
        "blog": 1.1, "email": 1.1, "letter": 1.1, "copy": 0.6,
        "creative": 1.2, "brainstorm": 1.0, "headline": 1.0,
        "newsletter": 1.1, "rewrite": 1.0, "summarise": 0.9, "summarize": 0.9,
        "draft": 1.0, "tone": 0.7,
    },
    TaskType.DATA: {
        "data": 0.9, "csv": 1.3, "json": 1.0, "parquet": 1.3,
        "dataset": 1.2, "statistics": 1.1, "chart": 1.0, "graph": 0.8,
        "parse": 1.0, "extract": 0.9, "table": 0.8, "pandas": 1.2,
        "aggregate": 1.0, "etl": 1.2, "pipeline": 0.9, "sql query": 1.2,
    },
    TaskType.PLANNING: {
        "plan": 1.2, "roadmap": 1.4, "milestone": 1.2, "architecture": 1.2,
        "design": 0.8, "project": 0.9, "schedule": 1.1, "break down": 1.1,
        "breakdown": 1.1, "steps": 1.0, "estimate": 0.9, "scope": 0.9,
        "sprint": 1.1, "timeline": 1.1, "migrate": 1.0, "rollout": 1.0,
    },
}

_CODE_TOKENS = _TASK_LEXICON[TaskType.CODE].keys() | _TASK_LEXICON[TaskType.DEBUGGING].keys()
_STRONG_CODE_TOKENS = (
    "python", "javascript", "typescript", "function", "implement", "refactor",
    "code", "script", "program", "module", "library", "sdk", "endpoint",
    "regex", "dockerfile", "compile", "algorithm", "class", "sql",
)
_ARITHMETIC_RE = re.compile(r"\d+\s*[-+*/^%]\s*\d+")
_MULTI_STEP_MARKERS = (" then ", " after that", " step 1", " step 2", " and then", " finally ")


def classify_task(description: str) -> Task:
    """Weighted-lexicon task classifier (deterministic, dependency-free).

    Scores every task type against a weighted keyword lexicon, applies
    structural adjustments (multi-step language bumps complexity; a code token
    plus a failure token promotes CODE -> DEBUGGING), and falls back to GENERAL
    when no signal reaches the confidence floor.
    """
    text = " " + description.lower().strip() + " "
    words = len(description.split())

    scores = {tt: keyword_score(text, lex) for tt, lex in _TASK_LEXICON.items()}

    # DEBUGGING only wins when there is also code context, otherwise
    # "fix my essay" would be miscategorised.
    if scores[TaskType.DEBUGGING] > 0 and keyword_score(text, {k: 1.0 for k in _CODE_TOKENS}) == 0:
        scores[TaskType.DEBUGGING] *= 0.5
    # Explicit build/engineering verbs are a stronger code signal than a
    # topical noun like "csv" or "data", so boost CODE when they are present.
    if scores[TaskType.CODE] > 0:
        strong = keyword_score(text, {k: 1.0 for k in _STRONG_CODE_TOKENS})
        if strong >= 1.0:
            scores[TaskType.CODE] *= 1.0 + min(0.6, 0.2 * strong)

    # Bare arithmetic is a reasoning task even with no verbal cue.
    if _ARITHMETIC_RE.search(text):
        scores[TaskType.REASONING] += 1.0

    # CODE and DEBUGGING overlap heavily; keep the stronger one.
    if scores[TaskType.DEBUGGING] >= scores[TaskType.CODE] and scores[TaskType.DEBUGGING] > 0:
        task_type = TaskType.DEBUGGING
    else:
        task_type = max(scores, key=lambda tt: (scores[tt], tt == TaskType.CODE))
        if scores[task_type] < 0.8:
            task_type = TaskType.GENERAL

    if words < 8:
        complexity = Complexity.TRIVIAL
    elif words < 25:
        complexity = Complexity.SIMPLE
    elif words < 60:
        complexity = Complexity.MODERATE
    elif words < 150:
        complexity = Complexity.COMPLEX
    else:
        complexity = Complexity.EXPERT

    markers = sum(1 for m in _MULTI_STEP_MARKERS if m in text)
    markers += max(0, text.count(" and ") - 1)
    if markers:
        order = [Complexity.TRIVIAL, Complexity.SIMPLE, Complexity.MODERATE, Complexity.COMPLEX, Complexity.EXPERT]
        idx = min(len(order) - 1, order.index(complexity) + min(markers, 2))
        complexity = order[idx]

    return Task(description=description, task_type=task_type, complexity=complexity)
