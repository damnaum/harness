"""Reasoning strategies — prompt shaping + output parsing per strategy.

Strategies implemented (see docs/STRATEGY.md for the full comparison):
- DIRECT: single pass, no scaffolding (trivial tasks).
- COT: chain-of-thought with explicit step markers.
- REACT: thought -> action -> observation loop (tool-using agents).
- TOT: tree-of-thoughts — generate N candidates, score, keep best.
- REFLEXION: act, then self-critique, then revise (up to K rounds).
- SELF_CONSISTENCY: sample N answers at temperature, majority-vote.
- PLAN_EXECUTE: plan first (DAG), then execute step by step.
- LATS: Monte-Carlo tree search over action sequences (budgeted).
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field

from ..types import ReasoningStrategy


@dataclass
class StrategySpec:
    strategy: ReasoningStrategy
    system_suffix: str
    user_template: str
    temperature: float = 0.7
    samples: int = 1  # for self-consistency / ToT
    max_reflections: int = 0
    description: str = ""


SPECS: dict[ReasoningStrategy, StrategySpec] = {
    ReasoningStrategy.DIRECT: StrategySpec(
        strategy=ReasoningStrategy.DIRECT,
        system_suffix="Answer directly and concisely.",
        user_template="{task}",
        temperature=0.3,
        description="Single-pass answer. Cheapest, for trivial lookups.",
    ),
    ReasoningStrategy.COT: StrategySpec(
        strategy=ReasoningStrategy.COT,
        system_suffix=(
            "Reason step by step. Use numbered steps, show intermediate results, "
            "then give the final answer clearly marked as FINAL:"
        ),
        user_template="Task: {task}\n\nThink step by step, then state FINAL: <answer>",
        temperature=0.5,
        description="Chain-of-thought: explicit intermediate steps before the answer.",
    ),
    ReasoningStrategy.REACT: StrategySpec(
        strategy=ReasoningStrategy.REACT,
        system_suffix=(
            "Use the ReAct loop: Thought: <reasoning> Action: <tool call or FINAL ANSWER>. "
            "After each tool result, continue with a new Thought. Never repeat an identical action."
        ),
        user_template="Task: {task}\n\nBegin with Thought:",
        temperature=0.5,
        description="Reason + Act loop with tool observations.",
    ),
    ReasoningStrategy.TOT: StrategySpec(
        strategy=ReasoningStrategy.TOT,
        system_suffix=(
            "Explore multiple candidate approaches. For each, briefly evaluate "
            "pros/cons, then pick the best and develop it fully."
        ),
        user_template="Task: {task}\n\nGenerate 3 candidate approaches, score each 0-10, then develop the winner.",
        temperature=0.8,
        samples=3,
        description="Tree-of-thoughts: branch, evaluate, prune, expand the best.",
    ),
    ReasoningStrategy.REFLEXION: StrategySpec(
        strategy=ReasoningStrategy.REFLEXION,
        system_suffix=(
            "First produce a draft answer. Then critique it honestly (correctness, "
            "completeness, edge cases). Then produce a revised final answer."
        ),
        user_template="Task: {task}\n\nDraft -> Critique -> Revised FINAL answer.",
        temperature=0.6,
        max_reflections=2,
        description="Self-critique and revise loop (Reflexion).",
    ),
    ReasoningStrategy.SELF_CONSISTENCY: StrategySpec(
        strategy=ReasoningStrategy.SELF_CONSISTENCY,
        system_suffix="Solve carefully. Multiple independent solutions will be compared; be consistent.",
        user_template="Task: {task}\n\nSolve step by step and end with FINAL: <answer>",
        temperature=0.9,
        samples=5,
        description="Sample N solutions, majority-vote the final answer.",
    ),
    ReasoningStrategy.PLAN_EXECUTE: StrategySpec(
        strategy=ReasoningStrategy.PLAN_EXECUTE,
        system_suffix=(
            "First produce a numbered plan of small verifiable steps. "
            "Then execute each step in order, marking progress."
        ),
        user_template="Task: {task}\n\nPlan (numbered steps), then execute step by step.",
        temperature=0.5,
        description="Plan-then-execute with a DAG of verifiable steps.",
    ),
    ReasoningStrategy.LATS: StrategySpec(
        strategy=ReasoningStrategy.LATS,
        system_suffix=(
            "Search over action sequences: consider several next actions, "
            "simulate their likely outcomes, and commit to the highest-value one."
        ),
        user_template="Task: {task}\n\nConsider 3 possible next actions, evaluate each, choose the best.",
        temperature=0.7,
        samples=3,
        description="Monte-Carlo tree search over actions (budgeted).",
    ),
}


def get_spec(strategy: ReasoningStrategy) -> StrategySpec:
    return SPECS[strategy]


def build_messages(spec: StrategySpec, task_description: str, system_base: str = "") -> list[dict[str, str]]:
    system = (system_base + "\n" + spec.system_suffix).strip()
    user = spec.user_template.format(task=task_description)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_final(text: str) -> str:
    """Pull the FINAL: answer out of CoT/Reflexion-style output, else return text."""
    for marker in ("FINAL:", "Final:", "final answer:"):
        if marker in text:
            return text.split(marker, 1)[-1].strip()
    return text.strip()


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ANSWER_FILLER = re.compile(
    r"^(?:the\s+)?(?:final\s+)?(?:answer|result|value)\s*(?:is|:|=)?\s*",
    re.IGNORECASE,
)


def normalise_answer(text: str) -> str:
    """Normalise an answer for voting/comparison.

    Exact string equality is too brittle for self-consistency: "40", "40.0",
    "40 km/h", "The answer is 40." are the same answer. We strip filler,
    punctuation, case and number formatting.
    """
    final = extract_final(text or "").strip()
    final = _ANSWER_FILLER.sub("", final)
    final = re.sub(r"\s+", " ", final).strip().strip(" .!*`\"'")
    final = final.lower()

    def _num(match: re.Match) -> str:
        raw = match.group(0)
        try:
            value = float(raw)
        except ValueError:
            return raw
        return str(int(value)) if value == int(value) else f"{value:g}"

    return _NUMBER_RE.sub(_num, final)


def majority_vote(answers: list[str]) -> str:
    """Self-consistency: the most common normalised answer wins.

    Ties break toward the earliest sample (deterministic). Returns the original
    (un-normalised) text so the caller sees real model output, not a key.
    """
    from collections import Counter

    pairs = [(normalise_answer(a), a) for a in answers if (a or "").strip()]
    if not pairs:
        return ""
    counts = Counter(key for key, _ in pairs)
    best_count = max(counts.values())
    for key, _original in pairs:  # first occurrence wins on a tie
        if counts[key] == best_count:
            for candidate_key, candidate in pairs:
                if candidate_key == key:
                    return extract_final(candidate).strip()
    return extract_final(pairs[0][1]).strip()


def best_of_n(answers: list[str], task_description: str = "") -> str:
    """Tree-of-thoughts / LATS style selection: score candidates, keep the best.

    Unlike majority voting this rewards a single well-structured, complete
    answer rather than the most popular one — which is the right objective when
    there is no ground truth to vote against.
    """
    from .critic import heuristic_score

    scored = []
    for idx, answer in enumerate(answers):
        if not (answer or "").strip():
            continue
        critique = heuristic_score(answer, task_description)
        # Deterministic tie-break: earlier sample, then longer answer.
        scored.append((critique.score, idx, len(answer), answer))
    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
    return scored[0][3].strip()
