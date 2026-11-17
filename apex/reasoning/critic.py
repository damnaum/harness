"""Critic — scores an answer for quality and decides accept / retry.

Two layers:
1. Heuristic scorer (deterministic, offline): length, structure, hedging,
   code fences, step markers, final-answer presence.
2. Model critic prompt (used when a real provider is available): asks the
   model to rate 0-1 with reasons.  The harness combines both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Critique:
    score: float  # 0..1
    accepted: bool
    reasons: list[str]
    suggestions: list[str]


def heuristic_score(text: str, task_description: str = "") -> Critique:
    reasons: list[str] = []
    suggestions: list[str] = []
    score = 0.5
    stripped = text.strip()

    # A short answer is not automatically a bad answer: "391" with a FINAL
    # marker can be exactly right. Penalise brevity only when there is no
    # verifiable result embedded.
    has_verifiable_result = bool(re.search(r"FINAL:|=\s*-?[\d.]+|```|^\s*\d+[.)]\s", stripped, re.M))
    if len(stripped) < 40 and not has_verifiable_result:
        score -= 0.25
        reasons.append("answer too short")
        suggestions.append("expand with more detail")
    elif len(stripped) < 40:
        reasons.append("concise, verifiable result")
    elif len(stripped) > 200:
        score += 0.05
        reasons.append("substantive length")

    structure_markers = (
        "FINAL:", "## ", "# ", "```", "|---", "| ",
        "1.", "2.", "- ", "* ", "1)", "•", "Subject:", "**",
    )
    has_structure = any(m in stripped for m in structure_markers) or bool(
        re.search(r"^\s*\d+[.)]\s", stripped, re.M)
    )
    if has_structure:
        score += 0.10
        reasons.append("structured output")
    else:
        suggestions.append("use headings/steps for structure")

    hedging = len(re.findall(r"\b(maybe|perhaps|possibly|might|I think)\b", stripped, re.I))
    if hedging > 3:
        score -= 0.10
        reasons.append(f"excessive hedging ({hedging})")
        suggestions.append("be decisive; state the answer plainly")

    if "mock" in stripped.lower() and "offline" in stripped.lower():
        # Mock answers are honest about being mock — don't over-penalise,
        # but cap so real-provider answers can still outrank them.
        score = min(score, 0.85)
        reasons.append("offline mock answer (honest labelling)")

    if task_description:
        # Task relevance: shared vocabulary between request and answer.
        # Vocabulary *and* numbers count as relevance signals: an answer that
        # reuses the task's figures is demonstrably working on the same problem.
        token_re = r"[a-z]{4,}|\d+(?:\.\d+)?"
        task_words = set(re.findall(token_re, task_description.lower()))
        answer_words = set(re.findall(token_re, stripped.lower()))
        overlap = len(task_words & answer_words) / len(task_words) if task_words else 0.0
        if overlap >= 0.2 or task_description.lower()[:30] in stripped.lower():
            score += 0.05
            reasons.append("addresses the task")
        elif len(task_words) >= 5 and overlap == 0.0:
            # Only penalise when the request is substantive enough that zero
            # shared vocabulary actually means something.
            score -= 0.10
            reasons.append("weak task relevance")
            suggestions.append("answer the question that was actually asked")

    score = max(0.0, min(1.0, score))
    return Critique(score=score, accepted=score >= 0.6, reasons=reasons, suggestions=suggestions)


CRITIC_SYSTEM = (
    "You are a strict reviewer. Rate the answer 0.0-1.0 for correctness, "
    "completeness and clarity. Reply with exactly: SCORE: <float> then "
    "REASONS: <short list> then SUGGESTIONS: <short list>."
)


def parse_model_critique(text: str) -> Critique:
    score_m = re.search(r"SCORE:\s*([0-9.]+)", text, re.I)
    score = float(score_m.group(1)) if score_m else 0.5
    score = max(0.0, min(1.0, score))
    reasons = []
    sugg = []
    for line in text.splitlines():
        low = line.lower()
        if "reason" in low:
            reasons.append(line.split(":", 1)[-1].strip())
        if "suggest" in low:
            sugg.append(line.split(":", 1)[-1].strip())
    return Critique(score=score, accepted=score >= 0.6, reasons=reasons, suggestions=sugg)


def combine(a: Critique, b: Critique) -> Critique:
    score = (a.score + b.score) / 2
    return Critique(
        score=score,
        accepted=score >= 0.6,
        reasons=a.reasons + b.reasons,
        suggestions=a.suggestions + b.suggestions,
    )
