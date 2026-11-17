"""Strategy selector — picks the reasoning strategy by task shape.

Decision logic (deterministic, explainable):
- trivial + no tools  -> DIRECT
- needs tools         -> REACT
- multi-step / planning -> PLAN_EXECUTE
- expert reasoning / math -> SELF_CONSISTENCY (or TOT for open-ended)
- code / debugging    -> REFLEXION (draft -> critique -> revise)
- research / compare  -> TOT
- default moderate    -> COT
"""

from __future__ import annotations

from ..types import Complexity, ReasoningStrategy, Task, TaskType


def select_strategy(task: Task, needs_tools: bool = False) -> tuple[ReasoningStrategy, str]:
    if needs_tools:
        return ReasoningStrategy.REACT, "task needs tool use -> ReAct loop"

    if task.complexity == Complexity.TRIVIAL and task.task_type in (TaskType.GENERAL, TaskType.WRITING):
        return ReasoningStrategy.DIRECT, "trivial task -> direct answer (cheapest)"

    if task.task_type in (TaskType.PLANNING,) or task.complexity == Complexity.EXPERT and task.task_type == TaskType.GENERAL:
        return ReasoningStrategy.PLAN_EXECUTE, "multi-step planning -> plan-and-execute DAG"

    if task.task_type in (TaskType.CODE, TaskType.DEBUGGING):
        return ReasoningStrategy.REFLEXION, "code task -> draft/critique/revise (Reflexion)"

    if task.task_type == TaskType.REASONING:
        if task.complexity in (Complexity.COMPLEX, Complexity.EXPERT):
            return ReasoningStrategy.SELF_CONSISTENCY, "hard reasoning -> sample + majority vote"
        return ReasoningStrategy.COT, "reasoning task -> chain-of-thought"

    if task.task_type == TaskType.RESEARCH:
        return ReasoningStrategy.TOT, "research/compare -> explore candidates, keep best"

    if task.task_type == TaskType.DATA:
        return ReasoningStrategy.COT, "data task -> step-by-step derivation"

    if task.complexity in (Complexity.COMPLEX, Complexity.EXPERT):
        return ReasoningStrategy.TOT, "complex open-ended -> tree-of-thoughts"

    return ReasoningStrategy.COT, "default -> chain-of-thought"
