# Reasoning Strategy Guide

APEX picks a strategy per task (see `apex/reasoning/selector.py`). This doc
explains when each wins and what it costs.

| Strategy | Mechanism | Cost | Use when |
|---|---|---|---|
| **Direct** | Single pass, no scaffolding | 1× | Trivial lookups, short answers |
| **Chain-of-Thought** | Numbered steps → FINAL answer | 1× | Default for moderate reasoning/data |
| **ReAct** | Thought → tool → observation loop | 1× + tools | Task needs files, search, calculation, shell |
| **Tree-of-Thoughts** | 3 candidates → score → develop best | ~3× | Open-ended research/comparison, complex tasks |
| **Reflexion** | Draft → critique → revise | ~2× | Code/debugging where self-review catches bugs |
| **Self-Consistency** | 5 samples → majority vote | ~5× | Hard math/logic where one sample may err |
| **Plan-and-Execute** | DAG plan → step-by-step swarm | 1× + agents | Multi-step, multi-skill, expert tasks |
| **LATS** | Search over action sequences | ~3× | Sequential decisions with branching futures |

## Selection rules (deterministic + explainable)

1. Needs tools → **ReAct**.
2. Trivial + general/writing → **Direct**.
3. Planning or expert-general → **Plan-and-Execute**.
4. Code/debugging → **Reflexion**.
5. Reasoning: hard → **Self-Consistency**, else **CoT**.
6. Research → **Tree-of-Thoughts**.
7. Data → **CoT**. Complex open-ended → **ToT**. Default → **CoT**.

Every decision returns a human-readable reason, surfaced in
`HarnessResult.metadata` and the `strategy_selected` event — no black boxes.

## Sampling and side effects

Self-Consistency (5 samples), ToT and LATS (3 branches) re-run the entire tool
loop per sample. That is the point for read-only tools, and unacceptable when
`write_file`, `bash` or `python_exec` are in scope — the same write or command
would happen N times. In that situation APEX refuses to sample: it runs once,
emits a `sampling_disabled` event and adds a warning to the result. If you want
branching with side-effecting tools, remove them for that task (`tools.remove(...)`)
or use Reflexion, which revises rather than repeats.

## Critic gating

After execution, the heuristic critic scores the output (structure, hedging,
length, task-addressing). Below `agents.critic_threshold` (default 0.6) the
harness retries once with feedback (Reflexion path) or flags it in warnings.
Swarm runs additionally get a model-based critic review blended with the
heuristic score.
