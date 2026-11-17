# APEX Architecture

```
User / CLI / SDK / HTTP API
        │
        ▼
┌──────────────────────── ApexHarness (orchestrator) ────────────────────────┐
│  safety → classify → memory recall → route model → select strategy         │
│  → plan DAG → execute (swarm | single agent) → critic → safety → persist   │
└───┬────────┬──────────┬───────────┬──────────┬───────────┬────────┬────────┘
    │        │          │           │          │           │        │
    ▼        ▼          ▼           ▼          ▼           ▼        ▼
  Router  Selector   Planner     Swarm     ReAct loop  Guardrails  Memory
  (22     (8         (DAG +     (9 agents  (tools,     (injection  (episodic/
  models)  strategies) topo-sort) + critic  retry,     PII, caps)  semantic/
                                            breaker,                procedural
                                            checkpoint)             + TF-IDF)
    │
    ▼
  Providers: OpenAI-compat · Anthropic · Google · Ollama · Mock (offline)
```

## Key design decisions

1. **Stdlib-only core.** No numpy, no SDKs, no framework. Providers are plain
   `urllib` HTTP. This makes APEX embeddable anywhere and trivially testable.
2. **Everything is injectable.** `model_fn`, `tool_executor`, `on_event` — the
   ReAct loop and agents are unit-testable with fakes, no network needed.
3. **Failures are typed.** Every failure mode is an `ApexError` subclass so
   callers can retry, fall back, or abort precisely.
4. **Observability by default.** Every run has a trace_id; spans, metrics and
   events are queryable via `/v1/traces`, `/v1/metrics` and `harness.events`.
5. **Memory learns.** Procedural memory records which (strategy, model) won per
   task type, so routing improves with use.
6. **Safety is layered.** Input injection scan → sandboxed tools with approval
   gates → output PII redaction + length caps. Defence in depth, not one wall.

## Production concerns, and where they live

| Concern | Implementation | Failure behaviour |
|---|---|---|
| Provider outage | Per-model `CircuitBreaker` (5 failures → 30 s open) + fallback chain | Next model is tried; run continues |
| Flaky call | `RetryPolicy` exponential backoff + jitter | Retries, then breaker |
| Hostile input | `Guardrails.inspect_input` signature + PII scan | Run refused, input never leaves the process |
| Unsafe tool | Approval gate + sandbox path/command validation | Tool returns failed result, loop continues |
| Hung tool | Wall-clock timeout in `Tool.run` | Failed observation, agent picks another action |
| Hallucinated tool | Registry returns a failed `ToolResult` | Model sees the error and recovers |
| Looping agent | Tool-call fingerprint with repeat detection | `LoopDetectedError`, step budget preserved |
| Long context | `ContextManager` budget + head/tail compaction between agents | Oldest material evicted, conclusion retained |
| Crash | JSON checkpoints per run | `Checkpointer.list_runs()` for recovery |
| Duplicate facts | Case-insensitive de-duplication in `MemoryStore` | Idempotent remembers |
| Index drift after eviction | TF-IDF rebuild on trim | Recall stays correct (regression-tested) |
| Corrupt memory file | Atomic write + safe load | Store starts empty, harness keeps running |

## Module map

| Module | Responsibility |
|---|---|
| `apex/types.py` | All core dataclasses (Task, Plan, AgentResult, HarnessResult…) |
| `apex/config.py` | Layered config (defaults < file < env < overrides) + profiles |
| `apex/models/` | Registry, router, providers (OpenAI/Anthropic/Google/Ollama/Mock) |
| `apex/reasoning/` | 8 strategies, selector, critic |
| `apex/agents/` | 9-role swarm (supervisor…critic) + DAG execution |
| `apex/tools/` | Registry + 10 builtin tools + `sandbox.py` (paths, env, commands, SSRF, limits) |
| `apex/memory/` | Episodic/semantic/procedural + TF-IDF index + atomic JSON persistence |
| `apex/context/` | Token budgeting, eviction, extractive summarisation |
| `apex/planning/` | Heuristic + model DAG planner, topological sort |
| `apex/execution/` | ReAct loop, retry/backoff, circuit breaker, checkpoints |
| `apex/safety/` | Injection detection (direct + tool output), PII redaction, output caps |
| `apex/eval/` | Benchmark suite + scorers + report |
| `apex/server/` | Stdlib HTTP API: auth, CORS, size caps, error mapping, trace grouping |
| `apex/cli.py`, `apex/sdk.py` | CLI + programmatic SDK |
