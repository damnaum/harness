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
  (23     (8         (DAG +     (9 agents  (tools,     (injection  (episodic/
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
6. **Safety is layered.** Input injection scan → tool-output (indirect) injection
   scan → sandboxed tools with deny-by-default approval gates (paths resolved with
   `Path.resolve`, child processes get an allow-listed env, URLs are resolved and
   IP-class-checked) → output PII redaction + length caps. Defence in depth, not
   one wall. See `docs/SECURITY.md` for what this does *not* protect against.

## Production concerns, and where they live

| Concern | Implementation | Failure behaviour |
|---|---|---|
| Provider outage | Per-model `CircuitBreaker` (5 failures → 30 s open) + fallback chain | Next model is tried; run continues |
| Flaky call | `RetryPolicy` exponential backoff + jitter, **status-aware** (`errors.is_retryable`) | Retries 429/5xx/timeouts; fails fast on 4xx, auth, budget, guardrails |
| Hostile input | `Guardrails.inspect_input` signature + PII scan | Run refused, input never leaves the process |
| Unsafe tool | Approval gate + sandbox path/command validation | Tool returns failed result, loop continues |
| Hung tool | Wall-clock timeout in `Tool.run` | Failed observation, agent picks another action |
| Hallucinated tool | Registry returns a failed `ToolResult` | Model sees the error and recovers |
| Looping agent | Tool-call fingerprint with repeat detection | `LoopDetectedError`, step budget preserved |
| Long context | `ContextManager` budget + head/tail compaction between agents **and inside the ReAct loop** | Oldest material evicted, task restated, tool turns kept valid |
| Crash | JSON checkpoints per run | `Checkpointer.list_runs()` for recovery |
| Duplicate facts | Case-insensitive de-duplication in `MemoryStore` | Idempotent remembers |
| Index drift after eviction | TF-IDF rebuild on trim | Recall stays correct (regression-tested) |
| Corrupt memory file | Atomic write (`fsync` + rename) + safe load | Store starts empty, harness keeps running |
| Unwritable memory file | Best-effort `save()` | Warning logged, run continues |
| Concurrent runs in one process | `MemoryStore` under an `RLock`; recall copies under the lock | Recall never observes a torn index |
| Tool output carries instructions | `Guardrails.check_tool_output` before the model sees it | Observation tagged `[UNTRUSTED DATA]`, event emitted |
| Server saturated | `/v1/run` semaphore (`APEX_MAX_CONCURRENCY`) | `503 overloaded` + `Retry-After`, no unbounded queue |
| Slow/idle client | 30 s socket timeout | Connection dropped |
| Shutdown | `SIGTERM`/`SIGINT` handler | In-flight request finishes, then exit |

## Module map

| Module | Responsibility |
|---|---|
| `apex/types.py` | All core dataclasses (Task, Plan, AgentResult, HarnessResult…) |
| `apex/config.py` | Layered config (defaults < file < env < overrides) + profiles |
| `apex/models/` | Registry (23 models), router, providers (OpenAI-compatible/Anthropic/Google/Ollama/Mock) |
| `apex/reasoning/` | 8 strategies, selector, critic |
| `apex/agents/` | 9-role swarm (supervisor…critic) + DAG execution |
| `apex/tools/` | Registry + 10 builtin tools + `sandbox.py` (paths, env, commands, SSRF, limits) |
| `apex/memory/` | Episodic/semantic/procedural + TF-IDF index + thread-safe, fsync-ed JSON persistence |
| `apex/context/` | Token budgeting, eviction, extractive summarisation |
| `apex/planning/` | Heuristic + model DAG planner, topological sort |
| `apex/execution/` | ReAct loop, retry/backoff, circuit breaker, checkpoints |
| `apex/safety/` | Injection detection (direct + tool output), PII redaction, output caps |
| `apex/eval/` | Benchmark suite, 5 scorers (incl. sandboxed code execution), report |
| `apex/server/` | Stdlib HTTP API: constant-time auth, CORS, size caps, concurrency limit, graceful shutdown, trace grouping |
| `apex/observability.py` | JSON logs, labelled metrics, nested spans, event bus |
| `apex/cli.py`, `apex/sdk.py` | CLI + programmatic SDK |

## Request lifecycle (one `/v1/run`)

1. Token check (constant time) → body parsed with a 1 MB cap → semaphore slot taken.
2. `Guardrails.check_input` — injection signatures, PII redaction.
3. `classify_task` — task type from a weighted lexicon; complexity from word
   count **plus a scope floor**: the number of distinct deliverables (distinct
   action verbs) raises the class, so a 22-word request to design, implement,
   benchmark and compare is EXPERT, not SIMPLE. The floor only ever raises.
4. Memory recall — facts first, then episodes; the current task echoing itself is filtered.
5. `ModelRouter.route` — score every available model, pick a winner and a fallback chain.
6. `select_strategy` — one of eight, with a human-readable reason.
7. Execute — single agent (strategy prompt → ReAct loop → critique) or swarm
   (DAG planner → specialist agents → critic → synthesis). Every tool observation
   passes through `check_tool_output` before the model sees it.
8. `Guardrails.check_output` — PII redaction, length cap.
9. Persist episode + procedure, write a checkpoint, export the span, emit metrics.
10. Slot released → JSON response (or `503` if the queue was already full).

Every one of those steps is individually testable, and the tests do exactly that.
