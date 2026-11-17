# HTTP API reference

Start the server:

```bash
apex --profile prod serve --host 0.0.0.0 --port 8000
```

All responses are JSON. Errors use a stable shape:

```json
{"error": "human readable message", "code": "bad_request"}
```

`code` is one of `bad_request`, `unauthorized`, `not_found`,
`method_not_allowed`, `overloaded`, `blocked`, `internal_error`.

Authentication: set `APEX_API_TOKEN` and send `Authorization: Bearer <token>` on
every request. When the variable is unset, auth is disabled (local development).

CORS: `Access-Control-Allow-Origin` is `*` by default; override with
`APEX_CORS_ORIGIN`. `OPTIONS` returns `204` with the allowed methods and headers,
so browsers and embedded iframes can preflight normally.

Methods: a known route with the wrong verb answers `405` and carries an `Allow`
header (`{"code": "method_not_allowed"}`), never the stdlib's HTML `501`. Any
verb on an unknown path answers `404`. A rejected request body is drained, so a
keep-alive connection stays usable. `HEAD` mirrors the `GET` routes: same status
and headers, no body — use it for probes and load balancers.

---

## `GET /health`

Liveness only — no dependency checks, cheapest possible probe.

```bash
curl -s localhost:8000/health
# {"status": "ok", "service": "apex-harness"}
```

## `GET /ready`

Readiness: the harness object exists, tools are registered, memory is usable.

```json
{"status": "ready", "tools": 10, "memory": {"episodes": 3, "facts": 1, "procedures": 2}}
```

## `GET /v1/models`

The full registry with capability scores and prices.

```bash
curl -s localhost:8000/v1/models | head -c 200
```

Each entry: `id`, `provider`, `display_name`, `context_window`, `max_output`,
`input_price_per_1m`, `output_price_per_1m`, the capability scores
(`reasoning`, `coding`, `tools`, `research`, `creativity`, `speed`,
`long_context`, `reliability`), `strengths`, `best_for`.

## `GET /v1/metrics`

```json
{
  "counters": {"runs.total": 12.0, "tokens.in": 4329.0, "cost.usd": 0.031},
  "histograms": {"runs.duration_ms": {"count": 12, "min": 3.1, "max": 812.4, "mean": 74.2}}
}
```

## `GET /v1/traces`

Spans grouped into traces, newest first:

```json
{
  "traces": [
    {"trace_id": "84509952c084486a", "started_at": 1794914000.1, "duration_ms": 12.4,
     "spans": [{"span_id": "...", "name": "harness.run", "status": "ok", "attributes": {"task": "..."}}]}
  ],
  "span_count": 14
}
```

## `GET /v1/memory`

```json
{"episodes": 5, "facts": 1, "procedures": 3}
```

## `POST /v1/run`

Request:

```json
{"task": "Write a Python function that reverses a string", "model": "anthropic/claude-sonnet-4.5", "strategy": "react"}
```

- `task` (required, non-empty string)
- `model` (optional) — force a model id from `/v1/models`
- `strategy` (optional) — `direct`, `chain_of_thought`, `react`, `tree_of_thoughts`,
  `reflexion`, `self_consistency`, `plan_execute`, `lats`. Unknown values return
  `400` with the allowed list.

Response: the serialised `HarnessResult` — `task_id`, `output`, `success`,
`model_used`, `strategy`, `plan`, `agent_results`, token counts, `total_cost_usd`,
`duration_ms`, `trace_id`, `warnings`, `metadata` (critic score, task type,
complexity, routing reason).

```bash
curl -s -X POST localhost:8000/v1/run \
  -H 'content-type: application/json' \
  -d '{"task":"What is 15% of 240?"}'
```

## `POST /v1/route`

Routing decision without running the task — useful for cost previews and for
explaining why a model was chosen.

```bash
curl -s -X POST localhost:8000/v1/route -H 'content-type: application/json' \
  -d '{"task":"Prove that sqrt(2) is irrational"}'
```

```json
{
  "model": {"id": "google/gemini-3-pro", "...": "..."},
  "fallbacks": [{"id": "google/gemini-2.5-pro", "...": "..."}],
  "score": 0.83,
  "reason": "mode=balanced task=reasoning complexity=simple capability=10/10 est_cost=$0.0160",
  "estimated_cost": 0.016,
  "classification": {"task_type": "reasoning", "complexity": "simple"}
}
```

## `POST /v1/remember`

```bash
curl -s -X POST localhost:8000/v1/remember -H 'content-type: application/json' \
  -d '{"fact":"Staging deploys use blue-green rollout","source":"ops"}'
```

Facts are de-duplicated case-insensitively and persisted to
`.apex/memory.json` relative to the workspace.

## `POST /v1/eval`

Runs the benchmark suite; optional `tasks` filters by id.

```bash
curl -s -X POST localhost:8000/v1/eval -H 'content-type: application/json' \
  -d '{"tasks":["math-add","code-basic","reason-basic"]}'
```

```json
{"total": 3, "passed": 3, "pass_rate": 1.0, "mean_score": 0.9,
 "total_tokens": 512, "total_cost": 0.0}
```

Per-task detail is intentionally omitted from the HTTP payload (it is large);
use `apex eval` in CI for the full table.

---

## Errors you should expect

| Situation | Status | `code` |
|---|---|---|
| Missing `task`/`fact` | 400 | `bad_request` |
| Malformed JSON, non-object body, body > 1 MB | 400 | `bad_request` |
| Unknown strategy | 400 | `bad_request` |
| Missing/invalid bearer token | 401 | `unauthorized` |
| Unknown route | 404 | `not_found` |
| Too many concurrent runs | 503 | `overloaded` (+ `Retry-After`) |
| `/v1/remember` fact fails the guardrail scan | 403 | `blocked` |
| Bug inside the harness | 500 | `internal_error` (type name only, no traceback) |

## Concurrency notes

The server is `ThreadingHTTPServer` with one thread per request, but `/v1/run`
is gated by a semaphore of `APEX_MAX_CONCURRENCY` slots (default 4). Beyond that,
requests get `503 overloaded` with `Retry-After: 5` — a bounded, explicit failure
rather than an unbounded queue of in-flight model calls. `/v1/route`, `/v1/eval`
and the read endpoints are not gated.

Each request shares the harness instance, so metrics, traces and memory are
process-wide. Memory writes are atomic *and* `fsync`-ed, so concurrent remembers
cannot corrupt the file, and a crash cannot lose a committed write. If you need
multi-process scaling, run several replicas with separate workspaces (or a shared
store) rather than raising thread counts.

Operational details: idle sockets are dropped after 30 s (slow-loris guard), the
listen backlog is 64, worker threads are daemons, and `SIGTERM`/`SIGINT` trigger a
graceful shutdown so a deploy does not cut a response in half.
