# Operations guide

Everything needed to run APEX in an environment that matters: configuration,
deployment, monitoring, and what to do when something breaks.

## 1. Deployment shapes

| Shape | Command | Notes |
|---|---|---|
| One-shot CLI | `apex run "<task>"` | Cron jobs, CI checks, local use |
| Library | `from apex.harness import ApexHarness` | Embed in an existing service |
| HTTP service | `apex serve --port 8000` | Stdlib server; put a TLS terminator in front |
| Evaluation job | `apex eval --json` | Wire into CI as a regression gate |

`apex serve` binds `0.0.0.0` by default so it works behind a container port
mapping or a preview proxy. It is a `ThreadingHTTPServer`: fine for tens of
concurrent requests, not a substitute for a process manager. Run it under
systemd/ECS/Kubernetes with a restart policy, and scale by replicas rather than
threads.

## 2. Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Enables OpenAI models | unset |
| `ANTHROPIC_API_KEY` | Enables Anthropic models | unset |
| `GOOGLE_API_KEY` | Enables Google models | unset |
| `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `QWEN_API_KEY`, `MISTRAL_API_KEY`, `MOONSHOT_API_KEY`, `ZHIPU_API_KEY` | Other providers | unset |
| `OLLAMA_HOST` | Local Ollama endpoint | `http://localhost:11434` |
| `APEX_PROFILE` | `fast` \| `balanced` \| `quality` \| `offline` \| `prod` | `balanced` |
| `APEX_CONFIG` | Path to a JSON config file | unset |
| `APEX_WORKSPACE` | Sandbox root for tools, memory and checkpoints | `.` |
| `APEX_CHECKPOINT_DIR` | Where per-run JSON checkpoints are written | `.apex/checkpoints` |
| `APEX_SESSION_ID` | Label written into checkpoints/metrics | `default` |
| `APEX_API_TOKEN` | If set, every HTTP route requires `Authorization: Bearer <token>` | unset |
| `APEX_CORS_ORIGIN` | `Access-Control-Allow-Origin` value | `*` |
| `APEX_MAX_CONCURRENCY` | Simultaneous `/v1/run` requests before `503` | `4` |
| `APEX_MAX_COST_USD` | Hard per-run spend cap; `0` disables | `0` |
| `APEX_MAX_SPANS` | Traces kept in memory (ring buffer) | `5000` |
| `APEX_TOOLS_ALWAYS_OFFER` | Send tool schemas on every request (`false` saves ~550 input tokens per call) | `true` |
| `APEX_LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARN` \| `ERROR` | `INFO` |
| `APEX_CONFIG` | Config file path (JSON or simple YAML) | unset |
| `APEX_MODEL`, `APEX_ROUTER_MODE`, `APEX_MAX_STEPS`, `APEX_PLAN_STEPS` | Direct overrides | profile defaults |
| `APEX_TRACE`, `APEX_MEMORY_PERSIST` | Toggle tracing / durable memory | `true` |
| `APEX_ENABLE_OLLAMA` | Treat a local Ollama server as routable | unset |
| `APEX_DEBUG` | Re-raise CLI exceptions instead of printing them | unset |

Providers without a key are simply not routable; the router degrades to
`mock/apex-mock` instead of failing a request. That is deliberate: a missing key
is a configuration state, not an outage. `apex doctor` reports
`degraded_offline` and the exact `available_providers` set, and every run in
that state carries a warning in `HarnessResult.warnings` so it can never be
mistaken for a real model's output. Ollama is opt-in (`OLLAMA_HOST` or
`APEX_ENABLE_OLLAMA=1`) because probing a missing local server would otherwise
cost a connection timeout on every task.

## 3. Configuration file

```json
{
  "profile": "prod",
  "workspace": "/srv/apex",
  "checkpoint_dir": "/srv/apex/.apex/checkpoints",
  "models": {"router_mode": "balanced", "max_retries": 3, "timeout_seconds": 120},
  "agents": {"max_steps": 25, "max_plan_steps": 12, "critic_threshold": 0.6},
  "memory": {"enabled": true, "persist": true, "path": ".apex/memory.json"},
  "context": {"max_tokens": 32000, "reserve_for_output": 8000},
  "safety": {
    "block_injection": true,
    "redact_pii": true,
    "require_approval_for": ["bash", "write_file"],
    "command_timeout": 30
  },
  "observability": {"log_level": "INFO", "trace_enabled": true}
}
```

Precedence: defaults → `APEX_CONFIG` file → environment → constructor overrides.
`ApexHarness` validates on construction and raises `ValueError` listing every
problem at once rather than failing one field at a time.

## 4. Monitoring

- `GET /v1/metrics` — counters (`runs.total`, `tokens.in`, `tokens.out`,
  `cost.usd`, `events.*`) and histograms (`runs.duration_ms`).
  Scrape it and alert on: error rate, p95 `runs.duration_ms`, `cost.usd` slope.
- `GET /v1/traces` — spans grouped by trace with durations. Correlate a user
  report to a `trace_id` from the CLI footer or `HarnessResult.trace_id`.
- `GET /ready` — tools registered + memory stats. Use it for readiness probes;
  `GET /health` is liveness only.
- Logs are single-line JSON on stdout: timestamp, level, message, then fields.

## 5. Runbook

**A provider is failing.** Per-model circuit breakers open after five
consecutive failures and stay open for 30 s, so the fallback chain takes over
automatically. Look for `model failed, trying fallback` in the logs; if all
fallbacks are open, `HARNESS_EVENT run failed` reports `circuit breaker is open`.
Check the provider status page, then either restore the key/endpoint or run with
`--profile offline` to keep serving.

**Runs are slow.** Check the routed model (`apex route "<task>"`) and
`runs.duration_ms`. Common causes: a reasoning model chosen for a trivial task
(set `router_mode: fast`), tool calls with long timeouts (`command_timeout`),
or max steps being exhausted by a loop (`repeated identical tool call`).

**Costs are high.** `cost.usd` counter plus `HarnessResult.total_cost_usd` per
run. Try `router_mode: cheapest`, lower `max_plan_steps`, and disable the
swarm for routine work (`chosen_strategy` shows `plan_execute` when it runs).

**A run was blocked.** Safety refusals return `success=false` with a
`blocked: [...]` warning. This is expected for injection or PII patterns; the
input is never sent to a model.

**Memory looks wrong.** `.apex/memory.json` is a plain JSON snapshot with a
version field. It is written atomically (tmp + rename), so a crash cannot
truncate it. Delete the file to reset; a corrupt file is ignored on load rather
than crashing the harness.

**A tool hung.** `Tool.run` enforces a wall-clock timeout and returns a failed
result with `timed out after Ns`; the agent receives that as an observation and
can choose another action. Tune `safety.command_timeout`.

## 6. Security posture

- Tools run inside a sandbox root; paths that escape it are rejected before
  execution, and `bash` refuses a list of destructive patterns.
- Dangerous tools require approval unless explicitly auto-approved. The
  non-interactive default is auto-approve **for local development only** —
  review `require_approval_for` before exposing the API.
- `APEX_API_TOKEN` should be set on any network-reachable deployment; without
  it the API is open to anyone who can reach the port.
- Inputs are scanned for prompt-injection signatures and PII before reaching a
  provider; outputs are redacted and length-capped before leaving the harness.
- The API caps request bodies at 1 MB and never returns stack traces.

## 7. Upgrades

The public surface is `apex.harness.ApexHarness`, `apex.sdk.Apex`,
`apex.config.ApexConfig` and the HTTP API. Internal modules may move between
minor versions. Pin a version in production, run `apex eval` against the new
version, and compare pass rate and mean score before rolling out — the eval
suite exists precisely so that upgrading is a measurement, not a leap of faith.

## Container deployment

A production `Dockerfile` ships in the repository root. It is deliberately
small — the harness has no runtime dependencies — and runs as an unprivileged
user:

```bash
docker build -t apex:1.0.0 .
docker run --rm -p 8000:8000 \
  -e APEX_API_TOKEN="$(openssl rand -hex 24)" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e APEX_MAX_CONCURRENCY=4 \
  -e APEX_MAX_COST_USD=2.00 \
  -v apex-state:/workspace \
  apex:1.0.0
```

What the image does for you:

| Concern | Setting |
|---|---|
| Non-root | uid/gid 10001, no shell |
| Shutdown | `SIGTERM` handled, in-flight request completes |
| Health | `HEALTHCHECK` hits `/health` (no curl needed) |
| State | `/workspace` volume holds memory + checkpoints |
| Safety | `APEX_PROFILE=prod` — guardrails on, **tool auto-approval off** |

The last row matters: with `prod`, `bash` and `write_file` are denied until an
approval handler is installed. That is the correct default for a network-facing
deployment. If you need them, mount a config with `safety.auto_approve_tools=true`
and accept that tool calls are then unguarded — read `docs/SECURITY.md` first.

Kubernetes: use the `/health` endpoint for liveness and `/ready` for readiness,
set `terminationGracePeriodSeconds` to at least your longest request, and do not
scale a replica above one writer per workspace volume (the memory file is
last-write-wins).
