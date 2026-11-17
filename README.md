# APEX — Agentic Peak Execution eXperience

A production-grade agentic harness: **model routing across the 2026 frontier**, eight
reasoning strategies with an automatic selector, a nine-role agent swarm over a DAG
planner, tool use with sandboxing and approvals, durable memory, guardrails,
tracing/metrics and an evaluation suite — **on the Python standard library alone**.

```bash
python -m apex --profile offline run "Write a Python function that returns the nth Fibonacci number"
python -m apex models                 # the 2026 model registry, scored
python -m apex route "Prove sqrt(2) is irrational" --all-available   # what frontier routing picks
python -m apex eval                   # 11-task benchmark (3 run the generated code)
python -m apex serve                  # HTTP API on :8000
```

No API keys are required to run any of the above: the bundled deterministic provider
(offline mode) actually solves arithmetic, word problems and common code requests, so the
whole pipeline — routing, planning, tools, memory, critique, tracing — is exercised for
free. Point `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` at it and the same
tasks are routed to real models.

---

## Why this exists

Most agent frameworks give you either a thin API wrapper with no control (you cannot see
why a model was chosen, or what it cost) or a heavyweight graph runtime that owns your
control flow. APEX is the middle: **explicit, inspectable decisions at every step**, with
production concerns (retries, circuit breaking, sandboxing, PII redaction, tracing,
budgeting) built in rather than bolted on.

| Concern | How APEX handles it |
|---|---|
| Which model? | Weighted scoring per task type/complexity (capability, quality, price, latency, reliability) with same-provider fallback chains |
| How to think? | 8 strategies (DIRECT, COT, ReAct, ToT, Reflexion, Self-Consistency, Plan-Execute, LATS) chosen by task shape, overridable |
| One agent or many? | Single-agent fast path, or a planner → specialist swarm → critic pipeline for complex work |
| Long tasks | DAG planner with topological execution, parallel waves, failure isolation, checkpointing |
| Context growth | Token-budgeted context manager, head+tail context compaction between agents |
| Memory | Episodic + semantic + procedural, TF-IDF recall, atomic JSON persistence |
| Safety | Direct + indirect (tool-output) injection scans, PII redaction, output caps, deny-by-default tool approval, path sandbox that survives symlinks and sibling prefixes, SSRF gate, credential-free child env |
| Cost | Per-run token and USD accounting, cheapest/quality/fast routing modes |
| Debuggability | Nested spans, counters/histograms, structured JSON logs, `/v1/traces` grouped by trace |
| Quality | Critic loop with retry-on-low-score, plus an 11-task eval suite — three tasks execute the generated code in a sandbox and probe the results — gateable in CI with `--min-pass-rate` |

## Install

```bash
pip install -e .          # or: pip install -e ".[dev]" for pytest/ruff
python -m unittest discover -s tests    # 385 tests, no network, no keys
```

Python 3.10+. Zero hard dependencies.

## Quickstart (SDK)

```python
from apex.sdk import Apex

apex = Apex(profile="offline")          # or "balanced" / "quality" / "fast" / "prod"

result = apex.run("Compare LangGraph, AutoGen and CrewAI for a production agent system")
print(result.output)
print(result.model_used, result.strategy.value, f"${result.total_cost_usd:.4f}")

apex.remember("Staging deploys use blue-green rollout behind a feature flag")
print(apex.run("What do you know about our staging deploy?").output)
print(apex.route("Refactor the payment module")["model"]["id"])
print(apex.doctor())
```

## CLI

```
apex run "<task>" [--json] [--strategy NAME] [--model ID]
apex models [--provider NAME] [--json]   # registry with scores and prices
apex route "<task>" [--json] [--all-available]   # decision + classification
apex eval [--json] [--min-pass-rate R]   # benchmark suite, CI gate
apex doctor                              # config, tools, providers, memory, health
apex remember "<fact>"                   # durable memory
apex serve [--host --port]               # HTTP API
```

Global flags (`--profile`, `--workspace`, `--model`, `--config`, `--version`) may appear
**before or after** the subcommand. `--config` loads a JSON/YAML file; precedence is
defaults < profile < file < environment < CLI flags. `apex eval --min-pass-rate 0.95`
exits non-zero when quality regresses, which makes it usable as a CI gate.

## HTTP API

```
GET  /health          liveness
GET  /ready           readiness: tools + memory
GET  /v1/models       registry
GET  /v1/metrics      counters + histograms
GET  /v1/traces       spans grouped by trace
GET  /v1/memory       memory stats
POST /v1/run          {task, model?, strategy?}
POST /v1/route        {task} -> model + fallbacks + classification
POST /v1/remember     {fact, source?}
POST /v1/eval         {tasks?: [ids]}
```

Set `APEX_API_TOKEN` to require `Authorization: Bearer <token>` on every route (compared in
constant time). `APEX_CORS_ORIGIN` controls `Access-Control-Allow-Origin` (default `*`), and
`APEX_MAX_CONCURRENCY` caps simultaneous `/v1/run` calls (default 4) — beyond that clients get
`503 overloaded` with `Retry-After` instead of an unbounded queue. Bodies are capped at 1 MB,
malformed JSON returns 400, unknown strategies return 400 with the allowed list, a fact that
fails the guardrail scan returns 403, and unexpected failures return 500 with a stable error
shape — never a stack trace. Idle sockets time out at 30 s and `SIGTERM` shuts down
gracefully.

## Configuration

Layered: defaults → JSON file (`APEX_CONFIG`) → environment (`APEX_*`) → constructor
overrides. Profiles: `fast`, `balanced` (default), `quality`, `offline`, `prod`.

```python
from apex.config import ApexConfig
from apex.harness import ApexHarness

config = ApexConfig(profile="quality", workspace="./sandbox")
config.models.router_mode = "quality"      # balanced | quality | fast | cheapest
config.safety.require_approval_for = ["bash", "write_file"]
harness = ApexHarness(config, api_keys={"openai": "..."})
```

Provider keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
(Ollama is opt-in via `OLLAMA_HOST`). **Only providers that have credentials are
routable** — a keyless install degrades to the offline provider with an explicit warning
instead of failing every task, and `doctor()` reports `degraded_offline` plus the set of
available providers.

## Architecture

```
task
 └─ safety.inspect_input        injection signatures, PII redaction
    └─ classify_task            task type + complexity (weighted lexicon)
       └─ memory.recall         facts first, then past episodes (TF-IDF, self-echo filtered)
          └─ ModelRouter.route   score every model, pick winner + fallback chain
             └─ select_strategy  8 strategies, task-shaped
                ├─ single agent  strategy prompt → ReAct loop → critique
                └─ swarm         DAG planner → agents (parallel waves) → critic → synthesis
                   └─ safety.check_output → memory.persist → checkpoint → traces/metrics
```

Key decisions, and why:

- **Stdlib-only core.** A harness that cannot start without a vendor SDK is a liability.
  Providers speak HTTP through `urllib`; swapping in `httpx` later is a one-file change.
- **Deterministic planner.** Clause decomposition and agent routing are inspectable and
  testable; an LLM plan is used when it parses and validates, never trusted blindly.
- **Per-model circuit breakers.** One provider's outage must not empty the fallback chain —
  a bug this repository fixed and now regression-tests.
- **Bounded context between agents.** Head+tail compaction under a hard character budget;
  unbounded context passing is how multi-agent runs blow up in cost and lose accuracy.
- **Honest offline provider.** The mock computes real answers and says "offline" instead of
  pretending to be frontier output, so demos and evals mean something.

## Evaluations

`python -m apex eval` runs eleven tasks and reports pass rate, mean score, tokens and cost.
Four scorers are supported — exact, contains, regex and heuristic quality — plus **`exec`**:
for the three coding tasks the generated code is extracted from its fence and actually run
in a subprocess with a restricted environment and a timeout, then probed for correctness
(`fibonacci(10) == 55`, `is_prime(28) is False`, …). "Contains `def`" is not a quality
signal; "runs and returns 55" is.

Offline it scores **100% pass, ~0.89 mean** — a regression baseline for the harness itself
(routing, planning, tools, critic, memory), not a claim about any model's intelligence.
`apex eval --min-pass-rate 0.95` makes it a CI gate.

## Reliability behaviour worth knowing

| Situation | What APEX does |
|---|---|
| No provider credentials | Routes to the offline provider, warns, keeps working |
| One provider down | Per-model breaker opens; fallback chain continues |
| Model rejects `temperature` | Retries once without it (never silently drops intent) |
| Tool name hallucinated | Failed observation returned to the model, run continues |
| Tool hangs | Wall-clock timeout, failed observation, loop continues |
| Agent loops | Repeated-call fingerprint detection stops it |
| Context grows | Biggest tool outputs summarised, oldest turns evicted, system kept |
| Tool output contains instructions | Observation tagged `[UNTRUSTED DATA]` and the attempt recorded |
| Memory file unwritable | Warning logged, run continues (persistence is best-effort) |
| Memory file corrupt | Ignored on load, harness starts clean instead of crashing |
| API saturated | `503 overloaded` + `Retry-After`, not an unbounded queue |
| Crash | JSON checkpoint per run |
| Corrupt memory file | Ignored on load; harness starts clean |

## Testing

```bash
python -m unittest discover -s tests -v     # 385 tests
python -m pytest tests -q                   # if pytest is installed
ruff check apex tests
```

385 tests across 18 files: models/routing/classification · registry integrity ·
tools + sandbox security · reasoning + memory · execution (retry/breaker/checkpoint) ·
harness end-to-end + guardrails · offline provider behaviour · real-provider
request/response contracts · memory persistence and eviction integrity ·
planner/swarm/synthesis · strategies and voting · config layering + SDK · observability ·
evaluation scorers · API hardening · API + CLI integration.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline, module map, design decisions
- [`docs/MODELS.md`](docs/MODELS.md) — the 2026 model scan, tiers, routing formula
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — when each reasoning strategy wins
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — APEX vs LangGraph, AutoGen, CrewAI, OpenAI
  Agents SDK, Claude Agent SDK, OpenHands, SWE-agent, smolagents, LlamaIndex
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — deployment shapes, env vars, monitoring, runbook
- [`docs/API.md`](docs/API.md) — HTTP endpoint reference with examples
- [`docs/SECURITY.md`](docs/SECURITY.md) — threat model, what is defended, what is not

## Limitations (stated plainly)

- The offline provider is a rule-based reasoner, not a general model: it handles arithmetic,
  word problems, common code asks, planning/comparison/analysis formats, and is honest
  elsewhere. Frontier quality requires a live provider.
- `web_search` scrapes DuckDuckGo's HTML endpoint. It is rate-limited and can break; treat it
  as best-effort with the tool result surfaced to the caller.
- The DAG planner can take a model-authored JSON plan, but the default path is heuristic.
  Truly open-ended task decomposition is where every harness is still weakest.
- Single-process by default. For multi-node execution you need an external queue; the
  checkpointer and JSON memory are local-disk designs.

## License

MIT.
