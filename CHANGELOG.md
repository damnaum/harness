# Changelog

All notable changes to APEX. Versioned with [SemVer](https://semver.org/).

## [Unreleased]

### Correctness (found by running the harness, not by reading it)
- **Tool tasks were answered blind.** Tool schemas were gated behind a keyword
  heuristic that missed concrete filenames, and the offline provider demanded a
  second keyword match, so "read notes.txt and tell me the release" produced a
  confident answer that never opened the file. Schemas are now offered every
  request (`agents.always_offer_tools`, `APEX_TOOLS_ALWAYS_OFFER`), the strategy
  heuristic recognises filename tokens, and the answer is grounded in the file.
- **The ReAct loop never compacted.** Prompts grew 881 → 20 881 → 40 881 chars
  and kept climbing; `fit_messages` now runs before every model call, restates
  the task after eviction and drops orphaned tool turns that providers reject.
- **Scope was judged by word count.** A 22-word request with five deliverables
  classified as SIMPLE and skipped the planner; complexity now has a floor based
  on the number of distinct deliverables.
- **Retry and the breaker treated every error as transient.** A 401 was retried
  three times with backoff, and five malformed requests could open the shared
  breaker. `errors.is_retryable` retries 429/5xx/timeouts only.
- **Denied privileged tools were invisible.** `success=True` with no warning when
  the approval gate refused `bash`; now `tool_denied` + a run warning, with
  `tool_failed` counting ordinary tool errors separately.
- **HTTP verbs answered with an HTML 501.** Wrong-verb requests now return 405
  with `Allow` and a JSON body; `HEAD` mirrors the GET routes.
- **Traces grew without bound** (a server-lifetime leak); the span list is now a
  ring buffer (`APEX_MAX_SPANS`).
- **The email redaction regex was quadratic** (3s on 50 KB of plain text); it is
  bounded and linear now, with the same redaction behaviour.

### Tests
- 556 tests across 24 suites, including docs-consistency checks that bind the
  README/CHANGELOG/API/SECURITY/OPERATIONS claims to the code.

## [1.0.0] — 2026-11-17

First production release. Standard-library only; no runtime dependencies.

### Harness
- Multi-model router over a 23-model / 12-provider 2026 registry, scored on
  capability, quality, price, latency and reliability, with same-provider-first
  fallback chains and per-model circuit breakers.
- Eight reasoning strategies (Direct, CoT, ReAct, ToT, Reflexion,
  Self-Consistency, Plan-and-Execute, LATS) with a deterministic, explainable
  selector; best-of-N and majority voting are used for the strategies that call
  for them.
- Nine-role swarm (supervisor, planner, executor, researcher, coder, tester,
  writer, analyst, critic) over a DAG planner with topological execution,
  failure isolation and critic-gated synthesis.
- Three-layer memory (episodic, semantic, procedural) with TF-IDF recall,
  `RLock`-guarded mutation, `fsync`-ed atomic persistence and eviction-safe
  index rebuilds.
- Typed error hierarchy, retry with jitter, circuit breaking, loop detection,
  checkpoints, token-budgeted context management.

### Safety
- Direct and indirect prompt-injection detection; tool observations are scanned
  before they reach the model and tagged `[UNTRUSTED DATA]`.
- PII and secret redaction that does not destroy dates, versions or numeric
  answers.
- Workspace path containment that survives symlinks and sibling-prefix escapes;
  deny-by-default tool approval; command normalisation and deny rules; SSRF
  resolution and IP-class checks; credential-free child environments; output and
  time caps; guarded calculator.

### Operations
- CLI (`run`, `models`, `route`, `eval`, `doctor`, `remember`, `serve`,
  `--version`), SDK facade, and a stdlib HTTP API with constant-time bearer auth,
  body caps, concurrency limiting (`503` + `Retry-After`), idle timeouts and
  graceful shutdown.
- Labelled metrics, nested spans, JSON logs, event bus, `/v1/metrics`,
  `/v1/traces`.
- Hard per-run spend cap (`models.max_cost_usd`) that binds on both the
  single-agent and swarm paths.
- 457 tests, offline and key-free; 11-task eval suite with three
  execution-verified coding tasks; `apex eval --min-pass-rate` as a CI gate.
- Production `Dockerfile` (non-root, healthcheck, volume, `prod` profile).
