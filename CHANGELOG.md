# Changelog

All notable changes to APEX. Versioned with [SemVer](https://semver.org/).

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
- 414 tests, offline and key-free; 11-task eval suite with three
  execution-verified coding tasks; `apex eval --min-pass-rate` as a CI gate.
- Production `Dockerfile` (non-root, healthcheck, volume, `prod` profile).
