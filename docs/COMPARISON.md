# APEX vs existing harnesses

Compared against the major agent harnesses as of late 2025 / 2026. The point is
not that APEX wins everywhere — it loses several of these comparisons, and those
are listed at the bottom. The point is that each design choice is deliberate and
each gap is known.

## Landscape

| Harness | Type | What it does better than everyone | Where it hurts |
|---|---|---|---|
| **LangGraph / LangChain** | Graph orchestration | Durable execution, checkpointing, streaming, ecosystem | Large dependency tree; model routing is DIY; abstractions leak into your code |
| **AutoGen (Microsoft)** | Multi-agent conversation | Flexible agent-to-agent dialogue, human-in-the-loop | Chat-centric rather than task-DAG-centric; sandboxing is your problem; cost control manual |
| **CrewAI** | Role-based crews | Fastest path from "a team of agents" to running code | Limited reasoning-strategy choice; no router; thin memory; observability DIY |
| **OpenAI Agents SDK** | Vendor SDK | Excellent tool loop, guardrails, built-in tracing | OpenAI-only; no cross-provider routing or fallback |
| **Claude Agent SDK / Claude Code** | Vendor agent | Best-in-class coding loop and subagents | Anthropic-only, closed harness |
| **OpenHands** | Coding agent platform | Real containerized SWE environment | Coding-only; heavy infrastructure |
| **SWE-agent** | SWE benchmark agent | Strong SWE-bench results, principled ACI | Single-task focus |
| **smolagents (HF)** | Minimal agents | Tiny, readable, code-writing agents | Minimal safety, observability, routing, eval |
| **LlamaIndex agents** | RAG-first agents | Best retrieval integration | Retrieval-first; orchestration secondary |
| **APEX** | Agentic harness | Router + strategy selector + swarm + memory + safety + eval in one stdlib-only core | Young ecosystem, no streaming UI, no OS-level sandbox |

## Dimension-by-dimension

| Dimension | LangGraph | AutoGen | CrewAI | OpenAI SDK | APEX |
|---|---|---|---|---|---|
| Hard dependencies | many | several | several | SDK | **zero (stdlib)** |
| Cross-provider routing | DIY | DIY | DIY | ✗ | **23 models, scored, with fallbacks** |
| Reasoning-strategy choice | DIY | prompt it | limited | prompt it | **8 strategies, auto-selected** |
| Task planning | you build the graph | conversation | task list | hand-offs | **DAG planner + specialists** |
| Memory | add-on | conversation state | short-term | session | **episodic/semantic/procedural + TF-IDF** |
| Safety layer | DIY | DIY | DIY | guardrails | **input + tool-output injection scan, PII, approval gate, SSRF** |
| Observability | LangSmith | built-in | events | built-in | **spans/metrics/events, `/v1/traces`** |
| Evaluation | DIY | DIY | DIY | DIY | **11 tasks incl. execution-verified code** |
| Offline/CI mode | ✗ | ✗ | ✗ | ✗ | **deterministic provider, no keys** |
| Durable execution | **best** | partial | partial | ✗ | JSON checkpoints (simpler) |
| Streaming | **best** | good | good | good | ✗ (complete responses) |
| Sandboxing | DIY | DIY | DIY | ✗ | logical sandbox, **not OS-level** |
| Ecosystem | **best** | large | large | large | minimal |

## What APEX took from each, and what it changed

1. **From LangGraph — the good idea is explicit state.** APEX keeps a real DAG
   planner with topological execution and per-step status, but drops the graph
   *framework*: the plan is data (`Plan`/`PlanStep`), so you can inspect, mutate
   and re-run it without learning an API.
2. **From AutoGen/CrewAI — the good idea is role specialisation.** APEX runs nine
   roles (supervisor, planner, executor, researcher, coder, tester, writer,
   analyst, critic), but attaches them to DAG steps rather than to a chat
   transcript. That removes the "everyone talks to everyone" token burn and makes
   failures isolatable: a failed step skips only its dependents.
3. **From the vendor SDKs — the good idea is a tight tool loop with guardrails.**
   APEX has the loop, the guardrails and the tracing, provider-agnostically. The
   trade was vendor polish for portability: no SDK, `urllib` only.
4. **From OpenHands/SWE-agent — the good idea is confinement.** APEX confines
   filesystem access to a workspace root (resolving symlinks and sibling-prefix
   escapes), blocks destructive shell patterns, deny-by-default gates privileged
   tools, and times out every tool. It does *not* provide a container.
5. **From smolagents — the good idea is a small core.** APEX's core imports
   nothing outside the standard library, so it runs in a minimal container, in
   CI, and behind a corporate proxy without a dependency audit.

## Where APEX genuinely improves on the field

- **Routing is a first-class subsystem.** Most harnesses assume one provider.
  APEX scores 23 models per task, explains the decision (`reason`), returns the
  fallback chain, and keeps per-model circuit breakers so an outage degrades
  instead of failing.
- **Strategy selection is explicit.** Direct / CoT / ReAct / ToT / Reflexion /
  Self-Consistency / Plan-Execute / LATS, chosen by task shape, overridable, with
  the reasoning recorded in `metadata.strategy_reason`.
- **Best-of-N is distinguished from voting.** ToT and LATS rank candidates by
  critic score; self-consistency normalises answers (`40`, `40.0`, "The answer is
  40." agree) and votes. Getting these two backwards is a common bug.
- **Offline mode is a real reasoner, not a stub.** Arithmetic, word problems,
  common code synthesis, comparison and planning formats all work with zero keys,
  which makes CI and demos meaningful. It labels itself as offline instead of
  impersonating a frontier model.
- **The eval suite executes generated code.** Three tasks extract the fenced code,
  run it in a subprocess with a restricted environment and a timeout, and probe
  the results. "Contains `def`" is not a quality signal; "`fibonacci(10) == 55`"
  is.
- **Indirect prompt injection is treated as a real attack.** Tool output is
  scanned before it reaches the model and tagged `[UNTRUSTED DATA]` when it
  contains instructions.

## Where APEX loses (and what to use instead)

- **Durable, resumable long-running workflows** → LangGraph. APEX checkpoints
  runs to JSON; it does not replay a partially executed graph after a crash.
- **Token streaming to a UI** → any of them. APEX returns complete results.
- **True isolation of generated code** → OpenHands (containers) or a VM. APEX's
  sandbox is logical: `python_exec` runs with the OS user's privileges.
- **A large integration ecosystem (vector stores, loaders, tools)** → LlamaIndex
  or LangChain. APEX ships ten tools and expects you to add your own.
- **Embeddings-based memory at scale** → LlamaIndex. APEX uses TF-IDF, which is
  fast and dependency-free but semantic-blind past a few thousand facts.
- **Vendor-optimised coding experience** → Claude Code / Codex. APEX is a
  harness, not a product.

## Design trade-offs, stated openly

| Decision | Benefit | Cost |
|---|---|---|
| Stdlib-only | Runs anywhere, easy audit, no supply-chain surface | Reimplementing HTTP clients, TF-IDF, YAML subset |
| Heuristic planner by default | Deterministic, inspectable, testable, free | Weaker on genuinely open-ended decomposition |
| Heuristic critic | Works offline, no cost | Cannot judge subtle correctness; model critique is optional |
| Logical sandbox | No infra required | Not a security boundary against hostile code |
| JSON memory | Simple, debuggable, portable | Not concurrent-safe across processes |
| Deny-list for shell | Catches accidents | Determined attackers will find another phrasing |
