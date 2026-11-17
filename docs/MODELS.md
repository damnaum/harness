# 2026 model scan — registry and per-model strategy

APEX ships **23 models across 12 providers**. This document explains what each
model is for, how the router decides between them, and where the numbers come
from — so the routing decisions are auditable instead of magic.

## Methodology (and its limits)

Scores are **0–10 expert estimates**, not measured benchmarks of your workload.
They are derived from published results (SWE-bench Verified, GPQA Diamond, AIME,
MATH, MMLU-Pro, LiveCodeBench, LMArena) plus hands-on agentic behaviour: tool-call
reliability, instruction adherence across long contexts, and how often a model
needs a retry. Prices are public list prices per 1M tokens at the time of writing.

Caveats, stated plainly:

- Vendor benchmarks are self-reported and optimistic; independent numbers are
  weighted higher.
- Ability decays with context length; a 400k-context model is not frontier-quality
  at 400k.
- Prices change monthly. Treat the registry as a starting point you can override —
  one row in `apex/models/registry.py`, or pass `--model` to bypass routing.
- The right way to settle a disagreement is `apex eval` against **your** tasks.

### Dimensions

| Dimension | Meaning | Used for |
|---|---|---|
| `reasoning` | Multi-step logic, proofs, hard analysis | reasoning, planning, data |
| `coding` | Code generation and refactoring quality | code, debugging |
| `tools` | Reliability of native tool/function calling | ReAct, agent loops |
| `research` | Breadth, synthesis, source handling | research |
| `creativity` | Prose quality and voice | writing |
| `speed` | Tokens/second and time-to-first-token | `fast` mode |
| `long_context` | Effective (not advertised) context use | long documents |
| `reliability` | Uptime, format adherence, low retry rate | every route |

## The registry

### Frontier tier

| Model | Context | $/1M in → out | Strongest at | Watch out for |
|---|---|---|---|---|
| **GPT-5.2** (OpenAI) | 400k | 1.75 → 14.00 | Agentic coding, tool use, planning | Cost on long agent runs |
| **Claude Opus 4.5** (Anthropic) | 200k | 5.00 → 25.00 | Long-horizon coding, careful writing | Most expensive per token |
| **Claude Sonnet 4.5** (Anthropic) | 500k | 3.00 → 15.00 | Balanced coding + writing at scale | Not the absolute ceiling |
| **Gemini 3 Pro** (Google) | 1M | 2.00 → 12.00 | Research, 1M-context synthesis, multimodal | Occasionally over-broad |
| **Grok 4.1** (xAI) | 256k | 3.00 → 15.00 | Deep reasoning, current-events knowledge | Smaller ecosystem |
| **o3** (OpenAI) | 200k | 2.00 → 8.00 | Hard reasoning, math | Slow; rejects `temperature` |
| **DeepSeek V4** | 256k | 0.27 → 1.10 | Frontier-adjacent quality at ~1/10 cost | Lower tool-call reliability |
| **Qwen 3.5 Max** (Alibaba) | 256k | 0.86 → 3.40 | Coding, multilingual | Data-residency questions |

### Value / fast tier

| Model | Context | $/1M in → out | Role in the fleet |
|---|---|---|---|
| **GPT-5.1** (OpenAI) | 400k | 1.25 → 10.00 | Quality-mode workhorse for code |
| **GPT-5 mini** (OpenAI) | 400k | 0.25 → 2.00 | Cheap agentic steps, bulk extraction |
| **o4-mini** (OpenAI) | 200k | 1.10 → 4.40 | Reasoning on a budget |
| **Claude Haiku 4.5** (Anthropic) | 200k | 1.00 → 5.00 | Near-frontier cheap writing/analysis |
| **Gemini 2.5 Pro / Flash** (Google) | 1M | 1.25 → 10.00 / 0.30 → 2.50 | Long context; Flash for latency |
| **Grok 4 Fast** (xAI) | 2M | 0.20 → 0.50 | Cheapest route for simple work |
| **Qwen3-Coder** (Alibaba) | 256k | 0.30 → 1.20 | Code-specialist value pick |
| **Kimi K2** (Moonshot) | 256k | 0.60 → 2.50 | Long-context agentic work |
| **GLM-4.6** (Zhipu) | 200k | 0.60 → 2.20 | Coding value, strong tool use |
| **Llama 4 Maverick** (Meta) | 1M | 0.20 → 0.60 | Open weights, very cheap, 1M context |
| **Mistral Large 3** (Mistral) | 256k | 2.00 → 6.00 | EU-hosted, dependable tool use |
| **DeepSeek R1** (DeepSeek) | 128k | 0.55 → 2.19 | Visible reasoning traces |

### Local / deterministic tier

| Model | Role |
|---|---|
| **Ollama Llama 3.1 8B** (Ollama) | Private, local, free — opt-in via `OLLAMA_HOST` |
| **APEX Mock** (`mock/apex-mock`) | Deterministic rule-based reasoner; powers offline mode, CI and the eval suite. Computes real answers and labels itself honestly |

## Per-model strategy

How the router actually uses each family:

- **OpenAI GPT-5.x** — first choice for *code* and *planning* in `quality` mode.
  Strong tool loops; expensive output tokens, so keep ReAct loops tight.
- **Anthropic Claude** — first choice for *writing* and long-horizon *coding*.
  Sonnet is the default writing route in `balanced`; Opus only when quality mode
  and the task is complex.
- **Google Gemini** — first choice for *research*, *data* and anything needing
  >200k context. Its 1M window is why long-document tasks land here.
- **xAI Grok** — strong reasoning plus current knowledge; Grok 4 Fast is the
  default `cheapest`/`fast` route for simple tasks.
- **DeepSeek / Qwen / GLM / Kimi** — value coding and agentic work. Qwen3-Coder
  wins *code* in `balanced` because its coding score is competitive and its cost
  is an order of magnitude lower.
- **Llama 4 / Mistral** — cheap generalists and EU/self-host constraints.
- **Mock** — the fallback of last resort, and the only provider that needs no
  credentials. It is always routable, so a keyless install degrades instead of
  failing.

## Observed routing (reproduce with `apex route "<task>" --all-available`)

With every provider credentialed (the default run degrades to whatever keys you
set, and to `mock/apex-mock` with none):

| Task (as classified) | `balanced` | `quality` | `cheapest` |
|---|---|---|---|
| code | qwen/qwen3-coder | openai/gpt-5.1 | qwen/qwen3-coder |
| debugging | qwen/qwen3-coder | openai/gpt-5.1 | qwen/qwen3-coder |
| reasoning | openai/gpt-5.2 | openai/gpt-5.2 | xai/grok-4-fast |
| research | google/gemini-3-pro | anthropic/claude-opus-4.5 | xai/grok-4-fast |
| writing | anthropic/claude-sonnet-4.5 | anthropic/claude-sonnet-4.5 | xai/grok-4-fast |
| data | google/gemini-3-pro | openai/gpt-5.2 | xai/grok-4-fast |
| planning | google/gemini-3-pro | openai/gpt-5.2 | xai/grok-4-fast |

Note `cheapest` converges on one model for everything — that is the honest result
of a cost-dominated objective. If you need *some* quality floor, use `balanced`
and set `models.default_model` only for a specific workload.

## The routing formula

```
score = capability·Wc + quality·Wq + reliability·0.1
        − cost_penalty·Wcost − latency_penalty·Wlat
```

- `capability` = model score for the classified task type / 10
- `quality` = complexity demand × mean(reasoning, coding, reliability)
- `cost_penalty` = `log10(cost_in_usd × 1000 + 1) / 3`, clipped to 1.0 —
  log-scaled so cheap-vs-cheap differences matter and expensive-vs-expensive
  differences do not dominate
- `latency_penalty` = `(10 − speed) / 10`

Weights by mode:

| Mode | capability | quality | cost | latency |
|---|---|---|---|---|
| balanced | 0.45 | 0.25 | 0.15 | 0.15 |
| quality | 0.55 | 0.35 | 0.05 | 0.05 |
| fast | 0.35 | 0.15 | 0.15 | 0.35 |
| cheapest | 0.35 | 0.10 | 0.45 | 0.10 |

**Fallbacks** are ordered same-provider-first (shared key, region and failure
domain) then global next-best. Each model has its own circuit breaker, so one
provider's outage does not consume the chain.

## Adding or overriding a model

```python
from apex.models.registry import MODELS, ModelInfo
from apex.types import TaskType

MODELS.append(ModelInfo(
    id="my/finetune-v3", provider="openai", display_name="Fine-tune v3",
    context_window=128_000, max_output=8_000,
    input_price_per_1m=0.5, output_price_per_1m=1.5,
    reasoning=7, coding=9, tools=8, research=5, creativity=5,
    speed=8, long_context=5, reliability=8,
))
```

Any provider that speaks the OpenAI chat-completions shape works with **no new
code** — set the matching `*_API_KEY` and the provider is routable. Adding a new
*wire format* means one `BaseProvider` subclass (see `apex/models/base.py`).
