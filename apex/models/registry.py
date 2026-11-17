"""2026 model registry — scanned frontier + open models with capability scores.

Scores are 0-10 expert estimates (reasoning/coding/tools/research/creativity/
speed/long_context/reliability).  Prices are public list prices per 1M tokens
(USD), refreshed for the late-2025 / 2026 generation.  The router uses these
to pick the best model per task under a cost/latency/quality policy.

See docs/MODELS.md for the full scan narrative and selection rationale.
"""

from __future__ import annotations

from ..types import ModelInfo

# id format: "<provider>/<model-id>" — provider prefix routes to a provider impl.
MODELS: list[ModelInfo] = [
    # ---------------------------------------------------------- OpenAI
    ModelInfo(
        id="openai/gpt-5.2", provider="openai", display_name="GPT-5.2",
        context_window=400_000, max_output=128_000,
        input_price_per_1m=1.75, output_price_per_1m=14.0,
        reasoning=10, coding=10, tools=10, research=9, creativity=9,
        speed=6, long_context=9, reliability=10,
        strengths=("agentic coding", "tool use", "reasoning depth", "instruction following"),
        best_for=("code", "reasoning", "planning", "debugging"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="openai/gpt-5.1", provider="openai", display_name="GPT-5.1",
        context_window=400_000, max_output=128_000,
        input_price_per_1m=1.25, output_price_per_1m=10.0,
        reasoning=9, coding=10, tools=10, research=9, creativity=9,
        speed=7, long_context=9, reliability=10,
        strengths=("coding", "agents", "verbosity control"),
        best_for=("code", "general", "planning"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="openai/o3", provider="openai", display_name="o3",
        context_window=200_000, max_output=100_000,
        input_price_per_1m=2.0, output_price_per_1m=8.0,
        reasoning=10, coding=9, tools=8, research=9, creativity=7,
        speed=4, long_context=7, reliability=9,
        strengths=("deep reasoning", "math", "science", "hard logic"),
        best_for=("reasoning", "data", "research"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="openai/o4-mini", provider="openai", display_name="o4-mini",
        context_window=200_000, max_output=100_000,
        input_price_per_1m=1.10, output_price_per_1m=4.40,
        reasoning=8, coding=8, tools=8, research=7, creativity=6,
        speed=8, long_context=7, reliability=9,
        strengths=("fast reasoning", "cheap", "STEM"),
        best_for=("reasoning", "general", "data"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="openai/gpt-5-mini", provider="openai", display_name="GPT-5 mini",
        context_window=400_000, max_output=128_000,
        input_price_per_1m=0.25, output_price_per_1m=2.0,
        reasoning=7, coding=8, tools=9, research=7, creativity=7,
        speed=9, long_context=9, reliability=9,
        strengths=("cheap agentic", "fast", "tool use"),
        best_for=("general", "code", "planning"),
    ),
    # ---------------------------------------------------------- Anthropic
    ModelInfo(
        id="anthropic/claude-opus-4.5", provider="anthropic", display_name="Claude Opus 4.5",
        context_window=200_000, max_output=64_000,
        input_price_per_1m=5.0, output_price_per_1m=25.0,
        reasoning=10, coding=10, tools=10, research=10, creativity=10,
        speed=5, long_context=8, reliability=10,
        strengths=("best coding", "long agentic runs", "nuanced writing", "safety"),
        best_for=("code", "debugging", "writing", "planning", "reasoning"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-4.5", provider="anthropic", display_name="Claude Sonnet 4.5",
        context_window=500_000, max_output=64_000,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
        reasoning=9, coding=10, tools=10, research=9, creativity=10,
        speed=7, long_context=10, reliability=10,
        strengths=("coding quality", "1M context beta", "balanced"),
        best_for=("code", "general", "research", "writing"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="anthropic/claude-haiku-4.5", provider="anthropic", display_name="Claude Haiku 4.5",
        context_window=200_000, max_output=64_000,
        input_price_per_1m=1.0, output_price_per_1m=5.0,
        reasoning=7, coding=8, tools=9, research=7, creativity=8,
        speed=9, long_context=8, reliability=9,
        strengths=("fast", "cheap", "near-frontier"),
        best_for=("general", "code", "planning"),
    ),
    # ---------------------------------------------------------- Google
    ModelInfo(
        id="google/gemini-3-pro", provider="google", display_name="Gemini 3 Pro",
        context_window=1_000_000, max_output=65_000,
        input_price_per_1m=2.0, output_price_per_1m=12.0,
        reasoning=10, coding=9, tools=9, research=10, creativity=9,
        speed=7, long_context=10, reliability=9,
        strengths=("1M context", "multimodal", "research", "reasoning"),
        best_for=("research", "reasoning", "data", "general"),
        supports_vision=True, supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="google/gemini-2.5-pro", provider="google", display_name="Gemini 2.5 Pro",
        context_window=1_000_000, max_output=65_000,
        input_price_per_1m=1.25, output_price_per_1m=10.0,
        reasoning=9, coding=9, tools=9, research=9, creativity=8,
        speed=7, long_context=10, reliability=9,
        strengths=("long context", "value", "multimodal"),
        best_for=("research", "general", "data"),
        supports_vision=True, supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="google/gemini-2.5-flash", provider="google", display_name="Gemini 2.5 Flash",
        context_window=1_000_000, max_output=65_000,
        input_price_per_1m=0.30, output_price_per_1m=2.50,
        reasoning=7, coding=7, tools=8, research=7, creativity=7,
        speed=10, long_context=10, reliability=8,
        strengths=("fastest", "cheap", "1M context"),
        best_for=("general", "research", "planning"),
    ),
    # ---------------------------------------------------------- xAI
    ModelInfo(
        id="xai/grok-4.1", provider="xai", display_name="Grok 4.1",
        context_window=256_000, max_output=64_000,
        input_price_per_1m=3.0, output_price_per_1m=15.0,
        reasoning=10, coding=9, tools=9, research=9, creativity=8,
        speed=7, long_context=8, reliability=8,
        strengths=("reasoning", "real-time knowledge", "humour"),
        best_for=("reasoning", "research", "general"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="xai/grok-4-fast", provider="xai", display_name="Grok 4 Fast",
        context_window=2_000_000, max_output=64_000,
        input_price_per_1m=0.20, output_price_per_1m=0.50,
        reasoning=8, coding=8, tools=8, research=8, creativity=7,
        speed=9, long_context=10, reliability=8,
        strengths=("2M context", "very cheap", "fast"),
        best_for=("general", "research", "data"),
    ),
    # ---------------------------------------------------------- DeepSeek
    ModelInfo(
        id="deepseek/deepseek-v4", provider="deepseek", display_name="DeepSeek V4",
        context_window=256_000, max_output=64_000,
        input_price_per_1m=0.27, output_price_per_1m=1.10,
        reasoning=9, coding=9, tools=8, research=8, creativity=7,
        speed=7, long_context=8, reliability=8,
        strengths=("open weights", "cheap frontier", "coding"),
        best_for=("code", "reasoning", "general"),
        supports_reasoning_mode=True,
    ),
    ModelInfo(
        id="deepseek/deepseek-r1", provider="deepseek", display_name="DeepSeek R1",
        context_window=128_000, max_output=32_000,
        input_price_per_1m=0.55, output_price_per_1m=2.19,
        reasoning=10, coding=8, tools=6, research=8, creativity=6,
        speed=5, long_context=6, reliability=8,
        strengths=("pure reasoning", "math", "open weights"),
        best_for=("reasoning", "data"),
        supports_reasoning_mode=True,
    ),
    # ---------------------------------------------------------- Alibaba Qwen
    ModelInfo(
        id="qwen/qwen3.5-max", provider="qwen", display_name="Qwen 3.5 Max",
        context_window=256_000, max_output=64_000,
        input_price_per_1m=0.86, output_price_per_1m=3.40,
        reasoning=9, coding=10, tools=9, research=8, creativity=8,
        speed=7, long_context=8, reliability=8,
        strengths=("coding", "agentic", "open weights"),
        best_for=("code", "debugging", "general"),
    ),
    ModelInfo(
        id="qwen/qwen3-coder", provider="qwen", display_name="Qwen3-Coder",
        context_window=256_000, max_output=64_000,
        input_price_per_1m=0.30, output_price_per_1m=1.20,
        reasoning=7, coding=10, tools=9, research=6, creativity=6,
        speed=8, long_context=8, reliability=8,
        strengths=("code generation", "repo-level edits", "cheap"),
        best_for=("code", "debugging"),
    ),
    # ---------------------------------------------------------- Meta / Mistral / others
    ModelInfo(
        id="meta/llama-4-maverick", provider="meta", display_name="Llama 4 Maverick",
        context_window=1_000_000, max_output=32_000,
        input_price_per_1m=0.20, output_price_per_1m=0.60,
        reasoning=8, coding=8, tools=8, research=8, creativity=7,
        speed=8, long_context=10, reliability=8,
        strengths=("open weights", "1M context", "cheap"),
        best_for=("general", "research", "data"),
    ),
    ModelInfo(
        id="mistral/mistral-large-3", provider="mistral", display_name="Mistral Large 3",
        context_window=256_000, max_output=32_000,
        input_price_per_1m=2.0, output_price_per_1m=6.0,
        reasoning=8, coding=9, tools=9, research=7, creativity=8,
        speed=7, long_context=8, reliability=9,
        strengths=("EU hosted", "tool use", "multilingual"),
        best_for=("code", "general", "writing"),
    ),
    ModelInfo(
        id="moonshot/kimi-k2", provider="moonshot", display_name="Kimi K2",
        context_window=256_000, max_output=64_000,
        input_price_per_1m=0.60, output_price_per_1m=2.50,
        reasoning=8, coding=9, tools=9, research=8, creativity=7,
        speed=7, long_context=8, reliability=8,
        strengths=("agentic", "open weights", "tool use"),
        best_for=("code", "general", "planning"),
    ),
    ModelInfo(
        id="zhipu/glm-4.6", provider="zhipu", display_name="GLM-4.6",
        context_window=200_000, max_output=64_000,
        input_price_per_1m=0.60, output_price_per_1m=2.20,
        reasoning=8, coding=9, tools=8, research=7, creativity=7,
        speed=7, long_context=8, reliability=8,
        strengths=("coding", "open weights", "cheap"),
        best_for=("code", "general"),
    ),
    # ---------------------------------------------------------- Mock / local
    ModelInfo(
        id="mock/apex-mock", provider="mock", display_name="APEX Mock (offline)",
        context_window=128_000, max_output=16_000,
        input_price_per_1m=0.0, output_price_per_1m=0.0,
        reasoning=5, coding=5, tools=5, research=5, creativity=5,
        speed=10, long_context=5, reliability=10,
        strengths=("deterministic", "offline", "free"),
        best_for=("general", "code", "planning", "research", "reasoning"),
    ),
    ModelInfo(
        id="ollama/llama3.1:8b", provider="ollama", display_name="Llama 3.1 8B (local)",
        context_window=128_000, max_output=8_000,
        input_price_per_1m=0.0, output_price_per_1m=0.0,
        reasoning=5, coding=6, tools=6, research=5, creativity=6,
        speed=8, long_context=6, reliability=7,
        strengths=("private", "local", "free"),
        best_for=("general", "code", "writing"),
    ),
]

_BY_ID = {m.id: m for m in MODELS}


def get_model(model_id: str) -> ModelInfo | None:
    return _BY_ID.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    if provider:
        return [m for m in MODELS if m.provider == provider]
    return list(MODELS)


def providers() -> list[str]:
    return sorted({m.provider for m in MODELS})


def best_for(task_type: str, top_n: int = 5) -> list[ModelInfo]:
    from ..types import TaskType

    tt = TaskType(task_type) if task_type in {t.value for t in TaskType} else TaskType.GENERAL
    ranked = sorted(MODELS, key=lambda m: m.capability(tt), reverse=True)
    return ranked[:top_n]
