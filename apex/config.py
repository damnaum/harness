"""APEX configuration — layered: defaults < file < env < explicit overrides.

Profiles: ``fast`` (cheap/fast models), ``quality`` (frontier models),
``balanced`` (default), ``offline`` (mock provider only), ``prod`` (strict
safety + checkpointing + tracing).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .errors import ConfigError


@dataclass
class ModelConfig:
    default_model: str = ""  # "" = auto-route by capability/cost/latency
    router_mode: str = "balanced"  # balanced | quality | fast | cheapest
    fallback_chain: list[str] = field(default_factory=list)
    max_retries: int = 3
    retry_base_delay: float = 0.5
    timeout_seconds: float = 120.0
    # Hard per-run spend cap in USD; 0 disables the cap. A long ReAct loop on a
    # frontier model can otherwise cost more than the task is worth.
    max_cost_usd: float = 0.0


@dataclass
class AgentConfig:
    max_steps: int = 25
    max_plan_steps: int = 12
    enable_critic: bool = True
    enable_reflection: bool = True
    enable_swarm: bool = True
    critic_threshold: float = 0.6  # below this, retry with feedback


@dataclass
class MemoryConfig:
    enabled: bool = True
    persist: bool = True
    path: str = ".apex/memory.json"  # relative to workspace
    episodic_limit: int = 200
    semantic_limit: int = 500
    compaction_threshold_tokens: int = 6000


@dataclass
class ContextConfig:
    max_tokens: int = 32000
    reserve_for_output: int = 8000
    summarise_threshold: float = 0.85


@dataclass
class SafetyConfig:
    enabled: bool = True
    block_injection: bool = True
    redact_pii: bool = True
    # Local development convenience: privileged tools (bash, write_file) run
    # without a prompt. The "prod" profile turns this off, and an approval
    # handler can be installed for interactive gating.
    auto_approve_tools: bool = True
    require_approval_for: list[str] = field(default_factory=lambda: ["bash", "write_file"])
    sandbox_root: str = "."
    command_timeout: float = 30.0
    max_output_chars: int = 200_000


@dataclass
class ObservabilityConfig:
    log_level: str = "INFO"
    trace_enabled: bool = True
    metrics_enabled: bool = True


@dataclass
class ApexConfig:
    profile: str = "balanced"
    models: ModelConfig = field(default_factory=ModelConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    workspace: str = "."
    checkpoint_dir: str = ".apex/checkpoints"
    session_id: str = "default"

    def __post_init__(self):
        self.apply_profile(self.profile)

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApexConfig":
        """Build from a dict. Precedence: profile defaults < explicit values."""
        cfg = cls()
        profile = data.get("profile", cfg.profile)
        # Profile first: it supplies defaults, and explicit values win over it.
        cfg.apply_profile(profile)
        cfg._apply_data(data)
        return cfg

    def _apply_data(self, data: dict[str, Any]) -> None:
        """Apply explicit section/field values without re-running the profile."""
        for section in ("models", "agents", "memory", "context", "safety", "observability"):
            payload = data.get(section)
            if isinstance(payload, dict):
                obj = getattr(self, section)
                for k, v in payload.items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        for k in ("profile", "workspace", "checkpoint_dir", "session_id"):
            if k in data:
                setattr(self, k, data[k])

    @classmethod
    def from_file(cls, path: str | Path) -> "ApexConfig":
        return cls.from_dict(_read_config_file(path))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ApexConfig":
        """Full layered load: defaults < profile < file < environment.

        ``APEX_CONFIG`` is honoured when ``path`` is not given, so the same code
        works in development (no file) and production (mounted config).
        """
        resolved = str(path) if path is not None else os.environ.get("APEX_CONFIG", "")
        data: dict[str, Any] = {}
        if resolved:
            data = _read_config_file(resolved)
        cfg = cls()
        profile = os.environ.get("APEX_PROFILE") or data.get("profile") or cfg.profile
        cfg.apply_profile(profile)
        cfg._apply_data(data)
        cfg._apply_env()
        return cfg

    def _apply_env(self, prefix: str = "APEX_") -> None:
        mapping = {
            "MAX_STEPS": ("agents", "max_steps", int),
            "PLAN_STEPS": ("agents", "max_plan_steps", int),
            "WORKSPACE": (None, "workspace", str),
            "MODEL": ("models", "default_model", str),
            "ROUTER_MODE": ("models", "router_mode", str),
            "MAX_COST_USD": ("models", "max_cost_usd", float),
            "CHECKPOINT_DIR": (None, "checkpoint_dir", str),
            "SESSION_ID": (None, "session_id", str),
            "LOG_LEVEL": ("observability", "log_level", str),
            "TRACE": ("observability", "trace_enabled", _as_bool),
            "MEMORY_PERSIST": ("memory", "persist", _as_bool),
        }
        for env_key, (section, attr, cast) in mapping.items():
            raw = os.environ.get(prefix + env_key)
            if raw is None:
                continue
            try:
                value = cast(raw)
            except (ValueError, TypeError):
                continue
            if section is None:
                setattr(self, attr, value)
            else:
                setattr(getattr(self, section), attr, value)

    @classmethod
    def from_env(cls, prefix: str = "APEX_") -> "ApexConfig":
        """Defaults < profile < environment (no config file)."""
        cfg = cls()
        profile = os.environ.get(prefix + "PROFILE")
        if profile:
            cfg.apply_profile(profile)
        cfg._apply_env(prefix)
        return cfg

    def apply_profile(self, profile: str) -> None:
        self.profile = profile
        if profile == "fast":
            self.models.router_mode = "fast"
            self.agents.max_steps = 15
        elif profile == "quality":
            self.models.router_mode = "quality"
            self.agents.max_steps = 40
            self.agents.enable_critic = True
            self.agents.enable_reflection = True
        elif profile == "offline":
            self.models.default_model = "mock/apex-mock"
            self.models.router_mode = "balanced"
        elif profile == "prod":
            self.safety.enabled = True
            self.safety.block_injection = True
            self.safety.auto_approve_tools = False
            self.observability.trace_enabled = True
            self.agents.enable_critic = True
        # balanced: defaults

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    PROFILES = ("fast", "balanced", "quality", "offline", "prod")
    ROUTER_MODES = ("balanced", "quality", "fast", "cheapest")
    LOG_LEVELS = ("DEBUG", "INFO", "WARN", "WARNING", "ERROR")

    def validate(self) -> list[str]:
        """Collect *every* configuration problem at once, not just the first.

        A harness that fails one field at a time wastes the operator's time;
        returning the full list lets a bad config be fixed in one pass.
        """
        problems: list[str] = []
        if self.profile not in self.PROFILES:
            problems.append(f"unknown profile '{self.profile}' (expected one of {list(self.PROFILES)})")
        if self.models.router_mode not in self.ROUTER_MODES:
            problems.append(
                f"unknown router_mode '{self.models.router_mode}' (expected one of {list(self.ROUTER_MODES)})"
            )
        if self.agents.max_steps < 1:
            problems.append("agents.max_steps must be >= 1")
        if self.agents.max_plan_steps < 1:
            problems.append("agents.max_plan_steps must be >= 1")
        if not 0.0 <= self.agents.critic_threshold <= 1.0:
            problems.append("agents.critic_threshold must be between 0 and 1")
        if self.context.max_tokens < 1000:
            problems.append("context.max_tokens too small (minimum 1000)")
        if self.context.reserve_for_output >= self.context.max_tokens:
            problems.append("context.reserve_for_output must be smaller than context.max_tokens")
        if self.models.max_retries < 0:
            problems.append("models.max_retries must be >= 0")
        if self.models.timeout_seconds <= 0:
            problems.append("models.timeout_seconds must be > 0")
        if self.safety.command_timeout <= 0:
            problems.append("safety.command_timeout must be > 0")
        if self.safety.max_output_chars < 100:
            problems.append("safety.max_output_chars too small (minimum 100)")
        if self.memory.episodic_limit < 1 or self.memory.semantic_limit < 1:
            problems.append("memory limits must be >= 1")
        if self.observability.log_level.upper() not in self.LOG_LEVELS:
            problems.append(f"unknown log_level '{self.observability.log_level}'")
        if not Path(self.workspace).exists():
            problems.append(f"workspace does not exist: {self.workspace}")
        return problems


def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _read_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {path}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return data


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML subset parser (2-space indent, key: value) — stdlib only."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(value)
    return root


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")
