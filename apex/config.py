"""APEX configuration — layered: defaults < file < env < explicit overrides.

Profiles: ``fast`` (cheap/fast models), ``quality`` (frontier models),
``balanced`` (default), ``offline`` (mock provider only), ``prod`` (strict
safety + checkpointing + tracing).
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

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
    # Offer the tool schemas on every request. The keyword heuristic that used to
    # gate this missed plainly tool-shaped tasks ("read notes.txt"), so the model
    # got no tools at all and had to invent an answer. ~550 tokens of schema is a
    # cheap price for not being blind; set false to save that on simple workloads.
    always_offer_tools: bool = True


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
    # Ring-buffer size for spans. A long-running server would otherwise grow the
    # span list forever (a slow, invisible leak).
    max_spans: int = 5_000


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
        self._load_problems: list[str] = []
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
        """Apply explicit section/field values, coercing to the declared types.

        Unknown keys and wrong types are *recorded*, not ignored: a typo like
        ``safety.auto_approve_tool`` would otherwise load a config that looks
        correct and silently leaves the safety default in place.
        """
        problems: list[str] = list(getattr(self, "_load_problems", []))

        for key in data:
            if key not in TOP_LEVEL_NAMES:
                problems.append(f"unknown config key '{key}'")

        for section in SECTION_NAMES:
            if section not in data:
                continue
            payload = data[section]
            if not isinstance(payload, dict):
                problems.append(
                    f"{section}: expected a mapping, got {type(payload).__name__}"
                )
                continue
            obj = getattr(self, section)
            hints = get_type_hints(type(obj))
            for name, raw in payload.items():
                if name not in hints:
                    problems.append(f"{section}.{name}: unknown setting")
                    continue
                try:
                    setattr(obj, name, _coerce_to(hints[name], raw, f"{section}.{name}"))
                except (TypeError, ValueError, ConfigError) as exc:
                    problems.append(str(exc))

        for key in ("profile", "workspace", "checkpoint_dir", "session_id"):
            if key in data:
                value = data[key]
                if not isinstance(value, str):
                    problems.append(f"{key}: expected a string, got {type(value).__name__}")
                    continue
                setattr(self, key, value)

        self._load_problems = problems

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
            "TOOLS_ALWAYS_OFFER": ("agents", "always_offer_tools", _as_bool),
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
            "MAX_SPANS": ("observability", "max_spans", int),
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
        problems.extend(getattr(self, "_load_problems", []))
        if not isinstance(self.agents.max_steps, int) or self.agents.max_steps < 1:
            problems.append("agents.max_steps must be an integer >= 1")
        if not isinstance(self.agents.max_plan_steps, int) or self.agents.max_plan_steps < 1:
            problems.append("agents.max_plan_steps must be an integer >= 1")
        if not isinstance(self.agents.critic_threshold, (int, float)) or not 0.0 <= float(self.agents.critic_threshold) <= 1.0:
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


def _coerce_to(annotation: Any, value: Any, where: str) -> Any:
    """Convert a parsed config value to the dataclass field's declared type."""
    if annotation in (bool, Optional[bool]):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ConfigError(f"{where}: expected a boolean (true/false), got {value!r}")
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ConfigError(f"{where}: expected a boolean, got {type(value).__name__}")

    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an integer, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        raise ConfigError(f"{where}: expected an integer, got {value!r}")

    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"{where}: expected a number, got {value!r}")

    if annotation is str:
        if isinstance(value, (dict, list)):
            raise ConfigError(f"{where}: expected a string, got {type(value).__name__}")
        return str(value)

    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        item_type = args[0] if args else str
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return [_coerce_to(item_type, item, where) for item in items]

    if isinstance(annotation, types.UnionType) or origin is not None:
        # Optional[x] / x | None: coerce against the first non-None type
        candidates = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        if candidates:
            return _coerce_to(candidates[0], value, where)

    return value


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


SECTION_NAMES = ("models", "agents", "memory", "context", "safety", "observability")
TOP_LEVEL_NAMES = SECTION_NAMES + ("profile", "workspace", "checkpoint_dir", "session_id")

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


def _strip_comment(line: str) -> str:
    """Remove a trailing ``# comment`` while respecting quoted scalars.

    A naive strip() split once turned ``auto_approve_tools: false   # deny`` into
    the *truthy string* ``'false   # deny'`` — i.e. a config that asked for deny
    silently enabled tool auto-approval.
    """
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _scalar(text: str, where: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part, where) for part in _split_inline(inner, where)]
    low = text.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if low in ("null", "~", "none"):
        return None
    if low == "{}":
        return {}
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_inline(text: str, where: str) -> list[str]:
    parts, current, quote = [], [], None
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char == ",":
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if quote:
        raise ConfigError(f"{where}: unterminated quote in inline list")
    parts.append("".join(current))
    return [part for part in (p.strip() for p in parts) if part]


def _tokenise(text: str) -> list[tuple[int, str, int]]:
    """(indent, content, line number) for every meaningful line."""
    tokens: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw.replace("\t", "    ")).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ConfigError(
                f"line {lineno}: indentation must be a multiple of 2 spaces (got {indent})"
            )
        tokens.append((indent, line.strip(), lineno))
    return tokens


def _parse_block(
    tokens: list[tuple[int, str, int]], index: int, indent: int
) -> tuple[Any, int]:
    if tokens[index][1].startswith("- "):
        return _parse_list(tokens, index, indent)
    return _parse_map(tokens, index, indent)


def _parse_list(tokens: list[tuple[int, str, int]], index: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while index < len(tokens):
        cur_indent, content, lineno = tokens[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ConfigError(f"line {lineno}: unexpected indentation inside a list")
        if not content.startswith("- "):
            break
        item = content[2:].strip()
        if not item:
            raise ConfigError(f"line {lineno}: empty list item")
        if ":" in item and not item.startswith(("'", '"', "[", "{")):
            raise ConfigError(
                f"line {lineno}: nested mappings inside a list are not supported "
                "(use an inline list, e.g. [a, b])"
            )
        out.append(_scalar(item, f"line {lineno}"))
        index += 1
    return out, index


def _parse_map(tokens: list[tuple[int, str, int]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while index < len(tokens):
        cur_indent, content, lineno = tokens[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ConfigError(f"line {lineno}: unexpected indentation")
        if content.startswith("- "):
            raise ConfigError(f"line {lineno}: list item where a 'key: value' was expected")
        key, sep, rest = content.partition(":")
        if not sep:
            raise ConfigError(f"line {lineno}: expected 'key: value' (got {content!r})")
        key = key.strip().strip("'\"")
        if not key:
            raise ConfigError(f"line {lineno}: empty key")
        rest = rest.strip()
        index += 1
        if rest:
            out[key] = _scalar(rest, f"line {lineno}")
        elif index < len(tokens) and tokens[index][0] > cur_indent:
            value, index = _parse_block(tokens, index, tokens[index][0])
            out[key] = value
        else:
            out[key] = {}
    return out, index


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """YAML subset: nested maps, block/inline lists, comments, quoted scalars.

    Deliberately strict. A permissive parser that silently produces the wrong
    shape (``require_approval_for`` turning into ``{'- bash': {}}``) is worse than
    a loud error, because the resulting config looks loaded but is not applied.
    """
    tokens = _tokenise(text)
    if not tokens:
        return {}
    if tokens[0][0] != 0:
        raise ConfigError(f"line {tokens[0][2]}: top-level keys must not be indented")
    value, index = _parse_block(tokens, 0, 0)
    if index != len(tokens):
        raise ConfigError(f"line {tokens[index][2]}: unexpected indentation")
    if not isinstance(value, dict):
        raise ConfigError("config file must contain a mapping at the top level")
    return value


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
