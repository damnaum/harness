"""APEX error hierarchy.  Every failure mode has a typed exception so callers
can react precisely (retry, fallback model, ask user, abort)."""

from __future__ import annotations


class ApexError(Exception):
    """Base class for all APEX errors."""

    code = "apex_error"


class ConfigError(ApexError):
    code = "config_error"


class ModelError(ApexError):
    code = "model_error"


class ModelUnavailableError(ModelError):
    code = "model_unavailable"


class RateLimitError(ModelError):
    code = "rate_limit"


class ContextOverflowError(ModelError):
    code = "context_overflow"


class BudgetExceededError(ApexError):
    """The run spent its configured USD budget (``models.max_cost_usd``)."""

    code = "budget_exceeded"


class ToolError(ApexError):
    code = "tool_error"


class ToolNotFoundError(ToolError):
    code = "tool_not_found"


class ToolTimeoutError(ToolError):
    code = "tool_timeout"


class ToolPermissionError(ToolError):
    code = "tool_permission"


class PlanningError(ApexError):
    code = "planning_error"


class ExecutionError(ApexError):
    code = "execution_error"


class LoopDetectedError(ExecutionError):
    code = "loop_detected"


class MaxStepsExceededError(ExecutionError):
    code = "max_steps_exceeded"


class SafetyError(ApexError):
    code = "safety_error"


class InjectionDetectedError(SafetyError):
    code = "injection_detected"


class GuardrailViolationError(SafetyError):
    code = "guardrail_violation"


class MemoryError_(ApexError):
    code = "memory_error"


class CheckpointError(ApexError):
    code = "checkpoint_error"


class CircuitOpenError(ApexError):
    code = "circuit_open"


class ValidationError(ApexError):
    code = "validation_error"
