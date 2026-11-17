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

    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message)
        # HTTP status, when the error came from a provider. Retry decisions need
        # it: a 500 is worth another attempt, a 400 never is.
        self.status = status


class ModelUnavailableError(ModelError):
    code = "model_unavailable"

    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message, status=status)


class RateLimitError(ModelError):
    code = "rate_limit"

    def __init__(self, message: str = "", *, status: int | None = 429) -> None:
        super().__init__(message, status=status)


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


#: Failures that mean "the request itself is wrong" — retrying cannot help and
#: counting them against a circuit breaker would let one bad caller take the
#: whole service down.
_NON_RETRYABLE = (
    ConfigError,
    ValidationError,
    BudgetExceededError,
    SafetyError,
    ToolPermissionError,
    ToolNotFoundError,
    CircuitOpenError,
)


def is_retryable(exc: BaseException) -> bool:
    """Should the caller try this exact call again?

    Retry is for *transient infrastructure*: rate limits, 5xx, timeouts, broken
    connections. A 400 from a malformed payload, a 401 from a bad key, a budget
    stop or a guardrail block will fail identically every time, so retrying only
    adds latency (and, for auth, looks like credential stuffing).
    """
    if isinstance(exc, _NON_RETRYABLE):
        return False
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429 or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    return True
