"""Structured logging + metrics + tracing + event bus (all stdlib).

- Logger: JSON lines to stderr/stdout, level-filtered, with trace_id context.
- Metrics: in-process counters/gauges/histograms with a snapshot() for /metrics.
- Tracer: span stack per trace_id; spans nest and export as dicts.
- EventBus: tiny pub/sub so UI, server and tests can observe harness events.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional


# ---------------------------------------------------------------- logger
class Logger:
    def __init__(self, level: str = "INFO", stream=None):
        self.levels = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
        self._level = self.levels.get(level.upper(), 20)
        self._stream = stream or sys.stderr
        self._lock = threading.Lock()

    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        if self.levels[level] < self._level:
            return
        record = {"ts": time.time(), "level": level, "msg": msg, **fields}
        with self._lock:
            self._stream.write(json.dumps(record, default=str) + "\n")
            self._stream.flush()

    def debug(self, msg: str, **f: Any) -> None:
        self._emit("DEBUG", msg, **f)

    def info(self, msg: str, **f: Any) -> None:
        self._emit("INFO", msg, **f)

    def warn(self, msg: str, **f: Any) -> None:
        self._emit("WARN", msg, **f)

    def error(self, msg: str, **f: Any) -> None:
        self._emit("ERROR", msg, **f)


# ---------------------------------------------------------------- metrics
class Metrics:
    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, Any] | None) -> str:
        """Prometheus-style label suffix: ``calls{model=gpt-5.2}``."""
        if not labels:
            return name
        rendered = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
        return f"{name}{{{rendered}}}"

    def incr(self, name: str, value: float = 1.0, labels: dict[str, Any] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    MAX_SAMPLES = 10_000  # per series; unbounded growth is a slow leak

    def observe(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            series = self._histograms[key]
            series.append(value)
            if len(series) > self.MAX_SAMPLES:
                del series[: len(series) - self.MAX_SAMPLES]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hist = {}
            for name, values in self._histograms.items():
                if values:
                    s = sorted(values)
                    hist[name] = {
                        "count": len(s),
                        "min": s[0],
                        "max": s[-1],
                        "mean": sum(s) / len(s),
                        "p50": s[len(s) // 2],
                        "p95": s[int(len(s) * 0.95)] if len(s) > 1 else s[0],
                    }
            return {"counters": dict(self._counters), "histograms": hist}


# ---------------------------------------------------------------- tracing
@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    parent_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
        }


class Tracer:
    MAX_SPANS = 5_000  # ring buffer: unbounded span lists are a slow memory leak

    def __init__(self, enabled: bool = True, max_spans: int | None = None) -> None:
        self.enabled = enabled
        self.spans: list[Span] = []
        self.max_spans = max(int(max_spans or self.MAX_SPANS), 1)
        self._lock = threading.Lock()
        self._local = threading.local()

    def new_trace(self) -> str:
        trace_id = uuid.uuid4().hex[:16]
        self._local.trace_id = trace_id
        self._local.stack = []
        return trace_id

    @property
    def trace_id(self) -> Optional[str]:
        return getattr(self._local, "trace_id", None)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        if not self.enabled:
            yield Span(trace_id="disabled", span_id="none", name=name)
            return
        # Ensure the trace (and its stack) exist *before* reading the stack:
        # new_trace() replaces the stack object, which used to detach the
        # first span and break parent linking for every nested span.
        if self.trace_id is None:
            self.new_trace()
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        parent = stack[-1].span_id if stack else None
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:12],
            name=name,
            parent_id=parent,
            attributes=dict(attrs),
        )
        stack.append(span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            span.status = "error"
            span.attributes["error"] = str(exc)
            raise
        finally:
            span.ended_at = time.time()
            stack.pop()
            with self._lock:
                self.spans.append(span)
                if len(self.spans) > self.max_spans:
                    del self.spans[: len(self.spans) - self.max_spans]

    def export(self, trace_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            spans = sorted(list(self.spans), key=lambda s: s.started_at)
        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        return [s.to_dict() for s in spans]


# ---------------------------------------------------------------- events
@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self.history: list[Event] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._subs[event_type].append(handler)

    def emit(self, event_type: str, **payload: Any) -> Event:
        event = Event(type=event_type, payload=payload)
        self.history.append(event)
        for handler in self._subs.get(event_type, []) + self._subs.get("*", []):
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - observers must never break the run
                pass
        return event
