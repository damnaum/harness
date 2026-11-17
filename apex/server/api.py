"""APEX HTTP API — stdlib ``ThreadingHTTPServer``, no framework needed.

Endpoints
---------
GET  ``/health``            liveness + version
GET  ``/ready``             readiness (tools registered, memory loaded)
GET  ``/v1/models``         model registry
GET  ``/v1/metrics``        counters + histograms
GET  ``/v1/traces``         spans grouped by trace
GET  ``/v1/memory``         memory statistics
POST ``/v1/run``            run a task            {task, model?, strategy?}
POST ``/v1/route``          routing decision      {task}
POST ``/v1/remember``       store a fact          {fact, source?}
POST ``/v1/eval``           run the eval suite    {suite?}

Production behaviour
--------------------
* Optional bearer-token auth (``APEX_API_TOKEN``); disabled when unset so local
  development stays frictionless.
* CORS is enabled for configured origins so browser clients (and preview
  iframes) can call the API directly.
* Request bodies are size-capped, JSON errors return 400 instead of dropping the
  connection, unknown strategies return 400, and unexpected failures return 500
  with a stable error shape (never a stack trace).
* ``Content-Length`` is always set, so keep-alive clients do not hang.
"""

from __future__ import annotations

import hmac
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..types import ReasoningStrategy

MAX_BODY_BYTES = 1_000_000  # 1 MB — a task prompt, not a file upload
DEFAULT_CORS = "*"
# ThreadingHTTPServer spawns a thread per connection with no upper bound, so a
# burst of /v1/run calls would fan out into unbounded model calls. Runs are gated
# by a semaphore; excess requests get 503 + Retry-After instead of a queue that
# grows until the box dies.
MAX_CONCURRENT_RUNS = int(os.environ.get("APEX_MAX_CONCURRENCY", "4"))
IDLE_SOCKET_TIMEOUT = 30.0  # slow-loris guard


def _error(message: str, code: str = "bad_request") -> dict[str, Any]:
    return {"error": message, "code": code}


def group_spans_by_trace(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat spans into per-trace records (API consumers think in traces)."""
    traces: dict[str, dict[str, Any]] = {}
    for span in spans:
        trace_id = str(span.get("trace_id", ""))
        entry = traces.setdefault(trace_id, {"trace_id": trace_id, "spans": []})
        entry["spans"].append(span)
    out = list(traces.values())
    for entry in out:
        starts = [s.get("started_at", 0) or 0 for s in entry["spans"]]
        ends = [s.get("ended_at", 0) or 0 for s in entry["spans"]]
        entry["started_at"] = min(starts) if starts else 0
        entry["duration_ms"] = round((max(ends) - min(starts)) * 1000, 3) if starts else 0.0
    out.sort(key=lambda t: t["started_at"], reverse=True)
    return out


def make_handler(
    harness,
    cors_origin: str | None = None,
    api_token: str | None = None,
    max_concurrent_runs: int = MAX_CONCURRENT_RUNS,
):
    # Accept either the Harness or the Apex SDK facade (the facade is what users
    # have in hand); unwrap so ``harness.metrics`` etc. resolve.
    if hasattr(harness, "harness") and not hasattr(harness, "metrics"):
        harness = harness.harness
    token = api_token if api_token is not None else os.environ.get("APEX_API_TOKEN", "")
    allow_origin = cors_origin or os.environ.get("APEX_CORS_ORIGIN", DEFAULT_CORS)
    run_slots = threading.BoundedSemaphore(max(1, max_concurrent_runs))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "APEX/1.0"
        timeout = IDLE_SOCKET_TIMEOUT  # drop idle/partial connections

        # -- plumbing -------------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib API
            """Route access logs through the harness logger, not stderr."""
            try:
                harness.logger.info("http", method=self.command, path=self.path)
            except Exception:  # noqa: BLE001 - logging must never break a request
                pass

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _send(self, code: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self._cors()
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:  # client disconnected mid-response; nothing to do
                pass

        def _authorised(self) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
            # Constant-time: a plain == leaks the token prefix through timing.
            return hmac.compare_digest(supplied.encode(), token.encode())

        def _read_json(self) -> tuple[dict[str, Any] | None, str | None]:
            length_raw = self.headers.get("Content-Length") or "0"
            try:
                length = int(length_raw)
            except ValueError:
                return None, "invalid Content-Length"
            if length > MAX_BODY_BYTES:
                return None, f"body exceeds {MAX_BODY_BYTES} bytes"
            if length <= 0:
                return {}, None
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None, "body is not valid JSON"
            if not isinstance(parsed, dict):
                return None, "body must be a JSON object"
            return parsed, None

        # -- verbs ----------------------------------------------------------
        def do_OPTIONS(self):  # noqa: N802 - stdlib naming
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if not self._authorised():
                self._send(401, _error("missing or invalid bearer token", "unauthorized"))
                return
            path = self.path.split("?", 1)[0]
            try:
                if path == "/health":
                    self._send(200, {"status": "ok", "service": "apex-harness"})
                elif path == "/ready":
                    self._send(200, {
                        "status": "ready",
                        "tools": len(harness.tools.names()),
                        "memory": harness.memory.stats(),
                    })
                elif path == "/v1/models":
                    from ..models.registry import list_models

                    self._send(200, {"models": [m.to_dict() for m in list_models()]})
                elif path == "/v1/metrics":
                    self._send(200, harness.metrics.snapshot())
                elif path == "/v1/traces":
                    spans = harness.tracer.export()
                    self._send(200, {"traces": group_spans_by_trace(spans), "span_count": len(spans)})
                elif path == "/v1/memory":
                    self._send(200, harness.memory.stats())
                else:
                    self._send(404, _error(f"no route for GET {path}", "not_found"))
            except Exception as exc:  # noqa: BLE001 - never leak a traceback
                self._send(500, _error(f"internal error: {type(exc).__name__}", "internal_error"))

        def do_POST(self):  # noqa: N802
            if not self._authorised():
                self._send(401, _error("missing or invalid bearer token", "unauthorized"))
                return
            data, problem = self._read_json()
            if problem is not None:
                self._send(400, _error(problem))
                return
            path = self.path.split("?", 1)[0]
            try:
                if path == "/v1/run":
                    self._handle_run(data or {})
                elif path == "/v1/route":
                    self._handle_route(data or {})
                elif path == "/v1/remember":
                    self._handle_remember(data or {})
                elif path == "/v1/eval":
                    self._handle_eval(data or {})
                else:
                    self._send(404, _error(f"no route for POST {path}", "not_found"))
            except Exception as exc:  # noqa: BLE001
                harness.logger.error("api request failed", path=path, error=str(exc))
                self._send(500, _error(f"internal error: {type(exc).__name__}", "internal_error"))

        # -- handlers -------------------------------------------------------
        def _handle_run(self, data: dict[str, Any]) -> None:
            task = data.get("task")
            if not isinstance(task, str) or not task.strip():
                self._send(400, _error("'task' (non-empty string) is required"))
                return
            strategy = None
            raw_strategy = data.get("strategy")
            if raw_strategy:
                try:
                    strategy = ReasoningStrategy(str(raw_strategy))
                except ValueError:
                    allowed = [s.value for s in ReasoningStrategy]
                    self._send(400, _error(f"unknown strategy; allowed: {allowed}"))
                    return
            model = data.get("model")
            if model is not None and not isinstance(model, str):
                self._send(400, _error("'model' must be a string"))
                return
            if not run_slots.acquire(blocking=False):
                self._send(
                    503,
                    _error(
                        f"server busy: {max_concurrent_runs} runs in flight, retry shortly",
                        "overloaded",
                    ),
                    headers={"Retry-After": "5"},
                )
                return
            try:
                result = harness.run(task, model=model, strategy=strategy)
            finally:
                run_slots.release()
            self._send(200, result.to_dict())

        def _handle_route(self, data: dict[str, Any]) -> None:
            from ..models.router import classify_task

            task = data.get("task")
            if not isinstance(task, str) or not task.strip():
                self._send(400, _error("'task' (non-empty string) is required"))
                return
            classified = classify_task(task)
            decision = harness.router.route(classified)
            payload = decision.to_dict()
            payload["classification"] = {
                "task_type": classified.task_type.value,
                "complexity": classified.complexity.value,
            }
            self._send(200, payload)

        def _handle_remember(self, data: dict[str, Any]) -> None:
            fact = data.get("fact")
            if not isinstance(fact, str) or not fact.strip():
                self._send(400, _error("'fact' (non-empty string) is required"))
                return
            source = str(data.get("source") or "api")
            fact = fact.strip()
            if getattr(harness.config.safety, "enabled", True):
                verdict = harness.guardrails.check_input(fact)
                if not verdict.allowed:
                    self._send(403, _error(
                        "fact rejected by guardrails: " + "; ".join(verdict.reasons[:3]),
                        "blocked",
                    ))
                    return
                fact = verdict.redacted_text
            harness.remember(fact, source)
            self._send(200, {"remembered": fact, "source": source,
                             "memory": harness.memory.stats()})

        def _handle_eval(self, data: dict[str, Any]) -> None:
            from ..eval.benchmark import DEFAULT_SUITE, run_benchmark

            suite = DEFAULT_SUITE
            ids = data.get("tasks")
            if isinstance(ids, list) and ids:
                wanted = {str(i) for i in ids}
                suite = [t for t in DEFAULT_SUITE if t.id in wanted]
                if not suite:
                    self._send(400, _error("no eval tasks matched 'tasks'"))
                    return
            report = run_benchmark(harness.run, suite=suite)
            report.pop("results", None)  # keep the payload small; details via CLI
            self._send(200, report)

    return Handler


class ApexHTTPServer(ThreadingHTTPServer):
    """Threaded server tuned for a container behind a proxy."""

    daemon_threads = True          # a stuck request cannot block shutdown
    allow_reuse_address = True     # fast restart after a deploy
    request_queue_size = 64        # default backlog of 5 drops bursts


def serve(harness, host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ApexHTTPServer((host, port), make_handler(harness))
    print(f"APEX API listening on http://{host}:{port}")

    def _shutdown(signum: int, _frame: Any) -> None:
        # SIGTERM is what systemd/docker send; without this the process only
        # stops on Ctrl-C and gets killed mid-request.
        harness.logger.info("shutdown signal", signal=signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _shutdown)
            except (ValueError, OSError):  # not the main thread / unsupported
                pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("shutting down")
