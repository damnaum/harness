"""APEX command line interface.

    apex run "<task>"            run a task and print the answer + run metrics
    apex models                  the 2026 model registry, scored
    apex route "<task>"          routing decision without executing
    apex eval                    benchmark suite (CI gate with --min-pass-rate)
    apex doctor                  self-diagnostics
    apex remember "<fact>"       durable memory
    apex serve                   HTTP API server

Global flags (``--profile``, ``--workspace``, ``--model``, ``--config``) may be
given before *or* after the subcommand: argv is normalised first, because
requiring a fixed position is a pointless papercut.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import ApexConfig

GLOBAL_FLAGS = ("--profile", "--workspace", "--model", "--config")
PROFILES = ["balanced", "fast", "quality", "offline", "prod"]


def _split_globals(argv: list[str]) -> tuple[list[str], list[str]]:
    """Move global flags (and their values) to the front of argv."""
    globals_: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in GLOBAL_FLAGS:
            globals_.append(token)
            if i + 1 < len(argv):
                globals_.append(argv[i + 1])
                i += 2
                continue
            i += 1
            continue
        rest.append(token)
        i += 1
    return globals_, rest


def _build_config(args: argparse.Namespace) -> ApexConfig:
    """Layered config: defaults < profile < file < env < CLI flags."""
    if getattr(args, "config", None):
        cfg = ApexConfig.load(args.config)
    else:
        cfg = ApexConfig.load()
    if args.profile:
        cfg.apply_profile(args.profile)
    if args.workspace:
        cfg.workspace = args.workspace
    if args.model:
        cfg.models.default_model = args.model
    return cfg


def _harness(args, strict: bool = True):
    from .harness import ApexHarness

    return ApexHarness(config=_build_config(args), strict=strict)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_run(args) -> int:
    if not args.task or not args.task.strip():
        print("error: task must be a non-empty string", file=sys.stderr)
        return 2
    h = _harness(args)
    result = h.run(args.task, model=args.model, strategy=args.strategy)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.success else 1

    print(result.output or "(no output)")
    print(
        f"\n---\nmodel={result.model_used} strategy={result.strategy.value} "
        f"tokens={result.total_tokens_in + result.total_tokens_out} "
        f"cost=${result.total_cost_usd:.4f} time={result.duration_ms:.0f}ms "
        f"success={result.success} trace={result.trace_id}"
    )
    if result.plan is not None:
        print(f"plan: {len(result.plan.steps)} steps, {len(result.agent_results)} agent runs")
    if result.warnings:
        print("warnings:", "; ".join(result.warnings))
    return 0 if result.success else 1


def cmd_models(args) -> int:
    from .models.registry import list_models

    models = list_models(args.provider)
    if getattr(args, "json", False):
        print(json.dumps([m.to_dict() for m in models], indent=2))
        return 0
    for m in models:
        print(
            f"{m.id:35s} ctx={m.context_window:>9,} "
            f"reason={m.reasoning} code={m.coding} tools={m.tools} "
            f"${m.input_price_per_1m:.2f}/${m.output_price_per_1m:.2f} per 1M"
        )
    print(f"\n{len(models)} models. Route a task with: apex route \"<task>\"")
    return 0


def cmd_route(args) -> int:
    from .models.router import ModelRouter, classify_task

    h = _harness(args)
    task = classify_task(args.task)
    router = h.router
    if getattr(args, "all_available", False):
        # Preview what routing would pick if every provider had credentials.
        router = ModelRouter(mode=h.config.models.router_mode, available_providers=None)
    decision = router.route(task, preferred=args.model)
    payload = decision.to_dict()
    payload["classification"] = {
        "task_type": task.task_type.value,
        "complexity": task.complexity.value,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print(f"task       : {args.task}")
    print(f"classified : {task.task_type.value} / {task.complexity.value}")
    print(f"model      : {decision.model.id}  (score {decision.score:.3f})")
    print(f"reason     : {decision.reason}")
    print(f"est cost   : ${decision.estimated_cost:.4f} for ~3k tokens")
    fallbacks = ", ".join(m.id for m in decision.fallbacks) or "(none)"
    print(f"fallbacks  : {fallbacks}")
    if getattr(args, "all_available", False):
        print("note       : preview of frontier routing (every provider assumed available)")
    elif h.degraded:
        print("note       : no provider credentials — this is the offline fallback; "
              "use --all-available to preview frontier routing")
    return 0


def cmd_eval(args) -> int:
    from .eval.benchmark import run_benchmark

    h = _harness(args)
    report = run_benchmark(lambda prompt: h.run(prompt, model=args.model))
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print(
            f"pass_rate={report['pass_rate']:.0%} mean_score={report['mean_score']:.2f} "
            f"tokens={report['total_tokens']} cost=${report['total_cost']:.4f}"
        )
        for r in report["results"]:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  [{mark}] {r['task_id']:15s} score={r['score']:.2f} {r['detail']}")
    if report["pass_rate"] < args.min_pass_rate:
        print(
            f"eval gate failed: pass_rate {report['pass_rate']:.0%} "
            f"< required {args.min_pass_rate:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_doctor(args) -> int:
    # strict=False: the whole point of doctor is to explain a broken config.
    h = _harness(args, strict=False)
    report = h.doctor()
    print(json.dumps(report, indent=2))
    return 0 if not report["config_problems"] else 1


def cmd_remember(args) -> int:
    h = _harness(args)
    h.remember(args.fact, source="cli")
    print(f"remembered: {args.fact}")
    return 0


def cmd_serve(args) -> int:
    from .server.api import serve

    h = _harness(args)
    try:
        serve(h, host=args.host, port=args.port)
    except OSError as exc:
        print(f"cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apex", description="APEX agentic harness")
    from . import __version__

    parser.add_argument("--version", action="version", version=f"apex {__version__}")
    parser.add_argument("--profile", choices=PROFILES, default=None,
                        help="fast | balanced | quality | offline | prod")
    parser.add_argument("--workspace", default=None, help="sandbox root (default: .)")
    parser.add_argument("--model", default=None, help="force a model id, e.g. mock/apex-mock")
    parser.add_argument("--config", default=None, help="path to a JSON/YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a task")
    p_run.add_argument("task")
    p_run.add_argument("--json", action="store_true", help="machine-readable output")
    p_run.add_argument("--strategy", default=None, help="force a reasoning strategy")
    p_run.set_defaults(func=cmd_run)

    p_models = sub.add_parser("models", help="list registered models")
    p_models.add_argument("--provider", default=None)
    p_models.add_argument("--json", action="store_true")
    p_models.set_defaults(func=cmd_models)

    p_route = sub.add_parser("route", help="show the routing decision for a task")
    p_route.add_argument("task")
    p_route.add_argument("--json", action="store_true")
    p_route.add_argument("--all-available", action="store_true", dest="all_available",
                         help="preview routing as if every provider had credentials")
    p_route.set_defaults(func=cmd_route)

    p_eval = sub.add_parser("eval", help="run the benchmark suite")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--min-pass-rate", type=float, default=0.0,
                        help="exit non-zero below this pass rate (CI gate)")
    p_eval.set_defaults(func=cmd_eval)

    p_doctor = sub.add_parser("doctor", help="self-diagnostics")
    p_doctor.set_defaults(func=cmd_doctor)

    p_rem = sub.add_parser("remember", help="store a durable fact in memory")
    p_rem.add_argument("fact")
    p_rem.set_defaults(func=cmd_remember)

    p_serve = sub.add_parser("serve", help="start the HTTP API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    globals_, rest = _split_globals(raw)
    parser = build_parser()
    args = parser.parse_args(globals_ + rest)

    if args.strategy if hasattr(args, "strategy") else False:
        from .types import ReasoningStrategy

        try:
            args.strategy = ReasoningStrategy(args.strategy)
        except ValueError:
            allowed = [s.value for s in ReasoningStrategy]
            print(f"error: unknown strategy '{args.strategy}'; allowed: {allowed}", file=sys.stderr)
            return 2

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a CLI must print, not traceback
        if _debug_enabled():
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _debug_enabled() -> bool:
    import os

    return os.environ.get("APEX_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _unused(*_: Any) -> None:  # pragma: no cover - keeps linters honest
    return None


if __name__ == "__main__":
    sys.exit(main())
