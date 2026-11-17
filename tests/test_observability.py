"""Observability tests: span nesting, metrics, event bus, log hygiene."""

from __future__ import annotations

import json
import unittest

from apex.observability import EventBus, Logger, Metrics, Tracer


class TestTracer(unittest.TestCase):
    def test_nested_spans_link_to_parent(self):
        tracer = Tracer(True)
        with tracer.span("outer"):
            with tracer.span("inner"):
                with tracer.span("deepest"):
                    pass
        spans = {s["name"]: s for s in tracer.export()}
        self.assertIsNone(spans["outer"]["parent_id"])
        self.assertEqual(spans["inner"]["parent_id"], spans["outer"]["span_id"])
        self.assertEqual(spans["deepest"]["parent_id"], spans["inner"]["span_id"])

    def test_all_spans_share_trace_id(self):
        tracer = Tracer(True)
        with tracer.span("a"):
            with tracer.span("b"):
                pass
        self.assertEqual(len({s["trace_id"] for s in tracer.export()}), 1)

    def test_errors_recorded(self):
        tracer = Tracer(True)
        with self.assertRaises(RuntimeError):
            with tracer.span("failing"):
                raise RuntimeError("boom")
        span = tracer.export()[0]
        self.assertEqual(span["status"], "error")
        self.assertIn("boom", span["attributes"]["error"])

    def test_durations_recorded(self):
        import time

        tracer = Tracer(True)
        with tracer.span("slow"):
            time.sleep(0.01)
        self.assertGreater(tracer.export()[0]["duration_ms"], 5)

    def test_disabled_tracer_records_nothing(self):
        tracer = Tracer(False)
        with tracer.span("x"):
            pass
        self.assertEqual(tracer.export(), [])

    def test_spans_sorted_by_start(self):
        tracer = Tracer(True)
        with tracer.span("first"):
            pass
        with tracer.span("second"):
            pass
        names = [s["name"] for s in tracer.export()]
        self.assertEqual(names, ["first", "second"])

    def test_separate_traces(self):
        tracer = Tracer(True)
        with tracer.span("one"):
            pass
        first = tracer.trace_id
        tracer.new_trace()
        with tracer.span("two"):
            pass
        self.assertNotEqual(first, tracer.trace_id)
        self.assertEqual(len({s["trace_id"] for s in tracer.export()}), 2)


class TestMetrics(unittest.TestCase):
    def test_counters(self):
        m = Metrics()
        m.incr("runs")
        m.incr("runs", 2)
        self.assertEqual(m.snapshot()["counters"]["runs"], 3)

    def test_histogram_summary(self):
        m = Metrics()
        for value in (10, 20, 30):
            m.observe("latency", value)
        hist = m.snapshot()["histograms"]["latency"]
        self.assertEqual(hist["count"], 3)
        self.assertEqual(hist["min"], 10)
        self.assertEqual(hist["max"], 30)

    def test_labels_do_not_collide(self):
        m = Metrics()
        m.incr("calls", labels={"model": "a"})
        m.incr("calls", labels={"model": "b"})
        self.assertEqual(len(m.snapshot()["counters"]), 2)

    def test_empty_snapshot(self):
        self.assertEqual(Metrics().snapshot(), {"counters": {}, "histograms": {}})


class TestEventBus(unittest.TestCase):
    def test_subscribe_and_emit(self):
        bus = EventBus()
        seen = []
        bus.subscribe("run", lambda event: seen.append(event.type))
        bus.emit("run", task="x")
        self.assertEqual(seen, ["run"])

    def test_wildcard(self):
        bus = EventBus()
        seen = []
        bus.subscribe("*", lambda event: seen.append(event.type))
        bus.emit("a")
        bus.emit("b")
        self.assertEqual(seen, ["a", "b"])

    def test_failing_subscriber_does_not_break_emit(self):
        bus = EventBus()
        seen = []

        def bad(event):
            raise RuntimeError("subscriber bug")

        bus.subscribe("*", bad)
        bus.subscribe("*", lambda event: seen.append(event.type))
        bus.emit("x")  # must not raise
        self.assertEqual(seen, ["x"])


class TestLogger(unittest.TestCase):
    def test_json_line_goes_to_stderr(self):
        # stdout stays clean for CLI output; JSON logs go to stderr.
        import io

        buffer = io.StringIO()
        Logger("DEBUG", stream=buffer).info("hello", a=1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["msg"], "hello")
        self.assertEqual(payload["a"], 1)

    def test_level_filtering(self):
        import io

        buffer = io.StringIO()
        logger = Logger("ERROR", stream=buffer)
        logger.info("hidden")
        logger.error("shown")
        output = buffer.getvalue()
        self.assertNotIn("hidden", output)
        self.assertIn("shown", output)

    def test_debug_hidden_at_info(self):
        import io

        buffer = io.StringIO()
        logger = Logger("INFO", stream=buffer)
        logger.debug("noise")
        logger.info("signal")
        self.assertNotIn("noise", buffer.getvalue())

    def test_non_serialisable_fields_do_not_crash(self):
        import io

        buffer = io.StringIO()
        Logger("DEBUG", stream=buffer).info("obj", value=object())
        self.assertIn("obj", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
