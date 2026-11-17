"""Guardrail cost bound.

Guardrails run on every prompt, before anything else. A quadratic regex here is a
denial-of-service vector: the original email pattern took 3 seconds on 50 KB of
plain letters, so a 1 MB prompt would have blocked a worker for many minutes.
These tests pin the cost at linear-ish and keep the matching correct.
"""

from __future__ import annotations

import time
import unittest

from apex.safety.guardrails import Guardrails


class TestRedactionCost(unittest.TestCase):
    def setUp(self):
        self.guardrails = Guardrails()

    def _timed(self, text: str) -> float:
        start = time.time()
        self.guardrails.redact(text)
        return time.time() - start

    def test_plain_word_run_is_fast(self):
        # Regression: this exact shape took ~3s at 50 KB before the fix.
        self.assertLess(self._timed("x" * 200_000), 1.0)

    def test_digit_run_is_fast(self):
        self.assertLess(self._timed("1" * 200_000), 1.0)

    def test_email_like_fragments_are_fast(self):
        self.assertLess(self._timed("a@b." * 40_000), 1.0)

    def test_punctuation_run_is_fast(self):
        self.assertLess(self._timed("(" * 50_000), 1.0)

    def test_check_input_and_output_bounded(self):
        text = "contact 99999 12345 or mail a@b.co " * 5_000
        start = time.time()
        self.guardrails.check_input(text)
        self.guardrails.check_output(text)
        self.assertLess(time.time() - start, 2.0)


class TestEmailMatching(unittest.TestCase):
    def setUp(self):
        self.guardrails = Guardrails()

    def test_common_emails_are_redacted(self):
        for address in (
            "jane.doe@example.com",
            "a@b.co",
            "user+tag@mail.example.co.uk",
            "UPPER@EXAMPLE.COM",
            "first_last-99@sub.domain.io",
        ):
            self.assertEqual(self.guardrails.redact(f"write to {address}"), "write to [EMAIL]", address)

    def test_non_addresses_are_left_alone(self):
        for text in ("not-an-email@", "@nope.com", "x@y", "a@b.c", "hello world"):
            self.assertEqual(self.guardrails.redact(text), text)

    def test_multiple_addresses(self):
        out = self.guardrails.redact("a@b.co, c.d@e.org")
        self.assertEqual(out, "[EMAIL], [EMAIL]")

    def test_redaction_is_idempotent(self):
        once = self.guardrails.redact("mail jane@example.com now")
        self.assertEqual(self.guardrails.redact(once), once)


class TestNoFalsePositivesOnNumbers(unittest.TestCase):
    """Dates, versions and plain numbers must survive redaction."""

    def setUp(self):
        self.guardrails = Guardrails()

    def test_structured_values_survive(self):
        for text in (
            "released 2024-11-17",
            "timestamp 2026-11-17 10:30",
            "version 1.2.345",
            "order 123456789",
            "amount 144",
            "id 12345678",
        ):
            self.assertEqual(self.guardrails.redact(text), text)

    def test_phones_still_redacted(self):
        for text in ("+91 98765 43210", "(555) 123-4567", "555 123 4567"):
            self.assertIn("[PHONE]", self.guardrails.redact(f"call {text}"))


if __name__ == "__main__":
    unittest.main()
