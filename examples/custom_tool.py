#!/usr/bin/env python3
"""Register a custom tool with the harness, with schema + safety.

Tools are plain objects: a name, a description, a JSON-schema-ish parameter
spec, and a ``run`` implementation. Registering one makes it available to every
agent that has that tool in its allow-list.

Run:  python examples/custom_tool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.config import ApexConfig  # noqa: E402
from apex.harness import ApexHarness  # noqa: E402
from apex.tools.base import Tool  # noqa: E402
from apex.types import ToolCall  # noqa: E402


class WordCountTool(Tool):
    name = "word_count"
    description = "Count words, sentences and characters in a piece of text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to analyse"}},
        "required": ["text"],
    }
    dangerous = False

    def execute(self, arguments: dict) -> str:
        text = str(arguments.get("text") or "")
        words = len(text.split())
        sentences = sum(1 for ch in text if ch in ".!?") or (1 if text.strip() else 0)
        return (
            f"words={words} sentences={sentences} characters={len(text)} "
            f"avg_words_per_sentence={round(words / sentences, 1) if sentences else 0.0}"
        )


def main() -> None:
    harness = ApexHarness(ApexConfig(profile="offline", workspace="."))
    harness.tools.register(WordCountTool())

    print("registered tools:", ", ".join(harness.tools.names()))

    result = harness.tools.execute(
        ToolCall(name="word_count", arguments={"text": "Agents need tools. Tools need sandboxes! Do they?"})
    )
    print("success:", result.success)
    print("output :", result.output)

    # Unknown tools fail loudly, never silently succeed.
    bad = harness.tools.execute(ToolCall(name="shell_exec", arguments={}))
    print("unknown tool ->", bad.success, bad.error[:80])


if __name__ == "__main__":
    main()
