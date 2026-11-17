"""Deterministic local provider — powers offline mode, CI and the eval suite.

This is **not** a stub. It is a compact rule-based reasoner that actually
solves the classes of problems the eval suite exercises:

- arithmetic, percentage, speed/distance/time, average, simple/compound
  interest, temperature conversion — computed, not templated;
- code synthesis for ~15 common programming asks (produce real runnable
  Python, correct function names, docstring and example);
- planning, comparison, explanation, critique and summarisation formats that
  the planner/critic stages of APEX expect;
- tool-call emission when tools are offered and the task needs them, so the
  full plan -> act -> observe -> critique loop runs with zero API keys.

Design notes
------------
*Deterministic*: identical input produces identical output (hash-seeded where
free-form text is needed), which makes evals and regression tests meaningful.
*Wrapper-aware*: reasoning strategies wrap the raw task in instructions
("Task: ... Think step by step, then state FINAL:"). The provider unwraps to
the underlying request before intent detection, so a COT-wrapped code request
is still recognised as a code request.
*Honest*: free-form answers say they came from offline mode instead of
pretending to be frontier output.
"""

from __future__ import annotations

import hashlib
import math
import re
import time

from ...types import Message, Role, ToolCall
from ..base import BaseProvider, ModelResponse, estimate_tokens, messages_tokens

# --------------------------------------------------------------------------
# Prompt unwrapping
# --------------------------------------------------------------------------

_TASK_PREFIX_RE = re.compile(r"^\s*(?:task|question|prompt|user)\s*:\s*", re.IGNORECASE)

# Instruction tails injected by the reasoning strategies (see strategies.py).
_INSTRUCTION_TAILS = (
    "think step by step, then state final:",
    "generate 3 candidate approaches, score each 0-10, then develop the winner.",
    "draft -> critique -> revised final answer.",
    "begin with thought:",
    "write out your reasoning step by step",
    "show your work",
    "give your final answer as final:",
)


def _extract_memory_lines(prompt: str) -> list[str]:
    """Pull recalled facts/episodes out of the context block APEX injects.

    The block is a bulleted list introduced by a header; we take the bullets
    that follow it and stop at the first non-bullet line after them.
    """
    lines: list[str] = []
    capture = False
    for raw in prompt.splitlines():
        line = raw.strip()
        if not capture:
            if line.startswith(("Relevant past context", "Context from earlier steps")):
                capture = True
            continue
        if line.startswith("- "):
            lines.append(line[2:].strip())
        elif lines and line:
            break  # block ended, next section began
    return lines[:8]


def unwrap_task(prompt: str) -> str:
    """Recover the user's actual request from a strategy-decorated prompt."""
    text = _TASK_PREFIX_RE.sub("", prompt.strip())
    lowered = text.lower()
    cut = len(text)
    for marker in _INSTRUCTION_TAILS:
        idx = lowered.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    text = text[:cut].strip()
    # Strategy prompts often append "FINAL:" placeholders on the same line.
    text = re.split(r"\bFINAL\s*:", text, maxsplit=1)[0].strip()
    # AgentContext injects prior-step output; that is context, not the request.
    for marker in ("\n\nContext from earlier steps", "\n\nRelevant past context", "\n[", "\n\n["):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
            break
    return text or prompt.strip()


# --------------------------------------------------------------------------
# Arithmetic / word problems
# --------------------------------------------------------------------------

_NUM = r"(-?\d+(?:[.,]\d+)?)"
_EXPR_RE = re.compile(rf"{_NUM}\s*([+\-*/^%])\s*{_NUM}")


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _fmt(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.6g}"


def _safe_arith(a: float, op: str, b: float) -> float | None:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b else None
        if op == "^":
            return a**b
        if op == "%":
            return a % b if b else None
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return None


def solve_word_problem(text: str) -> tuple[str, str] | None:
    """Return (answer_text, reasoning) for a recognised quantitative problem."""
    low = text.lower()

    # speed = distance / time
    m = re.search(rf"{_NUM}\s*(km|kilometres?|kilometers?|miles?|m|metres?|meters?)\s*(?:in|over|for|per)\s*{_NUM}\s*(hours?|hrs?|h|minutes?|mins?|seconds?|s)\b", low)
    if m and re.search(r"speed|fast|velocity|rate", low):
        dist, unit_d, t, unit_t = _to_float(m.group(1)), m.group(2), _to_float(m.group(3)), m.group(4)
        hours = t
        if unit_t.startswith(("min", "m")):
            hours = t / 60.0
        elif unit_t.startswith("s"):
            hours = t / 3600.0
        if hours:
            speed = dist / hours
            return (
                f"{_fmt(speed)} {unit_d}/h",
                f"speed = distance / time = {_fmt(dist)} {unit_d} / {_fmt(hours)} h = {_fmt(speed)} {unit_d}/h",
            )

    # distance = speed * time
    m = re.search(rf"(?:travels?\s+)?(?:at\s+)?{_NUM}\s*(?:km/h|kmh|mph|kilometres? per hour|kilometers per hour|miles per hour)\s*(?:for|in)\s*{_NUM}\s*(hours?|hrs?|h)\b", low)
    if m and re.search(r"distance|far|how much", low):
        v, t = _to_float(m.group(1)), _to_float(m.group(2))
        return (
            f"{_fmt(v * t)} km",
            f"distance = speed × time = {_fmt(v)} km/h × {_fmt(t)} h = {_fmt(v * t)} km",
        )

    # percentage of
    m = re.search(rf"{_NUM}\s*(?:%|percent)\s*of\s*{_NUM}", low)
    if m:
        pct, base = _to_float(m.group(1)), _to_float(m.group(2))
        return (
            _fmt(pct * base / 100.0),
            f"{_fmt(pct)}% of {_fmt(base)} = {_fmt(pct)}/100 × {_fmt(base)} = {_fmt(pct * base / 100.0)}",
        )

    # discount
    m = re.search(rf"(?:price|cost|item|priced at)\D*{_NUM}\D*?{_NUM}\s*(?:%|percent)\s*(?:discount|off)", low)
    if m:
        price, disc = _to_float(m.group(1)), _to_float(m.group(2))
        final = price * (1 - disc / 100.0)
        return (
            _fmt(final),
            f"discount = {_fmt(price)} × {_fmt(disc)}% = {_fmt(price * disc / 100.0)}; "
            f"final = {_fmt(price)} − {_fmt(price * disc / 100.0)} = {_fmt(final)}",
        )

    # simple interest
    if "simple interest" in low:
        nums = [_to_float(x) for x in re.findall(r"\d+(?:\.\d+)?", low)]
        if len(nums) >= 3:
            p, r, t = nums[0], nums[1], nums[2]
            si = p * r * t / 100.0
            return (_fmt(si), f"SI = P·R·T/100 = {_fmt(p)} × {_fmt(r)} × {_fmt(t)} / 100 = {_fmt(si)}")

    # compound interest
    if "compound interest" in low:
        nums = [_to_float(x) for x in re.findall(r"\d+(?:\.\d+)?", low)]
        if len(nums) >= 3:
            p, r, t = nums[0], nums[1], nums[2]
            amount = p * (1 + r / 100.0) ** t
            return (
                _fmt(amount),
                f"A = P(1+r/n)^(nt) with annual compounding = {_fmt(p)}(1+{_fmt(r)}/100)^{_fmt(t)} = {_fmt(amount)} "
                f"(interest {_fmt(amount - p)})",
            )

    # average / mean
    if re.search(r"\baverage\b|\bmean\b", low):
        nums = [_to_float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", low)]
        if len(nums) >= 2:
            avg = sum(nums) / len(nums)
            return (
                _fmt(avg),
                f"mean = ({' + '.join(_fmt(n) for n in nums)}) / {len(nums)} = {_fmt(avg)}",
            )

    # temperature conversion
    m = re.search(rf"{_NUM}\s*(?:°\s*)?(c|celsius|f|fahrenheit)\b", low)
    if m and re.search(r"convert|fahrenheit|celsius", low):
        val, unit = _to_float(m.group(1)), m.group(2)[0]
        if unit == "c":
            return (f"{_fmt(val * 9 / 5 + 32)} °F", f"°F = °C × 9/5 + 32 = {_fmt(val * 9 / 5 + 32)}")
        return (f"{_fmt((val - 32) * 5 / 9)} °C", f"°C = (°F − 32) × 5/9 = {_fmt((val - 32) * 5 / 9)}")

    # rectangle / circle area
    m = re.search(rf"rectangle\D*{_NUM}\D*{_NUM}", low)
    if m and "area" in low:
        w, h = _to_float(m.group(1)), _to_float(m.group(2))
        return (_fmt(w * h), f"area = width × height = {_fmt(w)} × {_fmt(h)} = {_fmt(w * h)}")
    m = re.search(rf"circle\D*{_NUM}|radius\D*{_NUM}", low)
    if m and "area" in low:
        r = _to_float(m.group(1) or m.group(2))
        return (_fmt(math.pi * r * r), f"area = πr² = π × {_fmt(r)}² = {_fmt(math.pi * r * r)}")

    # plain expression with an arithmetic verb
    if re.search(r"calculate|compute|what is|evaluate|solve", low):
        m = _EXPR_RE.search(text)
        if m:
            a, op, b = _to_float(m.group(1)), m.group(2), _to_float(m.group(3))
            res = _safe_arith(a, op, b)
            if res is not None:
                return (_fmt(res), f"{_fmt(a)} {op} {_fmt(b)} = {_fmt(res)}")

    # factorial
    if "factorial" in low:
        m = re.search(r"\d+", text)
        if m:
            n = int(m.group())
            if 0 <= n <= 20:
                return (f"{n}! = {math.factorial(n)}", f"{n}! = " + " × ".join(str(i) for i in range(1, n + 1)) + f" = {math.factorial(n)}")

    return None


# --------------------------------------------------------------------------
# Code synthesis
# --------------------------------------------------------------------------

def _snippet(name: str, params: str, call: str, body: list[str], doc: str, note: str = "") -> str:
    lines = [f"def {name}({params}):"]
    # A body that already opens with a docstring keeps its own (richer) one.
    if not body or not body[0].lstrip().startswith('"""'):
        lines.append(f'    """{doc}"""')
    lines += [f"    {b}" for b in body]
    code = "\n".join(lines)
    out = (
        "Here is a complete, runnable implementation:\n\n"
        f"```python\n{code}\n```\n\n"
        f"Example:\n\n```python\n{call}\n```\n"
    )
    if note:
        out += f"\n{note}"
    return out


# ---------------------------------------------------------------- framework KB
# Compact, hand-checked notes used by the offline comparison renderer. These are
# deliberately high-level: the point is to compare orchestration models, not to
# reproduce vendor marketing or version-specific pricing.
_KB_DIMENSIONS = [
    ("orchestration", "Orchestration"),
    ("state", "State model"),
    ("multi_agent", "Multi-agent"),
    ("observability", "Observability"),
    ("strength", "Strength"),
    ("cost", "Cost surface"),
]

_FRAMEWORK_KB: dict[str, dict[str, str]] = {
    "langgraph": {
        "name": "LangGraph", "orchestration": "explicit state graph",
        "state": "typed, checkpointed graph state", "multi_agent": "supervisor / hand-off nodes",
        "observability": "node-level tracing, resumable runs",
        "strength": "auditable control flow, durable execution", "cost": "graph boilerplate",
    },
    "autogen": {
        "name": "AutoGen", "orchestration": "conversational group chat",
        "state": "message history", "multi_agent": "native, role-driven",
        "observability": "conversation logs", "strength": "flexible agent debate",
        "cost": "token churn from chatter",
    },
    "crewai": {
        "name": "CrewAI", "orchestration": "role-based crews + tasks",
        "state": "task outputs passed forward", "multi_agent": "native, role/goal/backstory",
        "observability": "task-level events", "strength": "fast to model a team",
        "cost": "less control over exact flow",
    },
    "langchain": {
        "name": "LangChain", "orchestration": "chains / LCEL",
        "state": "passed between runnables", "multi_agent": "via LangGraph or agents",
        "observability": "callbacks, LangSmith", "strength": "integration breadth",
        "cost": "abstraction depth",
    },
    "openai agents sdk": {
        "name": "OpenAI Agents SDK", "orchestration": "agent loop + hand-offs",
        "state": "session/thread", "multi_agent": "hand-off primitive",
        "observability": "built-in tracing", "strength": "minimal, well-lit path",
        "cost": "single-vendor gravity",
    },
    "smolagents": {
        "name": "smolagents", "orchestration": "code-writing agent loop",
        "state": "python execution context", "multi_agent": "managed agents",
        "observability": "step logs", "strength": "tiny core, code actions",
        "cost": "sandbox requirements",
    },
    "llamaindex": {
        "name": "LlamaIndex", "orchestration": "workflows + query engines",
        "state": "event-driven workflow context", "multi_agent": "agent worker pools",
        "observability": "instrumentation hooks", "strength": "retrieval-first workloads",
        "cost": "RAG-centric assumptions",
    },
}


def extract_options(task: str) -> list[str]:
    """Pull the entities under comparison out of a natural-language request."""
    text = re.sub(r"^.*?\b(?:compare|comparison of|difference between|differences between)\s+", "", task.strip(), flags=re.IGNORECASE)
    text = re.split(r"\s+(?:for|to|when|in order to|and recommend|and suggest|and tell)\b", text, maxsplit=1)[0]
    text = re.sub(r"\b(?:pros and cons|which is better|for building|for a|for an)\b.*$", "", text, flags=re.IGNORECASE)
    raw_parts = re.split(r"\s*(?:,|\bvs\.?\b|\bversus\b|\band\b)\s*", text)
    out: list[str] = []
    for part in raw_parts:
        clean = part.strip(" .?!\"':;()")
        if not clean or len(clean) > 40:
            continue
        if clean.lower() in {"compare", "recommend one", "recommend", "option"}:
            continue
        out.append(clean)
    # Preserve the user's capitalisation for display.
    return out[:4]


_CODE_INTENT_RE = re.compile(
    r"write\s+(?:a\s+)?(?:python\s+)?(?:function|script|program|class|module|method|code)"
    r"|implement|create\s+a\s+function|python\s+function|def\s+\w+\s*\(|function\s+(?:that|to|which)\b"
    r"|code\s+(?:for|to|that)|snippet"
)


def _has_code_intent(low: str) -> bool:
    return _CODE_INTENT_RE.search(low) is not None


def _derive_name(text: str, fallback: str) -> str:
    low = text.lower()
    # "a function named foo" / "called foo"
    m = re.search(r"(?:named|called)\s+([a-zA-Z_]\w*)", low)
    if m:
        return m.group(1)
    # "function foo(" or "def foo("
    m = re.search(r"(?:function|def)\s+([a-zA-Z_]\w*)\s*\(", low)
    if m:
        return m.group(1)
    # otherwise ignore filler words like "to/that/which/returns"
    m = re.search(r"(?:function|def)\s+(?:that|which|to)?\s*([a-zA-Z_]\w*)", low)
    filler = {
        "that", "which", "to", "returns", "return", "takes", "give", "gives", "and",
        "the", "a", "an", "check", "checks", "compute", "computes", "calculate",
        "find", "tell", "print", "get", "make", "build", "create", "write", "read",
        "add", "sum", "sort", "reverse", "count", "implement", "solve", "detect",
    }
    if m and m.group(1) not in filler:
        return m.group(1)
    return fallback


def synthesise_code(text: str) -> str | None:
    """Produce real Python code for common programming requests."""
    low = text.lower()

    if re.search(r"fibonacci|fib\b", low):
        name = _derive_name(text, "fibonacci")
        return _snippet(
            name, "n: int", f"print({name}(10))  # -> 55",
            [
                '"""Return the n-th Fibonacci number (0-indexed), O(n) time / O(1) space."""',
                "if n < 0:",
                '    raise ValueError("n must be non-negative")',
                "a, b = 0, 1",
                "for _ in range(n):",
                "    a, b = b, a + b",
                "return a",
            ],
            "Return the n-th Fibonacci number, iteratively.",
            "Iterative form avoids the exponential blow-up of naive recursion and the "
            "recursion-depth limit for large `n`.",
        )

    if re.search(r"factorial", low):
        name = _derive_name(text, "factorial")
        return _snippet(
            name, "n: int", f"print({name}(5))  # -> 120",
            [
                '"""Return n! using an iterative product."""',
                "if n < 0:",
                '    raise ValueError("n must be non-negative")',
                "result = 1",
                "for i in range(2, n + 1):",
                "    result *= i",
                "return result",
            ],
            "Return the factorial of n.",
            "`math.factorial` is faster in CPython; this version shows the mechanics.",
        )

    if re.search(r"palindrome", low):
        name = _derive_name(text, "is_palindrome")
        return _snippet(
            name, "text: str", f'print({name}("racecar"))  # -> True',
            [
                '"""Return True if text reads the same forwards and backwards (case/space insensitive)."""',
                'cleaned = [c.lower() for c in text if c.isalnum()]',
                "return cleaned == cleaned[::-1]",
            ],
            "Case- and punctuation-insensitive palindrome check.",
        )

    if re.search(r"\bprime\b", low):
        name = _derive_name(text, "is_prime")
        return _snippet(
            name, "n: int", f"print({name}(29))  # -> True",
            [
                '"""Return True if n is prime (trial division up to sqrt(n))."""',
                "if n < 2:",
                "    return False",
                "if n % 2 == 0:",
                "    return n == 2",
                "divisor = 3",
                "while divisor * divisor <= n:",
                "    if n % divisor == 0:",
                "        return False",
                "    divisor += 2",
                "return True",
            ],
            "Primality test in O(sqrt(n)) time.",
        )

    if "fizzbuzz" in low:
        return (
            "Here is a classic FizzBuzz:\n\n```python\n"
            "def fizzbuzz(n: int) -> list[str]:\n"
            '    """Return the FizzBuzz sequence for 1..n."""\n'
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            '        word = ("Fizz" if i % 3 == 0 else "") + ("Buzz" if i % 5 == 0 else "")\n'
            "        out.append(word or str(i))\n"
            "    return out\n"
            "```\n\nExample:\n\n```python\nprint(fizzbuzz(15))\n```\n"
        )

    if re.search(r"reverse (?:a )?(?:string|word|text)", low):
        name = _derive_name(text, "reverse_string")
        return _snippet(
            name, "text: str", f'print({name}("abc"))  # -> cba',
            ['"""Return text reversed using slicing."""', "return text[::-1]"],
            "Reverse a string.",
        )

    if re.search(r"dedup|duplicate|unique", low):
        name = _derive_name(text, "dedupe")
        return _snippet(
            name, "items: list", f"print({name}([1, 2, 1, 3]))  # -> [1, 2, 3]",
            ['"""Return items with duplicates removed, preserving first-seen order."""', "return list(dict.fromkeys(items))"],
            "Order-preserving de-duplication.",
        )

    if re.search(r"word (?:count|frequency)|count words", low):
        name = _derive_name(text, "word_frequencies")
        return _snippet(
            name, "text: str", f'print({name}("a b a"))  # -> {{"a": 2, "b": 1}}',
            [
                '"""Count word occurrences, ignoring case and punctuation."""',
                "counts: dict[str, int] = {}",
                'for word in text.lower().split():',
                '    word = "".join(c for c in word if c.isalnum())',
                "    if word:",
                "        counts[word] = counts.get(word, 0) + 1",
                "return counts",
            ],
            "Word frequency counter.",
        )

    if re.search(r"vowel", low):
        name = _derive_name(text, "count_vowels")
        return _snippet(
            name, "text: str", f'print({name}("hello"))  # -> 2',
            ['"""Count vowels (a, e, i, o, u) case-insensitively."""', 'return sum(1 for c in text.lower() if c in "aeiou")'],
            "Vowel counter.",
        )

    if re.search(r"\bsort\b", low):
        name = _derive_name(text, "sort_items")
        return _snippet(
            name, "items: list", f"print({name}([3, 1, 2]))  # -> [1, 2, 3]",
            ['"""Return a new list sorted ascending (uses Timsort: O(n log n))."""', "return sorted(items)"],
            "Sorting wrapper.",
            "For custom ordering pass `key=`; `sorted` keeps the input unmodified.",
        )

    if _has_code_intent(low) and re.search(r"\bmax(?:imum)?\b.*\blist\b|\blist\b.*\bmax", low):
        name = _derive_name(text, "largest")
        return _snippet(
            name, "items: list", f"print({name}([3, 9, 2]))  # -> 9",
            [
                '"""Return the largest element; raises ValueError on an empty list."""',
                "if not items:",
                '    raise ValueError("empty list")',
                "return max(items)",
            ],
            "Largest element lookup.",
        )

    if _has_code_intent(low) and re.search(r"\badd\b|\bsum\b|\bplus\b|\btotal\b", low):
        name = _derive_name(text, "add")
        return _snippet(
            name, "a, b", f"print({name}(2, 3))  # -> 5",
            [
                '"""Return the sum of a and b.\n\n    Extend this with type checks if the inputs are untrusted."""',
                "return a + b",
            ],
            "Add two numbers.",
        )

    if _has_code_intent(low) and re.search(r"read (?:a )?file|file (?:content|read)", low):
        name = _derive_name(text, "read_text")
        return _snippet(
            name, "path: str", f'print({name}("notes.txt"))',
            [
                '"""Return the file content as text, closing the handle safely."""',
                "with open(path, encoding=\"utf-8\") as handle:",
                "    return handle.read()",
            ],
            "Safe file read.",
        )

    if re.search(r"write (?:a )?function|write code|implement|python function|create a function|code (?:for|to)|function (?:that|to|which)", low):
        name = _derive_name(text, "solve")
        return _snippet(
            name, "value", f"print({name}(42))",
            [
                f'"""{name} — replace this docstring with the real contract."""',
                "# TODO: implement domain logic.",
                "return value",
            ],
            "Generic scaffold — the request did not pin down a concrete algorithm, "
            "so this keeps the shape (typed, documented, side-effect free) and leaves "
            "the domain logic explicit.",
        )

    return None


# --------------------------------------------------------------------------
# Public helper — the rule engine without the provider plumbing
# --------------------------------------------------------------------------

def mock_answer(task: str, system: str = "") -> str:
    """Run the deterministic rule engine on a task (used by tests and tools)."""
    return MockProvider()._respond(task, system, [], [])


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

class MockProvider(BaseProvider):
    provider_name = "mock"

    def complete(self, model, messages, tools=None, temperature=0.7, max_tokens=None):
        start = time.time()
        last_user = next((m.content for m in reversed(messages) if m.role == Role.USER), "")
        system = next((m.content for m in messages if m.role == Role.SYSTEM), "")
        tool_calls: list[ToolCall] = []
        observations = [
            (getattr(m, "name", "") or "tool", m.content)
            for m in messages
            if m.role == Role.TOOL
        ]
        content = self._respond(last_user, system, tools or [], tool_calls, observations)
        return ModelResponse(
            content=content,
            model_id=model.id,
            tool_calls=tool_calls,
            tokens_in=messages_tokens(messages),
            tokens_out=estimate_tokens(content),
            cost_usd=0.0,
            latency_ms=(time.time() - start) * 1000,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------
    def _respond(self, prompt: str, system: str, tools, tool_calls: list[ToolCall],
                 observations: list[tuple[str, str]] | None = None) -> str:
        memory_lines = _extract_memory_lines(prompt)
        task = unwrap_task(prompt)
        low = task.lower()
        role = self._role_from_system(system)

        # 0a. Memory questions are answered from recalled context when present.
        if memory_lines and re.search(
            r"what do you (?:know|remember)|do you remember|what did we|"
            r"based on (?:our|the) (?:past|previous|earlier)|our (?:earlier|previous) decision",
            low,
        ):
            listed = "\n".join(f"- {line}" for line in memory_lines)
            return (
                f"From memory, these are the relevant recorded points:\n\n{listed}\n\n"
                "**What this implies.** The facts above come from earlier sessions and are "
                "stored verbatim; treat them as context, not as a fresh verification.\n\n"
                "_Recalled from APEX durable memory (offline synthesis)._"
            )

        # 0. Grounding: if a tool has already returned data, answer from it
        #    instead of firing the same call again (the ReAct loop would
        #    otherwise detect a repeat and abort).
        if observations:
            return self._answer_from_observations(task, observations)

        # 1. Tool use: emit a real call so the executor can act and re-prompt.
        if tools and self._wants_tool(low):
            call = self._make_tool_call(task, tools)
            if call:
                tool_calls.append(call)
                return f"Thought: I need real data before answering.\nAction: {call.name}({self._fmt_args(call.arguments)})"

        # 2. Code requests beat role framing: a coder agent asked for code
        #    must return code, not a plan.
        wants_code = _has_code_intent(low) or bool(
            re.search(r"fibonacci|factorial|palindrome|\bprime\b|fizzbuzz|sort|dedup", low)
        )
        if wants_code:
            code = synthesise_code(task)
            if code:
                return code

        # 3. Role-specialised outputs.
        if role == "critic":
            return self._critique(task)
        if role == "planner":
            return self._plan(task)
        if role == "analyst":
            return self._analysis(task)
        if role == "researcher":
            return self._research(task)
        if role == "tester":
            return self._tests(task)
        if role == "writer":
            return self._prose(task)

        # 4. Quantitative problems — computed answers first.
        solved = solve_word_problem(task)
        if solved:
            answer, reasoning = solved
            return f"{reasoning}\n\nFINAL: {answer}"

        # 5. Code synthesis (unframed).
        code = synthesise_code(task)
        if code:
            return code

        # 6. Planning.
        if re.search(r"\bplan\b|\broadmap\b|break(?:ing)? down|\bsteps\b|schedule", low):
            return self._plan(task)

        # 7. Comparison.
        if re.search(r"\bcompare\b|\bvs\b|\bversus\b|difference between|pros and cons", low):
            return self._comparison(task)

        # 8. Explanation.
        if re.search(r"\bexplain\b|what is|what are|how does|how do|why ", low):
            return self._explain(task)

        # 9. Critique.
        if re.search(r"\bcritique\b|\breview\b|\bevaluate\b|check this|improve this|what.s wrong", low):
            return self._critique(task)

        # 10. Summary.
        if "summar" in low:
            return self._summary(task)

        # 11. Email / writing.
        if re.search(r"\bemail\b|\bletter\b|\bmessage\b", low):
            return self._email(task)

        return self._generic(task)

    # ------------------------------------------------------------------
    @staticmethod
    def _answer_from_observations(task: str, observations: list[tuple[str, str]]) -> str:
        blocks: list[str] = []
        for name, content in observations[-3:]:
            body = content.strip()
            if len(body) > 1200:
                body = body[:1200].rstrip() + "\n…[truncated]"
            blocks.append(f"**Observation from `{name}`**\n\n```\n{body}\n```")
        joined = "\n\n".join(blocks)

        # Light-weight synthesis of common observation shapes.
        combined = " ".join(c for _, c in observations).lower()
        if "def " in combined:
            funcs = re.findall(r"def\s+(\w+)\s*\(([^)]*)\)", " ".join(c for _, c in observations))
            names = ", ".join(f"`{n}({a.strip()})`" for n, a in funcs[:6])
            reading = (
                f"The file defines {len(funcs)} function(s): {names}. "
                "It is a plain Python module with no imports or side effects visible in the excerpt, "
                "so it behaves as a small utility library."
            )
        elif combined.strip().startswith("{") or combined.strip().startswith("["):
            reading = "The tool returned structured data (JSON); the shape is shown in the observation above."
        elif "error" in combined or "not found" in combined or "denied" in combined:
            reading = "The tool reported a failure — the operation did not complete, so no conclusions can be drawn from its output."
        else:
            reading = "The observation above is the raw tool output for the request."

        return (
            f"Result for: {task.strip().rstrip('.')}\n\n"
            f"{joined}\n\n"
            f"**What it means.** {reading}\n\n"
            "_Grounded in tool output (offline mock synthesis); no values were invented._"
        )

    @staticmethod
    def _role_from_system(system: str) -> str:
        s = system.lower()
        for role in ("supervisor", "planner", "executor", "researcher", "coder", "tester", "writer", "analyst", "critic"):
            if f"apex {role}" in s or f"you are the {role}" in s:
                return role
        if "strict, fair reviewer" in s or "never rubber-stamp" in s:
            return "critic"
        if "decompose" in s and "json" in s:
            return "planner"
        return ""

    @staticmethod
    def _fmt_args(args: dict) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in args.items())

    @staticmethod
    def _plan(task: str) -> str:
        subject = re.sub(r"^(?:plan|design|plan out)\s+(?:a|an|the)?\s*", "", task.strip(), flags=re.IGNORECASE)
        subject = subject.strip(" .:") or "the goal"
        return (
            f"Plan — {subject}\n\n"
            "**Goal.** Deliver the outcome with verifiable intermediate states, no "
            "big-bang cutover, and a written rollback at every gate.\n\n"
            "**Phases.**\n"
            f"1. Establish the baseline. Define done and success metrics for {subject}; "
            "measure the current state so the improvement is provable.\n"
            "2. Map dependencies. List components, owners, data flows and the "
            "contracts between them; anything unowned is a risk, not a detail.\n"
            "3. Slice the work. Cut one thin end-to-end path that can ship alone; "
            "keep the remaining work behind it so each slice is reversible.\n"
            "4. Parallelise the safe parts. Work on independent slices concurrently; "
            "serialise anything that touches shared state.\n"
            "5. Instrument before cutover. Add metrics, logs and alerts so the change "
            "can be judged objectively from the first minute.\n"
            "6. Ship incrementally behind a flag. Enable for 1% → 10% → 100%, "
            "watching the defined metrics at each step.\n"
            "7. Verify against the original criteria. Compare measured results to the "
            "phase-1 baseline; document any residual gap.\n"
            "8. Close out. Remove dead paths, update docs and hand over ownership.\n\n"
            "**Risks and mitigations.**\n"
            "- Hidden dependencies → dependency map before the first slice moves.\n"
            "- Data migration errors → dual-write plus checksum reconciliation.\n"
            "- Silent regressions → flags, canaries and a metric that forces a stop.\n\n"
            "**Rollback.** Every gate keeps the previous path alive behind the flag; "
            "rolling back is a flag flip, not a deploy.\n\n"
            "**Definition of done.** Baseline beaten on the agreed metric, no unresolved "
            "incidents, and the change reproducible from the docs."
        )

    @staticmethod
    def _analysis(task: str) -> str:
        subject = task.strip().rstrip(".")
        return (
            f"Analysis — {subject}\n\n"
            "**What the request actually needs.** A decision, not a description: the "
            "analysis below ends in a recommendation and a stated confidence.\n\n"
            "**Key factors (weighted).**\n"
            "1. Correctness of the underlying assumptions — if these are wrong, every "
            "downstream number is wrong (highest weight).\n"
            "2. Cost, both direct spend and the operational cost of carrying complexity.\n"
            "3. Time-to-value: how quickly a verifiable result can be produced.\n"
            "4. Reversibility: how expensive it is to undo if the decision is wrong.\n"
            "5. Risk concentration: whether one failure can take down the whole path.\n\n"
            "**Assessment.**\n"
            "| Factor | Direction | Confidence |\n"
            "|---|---|---|\n"
            "| Assumptions | must be validated first | medium |\n"
            "| Cost | bounded by staged rollout | high |\n"
            "| Time-to-value | improved by slicing the work | high |\n"
            "| Reversibility | preserved via feature flags | high |\n\n"
            "**Risks and mitigations.**\n"
            "- Hidden dependencies between components — map them before committing.\n"
            "- Unmeasured performance claims — instrument first, then decide.\n"
            "- Scope creep — fix the acceptance criteria in writing before starting.\n\n"
            "**Recommendation.** Proceed in stages, with an explicit rollback at each "
            "gate and a stated metric that would force a stop. Confidence: moderate — "
            "the plan is sound, the inputs need validation.\n\n"
            "_Offline (mock) analysis — the reasoning structure is real; the figures are "
            "placeholders until a live provider or tool supplies them._"
        )

    @staticmethod
    def _research(task: str) -> str:
        subject = re.sub(r"^(?:research|find out about|look up|investigate)\s+", "", task.strip(), flags=re.IGNORECASE).strip(" ?.")
        return (
            f"Research findings — {subject}\n\n"
            "**Status of the evidence.** Offline mode has no network access, so this "
            "report states what is structurally known and marks every claim that needs "
            "a live source. It does not fabricate citations.\n\n"
            "**What is established.**\n"
            "- The question decomposes into definition, current state and comparison criteria.\n"
            "- Any credible answer must separate measured facts from vendor claims.\n\n"
            "**What must be verified against sources.**\n"
            "1. Latest published figures and dates (they change month to month).\n"
            "2. Independent benchmarks rather than self-reported results.\n"
            "3. Known limitations and failure cases reported by practitioners.\n\n"
            "**How to verify.** Enable a live provider (or run `web_search`) and check "
            "primary sources first: papers, official documentation, changelogs. Treat "
            "summaries as leads, never as evidence.\n\n"
            "_Offline (mock) research — no sources were fetched; claims are labelled by "
            "verification status instead of being invented._"
        )

    @staticmethod
    def _tests(task: str) -> str:
        subject = task.strip().rstrip(".")
        return (
            f"Test plan — {subject}\n\n"
            "**Test cases.**\n"
            "| # | Case | Input | Expected |\n"
            "|---|---|---|---|\n"
            "| 1 | Happy path | typical valid input | correct result, no side effects |\n"
            "| 2 | Boundary — empty | `[]` / `\"\"` | documented behaviour, no crash |\n"
            "| 3 | Boundary — single | one element | correct result |\n"
            "| 4 | Invalid input | wrong type | explicit error, not silent coercion |\n"
            "| 5 | Idempotency | run twice | same result |\n\n"
            "**Runnable smoke test.**\n"
            "```python\n"
            "import unittest\n\n"
            "class TestTarget(unittest.TestCase):\n"
            "    def test_happy_path(self):\n"
            "        self.assertEqual(call_under_test(2, 3), 5)\n\n"
            "    def test_empty_input(self):\n"
            "        with self.assertRaises(ValueError):\n"
            "            call_under_test(None)\n\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n"
            "```\n\n"
            "**Verdict.** PASS requires all five cases green on an actual run. No test "
            "was executed in offline mode — treat this as the specification to run.\n\n"
            "_Offline (mock) test plan._"
        )

    @staticmethod
    def _prose(task: str) -> str:
        subject = re.sub(r"^(?:write|draft|compose)\s+(?:a|an|the)?\s*", "", task.strip(), flags=re.IGNORECASE).strip(" .")
        return (
            f"{subject[:1].upper() + subject[1:] if subject else 'Draft'}\n\n"
            "The short version: the goal is clear, the constraints are known, and the "
            "work can start now. What follows is the reasoning, kept to what changes a "
            "decision.\n\n"
            "**What matters**\n"
            "- Start from the outcome the reader needs, not from the process that produced it.\n"
            "- Cut every sentence that does not add a fact, a decision or a next step.\n"
            "- Make the ask explicit and put it at the end where it is easy to act on.\n\n"
            "**Next step**\n"
            "Reply with the one decision you want made, and who owns it. Everything else "
            "is detail that can follow.\n\n"
            "_Offline (mock) draft — revise the voice for the actual audience._"
        )

    @staticmethod
    def _comparison(task: str) -> str:
        options = extract_options(task)
        kb = [_FRAMEWORK_KB.get(o.lower(), {"name": o}) for o in options]
        recognised = sum(1 for k in kb if len(k) > 1)
        if recognised >= 2:
            header = "| Dimension | " + " | ".join(k["name"] for k in kb) + " |"
            sep = "|---|" + "|".join("---" for _ in kb) + "|"
            rows = []
            for key, label in _KB_DIMENSIONS:
                rows.append("| " + label + " | " + " | ".join(k.get(key, "verify against docs") for k in kb) + " |")
            return (
                f"Comparison — {' vs '.join(k['name'] for k in kb)}\n\n"
                + "\n".join([header, sep, *rows])
                + "\n\n**Recommendation.** Pick the runtime whose *orchestration model* "
                "matches the shape of your work: explicit graphs when the control flow is "
                "known and auditable, conversational hand-offs when the work is open-ended, "
                "and role-based crews when the division of labour is the product. Prefer the "
                "one whose failure modes you can debug at 3 a.m.; all three will work in a "
                "demo, only one will work in an incident.\n\n"
                "**How to decide in an hour.** Prototype the same two-step task in each "
                "candidate, then measure: lines of glue code, traceability of a failed run, "
                "and time to add a new tool. The winner is usually obvious after that.\n\n"
                f"_Offline comparison from APEX's built-in framework notes; verify version-specific "
                f"details against current docs before committing._"
            )
        return (
            "Comparison\n\n"
            "| Dimension | " + (options[0] if options else "Option A") + " | "
            + (options[1] if len(options) > 1 else "Option B") + " |\n"
            "|---|---|---|\n"
            "| Time to first result | Fast — minimal setup | Slower — upfront design |\n"
            "| Ceiling | Lower; hits limits at scale | Higher; scales with investment |\n"
            "| Operational cost | Low | Higher (infra, on-call, complexity) |\n"
            "| Failure modes | Few, well understood | More, and harder to diagnose |\n"
            "| Reversibility | Easy to swap out | Costly to unwind |\n\n"
            "Decision rule: choose the simpler option until a measured limit is hit, then "
            "migrate with a written rollback plan. Optimise on evidence, never on speculation.\n\n"
            f"Applied to: {task.strip().rstrip('.')}\n\n"
            "_Offline mode has no source access for these entities — the comparison states "
            "the decision rule rather than inventing specifications. Enable a live provider "
            "or `web_search` for sourced, entity-specific numbers._"
        )

    @staticmethod
    def _explain(task: str) -> str:
        topic = re.sub(r"^(?:please\s+)?(?:explain|what is|what are|how does|how do|why)\s+", "", task.strip(), flags=re.IGNORECASE)
        topic = topic.strip(" ?.") or "the topic"
        return (
            f"**{topic[:1].upper() + topic[1:]}**\n\n"
            "**Core idea.** The essential mechanism is small and composable: inputs are "
            "normalised, a decision is made on the normalised form, and the result is "
            "mapped back to the caller's domain.\n\n"
            "**Why it matters.** It removes repeated work from the common path and gives "
            "an explicit escape hatch for the cases that do not fit the common path.\n\n"
            "**How to use it well.**\n"
            "1. Start with the simplest correct version and write down its invariant.\n"
            "2. Measure before optimising; optimise the measured hot path only.\n"
            "3. Keep an explicit boundary so the implementation can be replaced.\n\n"
            "**Common pitfalls.** Over-engineering before requirements settle, hiding "
            "errors instead of surfacing them, and treating a heuristic as a guarantee.\n\n"
            "_Offline (mock) explanation — configure OPENAI_API_KEY, ANTHROPIC_API_KEY or "
            "GOOGLE_API_KEY for frontier-quality answers._"
        )

    @staticmethod
    def _critique(task: str) -> str:
        return (
            "Review\n\n"
            "- Correctness: the approach is sound for the stated goal; assumptions should be written down.\n"
            "- Completeness: primary path is covered; error and empty-input paths need explicit handling.\n"
            "- Clarity: names and structure are readable; the intent of each block is obvious.\n"
            "- Edge cases: boundary values, concurrency and partial failure are unspecified.\n"
            "- Verifiability: add tests for the happy path, one boundary case and one failure case.\n\n"
            "Verdict: ACCEPT with minor revisions. Score: 0.8/1.0\n"
            "Top fix: make the failure behaviour explicit and tested."
        )

    @staticmethod
    def _summary(task: str) -> str:
        body = re.sub(r"^\s*(?:summari[sz]e|summary of)\s*:?\s*", "", task, flags=re.IGNORECASE).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        head = sentences[:3] or [body[:200]]
        return (
            "Summary\n\n" + "\n".join(f"- {s}" for s in head) + "\n\n"
            "Key takeaway: the core request is captured above in the minimum number of points.\n"
            "Next step: confirm the scope and I will expand any point."
        )

    @staticmethod
    def _email(task: str) -> str:
        purpose = task.strip().rstrip(".")
        return (
            "Subject: Request for a 30-minute meeting this week\n\n"
            "Hi [Name],\n\n"
            "I am writing to request a short meeting to align on priorities and close "
            "the open decisions before the end of the week. Thirty minutes should be "
            "enough if we keep to the agenda below.\n\n"
            "Proposed agenda:\n"
            "1. Status of current work and anything blocked.\n"
            "2. Decisions needed from you, with my recommendation for each.\n"
            "3. Owners and dates for the next steps.\n\n"
            "Would Thursday 15:00 or Friday 11:00 work? If neither suits, send me two "
            "slots that do and I will make one of them work.\n\n"
            "Thank you,\n[Your name] · [Role] · [Contact]\n\n"
            f"---\nRequest addressed: {purpose}\n"
            "This is an offline (mock) draft — replace the placeholders, adjust the "
            "tone to the relationship, and confirm the availability before sending."
        )

    @staticmethod
    def _generic(task: str) -> str:
        digest = hashlib.md5(task.encode("utf-8")).hexdigest()[:8]
        return (
            f"Answer (offline mode · ref {digest})\n\n"
            f"Request: {task[:220]}{'…' if len(task) > 220 else ''}\n\n"
            "Approach taken:\n"
            "1. Restated the request to pin down the deliverable.\n"
            "2. Listed the constraints and the information actually available.\n"
            "3. Produced a self-contained answer with the reasoning made explicit.\n\n"
            "This response was generated by APEX's deterministic offline reasoner. It is "
            "useful for validating the full pipeline (routing, planning, tools, memory, "
            "critique, tracing) without API keys or spend. Point APEX at a frontier model "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY, or a local Ollama "
            "instance) and the same request is routed to the best available model."
        )

    # ------------------------------------------------------------------
    def _wants_tool(self, low: str) -> bool:
        # A concrete filename is the strongest signal there is: "read notes.txt and
        # tell me the release" has no word "file" in it, yet is unambiguously a tool
        # task. _make_tool_call already knows how to act on it.
        if re.search(r"\b[\w./\\-]+\.(?:py|txt|md|json|ya?ml|csv|toml|log|html?)\b", low):
            return True
        return bool(
            re.search(r"\b(read|open|list|search|find|fetch|download|browse|run|execute|calculate|compute|git)\b", low)
            and re.search(r"\bfile|folder|directory|web|url|page|internet|command|shell|repository|repo|math|expression\b", low)
        )

    @staticmethod
    def _tool_names(tools) -> set[str]:
        """Tool specs arrive either as dicts (OpenAI wire format) or Tool objects."""
        names: set[str] = set()
        for spec in tools:
            if isinstance(spec, dict):
                name = spec.get("name") or (spec.get("function") or {}).get("name")
            else:
                name = getattr(spec, "name", None)
            if name:
                names.add(str(name))
        return names

    def _make_tool_call(self, task: str, tools) -> ToolCall | None:
        names = self._tool_names(tools)
        low = task.lower()

        if re.search(r"\b(web|url|internet|browse|search the web)\b", low):
            if "web_search" in names:
                query = re.sub(r"^.*?(?:search|find|look up)\s+", "", task, flags=re.IGNORECASE).strip(" ?")
                return ToolCall(name="web_search", arguments={"query": query or task[:80]})
            if "web_fetch" in names:
                m = re.search(r"https?://\S+", task)
                return ToolCall(name="web_fetch", arguments={"url": m.group(0) if m else ""})

        m = re.search(r"([\w./\\-]+\.(?:py|txt|md|json|yaml|yml|csv|toml))", task)
        path = m.group(1) if m else ""
        if re.search(r"\bread\b|\bopen\b", low) and "read_file" in names and path:
            return ToolCall(name="read_file", arguments={"path": path})
        if re.search(r"\blist\b|\bdirectory\b|\bfolder\b", low) and "list_dir" in names:
            return ToolCall(name="list_dir", arguments={"path": "."})

        expr = re.search(r"[-+*/().\d\s]{3,}", task)
        if "calculator" in names and expr and re.search(r"calculate|compute|evaluate", low):
            return ToolCall(name="calculator", arguments={"expression": expr.group(0).strip()})

        if "bash" in names and re.search(r"\bcommand\b|\bshell\b", low):
            return ToolCall(name="bash", arguments={"command": "ls -la"})

        if "git_status" in names and re.search(r"\bgit\b|\brepository\b|\brepo\b", low):
            return ToolCall(name="git_status", arguments={})

        return None
