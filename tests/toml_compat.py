"""TOML reading for the test suite on every supported Python.

``tomllib`` arrived in Python 3.11, but ``pyproject.toml`` requires-python is
``>=3.10`` — so the packaging tests crashed at import time on 3.10 with
``ModuleNotFoundError`` and the CI matrix went red on exactly one interpreter.

Rather than skip the assertions on 3.10 (which is how packaging bugs slip
through), this module provides a small TOML reader covering the subset a
pyproject file actually uses: comments, ``[table]``/``[table.sub]`` headers,
``[[array.of.tables]]``, quoted strings, integers/floats, booleans, arrays
(including multiline and mixed), and inline tables.

On 3.11+ ``tomllib`` is preferred, and the test suite asserts that this fallback
produces *identical* output for the real pyproject — so the code path 3.10 runs
is continuously verified by the interpreters that have the real parser.
"""

from __future__ import annotations

import re
from typing import Any

try:  # Python 3.11+
    import tomllib as _tomllib
except ImportError:  # pragma: no cover - exercised on 3.10 only
    # ImportError, not just ModuleNotFoundError: a partially installed or
    # shadowed module must fall back too, not crash the suite.
    _tomllib = None


class TomlError(ValueError):
    """Raised when the fallback reader cannot make sense of the input."""


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_NUMBER = re.compile(r"[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?")


def loads_fallback(text: str) -> dict[str, Any]:
    """Parse the TOML subset used by pyproject files (see module docstring)."""
    root: dict[str, Any] = {}
    table: dict[str, Any] = root
    for raw_line in _logical_lines(text):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            path = _split_key(line[2:-2].strip())
            parent = _walk(root, path[:-1], create=True)
            bucket = parent.setdefault(path[-1], [])
            if not isinstance(bucket, list):
                raise TomlError(f"array-of-tables collides with a value: {path}")
            table = {}
            bucket.append(table)
            continue
        if line.startswith("[") and line.endswith("]"):
            path = _split_key(line[1:-1].strip())
            table = _walk(root, path, create=True)
            continue
        if "=" not in line:
            raise TomlError(f"expected 'key = value', got: {line!r}")
        key_part, _, value_part = line.partition("=")
        key = _parse_key(key_part.strip())
        value, rest = _parse_value(value_part.strip(), 0)
        if rest.strip():
            raise TomlError(f"trailing characters after value: {rest!r}")
        table[key] = value
    return root


def loads(text: str) -> dict[str, Any]:
    """Parse TOML with the interpreter's parser when available."""
    if _tomllib is not None:
        return _tomllib.loads(text)
    return loads_fallback(text)


def load_pyproject(path) -> dict[str, Any]:
    from pathlib import Path

    return loads(Path(path).read_text(encoding="utf-8"))


# -- internals -------------------------------------------------------------


def _logical_lines(text: str) -> list[str]:
    """Strip comments and join multi-line arrays into single logical lines."""
    out: list[str] = []
    buffer = ""
    depth = 0
    for physical in text.splitlines():
        stripped = _strip_comment(physical)
        buffer = f"{buffer} {stripped.strip()}" if buffer else stripped
        depth += _bracket_delta(stripped)
        if depth <= 0:
            if buffer.strip():
                out.append(buffer)
            buffer = ""
            depth = 0
    if buffer.strip():
        out.append(buffer)
    return out


def _strip_comment(line: str) -> str:
    in_string: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\" and in_string == '"':
                escaped = True
            elif char == in_string:
                in_string = None
        elif char in "\"'":
            in_string = char
        elif char == "#":
            return line[:index]
    return line


def _bracket_delta(line: str) -> int:
    """Count unclosed [ and { so a multi-line array can be joined."""
    delta = 0
    in_string: str | None = None
    escaped = False
    for char in line:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\" and in_string == '"':
                escaped = True
            elif char == in_string:
                in_string = None
        elif char in "\"'":
            in_string = char
        elif char in "[{":
            delta += 1
        elif char in "]}":
            delta -= 1
    return delta


def _split_key(key: str) -> list[str]:
    return [_parse_key(part.strip()) for part in _split_top_level(key, ".")]


def _parse_key(key: str) -> str:
    key = key.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        return key[1:-1]
    if not _BARE_KEY.fullmatch(key):
        raise TomlError(f"invalid key: {key!r}")
    return key


def _walk(root: dict[str, Any], path: list[str], *, create: bool) -> dict[str, Any]:
    node = root
    for part in path:
        if create:
            node = node.setdefault(part, {})
        else:
            node = node[part]
        if not isinstance(node, dict):
            raise TomlError(f"key {part!r} is not a table")
    return node


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    in_string: str | None = None
    escaped = False
    current = ""
    for char in text:
        if in_string:
            current += char
            if escaped:
                escaped = False
            elif char == "\\" and in_string == '"':
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in "\"'":
            in_string = char
            current += char
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == separator and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    if current.strip() or parts:
        parts.append(current)
    return parts


def _parse_value(text: str, index: int) -> tuple[Any, str]:
    """Return (value, remaining_text) starting at ``index``."""
    text_rest = text[index:]
    stripped = text_rest.lstrip()
    consumed = len(text_rest) - len(stripped)
    index += consumed
    if not stripped:
        raise TomlError("missing value")

    char = stripped[0]
    if char in "\"'":
        value, used = _parse_string(text, index)
        return value, text[index + used:]
    if char == "[":
        return _parse_array(text, index)
    if char == "{":
        return _parse_inline_table(text, index)

    for literal, value in (("true", True), ("false", False)):
        if stripped.startswith(literal):
            return value, text[index + len(literal):]
    match = _NUMBER.match(stripped)
    if match:
        number_text = match.group(0)
        end = index + len(number_text)
        if re.fullmatch(r"[+-]?\d+", number_text):
            return int(number_text), text[end:]
        return float(number_text), text[end:]
    raise TomlError(f"unsupported value at: {stripped[:40]!r}")


def _parse_string(text: str, index: int) -> tuple[str, int]:
    quote = text[index]
    if quote == "'":
        end = text.index("'", index + 1)
        return text[index + 1:end], end - index + 1
    out: list[str] = []
    position = index + 1
    while position < len(text):
        char = text[position]
        if char == "\\":
            nxt = text[position + 1] if position + 1 < len(text) else ""
            out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}.get(nxt, nxt))
            position += 2
            continue
        if char == '"':
            return "".join(out), position - index + 1
        out.append(char)
        position += 1
    raise TomlError("unterminated string")


def _parse_array(text: str, index: int) -> tuple[list[Any], str]:
    items: list[Any] = []
    position = index + 1
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text):
            raise TomlError("unterminated array")
        if text[position] == "]":
            return items, text[position + 1:]
        value, rest = _parse_value(text, position)
        items.append(value)
        position = len(text) - len(rest)


def _parse_inline_table(text: str, index: int) -> tuple[dict[str, Any], str]:
    table: dict[str, Any] = {}
    position = index + 1
    while True:
        while position < len(text) and text[position] in " \t,\r\n":
            position += 1
        if position >= len(text):
            raise TomlError("unterminated inline table")
        if text[position] == "}":
            return table, text[position + 1:]
        pivot = position
        while pivot < len(text) and text[pivot] != "=":
            pivot += 1
        key = _parse_key(text[position:pivot].strip())
        value, rest = _parse_value(text, pivot + 1)
        table[key] = value
        position = len(text) - len(rest)
