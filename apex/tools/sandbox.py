"""Sandbox primitives shared by every builtin tool.

Keeping these in one module matters: a path check copied into five tools drifts
apart, and the failure mode is silent (a tool quietly reading outside the
workspace). The functions here are:

``safe_path``           containment check that survives ``..`` and symlinks
``restricted_env``      child-process environment without credentials
``normalise_command``   collapse whitespace/quoting before deny-list matching
``check_command``       deny destructive shell patterns (best-effort defence)
``check_url``           block SSRF targets (loopback, private, link-local, metadata)
``read_limited``        bounded reads so a 10 GB response cannot exhaust memory
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from pathlib import Path

# --------------------------------------------------------------------------
# filesystem
# --------------------------------------------------------------------------

def safe_path(root: str | Path, user_path: str | None) -> Path:
    """Resolve ``user_path`` inside ``root`` or raise ``PermissionError``.

    Uses ``relative_to`` rather than a string prefix check: the prefix form
    wrongly accepts ``/tmp/workspace-evil`` for root ``/tmp/workspace``, which
    is exactly the shape of a directory traversal attack.
    """
    root_path = Path(root).resolve()
    raw = (user_path or ".").strip() or "."
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else root_path / candidate).resolve()

    if resolved == root_path:
        return resolved
    try:
        resolved.relative_to(root_path)
    except ValueError:
        raise PermissionError(f"path escapes sandbox: {user_path}") from None
    return resolved


# --------------------------------------------------------------------------
# process environment
# --------------------------------------------------------------------------

_SECRET_HINTS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
    "SESSION", "COOKIE", "PRIVATE",
)
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "TMPDIR")


def restricted_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal environment for child processes.

    Anything whose name looks like a credential is dropped unless explicitly
    allow-listed. A tool that can read ``os.environ`` in a subprocess would
    otherwise exfiltrate every provider API key the harness holds.
    """
    import os

    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if k in _ENV_ALLOWLIST}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_HINTS)


# --------------------------------------------------------------------------
# shell commands
# --------------------------------------------------------------------------

# Destructive patterns. This is defence in depth, not a jail: the bash tool is
# privileged by nature and must stay approval-gated. Matching is done on a
# normalised form so trivial spacing/quoting variations do not slip through.
_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+(/|/\*|~|\$HOME)(\s|$)", "recursive delete of root or home"),
    (r"\brm\s+-[a-z]*r[a-z]*f?\b.*\s/(\s|$)", "recursive delete at filesystem root"),
    (r"\bmkfs(\.|\s|$)", "filesystem format"),
    (r"\bdd\b[^\n]*\bof=/dev/", "raw device write"),
    (r">\s*/dev/(sd|hd|nvme|vd)", "raw device write"),
    (r"\b(shutdown|reboot|halt|poweroff|init\s+0|init\s+6)\b", "host power control"),
    (r":\(\)\s*\{.*\}\s*;?\s*:", "fork bomb"),
    (r"\b(curl|wget)\b[^\n|]*\|\s*(ba|z|da)?sh\b", "pipe download into a shell"),
    (r"\bchmod\s+(-R\s+)?777\s+/(\s|$)", "world-writable root"),
    (r"\bchown\s+-R\b[^\n]*\s/(\s|$)", "recursive ownership change at root"),
    (r"\bsudo\b", "privilege escalation"),
    (r"/etc/(passwd|shadow|sudoers)", "credential file access"),
    (r"\bhistory\s+-c\b", "audit trail destruction"),
    (r"\bcat\s+/dev/(zero|random|urandom)\b", "unbounded device read"),
)

_DENY_RE = tuple((re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in _DENY_PATTERNS)


def normalise_command(cmd: str) -> str:
    """Collapse whitespace and repeated quotes so matching is robust."""
    text = cmd.replace("\\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"['\"`]{2,}", "", text)
    return text.strip()


def check_command(cmd: str) -> None:
    """Raise ``PermissionError`` for a destructive command."""
    normalised = normalise_command(cmd)
    for pattern, reason in _DENY_RE:
        if pattern.search(normalised):
            raise PermissionError(f"blocked command ({reason}): {cmd[:120]}")


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

_BLOCKED_HOSTNAMES = {
    "localhost", "localhost.localdomain", "ip6-localhost", "metadata",
    "metadata.google.internal", "instance-data",
}
_MAX_REDIRECTS = 3


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def check_url(url: str, allow_private: bool = False) -> str:
    """Validate a URL for outbound fetching; returns the normalised URL.

    Blocks non-HTTP schemes and any host resolving to a private, loopback or
    link-local address (SSRF: the metadata endpoint on a cloud VM is the
    classic target of a prompt-injected fetch).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("url must start with http:// or https://")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("url has no host")

    if allow_private:
        return url

    if host in _BLOCKED_HOSTNAMES:
        raise PermissionError(f"blocked host: {host}")

    # Literal IP?
    try:
        if _ip_is_blocked(ipaddress.ip_address(host)):
            raise PermissionError(f"blocked address: {host}")
        return url
    except ValueError:
        pass  # not an IP literal; resolve below

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host {host}: {exc}") from None
    for info in infos:
        address = info[4][0]
        try:
            if _ip_is_blocked(ipaddress.ip_address(address)):
                raise PermissionError(f"blocked address for {host}: {address}")
        except ValueError:
            continue
    return url


def read_limited(response, limit: int = 1_000_000) -> bytes:
    """Read at most ``limit`` bytes from an HTTP response."""
    return response.read(limit)


# --------------------------------------------------------------------------
# arithmetic safety (calculator tool)
# --------------------------------------------------------------------------

MAX_EXPONENT = 1_000
MAX_ABS_RESULT = 1e100


def guard_power(base: float, exponent: float) -> float:
    """Reject tower-of-power inputs that would hang the process."""
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent too large (max {MAX_EXPONENT})")
    result = base**exponent
    if isinstance(result, complex) or abs(result) > MAX_ABS_RESULT:
        raise ValueError("result out of range")
    return result
