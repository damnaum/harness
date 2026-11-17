# Security

The threat model, the defences and — importantly — the things APEX does **not**
protect against are documented in **[`docs/SECURITY.md`](docs/SECURITY.md)**.

Short version:

- **Defended:** direct and indirect (tool-output) prompt injection, secret/PII
  leakage in inputs, outputs and child processes, filesystem escape from the
  workspace, destructive shell commands, SSRF, runaway tool output, unapproved
  tool use.
- **Not defended:** arbitrary code executed by `python_exec`/`bash` runs with the
  OS user's privileges. APEX is a logical sandbox, not a container. If you run
  untrusted code, run the whole harness inside a container or VM.

Reporting a vulnerability: open a private security advisory on the repository
rather than a public issue.
