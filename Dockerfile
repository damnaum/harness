# APEX — production image.
#
# The harness has zero runtime dependencies, so the image is the interpreter,
# the source and nothing else: no compiler, no build tools, no pip cache.
FROM python:3.12-slim AS runtime

# The prod profile enables all safety checks and turns OFF tool auto-approval:
# bash/write_file are denied until you install an approval handler. That is the
# intended default for a network-reachable deployment (see docs/SECURITY.md).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    APEX_PROFILE=prod \
    APEX_WORKSPACE=/workspace \
    APEX_MEMORY_PERSIST=true

WORKDIR /app

# Install the package first (best layer caching); no network deps to resolve.
COPY pyproject.toml README.md ./
COPY apex ./apex
RUN pip install --no-cache-dir . && rm -rf /app/apex

# Unprivileged runtime: an agent that can run shell commands should not be root.
RUN groupadd --system --gid 10001 apex \
 && useradd --system --uid 10001 --gid apex --home /home/apex --shell /usr/sbin/nologin apex \
 && mkdir -p /workspace /home/apex/.apex \
 && chown -R apex:apex /workspace /home/apex

USER apex

# Runtime state (memory, checkpoints) lives here; mount a volume to keep it.
VOLUME ["/workspace"]

EXPOSE 8000

# Readiness without curl: the slim image has no HTTP client binary.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# SIGTERM is handled by the server (graceful shutdown); do not wrap in a shell.
ENTRYPOINT ["apex"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
