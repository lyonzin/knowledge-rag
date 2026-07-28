"""
╭─╴ KNOWLEDGE RAG TELEMETRY ╶────────────────────────────────────╮
│                                                                │
│   Structured JSON logging + OpenTelemetry tracing.             │
│   Both features are opt-in — the module ships no-op tracers    │
│   and plain-text logging by default so the server keeps the    │
│   pre-A2.6 behaviour when no telemetry knob is set.            │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Design contract:
    - Logs go to stderr. Never to stdout — MCP stdio uses stdout for JSON-RPC
      and any stray byte on it corrupts the framing.
    - Default: plain-text ``%(message)s`` formatter at INFO level. Users who
      never touched ``config.yaml`` see the same output shape as ``print()``.
    - Opt-in ``json_logs: true``: every log record becomes a single-line
      JSON object with ``timestamp`` / ``level`` / ``logger`` / ``message``
      plus any structured ``extra`` fields the caller supplied.
    - Tracing is a no-op unless ``telemetry.enabled: true``. The OTel SDK is
      an OPTIONAL dependency shipped in the ``[otel]`` extra — importing this
      module never requires ``opentelemetry``. When tracing is asked for but
      the SDK is missing the caller is warned and the no-op tracer is used.
    - Spans added around ``KnowledgeOrchestrator.query`` and each MCP tool
      handler stay zero-cost when tracing is disabled: ``_NoopTracer`` short
      circuits into a plain context manager that does no allocation on the
      hot path.

The ``@trace`` decorator is a convenience wrapper — it always resolves the
current tracer at call time so callers can enable tracing after import.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional

# ╭─╴ STRUCTURED JSON LOGGING ╶───────────────────────────────────╮
# │       Single-line JSON records with optional extra fields     │
# ╰───────────────────────────────────────────────────────────────╯


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON documents.

    The output shape is intentionally flat and deterministic so downstream
    log aggregators (Loki, ELK, Datadog, Cortex XDR ingest) can index it
    without a custom parser. Extra fields are merged at the top level via
    ``record.extra_fields`` — attach them by passing ``extra={"extra_fields":
    {...}}`` to any logger call.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single JSON line.

        Args:
            record: Standard logging record produced by any ``logging`` call.

        Returns:
            str: JSON-encoded log entry (no trailing newline — the handler
            adds one).
        """
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Optional correlation ID injected by the caller (e.g. request handler
        # that pulls the active OTel span context and attaches its trace_id).
        trace_id = getattr(record, "trace_id", None)
        if trace_id is not None:
            payload["trace_id"] = trace_id

        # Free-form structured payload. Kept as a nested key so it never
        # collides with the reserved top-level fields above.
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            for key, value in extra_fields.items():
                if key not in payload:
                    payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


# Idempotency guard — repeated ``setup_logging`` calls (tests, embedded use)
# should not stack handlers on the root logger. Any prior knowledge-rag
# handler is removed before installing the new one.
_LOG_HANDLER_MARKER = "_knowledge_rag_telemetry_handler"


def setup_logging(json_format: bool = False, level: str = "INFO") -> None:
    """Configure the root logger for the knowledge-rag process.

    Attaches a single ``StreamHandler`` pointed at ``sys.stderr`` — never at
    ``sys.stdout``, which is reserved for MCP JSON-RPC. Safe to call multiple
    times: any prior handler registered by this function is removed before
    the new one lands, so re-configuration inside tests does not multiply
    output.

    Args:
        json_format: When ``True`` the handler uses :class:`JsonFormatter`,
            otherwise a plain ``%(message)s`` format that matches the pre-A2.6
            ``print()`` output shape byte-for-byte for the common case.
        level: Root log level as a string (``DEBUG`` / ``INFO`` / ``WARNING``
            / ``ERROR`` / ``CRITICAL``). Invalid values silently fall back to
            ``INFO`` so a typo in ``config.yaml`` cannot silence the server.

    Returns:
        None: The root logger is mutated in place.
    """
    root = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(resolved_level, int):
        resolved_level = logging.INFO
    root.setLevel(resolved_level)

    # Drop any handler we previously installed. Third-party handlers stay
    # in place so uvicorn / chromadb loggers keep flushing their output.
    for existing in list(root.handlers):
        if getattr(existing, _LOG_HANDLER_MARKER, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stderr)
    setattr(handler, _LOG_HANDLER_MARKER, True)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        # Match the pre-migration ``print()`` output shape.
        handler.setFormatter(logging.Formatter("%(message)s"))

    root.addHandler(handler)


# ╭─╴ OPENTELEMETRY TRACING ╶─────────────────────────────────────╮
# │         No-op tracer by default — SDK is an optional dep      │
# ╰───────────────────────────────────────────────────────────────╯


class _NoopSpan:
    """Zero-cost span used when tracing is disabled.

    Implements the tiny subset of the OTel span API the codebase actually
    calls (``set_attribute`` plus context-manager protocol). Anything else is
    silently ignored via ``__getattr__``.
    """

    __slots__ = ()

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401 — mirror OTel signature
        """Ignore attribute writes — no span is being recorded."""
        return None

    def set_status(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Ignore status writes — no span is being recorded."""
        return None

    def record_exception(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Ignore exception recording — no span is being recorded."""
        return None

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Absorb any OTel span method we do not model explicitly."""

        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


class _NoopTracer:
    """Zero-cost tracer returned when OTel is disabled or missing.

    Only ``start_as_current_span`` is exercised by the rest of the codebase.
    Extra methods are absorbed by ``__getattr__`` so future callers do not
    crash if they reach for something the real SDK exposes.
    """

    def start_as_current_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> _NoopSpan:  # noqa: D401 — mirror OTel signature
        """Return a no-op span — the name/attributes are discarded."""
        return _NoopSpan()

    def __getattr__(self, name: str) -> Callable[..., _NoopSpan]:
        """Any unknown method returns a no-op factory yielding a no-op span."""

        def _factory(*args: Any, **kwargs: Any) -> _NoopSpan:
            return _NoopSpan()

        return _factory


_tracer: Optional[Any] = None
_TRACING_LOG = logging.getLogger(__name__)


def setup_tracing(
    enabled: bool = False,
    exporter: str = "otlp",
    service_name: str = "knowledge-rag",
    endpoint: Optional[str] = None,
) -> None:
    """Install the global tracer.

    Real OTel SDK wiring only kicks in when ``enabled`` is truthy AND the
    ``opentelemetry`` packages are importable. Every other path installs the
    no-op tracer so :func:`get_tracer` always returns something usable.

    Args:
        enabled: Master switch — ``False`` short-circuits into the no-op path
            even when the SDK is installed. This is the default so a stray
            ``import mcp_server`` never opens a network exporter.
        exporter: One of ``otlp`` (gRPC OTLP exporter, default), ``console``
            (batch-print spans to stderr — useful for local debugging), or
            ``none`` (register a tracer provider but ship no exporter).
        service_name: Value assigned to the ``service.name`` resource
            attribute — this is the field APMs group by in the UI.
        endpoint: Override for the OTLP collector endpoint. When ``None``
            the exporter uses its own default (``http://localhost:4317``)
            and honours the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.

    Returns:
        None: The module-level tracer singleton is mutated in place.
    """
    global _tracer

    if not enabled:
        _tracer = _NoopTracer()
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        _TRACING_LOG.warning(
            "telemetry.enabled=true but the opentelemetry SDK is not installed. "
            "Install the extra with: pip install knowledge-rag[otel]"
        )
        _tracer = _NoopTracer()
        return

    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        normalized_exporter = (exporter or "otlp").strip().lower()
        if normalized_exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError:
                _TRACING_LOG.warning(
                    "OTLP exporter requested but opentelemetry-exporter-otlp is not installed. "
                    "Falling back to the console exporter — install knowledge-rag[otel] for OTLP."
                )
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            else:
                span_exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(span_exporter))
        elif normalized_exporter == "console":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif normalized_exporter == "none":
            # Provider without an exporter — spans are still emitted (useful
            # for downstream sidecars that hook into the OTel API) but the
            # process ships nothing on its own.
            pass
        else:
            _TRACING_LOG.warning(
                "Unknown telemetry.exporter=%r — falling back to the console exporter.",
                exporter,
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
    except Exception as exc:
        # OTel setup should never crash the server. Log and fall back.
        _TRACING_LOG.warning("Failed to initialize OpenTelemetry: %s — tracing disabled.", exc)
        _tracer = _NoopTracer()


def get_tracer() -> Any:
    """Return the process-wide tracer.

    Falls back to :class:`_NoopTracer` when :func:`setup_tracing` has not been
    called yet, so unit tests that touch instrumented code without calling
    ``setup_tracing`` still work.

    Returns:
        Any: A tracer with ``start_as_current_span(name, attributes=...)``.
    """
    global _tracer
    if _tracer is None:
        _tracer = _NoopTracer()
    return _tracer


def reset_tracer_for_tests() -> None:
    """Reset the tracer singleton back to no-op.

    Only intended for test teardown — production code should not touch this.
    """
    global _tracer
    _tracer = _NoopTracer()


# ╭─╴ @trace DECORATOR ╶──────────────────────────────────────────╮
# │       Convenience wrapper — resolves tracer at call time      │
# ╰───────────────────────────────────────────────────────────────╯


def trace(name: str, attributes: Optional[Dict[str, Any]] = None) -> Callable[..., Any]:
    """Wrap a function in an OpenTelemetry span.

    The tracer is resolved at *call* time via :func:`get_tracer`, so tracing
    can be enabled after the decorator has already been applied at import
    time. When tracing is disabled the wrapper collapses to a plain function
    call because ``_NoopSpan.__enter__`` / ``__exit__`` do no work.

    Args:
        name: Span name — usually ``"<subsystem>.<operation>"``, e.g.
            ``"knowledge_rag.query.semantic"``.
        attributes: Optional static attributes attached at span start.

    Returns:
        Callable[..., Any]: Decorator preserving ``functools.wraps`` metadata.

    Example:
        >>> @trace("knowledge_rag.query", {"component": "orchestrator"})
        ... def query(self, text: str) -> list:
        ...     return self._run(text)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_tracer().start_as_current_span(name, attributes=attributes):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
