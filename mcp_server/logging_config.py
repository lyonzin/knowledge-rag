"""Opt-in structured JSON logging for knowledge-rag.

Off by default. Enabled by ``server.logging.format: "json"`` in
``config.yaml``. When active, installs a :class:`logging.StreamHandler` on
the ``knowledge_rag`` logger with a :class:`JsonFormatter` that emits one
JSON object per record on stderr — ready for ELK, Loki, Datadog, CloudWatch
Logs or any structured pipeline.

Existing ``print(...)`` calls throughout the codebase are intentionally
left alone: they continue to write plain lines to stderr as before. New
call sites should use ``get_logger(__name__)`` so they benefit from JSON
output when the operator opts in. Migration of legacy prints is
incremental — do it in the PR that touches the surrounding code, not in
a big-bang rewrite.

Example — enable JSON logs in ``config.yaml``:

    server:
      logging:
        format: "json"
        level: "INFO"

Then a call like::

    log = get_logger(__name__)
    log.info("Indexed document", extra={"doc_id": "abc", "chunks": 42})

produces on stderr::

    {"timestamp":"2026-08-13T21:45:12.891Z","level":"INFO",
     "logger":"knowledge_rag.ingestion","message":"Indexed document",
     "module":"ingestion","line":214,"doc_id":"abc","chunks":42}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

__all__ = ["JsonFormatter", "setup_logging", "get_logger", "ROOT_LOGGER_NAME"]


ROOT_LOGGER_NAME = "knowledge_rag"

# Standard LogRecord attributes we must NOT re-emit as extra fields — anything
# not in this set that appears on the record was passed via ``extra={...}`` and
# is safe to include in the JSON payload.
_LOGRECORD_BUILTINS = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    ]
)


class JsonFormatter(logging.Formatter):
    """Format a :class:`logging.LogRecord` as a single-line JSON object.

    The payload includes a stable core (timestamp, level, logger, message,
    module, line) plus any ``extra={...}`` fields the caller supplied. When
    ``record.exc_info`` is set, the formatted traceback ships as an
    ``exception`` field.

    Timestamps use ISO 8601 with millisecond precision in UTC — matching
    the format consumed by ELK / Loki / Datadog ingestion out of the box.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to a compact JSON string."""
        payload: dict[str, Any] = {
            "timestamp": _format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        for key, value in record.__dict__.items():
            if key in _LOGRECORD_BUILTINS or key.startswith("_"):
                continue
            if key in payload:  # do not let extras override the stable core
                continue
            payload[key] = _coerce(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, separators=(",", ":"), default=str)


def _format_timestamp(created: float) -> str:
    """Return an ISO 8601 UTC timestamp with millisecond precision."""
    dt = datetime.fromtimestamp(created, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _coerce(value: Any) -> Any:
    """Coerce arbitrary values to something ``json.dumps`` can handle."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    return str(value)


def setup_logging(fmt: str = "text", level: str = "INFO") -> logging.Logger:
    """Configure the ``knowledge_rag`` logger tree.

    Idempotent — safe to call multiple times. Existing handlers we own
    (marked with ``_kr_managed``) are replaced; foreign handlers are left
    alone so applications embedding knowledge-rag as a library can install
    their own handlers without being clobbered.

    Args:
        fmt: ``"text"`` (default, no logger changes; prints continue as
            before) or ``"json"`` (install :class:`JsonFormatter` on the
            ``knowledge_rag`` logger).
        level: Standard logging level name.

    Returns:
        The configured root logger for the ``knowledge_rag`` tree.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    _remove_managed_handlers(logger)

    if fmt == "json":
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._kr_managed = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    else:
        # ``text`` mode: keep the logger silent so the existing print(...)
        # sites continue to be the only source of stderr output. No handler
        # installed; propagation stays enabled for libraries that opt-in.
        logger.setLevel(level)

    return logger


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Drop only handlers we previously installed. Preserve foreign ones."""
    logger.handlers = [h for h in logger.handlers if not getattr(h, "_kr_managed", False)]


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``knowledge_rag`` tree.

    Args:
        name: Usually ``__name__``. When called from
            ``mcp_server.ingestion`` returns ``knowledge_rag.ingestion``.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    suffix = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{suffix}")
