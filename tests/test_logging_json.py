"""Structured JSON logging — regression tests.

Covers the opt-in gate (default ``text`` is a no-op), the JSON payload
shape (stable core + extras), handler idempotency, and interaction with
``get_logger`` naming.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from mcp_server.logging_config import (
    ROOT_LOGGER_NAME,
    JsonFormatter,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Each test starts with a pristine knowledge_rag logger."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    try:
        yield
    finally:
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate


def _capture_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace sys.stderr with an in-memory buffer for one test."""
    buf = io.StringIO()
    import sys

    monkeypatch.setattr(sys, "stderr", buf)
    return buf


class TestOptInGate:
    def test_text_format_installs_no_handler(self):
        """Default ``text`` mode keeps the logger silent — prints still rule."""
        logger = setup_logging(fmt="text")
        assert logger.name == ROOT_LOGGER_NAME
        assert logger.handlers == [], "text mode must not install a handler"

    def test_json_format_installs_stream_handler(self):
        """``json`` mode installs exactly one managed StreamHandler."""
        logger = setup_logging(fmt="json")
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert isinstance(handler.formatter, JsonFormatter)
        assert getattr(handler, "_kr_managed", False) is True

    def test_calling_setup_twice_is_idempotent(self):
        """Repeated calls replace managed handlers, never accumulate."""
        setup_logging(fmt="json")
        setup_logging(fmt="json")
        setup_logging(fmt="json")
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        managed = [h for h in logger.handlers if getattr(h, "_kr_managed", False)]
        assert len(managed) == 1

    def test_switching_json_to_text_removes_managed_handler(self):
        """text → json → text leaves zero managed handlers."""
        setup_logging(fmt="json")
        setup_logging(fmt="text")
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        managed = [h for h in logger.handlers if getattr(h, "_kr_managed", False)]
        assert managed == []

    def test_foreign_handlers_are_preserved(self):
        """Non-managed handlers (installed by embedding apps) survive."""
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        foreign = logging.NullHandler()
        logger.addHandler(foreign)
        setup_logging(fmt="json")
        assert foreign in logger.handlers


class TestJsonPayload:
    def _log_and_parse(self, monkeypatch, level: str, msg: str, **extra):
        buf = _capture_stderr(monkeypatch)
        setup_logging(fmt="json", level=level)
        log = get_logger("mcp_server.ingestion")
        log.info(msg, extra=extra)
        for h in logging.getLogger(ROOT_LOGGER_NAME).handlers:
            h.flush()
        line = buf.getvalue().strip()
        assert line, "handler must have emitted a line"
        return json.loads(line)

    def test_payload_has_stable_core(self, monkeypatch):
        payload = self._log_and_parse(monkeypatch, "INFO", "hello")
        for key in ("timestamp", "level", "logger", "message", "module", "line"):
            assert key in payload
        assert payload["level"] == "INFO"
        assert payload["message"] == "hello"
        assert payload["logger"] == f"{ROOT_LOGGER_NAME}.ingestion"

    def test_extras_flow_into_payload(self, monkeypatch):
        payload = self._log_and_parse(
            monkeypatch, "INFO", "indexed", doc_id="abc", chunks=42
        )
        assert payload["doc_id"] == "abc"
        assert payload["chunks"] == 42

    def test_extras_cannot_override_stable_core(self, monkeypatch):
        # Extras named after our formatter-produced core keys (``timestamp``,
        # ``logger``) must not overwrite the real values. Python's
        # ``logging.makeRecord`` already blocks LogRecord-reserved names
        # (``message``, ``asctime``, and any attribute already on the
        # record), so we only need to defend against formatter-owned keys.
        buf = _capture_stderr(monkeypatch)
        setup_logging(fmt="json")
        log = get_logger("mcp_server.ingestion")
        log.info("collide", extra={"timestamp": "fake", "logger": "spoofed"})
        for h in logging.getLogger(ROOT_LOGGER_NAME).handlers:
            h.flush()
        payload = json.loads(buf.getvalue().strip())
        # Real ISO timestamp, not the fake string
        assert payload["timestamp"].endswith("Z")
        assert payload["timestamp"] != "fake"
        # Real logger name, not the extras-supplied ``spoofed``
        assert payload["logger"] == f"{ROOT_LOGGER_NAME}.ingestion"

    def test_timestamp_iso8601_utc_ms(self, monkeypatch):
        payload = self._log_and_parse(monkeypatch, "INFO", "ts")
        # Format: YYYY-MM-DDTHH:MM:SS.mmmZ
        assert payload["timestamp"].endswith("Z")
        assert "T" in payload["timestamp"]
        assert "." in payload["timestamp"]
        ms_part = payload["timestamp"].split(".")[1].rstrip("Z")
        assert len(ms_part) == 3, f"expected 3-digit ms, got {ms_part!r}"

    def test_exception_serialised(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        setup_logging(fmt="json")
        log = get_logger("mcp_server.query")
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("query failed")
        for h in logging.getLogger(ROOT_LOGGER_NAME).handlers:
            h.flush()
        payload = json.loads(buf.getvalue().strip())
        assert payload["level"] == "ERROR"
        assert "exception" in payload
        assert "ValueError: boom" in payload["exception"]

    def test_non_json_serialisable_extras_coerced_to_str(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        setup_logging(fmt="json")
        log = get_logger("mcp_server.x")

        class NotJson:
            def __repr__(self):
                return "<NotJson>"

        log.info("weird", extra={"obj": NotJson()})
        for h in logging.getLogger(ROOT_LOGGER_NAME).handlers:
            h.flush()
        payload = json.loads(buf.getvalue().strip())
        assert payload["obj"] == "<NotJson>"

    def test_nested_dict_and_list_preserved(self, monkeypatch):
        payload_json = self._log_and_parse(
            monkeypatch,
            "INFO",
            "nested",
            meta={"a": 1, "b": [1, 2, 3]},
            tags=["retrieval", "hybrid"],
        )
        assert payload_json["meta"] == {"a": 1, "b": [1, 2, 3]}
        assert payload_json["tags"] == ["retrieval", "hybrid"]


class TestGetLogger:
    def test_child_logger_reuses_root(self):
        log = get_logger("mcp_server.ingestion")
        assert log.name == f"{ROOT_LOGGER_NAME}.ingestion"

    def test_top_level_name_falls_back_to_last_component(self):
        log = get_logger("some_package")
        assert log.name == f"{ROOT_LOGGER_NAME}.some_package"

    def test_name_already_under_root_returned_as_is(self):
        log = get_logger(f"{ROOT_LOGGER_NAME}.custom.deep")
        assert log.name == f"{ROOT_LOGGER_NAME}.custom.deep"

    def test_root_logger_name_returned_as_is(self):
        log = get_logger(ROOT_LOGGER_NAME)
        assert log.name == ROOT_LOGGER_NAME
