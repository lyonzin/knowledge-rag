"""Liveness (/health, /healthz) against readiness (/readyz) on the HTTP health middleware.

The probes report this process's index state (``_writer_role.index_status``):
liveness stays 200 so a supervisor never kills a long repair, readiness is 200
only when tools can be served. Both are anonymous, so neither returns a path,
an exception message, a token or a PID. The ASGI app is called in-process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mcp_server import _writer_role
from mcp_server.health import READINESS_PATH, HealthMiddleware

PRIVATE_FAILURE = "Could not open the index in C:/Users/someone/private/data: Errno 13 (pid 4242, token s3cret)"
PRIVATE_WORDS = ("private", "4242", "s3cret", "Errno")
LIVENESS_KEYS = ["cache", "index", "status", "uptime_seconds", "version"]


def _set_index(monkeypatch, status):
    """This process's index in ``status``: "ready", "repairing", or "error" with a private failure text."""
    state = _writer_role.RoleState(failure=PRIVATE_FAILURE if status == _writer_role.ERROR else None)
    if status == _writer_role.REPAIRING:
        state.ready.clear()
    monkeypatch.setattr(_writer_role, "STATE", state)


def _probe(path, get_orchestrator=None):
    """Status code, headers and JSON body of one GET on the middleware, as uvicorn would deliver it."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.disconnect"}

    async def downstream(scope, receive, send):
        pytest.fail("a probe must not reach the MCP app")

    middleware = HealthMiddleware(downstream, get_orchestrator=get_orchestrator)
    scope = {"type": "http", "path": path, "method": "GET", "headers": []}
    asyncio.new_event_loop().run_until_complete(middleware(scope, receive, send))
    start, body = sent
    return start["status"], dict(start["headers"]), json.loads(body["body"])


@pytest.mark.parametrize(
    "status, expected", [("ready", (200, None)), ("repairing", (503, b"5")), ("error", (503, None))]
)
def test_given_each_index_state_when_readyz_is_probed_then_only_a_ready_index_answers_200(
    monkeypatch, status, expected
):
    """Retry-After only while repairing: a startup error does not heal by waiting."""
    _set_index(monkeypatch, status)

    code, headers, payload = _probe(READINESS_PATH)

    assert (code, headers.get(b"retry-after"), payload) == (*expected, {"status": status})


@pytest.mark.parametrize("path", ["/health", "/healthz"])
@pytest.mark.parametrize("status", ["ready", "repairing", "error"])
def test_given_each_index_state_when_liveness_is_probed_then_it_stays_200_and_reports_the_state(
    monkeypatch, path, status
):
    _set_index(monkeypatch, status)

    code, _headers, payload = _probe(path)

    assert (code, payload["status"], payload["index"]) == (200, "ok", status)


def test_given_a_repairing_index_when_health_is_probed_then_it_never_asks_the_orchestrator(monkeypatch):
    _set_index(monkeypatch, _writer_role.REPAIRING)

    def must_not_open():
        pytest.fail("a probe must not reach the index while the startup runs")

    _code, _headers, payload = _probe("/health", get_orchestrator=must_not_open)

    assert payload["cache"] is None


def test_given_a_ready_index_whose_cache_statistics_fail_when_probed_then_both_probes_still_answer(monkeypatch):
    _set_index(monkeypatch, _writer_role.READY)

    def broken_orchestrator():
        raise RuntimeError("cache statistics unavailable")

    liveness = _probe("/health", get_orchestrator=broken_orchestrator)
    readiness = _probe(READINESS_PATH, get_orchestrator=broken_orchestrator)

    assert (liveness[0], liveness[2]["cache"], readiness[0]) == (200, None, 200)


@pytest.mark.parametrize("path", ["/health", "/healthz", READINESS_PATH])
def test_given_a_startup_failure_with_private_details_when_probed_anonymously_then_nothing_private_leaks(
    monkeypatch, path
):
    _set_index(monkeypatch, _writer_role.ERROR)

    _code, _headers, payload = _probe(path)

    keys = ["status"] if path == READINESS_PATH else LIVENESS_KEYS
    assert (sorted(payload), any(word in json.dumps(payload) for word in PRIVATE_WORDS)) == (keys, False)
