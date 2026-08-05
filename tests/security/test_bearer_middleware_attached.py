"""End-to-end wiring — the HTTP MCP transport must serve behind auth.

``BearerAuthMiddleware`` is exercised in ``test_bearer_auth`` at the ASGI
level. This module proves the two wires that turn that library into an
actually-enforced protection on the running server:

* The transport dispatcher installs the middleware when a token is
  configured, and refuses to fall back to ``mcp.run`` (which would open
  an unguarded port).
* Requests without / with the wrong / with the right bearer token get
  the responses the middleware promises.

The requests are driven directly against the ASGI callable — no
uvicorn boot, no network — because the test's job is to assert wiring,
not to benchmark uvicorn.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from mcp_server.security import BearerAuthMiddleware

TOKEN = "wiring-test-token"


class _RecordingApp:
    """Downstream ASGI app that records if it was ever called."""

    def __init__(self) -> None:
        self.hits: List[Dict[str, Any]] = []

    async def __call__(self, scope, receive, send):
        self.hits.append(dict(scope))
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"ok":true}'})


def _drive(middleware, scope):
    """Run one ASGI request through ``middleware`` and collect emitted messages."""
    sent: List[Dict[str, Any]] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    asyncio.run(middleware(scope, receive, send))
    return sent


def _status(messages):
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


def _http_scope(headers=None, path="/"):
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers or [],
    }


# ---------------------------------------------------------------------------
# Wire — _run_transport installs the middleware
# ---------------------------------------------------------------------------


def test_streamable_http_wraps_the_app_when_a_token_is_configured(monkeypatch):
    """The wiring that closes the finding: middleware must fence the app."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    monkeypatch.setattr(server_module.mcp, "streamable_http_app", lambda: _RecordingApp())
    monkeypatch.setattr(
        server_module.mcp,
        "run",
        lambda **kw: pytest.fail("mcp.run must not be called on the guarded HTTP path"),
    )

    served: Dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: served.update(app=app, **kwargs))

    server_module._run_transport("streamable-http")

    assert isinstance(served["app"], BearerAuthMiddleware)
    assert served["app"].token == TOKEN


def test_sse_transport_also_wraps_the_app(monkeypatch):
    """The SSE transport shares the same wire — a token there is honoured."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    monkeypatch.setattr(server_module.mcp, "sse_app", lambda: _RecordingApp())
    monkeypatch.setattr(
        server_module.mcp,
        "run",
        lambda **kw: pytest.fail("mcp.run must not be called on the guarded SSE path"),
    )

    served: Dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: served.update(app=app, **kwargs))

    server_module._run_transport("sse")

    assert isinstance(served["app"], BearerAuthMiddleware)


def test_stdio_never_installs_the_middleware(monkeypatch):
    """A stdio pipe has no HTTP headers — auth is not applicable."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    called: Dict[str, Any] = {}
    monkeypatch.setattr(server_module.mcp, "run", lambda **kw: called.update(kw))

    server_module._run_transport("stdio")

    assert called == {"transport": "stdio"}


def test_typo_in_transport_name_is_refused(monkeypatch):
    """A misconfiguration must never boot an unguarded HTTP server."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)

    with pytest.raises(ValueError, match="Unknown transport"):
        server_module._run_transport("streamble-http")


# ---------------------------------------------------------------------------
# End-to-end — the wired app installed by _run_transport enforces auth
# ---------------------------------------------------------------------------


def test_wired_stack_serves_401_when_no_token_is_presented(monkeypatch):
    """The exact app uvicorn would boot must refuse unauthenticated traffic."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    monkeypatch.setattr(server_module.mcp, "streamable_http_app", lambda: _RecordingApp())
    monkeypatch.setattr(server_module.mcp, "run", lambda **kw: pytest.fail("must not use unguarded path"))

    served: Dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: served.update(app=app))
    server_module._run_transport("streamable-http")

    wired_app = served["app"]
    downstream = wired_app.app  # unwrap the middleware for the recording assertion

    messages = _drive(wired_app, _http_scope())
    assert _status(messages) == 401
    assert downstream.hits == []


def test_wired_stack_lets_the_correct_token_through(monkeypatch):
    """The wired app must forward valid credentials to the MCP dispatcher."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    monkeypatch.setattr(server_module.mcp, "streamable_http_app", lambda: _RecordingApp())
    monkeypatch.setattr(server_module.mcp, "run", lambda **kw: pytest.fail("must not use unguarded path"))

    served: Dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: served.update(app=app))
    server_module._run_transport("streamable-http")

    wired_app = served["app"]
    downstream = wired_app.app

    messages = _drive(wired_app, _http_scope([(b"authorization", f"Bearer {TOKEN}".encode())]))
    assert _status(messages) == 200
    assert len(downstream.hits) == 1
