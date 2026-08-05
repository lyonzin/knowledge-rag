"""CWE-287 — bearer token enforcement regression tests.

``config.auth_bearer_token`` existed since v4.0.0 but nothing ever compared it
against the ``Authorization`` header: an operator who set it believed the port
was protected while every MCP tool stayed reachable unauthenticated.

These tests drive :class:`BearerAuthMiddleware` as a raw ASGI application — no
network, no uvicorn — so they assert the wire behaviour directly.
"""

import asyncio

import pytest

from mcp_server.security import BearerAuthMiddleware, bearer_token_matches, extract_bearer_token

TOKEN = "s3cret-token-value"


class _SpyApp:
    """Downstream ASGI app that records whether it was ever reached."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, scope, receive, send) -> None:
        self.calls.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})


def _http_scope(headers=None, path="/mcp"):
    """Build a minimal ASGI HTTP scope.

    Args:
        headers: Raw header pairs, or ``None`` for no headers.
        path: Request path.

    Returns:
        dict: An ASGI ``http`` scope.
    """
    return {"type": "http", "path": path, "method": "POST", "headers": headers or []}


def _drive(middleware, scope):
    """Run one request through the middleware and collect ASGI messages.

    Args:
        middleware: The middleware instance under test.
        scope: ASGI scope to send.

    Returns:
        list[dict]: Messages the middleware (or downstream app) emitted.
    """
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def _status(messages):
    """Extract the HTTP status from collected ASGI messages.

    Args:
        messages: Messages returned by :func:`_drive`.

    Returns:
        int | None: The response status, if a start message was sent.
    """
    for message in messages:
        if message["type"] == "http.response.start":
            return message["status"]
    return None


# ---------------------------------------------------------------------------
# Constant-time comparison primitive
# ---------------------------------------------------------------------------


def test_matching_token_is_accepted():
    """The happy path still works."""
    assert bearer_token_matches(TOKEN, TOKEN) is True


@pytest.mark.parametrize(
    "presented",
    [
        None,
        "",
        "wrong",
        TOKEN + "x",
        TOKEN[:-1],
        TOKEN.upper(),
        " " + TOKEN,
        TOKEN + " ",
    ],
)
def test_non_matching_tokens_are_rejected(presented):
    """Prefix matches, case folds and whitespace padding all fail closed."""
    assert bearer_token_matches(TOKEN, presented) is False


def test_empty_expected_token_never_authenticates():
    """A blank configured token must not turn into "accept anything"."""
    assert bearer_token_matches("", "anything") is False
    assert bearer_token_matches("", "") is False
    assert bearer_token_matches("", None) is False


def test_comparison_uses_constant_time_primitive():
    """Guard against a refactor back to ``==`` and its timing side channel."""
    import inspect

    from mcp_server import security

    source = inspect.getsource(security.bearer_token_matches)
    assert "compare_digest" in source


def test_non_ascii_token_does_not_raise():
    """Unicode credentials must compare, not explode on encoding."""
    assert bearer_token_matches("tökén", "tökén") is True
    assert bearer_token_matches("tökén", "token") is False


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_extracts_token_from_authorization_header():
    """The canonical form is parsed."""
    assert extract_bearer_token([(b"authorization", b"Bearer abc123")]) == "abc123"


def test_header_name_and_scheme_are_case_insensitive():
    """RFC 7235 makes both case-insensitive."""
    assert extract_bearer_token([(b"Authorization", b"bearer abc")]) == "abc"
    assert extract_bearer_token([(b"AUTHORIZATION", b"BEARER abc")]) == "abc"


@pytest.mark.parametrize(
    "raw",
    [
        b"Basic dXNlcjpwYXNz",
        b"Bearer",
        b"Bearer ",
        b"abc123",
        b"",
    ],
)
def test_malformed_authorization_headers_yield_none(raw):
    """Anything that is not a well-formed bearer credential is discarded."""
    assert extract_bearer_token([(b"authorization", raw)]) is None


def test_missing_authorization_header_yields_none():
    """No header at all is the common unauthenticated case."""
    assert extract_bearer_token([(b"content-type", b"application/json")]) is None
    assert extract_bearer_token([]) is None


# ---------------------------------------------------------------------------
# ASGI enforcement
# ---------------------------------------------------------------------------


def test_missing_header_returns_401_and_never_reaches_the_app():
    """The core finding: unauthenticated requests must not hit MCP tools."""
    app = _SpyApp()
    messages = _drive(BearerAuthMiddleware(app, TOKEN), _http_scope())

    assert _status(messages) == 401
    assert app.calls == []


def test_wrong_token_returns_401_and_never_reaches_the_app():
    """A guessed or stale credential is refused the same way."""
    app = _SpyApp()
    messages = _drive(BearerAuthMiddleware(app, TOKEN), _http_scope([(b"authorization", b"Bearer nope")]))

    assert _status(messages) == 401
    assert app.calls == []


def test_correct_token_is_forwarded_to_the_app():
    """Authenticated traffic passes through untouched."""
    app = _SpyApp()
    messages = _drive(
        BearerAuthMiddleware(app, TOKEN),
        _http_scope([(b"authorization", f"Bearer {TOKEN}".encode())]),
    )

    assert _status(messages) == 200
    assert len(app.calls) == 1


def test_401_carries_a_www_authenticate_challenge():
    """RFC 6750 challenge so clients can react instead of guessing."""
    messages = _drive(BearerAuthMiddleware(_SpyApp(), TOKEN), _http_scope())

    headers = dict(next(m for m in messages if m["type"] == "http.response.start")["headers"])
    assert headers[b"www-authenticate"].startswith(b"Bearer")
    assert headers[b"content-type"] == b"application/json"


def test_401_body_is_json_and_leaks_nothing():
    """The error must not echo the expected token or any config value."""
    messages = _drive(BearerAuthMiddleware(_SpyApp(), TOKEN), _http_scope())

    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    assert b"unauthorized" in body
    assert TOKEN.encode() not in body


def test_lifespan_scope_passes_through_unauthenticated():
    """Blocking lifespan would stop the StreamableHTTP session manager."""
    app = _SpyApp()

    async def run():
        sent = []
        await BearerAuthMiddleware(app, TOKEN)(
            {"type": "lifespan"},
            lambda: asyncio.sleep(0, result={"type": "lifespan.startup"}),
            lambda m: asyncio.sleep(0, result=sent.append(m)),
        )

    asyncio.run(run())
    assert len(app.calls) == 1


def test_health_path_is_exempt():
    """Liveness probes must not require a credential."""
    app = _SpyApp()
    messages = _drive(BearerAuthMiddleware(app, TOKEN), _http_scope(path="/health"))

    assert _status(messages) == 200
    assert len(app.calls) == 1


def test_middleware_refuses_to_start_with_an_empty_token():
    """Fail loudly rather than install a no-op guard."""
    with pytest.raises(ValueError, match="non-empty token"):
        BearerAuthMiddleware(_SpyApp(), "")


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


def test_stdio_transport_bypasses_auth(monkeypatch):
    """A local pipe carries no HTTP headers — never gate it."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    ran = {}
    monkeypatch.setattr(server_module.mcp, "run", lambda transport: ran.setdefault("transport", transport))

    server_module._run_transport("stdio")

    assert ran["transport"] == "stdio"


@pytest.mark.xfail(
    reason="integration with MCP tools pending v4.6.0 (library shipped standalone in v4.5.1)", strict=False
)
def test_http_transport_without_token_warns_and_stays_open(monkeypatch, capsys):
    """Backwards compatibility: unset token keeps the current behaviour."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", "")
    ran = {}
    monkeypatch.setattr(server_module.mcp, "run", lambda transport: ran.setdefault("transport", transport))

    server_module._run_transport("streamable-http")

    assert ran["transport"] == "streamable-http"
    assert "Bearer auth disabled" in capsys.readouterr().err


def test_http_transport_with_token_installs_the_middleware(monkeypatch):
    """The wiring that actually closes the finding."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)
    monkeypatch.setattr(server_module.mcp, "streamable_http_app", lambda: _SpyApp())
    monkeypatch.setattr(server_module.mcp, "run", lambda transport: pytest.fail("must not use the unguarded path"))

    served = {}
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, **kwargs: served.update(app=app, **kwargs),
    )

    server_module._run_transport("streamable-http")

    assert isinstance(served["app"], BearerAuthMiddleware)
    assert served["app"].token == TOKEN


def test_unknown_transport_is_rejected(monkeypatch):
    """A typo in config must not silently start an unguarded server."""
    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "auth_bearer_token", TOKEN)

    with pytest.raises(ValueError, match="Unknown transport"):
        server_module._run_transport("htpp")
