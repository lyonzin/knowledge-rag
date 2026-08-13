"""Health endpoint ASGI middleware — regression tests.

Covers the payload shape, delegation to the downstream app, best-effort
cache stats, and interaction with the bearer auth middleware (probes must
not require authentication).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, MutableMapping

import pytest

from mcp_server import __version__
from mcp_server.health import DEFAULT_HEALTH_PATHS, HealthMiddleware
from mcp_server.security import BearerAuthMiddleware


class _Recorder:
    """Collect ASGI send events into a list for assertions."""

    def __init__(self) -> None:
        self.events: list[MutableMapping[str, Any]] = []

    async def __call__(self, event: MutableMapping[str, Any]) -> None:
        self.events.append(event)


async def _noop_receive() -> MutableMapping[str, Any]:
    return {"type": "http.disconnect"}


def _http_scope(path: str) -> dict[str, Any]:
    return {"type": "http", "path": path, "method": "GET", "headers": []}


def _run(coro: Any) -> None:
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _run_middleware(middleware: HealthMiddleware, scope: dict[str, Any]) -> _Recorder:
    send = _Recorder()
    asyncio.new_event_loop().run_until_complete(middleware(scope, _noop_receive, send))
    return send


def _body_of(recorder: _Recorder) -> dict[str, Any]:
    starts = [e for e in recorder.events if e["type"] == "http.response.start"]
    bodies = [e for e in recorder.events if e["type"] == "http.response.body"]
    assert starts, "no response.start event"
    assert bodies, "no response.body event"
    return json.loads(bodies[0]["body"].decode("utf-8"))


class TestHealthPayload:
    def test_health_returns_ok_with_version_and_uptime(self):
        """/health returns status ok + version + uptime + null cache."""
        downstream_calls: list[str] = []

        async def _downstream(scope, receive, send):
            downstream_calls.append(scope["path"])

        middleware = HealthMiddleware(_downstream, get_orchestrator=None)
        recorder = _run_middleware(middleware, _http_scope("/health"))

        payload = _body_of(recorder)
        assert payload["status"] == "ok"
        assert payload["version"] == __version__
        assert isinstance(payload["uptime_seconds"], int)
        assert payload["uptime_seconds"] >= 0
        assert payload["cache"] is None
        assert not downstream_calls, "downstream must not be invoked on /health"

    def test_healthz_alias_also_served(self):
        """/healthz responds identically to /health."""

        async def _downstream(scope, receive, send):
            pytest.fail("downstream must not be invoked")

        middleware = HealthMiddleware(_downstream, get_orchestrator=None)
        recorder = _run_middleware(middleware, _http_scope("/healthz"))
        payload = _body_of(recorder)
        assert payload["status"] == "ok"

    def test_response_headers_are_json_no_store(self):
        """Content-type application/json + cache-control no-store."""

        async def _downstream(scope, receive, send):
            pytest.fail("downstream must not be invoked")

        middleware = HealthMiddleware(_downstream, get_orchestrator=None)
        recorder = _run_middleware(middleware, _http_scope("/health"))
        starts = [e for e in recorder.events if e["type"] == "http.response.start"]
        headers = dict(starts[0]["headers"])
        assert starts[0]["status"] == 200
        assert headers[b"content-type"] == b"application/json"
        assert headers[b"cache-control"] == b"no-store"


class TestDelegation:
    def test_non_health_path_delegates_to_downstream(self):
        """Any path other than /health*/healthz reaches the wrapped app."""
        downstream_calls: list[str] = []

        async def _downstream(scope, receive, send):
            downstream_calls.append(scope["path"])

        middleware = HealthMiddleware(_downstream, get_orchestrator=None)
        recorder = _run_middleware(middleware, _http_scope("/mcp/search"))

        assert downstream_calls == ["/mcp/search"]
        assert not recorder.events, "middleware must not respond directly"

    def test_lifespan_scope_delegates(self):
        """lifespan scopes bypass the middleware entirely."""
        downstream_calls: list[str] = []

        async def _downstream(scope, receive, send):
            downstream_calls.append(scope["type"])

        middleware = HealthMiddleware(_downstream, get_orchestrator=None)
        scope = {"type": "lifespan"}
        asyncio.new_event_loop().run_until_complete(middleware(scope, _noop_receive, _Recorder()))
        assert downstream_calls == ["lifespan"]

    def test_websocket_scope_delegates(self):
        """websocket scopes bypass the middleware entirely."""
        downstream_calls: list[str] = []

        async def _downstream(scope, receive, send):
            downstream_calls.append(scope["type"])

        middleware = HealthMiddleware(_downstream, get_orchestrator=None)
        scope = {"type": "websocket", "path": "/health"}
        asyncio.new_event_loop().run_until_complete(middleware(scope, _noop_receive, _Recorder()))
        assert downstream_calls == ["websocket"]


class TestCacheStats:
    def test_orchestrator_cache_stats_populate_payload(self):
        """When orchestrator exposes a query_cache with stats(), include them."""

        class _Cache:
            def stats(self):
                return {"size": 42, "hits": 100, "misses": 5}

        class _Orch:
            query_cache = _Cache()

        async def _downstream(scope, receive, send):
            pytest.fail("downstream must not be invoked")

        middleware = HealthMiddleware(_downstream, get_orchestrator=lambda: _Orch())
        recorder = _run_middleware(middleware, _http_scope("/health"))
        payload = _body_of(recorder)
        assert payload["cache"] == {"size": 42, "hits": 100, "misses": 5}

    def test_orchestrator_getter_raising_yields_null_cache(self):
        """A raising orchestrator getter must not turn the probe negative."""

        def _boom():
            raise RuntimeError("cold start")

        async def _downstream(scope, receive, send):
            pytest.fail("downstream must not be invoked")

        middleware = HealthMiddleware(_downstream, get_orchestrator=_boom)
        recorder = _run_middleware(middleware, _http_scope("/health"))
        payload = _body_of(recorder)
        assert payload["status"] == "ok"
        assert payload["cache"] is None

    def test_missing_cache_yields_null(self):
        """Orchestrator without query_cache still produces status=ok."""

        class _Orch:
            pass  # no query_cache attribute

        async def _downstream(scope, receive, send):
            pytest.fail("downstream must not be invoked")

        middleware = HealthMiddleware(_downstream, get_orchestrator=lambda: _Orch())
        recorder = _run_middleware(middleware, _http_scope("/health"))
        payload = _body_of(recorder)
        assert payload["cache"] is None


class TestBearerAuthInteraction:
    def test_health_bypasses_bearer_auth(self):
        """Order: HealthMiddleware wraps BearerAuthMiddleware — probe never hits auth."""
        auth_calls: list[str] = []

        async def _downstream(scope, receive, send):
            auth_calls.append(scope["path"])

        guarded = BearerAuthMiddleware(_downstream, "s3cret")
        middleware = HealthMiddleware(guarded, get_orchestrator=None)
        recorder = _run_middleware(middleware, _http_scope("/health"))

        payload = _body_of(recorder)
        assert payload["status"] == "ok"
        assert not auth_calls, "bearer auth must never see the probe"

    def test_non_health_path_still_requires_auth(self):
        """When health middleware wraps bearer, non-health paths still require auth."""

        async def _downstream(scope, receive, send):
            pytest.fail("unauthenticated request reached the app")

        guarded = BearerAuthMiddleware(_downstream, "s3cret")
        middleware = HealthMiddleware(guarded, get_orchestrator=None)
        recorder = _run_middleware(middleware, _http_scope("/mcp/tools"))

        starts = [e for e in recorder.events if e["type"] == "http.response.start"]
        assert starts and starts[0]["status"] == 401


class TestDefaults:
    def test_default_paths_expose_both_health_and_healthz(self):
        assert DEFAULT_HEALTH_PATHS == ("/health", "/healthz")

    def test_custom_paths_override_defaults(self):
        """Callers can supply custom paths — /status, /ping, etc."""

        async def _downstream(scope, receive, send):
            pytest.fail("downstream must not be invoked")

        middleware = HealthMiddleware(_downstream, get_orchestrator=None, paths=("/status",))
        recorder = _run_middleware(middleware, _http_scope("/status"))
        payload = _body_of(recorder)
        assert payload["status"] == "ok"
