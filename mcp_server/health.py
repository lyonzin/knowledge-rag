"""Health check endpoints for HTTP-mode MCP servers.

``/health`` and ``/healthz`` are liveness probes: they answer 200 while the
process runs, with the server version, uptime, the index state (``ready``,
``repairing`` or ``error``) and cache statistics. ``/readyz`` is the
readiness probe: 200 only when tools can be served, otherwise 503, with
``Retry-After`` while the index is still being checked, opened or repaired.
A liveness probe that restarted the server on ``repairing`` would cut a long
repair short, so only ``/readyz`` turns negative. None of them needs a token,
and none returns a path, an exception message, a token or a PID.

Wired in ``server._run_transport`` in front of the bearer auth middleware so
probes always succeed regardless of the auth configuration.
"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable, MutableMapping, Optional, Sequence

from . import __version__, _writer_role

__all__ = ["HealthMiddleware", "DEFAULT_HEALTH_PATHS", "READINESS_PATH"]


DEFAULT_HEALTH_PATHS: tuple[str, ...] = ("/health", "/healthz")
READINESS_PATH = "/readyz"
_NOT_READY_STATUS = 503
_Headers = list[tuple[bytes, bytes]]


class HealthMiddleware:
    """ASGI middleware that answers health probes before the MCP dispatcher.

    Wraps any downstream ASGI application. Only ``http`` scopes matching one
    of the configured paths or ``/readyz`` are intercepted — everything else
    (including ``lifespan`` and ``websocket``) passes straight through.

    The response payload is intentionally small and cheap to compute so the
    endpoint stays useful under load. The index state comes from this
    process's startup (``_writer_role.index_status``). Cache statistics are
    best-effort and read only from a ready index — if the orchestrator getter
    raises, the payload still ships with ``cache: null`` and status ``ok``
    so the probe never flaps on a non-critical failure.

    Attributes:
        app: The wrapped ASGI application.
        get_orchestrator: Callable returning the live orchestrator, or
            ``None`` when it is not yet initialised.
        paths: Tuple of URL paths served by this middleware.
        started_at: Monotonic timestamp captured at construction.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        get_orchestrator: Optional[Callable[[], Any]] = None,
        paths: Sequence[str] = DEFAULT_HEALTH_PATHS,
    ) -> None:
        """Initialise the middleware.

        Args:
            app: Downstream ASGI application.
            get_orchestrator: Optional callable returning the orchestrator
                instance. Used to include cache statistics in the payload.
                Failures are swallowed so a broken orchestrator does not
                turn health probes into false negatives.
            paths: Paths that trigger the liveness response.
        """
        self.app = app
        self.get_orchestrator = get_orchestrator
        self.paths = tuple(paths)
        self.started_at = time.monotonic()

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """Serve the liveness or readiness payload, or delegate to the downstream app.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        path = scope.get("path")
        if scope.get("type") != "http" or (path not in self.paths and path != READINESS_PATH):
            await self.app(scope, receive, send)
            return

        status, payload, headers = self._readiness() if path == READINESS_PATH else (200, self._build_payload(), [])
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                    *headers,
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _readiness() -> tuple[int, dict[str, Any], _Headers]:
        """200 when tools can be served; 503 otherwise, with Retry-After only while repairing. By Lyon."""
        state = _writer_role.index_status()
        if state == _writer_role.READY:
            return 200, {"status": state}, []
        retry = [(b"retry-after", str(_writer_role.RETRY_AFTER_SECONDS).encode("ascii"))]
        return _NOT_READY_STATUS, {"status": state}, retry if state == _writer_role.REPAIRING else []

    def _build_payload(self) -> dict[str, Any]:
        """Assemble the liveness payload with the index state and best-effort cache stats."""
        uptime_seconds = int(time.monotonic() - self.started_at)
        index = _writer_role.index_status()
        payload: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": uptime_seconds,
            "index": index,
            "cache": None,
        }
        if self.get_orchestrator is None or index != _writer_role.READY:
            return payload
        try:
            orchestrator = self.get_orchestrator()
        except Exception:
            return payload
        if orchestrator is None:
            return payload
        cache = getattr(orchestrator, "query_cache", None)
        stats_fn = getattr(cache, "stats", None) if cache is not None else None
        if callable(stats_fn):
            try:
                payload["cache"] = stats_fn()
            except Exception:
                payload["cache"] = None
        return payload
