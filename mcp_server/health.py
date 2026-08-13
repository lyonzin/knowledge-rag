"""Health check endpoint for HTTP-mode MCP servers.

Exposes ``/health`` and ``/healthz`` returning a JSON payload with server
status, version, uptime and cache statistics — allowing liveness/readiness
probes (Kubernetes, load balancers, monitoring) without authentication.

Wired in ``server._run_transport`` in front of the bearer auth middleware so
probes always succeed regardless of the auth configuration.
"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable, MutableMapping, Optional, Sequence

from . import __version__

__all__ = ["HealthMiddleware", "DEFAULT_HEALTH_PATHS"]


DEFAULT_HEALTH_PATHS: tuple[str, ...] = ("/health", "/healthz")


class HealthMiddleware:
    """ASGI middleware that answers health probes before the MCP dispatcher.

    Wraps any downstream ASGI application. Only ``http`` scopes matching one
    of the configured paths are intercepted — everything else (including
    ``lifespan`` and ``websocket``) passes straight through.

    The response payload is intentionally small and cheap to compute so the
    endpoint stays useful under load. Cache statistics are best-effort — if
    the orchestrator getter raises, the payload still ships with
    ``cache: null`` and status ``ok`` so the probe never flaps on a
    non-critical failure.

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
            paths: Paths that trigger the health response.
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
        """Serve the health payload or delegate to the downstream app.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope.get("type") != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        payload = self._build_payload()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _build_payload(self) -> dict[str, Any]:
        """Assemble the health payload with best-effort cache stats."""
        uptime_seconds = int(time.monotonic() - self.started_at)
        payload: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": uptime_seconds,
            "cache": None,
        }
        if self.get_orchestrator is None:
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
