"""Optional per-client rate limiting for MCP tool calls."""

import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable

from .config import config


class SlidingWindowCounter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, window_seconds: int = 60, max_requests: int = 60, burst: int = 10):
        self.window = window_seconds
        self.max_requests = max_requests
        self.burst = burst
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, client_id: str = "default") -> bool:
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests[client_id]
            cutoff = now - self.window
            self._requests[client_id] = [t for t in timestamps if t > cutoff]
            timestamps = self._requests[client_id]
            if len(timestamps) < self.max_requests + self.burst:
                timestamps.append(now)
                return True
            return False


_limiter: SlidingWindowCounter | None = None


def _get_limiter() -> SlidingWindowCounter:
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindowCounter(
            window_seconds=60,
            max_requests=config.rate_limit_rpm,
            burst=config.rate_limit_burst,
        )
    return _limiter


def rate_limited(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that enforces rate limiting if enabled in config."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not config.rate_limit_enabled:
            return fn(*args, **kwargs)
        limiter = _get_limiter()
        if not limiter.allow("default"):
            import json

            return json.dumps(
                {
                    "status": "error",
                    "message": "Rate limit exceeded. Retry after a short delay.",
                    "retry_after_seconds": 60,
                }
            )
        return fn(*args, **kwargs)

    return wrapper
