"""Optional Prometheus-compatible metrics for knowledge-rag server."""

import sys
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable

from .config import config


class MetricsCollector:
    """Lightweight Prometheus-compatible metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = defaultdict(float)

    def inc(self, name: str, labels: str = "", value: float = 1.0) -> None:
        with self._lock:
            self._counters[f"{name}{labels}"] += value

    def set_gauge(self, name: str, value: float, labels: str = "") -> None:
        with self._lock:
            self._gauges[f"{name}{labels}"] = value

    def observe(self, name: str, value: float, labels: str = "") -> None:
        with self._lock:
            self._histograms[f"{name}{labels}"].append(value)

    def exposition(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"{key} {val}")
            for key, val in sorted(self._gauges.items()):
                lines.append(f"{key} {val}")
            for key, observations in sorted(self._histograms.items()):
                if observations:
                    lines.append(f"{key}_count {len(observations)}")
                    lines.append(f"{key}_sum {sum(observations):.6f}")
        return "\n".join(lines) + "\n"


_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics


def instrument(tool_name: str) -> Callable[..., Callable[..., Any]]:
    """Decorator to instrument a tool function with call count and latency."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not config.metrics_enabled:
                return fn(*args, **kwargs)
            _metrics.inc("knowledge_rag_tool_calls_total", f'{{tool="{tool_name}"}}')
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                _metrics.inc("knowledge_rag_tool_errors_total", f'{{tool="{tool_name}"}}')
                raise
            finally:
                elapsed = time.monotonic() - start
                _metrics.observe("knowledge_rag_tool_duration_seconds", elapsed, f'{{tool="{tool_name}"}}')

        return wrapper

    return decorator


def start_metrics_server(port: int) -> None:
    """Start a lightweight HTTP server for Prometheus metrics scraping."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/metrics":
                body = get_metrics().exposition().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[METRICS] Prometheus endpoint at http://0.0.0.0:{port}/metrics", file=sys.stderr)
