"""Optional Prometheus-compatible metrics for knowledge-rag server."""

import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

from .config import config

# Canonical metric names for the FTS5 lexical fast-path (v4.8.2+).
# Kept as module-level constants so callers reference a single source of
# truth instead of duplicating string literals across ``server.py``.
# See TechSpec §Monitoring and Observability for label semantics.
FAST_PATH_HITS_TOTAL = "knowledge_rag_fast_path_hits_total"
FAST_PATH_FALLBACK_TOTAL = "knowledge_rag_fast_path_fallback_total"
FAST_PATH_LATENCY_SECONDS = "knowledge_rag_fast_path_latency_seconds"
FAST_PATH_ERRORS_TOTAL = "knowledge_rag_fast_path_errors_total"
FAST_PATH_RERANK_SKIPPED_TOTAL = "knowledge_rag_fast_path_rerank_skipped_total"
FAST_PATH_MIGRATION_DOCS_INDEXED = "knowledge_rag_fast_path_migration_docs_indexed"
FAST_PATH_MIGRATION_DOCS_TOTAL = "knowledge_rag_fast_path_migration_docs_total"

# Histogram buckets pinned by TechSpec — cover fast-path targets (p95 <=10ms)
# plus a long tail so runaway queries are still visible in the +Inf bucket.
FAST_PATH_LATENCY_BUCKETS: tuple[float, ...] = (0.001, 0.005, 0.010, 0.050, 0.100, 0.500)


@dataclass(slots=True)
class _Histogram:
    """Keep sufficient statistics, never the individual observations."""

    count: int = 0
    total: float = 0.0
    bucket_counts: list[int] = field(default_factory=list)


class MetricsCollector:
    """Lightweight Prometheus-compatible metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, _Histogram] = {}
        self._gauges: dict[str, float] = defaultdict(float)
        self._histogram_buckets: dict[str, tuple[float, ...]] = {}

    def inc(self, name: str, labels: str = "", value: float = 1.0) -> None:
        with self._lock:
            self._counters[f"{name}{labels}"] += value

    def set_gauge(self, name: str, value: float, labels: str = "") -> None:
        with self._lock:
            self._gauges[f"{name}{labels}"] = value

    def observe(self, name: str, value: float, labels: str = "") -> None:
        with self._lock:
            key = f"{name}{labels}"
            buckets = self._histogram_buckets.get(name, ())
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = self._histograms[key] = _Histogram(bucket_counts=[0] * len(buckets))
            histogram.count += 1
            histogram.total += value
            for index, boundary in enumerate(buckets):
                if value <= boundary:
                    histogram.bucket_counts[index] += 1

    def register_histogram_buckets(self, name: str, buckets: tuple[float, ...]) -> None:
        """Enable Prometheus-style bucketed emission for ``name``.

        Idempotent — re-registering with the same buckets is a no-op.
        Register before observing: past bucket counts cannot be reconstructed
        from bounded aggregates. Unregistered histograms emit count and sum.
        """
        sorted_buckets = tuple(sorted(set(buckets)))
        with self._lock:
            if self._histogram_buckets.get(name) == sorted_buckets:
                return
            if any(_split_metric_key(key)[0] == name for key in self._histograms):
                raise ValueError(f"Register histogram buckets before observing {name}")
            self._histogram_buckets[name] = sorted_buckets

    def exposition(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"{key} {val}")
            for key, val in sorted(self._gauges.items()):
                lines.append(f"{key} {val}")
            for key, histogram in sorted(self._histograms.items()):
                name, base_labels = _split_metric_key(key)
                buckets = self._histogram_buckets.get(name)
                if buckets:
                    lines.extend(_format_histogram_buckets(name, base_labels, histogram, buckets))
                lines.append(f"{name}_count{base_labels} {histogram.count}")
                lines.append(f"{name}_sum{base_labels} {histogram.total:.6f}")
        return "\n".join(lines) + "\n"


def _split_metric_key(key: str) -> tuple[str, str]:
    """Return ``(name, label_suffix)`` from a stored histogram key.

    The stored key is ``<name><label_suffix>`` where label_suffix is either
    empty or the ``{k="v",...}`` block prepended in ``observe``.
    """
    if "{" not in key:
        return key, ""
    idx = key.index("{")
    return key[:idx], key[idx:]


def _format_histogram_buckets(
    name: str,
    base_labels: str,
    histogram: _Histogram,
    buckets: tuple[float, ...],
) -> list[str]:
    """Return Prometheus histogram bucket lines with cumulative counts."""
    lines: list[str] = []
    for boundary, count in zip(buckets, histogram.bucket_counts):
        bucket_labels = _merge_labels(base_labels, f'le="{boundary}"')
        lines.append(f"{name}_bucket{bucket_labels} {count}")
    inf_labels = _merge_labels(base_labels, 'le="+Inf"')
    lines.append(f"{name}_bucket{inf_labels} {histogram.count}")
    return lines


def _merge_labels(base_labels: str, extra: str) -> str:
    """Combine an existing ``{...}`` block with a new ``le="..."`` label."""
    if not base_labels:
        return "{" + extra + "}"
    # base_labels looks like '{k="v"}'; splice extra before the closing brace.
    return base_labels[:-1] + "," + extra + "}"


_metrics = MetricsCollector()
_metrics.register_histogram_buckets(FAST_PATH_LATENCY_SECONDS, FAST_PATH_LATENCY_BUCKETS)


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
