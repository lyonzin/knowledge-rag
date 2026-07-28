"""
╭─╴ QUERY CACHE ╶────────────────────────────────────────────────╮
│                                                                │
│   LRU cache with TTL for search queries — avoids redundant     │
│   pipelines on repeat queries. Extracted from server.py in     │
│   the A2.1 refactor without behaviour changes.                 │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class QueryCache:
    """
    LRU cache with TTL for search queries.

    Avoids redundant searches when the same query is executed multiple times.
    Uses OrderedDict for O(1) LRU eviction.

    Args:
        max_size: Maximum number of cached entries (default: 100)
        ttl_seconds: Time-to-live for cache entries in seconds (default: 300)
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> str:
        """Generate cache key from query parameters"""
        raw = f"{query}|{max_results}|{category}|{hybrid_alpha}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        key = self._make_key(query, max_results, category, hybrid_alpha)

        with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return result
                else:
                    del self._cache[key]

            self._misses += 1
            return None

    def put(self, query: str, max_results: int, category: Optional[str], hybrid_alpha: float, result: Any) -> None:
        """Store result in cache"""
        key = self._make_key(query, max_results, category, hybrid_alpha)
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (time.time(), result)

    def invalidate(self) -> None:
        """Clear entire cache (call after reindex)"""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "0%",
        }
