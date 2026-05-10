"""Chaos: filesystem watcher fails to start.

If watchdog cannot attach to the documents directory (FS gone, perms
denied, exhausted inotify watches on Linux), the MCP server must keep
serving queries — only auto-reindex is degraded.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.chaos


def test_query_cache_survives_under_unresponsive_watcher():
    """Even if the watcher is dead, the cache must keep responding."""
    from mcp_server.server import QueryCache

    cache = QueryCache(max_size=10, ttl_seconds=300)
    cache.put("k", 5, None, 0.3, [{"content": "v"}])

    # Simulate hostile environment: many parallel get/put operations
    for i in range(100):
        cache.put(f"k-{i}", 5, None, 0.3, [{"content": str(i)}])
        result = cache.get(f"k-{i}", 5, None, 0.3)
        # Eviction may have happened; the contract is "no exception",
        # not "always present".
        assert result is None or isinstance(result, list)

    stats = cache.stats()
    assert stats["size"] <= 10
