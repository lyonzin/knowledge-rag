"""GH #316: loud early exit in BM25 rebuild + observable status in stats.

Three regressions guard the fix:
  1. ``_rebuild_bm25_from_collection`` logs when an empty page aborts the
     build and never returns silently. Collection shrinking mid-build
     publishes the partial corpus rather than leaving the orchestrator
     with ``_bm25_initialized=False`` and nothing in logs.
  2. ``get_stats()`` exposes a ``bm25`` dict with status/chunk_count/
     last_built_at/stale/error.
  3. ``_bm25_last_built_at`` is populated after a successful publish.

Uses the shared ``isolated_orchestrator`` fixture (mock embeddings +
EphemeralClient), so these tests never touch the real FastEmbed model.
"""

from __future__ import annotations

import time
from unittest.mock import patch


def _seed_chunks(orch, ids_and_text):
    """Add synthetic chunks directly to the fixture collection and corpus."""
    ids = [cid for cid, _ in ids_and_text]
    docs = [doc for _, doc in ids_and_text]
    metadatas = [{"source": f"{cid}.md"} for cid in ids]
    orch.collection.add(ids=ids, documents=docs, metadatas=metadatas)


def test_rebuild_logs_when_empty_page_with_chunks_remaining(isolated_orchestrator, capsys):
    """count says N but a page returns no ids: must log and abandon loudly."""
    orch = isolated_orchestrator
    _seed_chunks(orch, [("c1", "alpha payload"), ("c2", "beta payload")])

    real_get = orch.collection.get
    call_counter = {"n": 0}

    def flaky_get(**kwargs):
        if kwargs.get("offset", 0) == 0:
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                return {"ids": [], "documents": []}
        return real_get(**kwargs)

    with patch.object(orch.collection, "get", side_effect=flaky_get):
        orch._rebuild_bm25_from_collection()

    err = capsys.readouterr().err
    assert "empty page at offset 0" in err, f"early exit must be logged, got: {err!r}"
    assert orch._bm25_initialized is False
    assert orch._bm25_last_error and "offset 0" in orch._bm25_last_error


def test_rebuild_publishes_partial_when_collection_shrank(isolated_orchestrator, capsys):
    """Watcher deleted docs mid-build: publish what we already read, logged."""
    orch = isolated_orchestrator
    _seed_chunks(orch, [("c1", "alpha payload")])

    count_calls = {"n": 0}

    def shrinking_count():
        count_calls["n"] += 1
        return 501 if count_calls["n"] == 1 else 0

    real_get = orch.collection.get

    def empty_after_offset_500(**kwargs):
        if kwargs.get("offset", 0) >= 500:
            return {"ids": [], "documents": []}
        return real_get(**kwargs)

    with (
        patch.object(orch.collection, "count", side_effect=shrinking_count),
        patch.object(orch.collection, "get", side_effect=empty_after_offset_500),
    ):
        orch._rebuild_bm25_from_collection()

    err = capsys.readouterr().err
    assert "collection shrank" in err, f"shrink must be logged, got: {err!r}"
    assert orch._bm25_initialized is True
    assert orch._bm25_last_built_at is not None


def test_get_stats_exposes_bm25_status_on_empty(isolated_orchestrator):
    orch = isolated_orchestrator
    stats = orch.get_stats()
    assert "bm25" in stats
    bm25 = stats["bm25"]
    for key in ("status", "chunk_count", "last_built_at", "stale", "error"):
        assert key in bm25, f"missing bm25.{key}"
    assert bm25["status"] == "empty"
    assert bm25["chunk_count"] == 0
    assert bm25["last_built_at"] is None


def test_get_stats_bm25_ready_after_build(isolated_orchestrator):
    orch = isolated_orchestrator
    _seed_chunks(orch, [("c1", "hello world payload")])
    orch._ensure_bm25_index()
    bm25 = orch.get_stats()["bm25"]
    assert bm25["status"] == "ready"
    assert bm25["chunk_count"] >= 1
    assert bm25["last_built_at"] is not None
    assert bm25["last_built_at"] <= time.time()
    assert bm25["stale"] is False
    assert bm25["error"] is None
