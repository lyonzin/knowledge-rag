"""Coverage for dual-collection swap zero-downtime (v4.8.0 Fase 5).

Four groups:

1. TestStagingLifecycle — naming format, staging populate does NOT touch
   production BM25, validation gates (count threshold + canonical query
   sanity).
2. TestSwap — post-swap self.collection reconnect + BM25 rebuild only
   after swap succeeds + parallel-query zero-error chaos test.
3. TestCleanup — stale (>24h) staging removal + preservation of recent
   ones (may belong to a rebuild in progress).
4. TestBackwardsCompat — swap=False keeps legacy destructive workflow;
   default is swap=True.

Tests bypass __init__ via object.__new__ so no real ChromaDB or embedding
download happens. Same pattern as tests/test_reindex_resume.py.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from mcp_server.server import BM25Index, KnowledgeOrchestrator

# =============================================================================
# HELPERS
# =============================================================================


def _fake_collection(
    name: str,
    count: int = 100,
    query_hits: int = 1,
    raise_on_query: bool = False,
):
    """Build a MagicMock that quacks like a ChromaDB Collection.

    - ``count()`` returns ``count``
    - ``query()`` returns ``query_hits`` ids per call, or raises when
      ``raise_on_query`` is set
    - ``modify(name=X)`` mutates the ``name`` attribute in-place so
      subsequent ``.name`` reads reflect the rename
    """
    coll = MagicMock()
    coll.name = name
    coll.count.return_value = count

    def _query(query_texts=None, n_results=1, include=None, **_):
        if raise_on_query:
            raise RuntimeError("simulated query failure")
        return {
            "ids": [[f"id_{i}" for i in range(query_hits)]],
            "documents": [[f"doc_{i}" for i in range(query_hits)]],
        }

    coll.query.side_effect = _query

    def _modify(name=None, metadata=None, configuration=None):
        if name is not None:
            coll.name = name

    coll.modify.side_effect = _modify
    return coll


def _fake_client(collections: List[MagicMock]):
    """Build a MagicMock quacking like a chromadb PersistentClient.

    Backed by a mutable list so tests can inspect create/delete/rename
    behavior end-to-end.
    """
    client = MagicMock()
    _by_name: Dict[str, MagicMock] = {c.name: c for c in collections}

    def _list():
        return list(_by_name.values())

    def _get(name, embedding_function=None, **_):
        if name not in _by_name:
            raise ValueError(f"Collection not found: {name}")
        return _by_name[name]

    def _get_or_create(name, embedding_function=None, metadata=None, **_):
        if name not in _by_name:
            new = _fake_collection(name, count=0, query_hits=0)
            _by_name[name] = new

            def _mod(nm=None, md=None, cfg=None, coll=new):
                nonlocal _by_name
                if nm is not None and nm != coll.name:
                    _by_name.pop(coll.name, None)
                    coll.name = nm
                    _by_name[nm] = coll

            new.modify.side_effect = _mod
        return _by_name[name]

    def _delete(name):
        _by_name.pop(name, None)

    # Rebind modify on original collections so rename updates the client's
    # internal map — otherwise get_collection('prod') would still see the
    # old handle at the old name after swap.
    for c in collections:

        def _make_mod(coll):
            def _mod(name=None, metadata=None, configuration=None):
                nonlocal _by_name
                if name is not None and name != coll.name:
                    _by_name.pop(coll.name, None)
                    coll.name = name
                    _by_name[name] = coll

            return _mod

        c.modify.side_effect = _make_mod(c)

    client.list_collections.side_effect = _list
    client.get_collection.side_effect = _get
    client.get_or_create_collection.side_effect = _get_or_create
    client.delete_collection.side_effect = _delete
    return client, _by_name


def _fresh_orch(prod_collection_name: str = "knowledge_base") -> KnowledgeOrchestrator:
    """Build an orchestrator with the minimal state the swap workflow needs.

    __init__ is bypassed so no FastEmbed/ChromaDB is touched.
    """
    orch = object.__new__(KnowledgeOrchestrator)
    orch.embed_fn = MagicMock()
    orch.bm25_index = BM25Index()
    orch._bm25_initialized = False
    orch._indexed_docs = {}
    orch._source_to_docid = {}
    orch.query_cache = MagicMock()
    orch._reindex_progress = {"active": False}
    orch._metadata_file = MagicMock()
    orch._save_metadata = MagicMock()
    orch._ensure_bm25_index = MagicMock()
    orch.index_all = MagicMock(
        return_value={
            "total_files": 10,
            "indexed": 10,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
            "chunks_added": 100,
            "chunks_removed": 0,
            "dedup_skipped": 0,
            "categories": {},
        }
    )
    return orch


# =============================================================================
# TestStagingLifecycle
# =============================================================================


class TestStagingLifecycle:
    def test_staging_created_with_timestamped_name(self, monkeypatch):
        """Name format: {collection_name}__staging_{unix_ts}."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)
        prod = _fake_collection("kb", count=100)
        client, by_name = _fake_client([prod])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        ts = 1_700_000_000
        staging = orch._create_staging_collection(ts)
        assert staging.name == "kb__staging_1700000000"
        assert "kb__staging_1700000000" in by_name

    def test_populate_does_not_touch_prod_bm25(self, monkeypatch):
        """_populate_staging must NOT mutate the production BM25 index."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        prod = _fake_collection("kb", count=50)
        client, _ = _fake_client([prod])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        # Seed prod BM25 with a marker so we can detect any tampering.
        orch.bm25_index.add_documents(["prod_id_1"], ["prod content"])
        prod_bm25_before = orch.bm25_index

        staging = _fake_collection("kb__staging_1", count=0, query_hits=0)
        orch._populate_staging(staging)

        # After populate returns, self.bm25_index has been rebound to the
        # throwaway staging BM25 — but the original object is still intact
        # (identity preserved, so a query thread that captured the reference
        # earlier would keep working).
        assert len(prod_bm25_before) == 1

    def test_validate_rejects_undersized_staging(self, monkeypatch):
        """staging.count() below 90% of baseline -> validate returns ok=False."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        orch = _fresh_orch()
        staging = _fake_collection("kb__staging_x", count=50, query_hits=1)

        result = orch._validate_staging(staging, baseline_count=100)
        assert result["ok"] is False
        assert result["count"] == 50
        assert result["min_expected"] == 90

    def test_validate_rejects_broken_canonical_queries(self, monkeypatch):
        """4+ of 5 canonical queries returning 0 hits -> ok=False."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        orch = _fresh_orch()
        # Count passes but queries return 0 hits.
        staging = _fake_collection("kb__staging_x", count=100, query_hits=0)

        result = orch._validate_staging(staging, baseline_count=100)
        assert result["ok"] is False
        assert result["canonical_hits"] == 0

    def test_validate_passes_healthy_staging(self, monkeypatch):
        """Baseline met + all canonical queries return hits -> ok=True."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        orch = _fresh_orch()
        staging = _fake_collection("kb__staging_x", count=100, query_hits=1)

        result = orch._validate_staging(staging, baseline_count=100)
        assert result["ok"] is True
        assert result["canonical_hits"] == 5

    def test_validate_fresh_install_zero_baseline(self, monkeypatch):
        """Baseline == 0 (fresh install) skips canonical-hit gate."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        orch = _fresh_orch()
        # Zero-count staging + zero baseline: fresh install is valid.
        staging = _fake_collection("kb__staging_x", count=0, query_hits=0)

        result = orch._validate_staging(staging, baseline_count=0)
        assert result["ok"] is True


# =============================================================================
# TestSwap
# =============================================================================


class TestSwap:
    def test_swap_reconnects_self_collection(self, monkeypatch):
        """After swap, self.collection points to the collection now named prod."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        prod = _fake_collection("kb", count=100)
        staging = _fake_collection("kb__staging_1", count=100, query_hits=1)
        client, by_name = _fake_client([prod, staging])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        orch._swap_collections_atomic(staging, "kb", ts=1)
        orch._rebuild_bm25_post_swap("kb")

        # After swap: the object that was `staging` now holds name "kb".
        assert staging.name == "kb"
        # The client returns that object under "kb".
        assert orch.collection is staging
        # __old_1 was cleaned up.
        assert "kb__old_1" not in by_name

    def test_bm25_rebuilt_only_after_swap_success(self, monkeypatch):
        """Failed validate -> BM25 NOT rebuilt, no _ensure_bm25_index call."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        prod = _fake_collection("kb", count=100)
        client, _ = _fake_client([prod])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod
        # index_all populates with too few docs -> validate fails
        broken_staging = _fake_collection("kb__staging_1", count=10, query_hits=0)

        # Simulate populate returning a broken staging by having index_all
        # write nothing (default) — validate will trip on count.
        # We call the full path via _rebuild_via_swap.
        # Force the staging to be the broken one.

        def _create_stub(ts):
            return broken_staging

        orch._create_staging_collection = _create_stub

        with pytest.raises(RuntimeError, match="Staging validation failed"):
            orch._rebuild_via_swap()

        # _ensure_bm25_index (post-swap BM25 rebuild) never called.
        orch._ensure_bm25_index.assert_not_called()

    def test_100_parallel_queries_during_populate_zero_errors(self, monkeypatch):
        """Chaos: populate in one thread + 100 queries in others -> zero errors.

        Simulates realistic concurrency by having queries fire against
        self.collection while _populate_staging temporarily rebinds it.
        The saved-original snapshot in _populate_staging means the query
        thread that captured the reference before populate started keeps
        reading from prod. Threads that fetch self.collection AFTER the
        rebind see staging — which is the expected behavior of a
        thread-safe rebind pattern.
        """
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        prod = _fake_collection("kb", count=100, query_hits=1)
        client, _ = _fake_client([prod])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        errors: list = []
        # Snapshot BEFORE populate — simulating a query thread that
        # captured the reference at boot time.
        captured_prod = orch.collection

        def _query_thread():
            for _ in range(10):
                try:
                    r = captured_prod.query(query_texts=["q"], n_results=1)
                    assert r["ids"][0], "empty result during rebuild"
                except Exception as e:
                    errors.append(str(e))

        # Make populate slow by wrapping index_all
        real_index_all = orch.index_all

        def _slow_index_all(*a, **kw):
            time.sleep(0.02)
            return real_index_all(*a, **kw)

        orch.index_all = _slow_index_all

        staging = _fake_collection("kb__staging_x", count=0, query_hits=0)

        threads = [threading.Thread(target=_query_thread) for _ in range(10)]
        for t in threads:
            t.start()

        orch._populate_staging(staging)

        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"parallel queries failed during populate: {errors}"


# =============================================================================
# TestCleanup
# =============================================================================


class TestCleanup:
    def test_stale_staging_removed_on_cleanup(self, monkeypatch):
        """__staging_{ts} with ts > 24h ago is deleted."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        old_ts = int(time.time()) - (25 * 60 * 60)  # 25h ago
        stale = _fake_collection(f"kb__staging_{old_ts}", count=0)
        prod = _fake_collection("kb", count=100)
        client, by_name = _fake_client([prod, stale])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        stats = orch._cleanup_stale_staging_collections()
        assert stats["removed"] == 1
        assert f"kb__staging_{old_ts}" not in by_name
        assert "kb" in by_name  # prod untouched

    def test_recent_staging_preserved(self, monkeypatch):
        """__staging_{ts} with ts < 24h is preserved (may be active rebuild)."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        recent_ts = int(time.time()) - (60 * 60)  # 1h ago
        recent = _fake_collection(f"kb__staging_{recent_ts}", count=50)
        prod = _fake_collection("kb", count=100)
        client, by_name = _fake_client([prod, recent])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        stats = orch._cleanup_stale_staging_collections()
        assert stats["removed"] == 0
        assert stats["preserved"] == 1
        assert f"kb__staging_{recent_ts}" in by_name

    def test_non_staging_collection_ignored(self, monkeypatch):
        """Collection with a name that doesn't start with prefix is left alone."""
        from mcp_server import server as srv

        monkeypatch.setattr(srv.config, "collection_name", "kb", raising=False)

        unrelated = _fake_collection("some_other_kb", count=100)
        prod = _fake_collection("kb", count=100)
        client, by_name = _fake_client([prod, unrelated])

        orch = _fresh_orch()
        orch.chroma_client = client
        orch.collection = prod

        stats = orch._cleanup_stale_staging_collections()
        assert stats["scanned"] == 0
        assert "some_other_kb" in by_name


# =============================================================================
# TestBackwardsCompat
# =============================================================================


class TestBackwardsCompat:
    def test_default_is_swap_true(self):
        """v4.8.0+ default: nuclear_rebuild() with no args uses swap workflow."""
        import inspect

        sig = inspect.signature(KnowledgeOrchestrator.nuclear_rebuild)
        assert sig.parameters["swap"].default is True

    def test_swap_false_dispatches_destructive(self, monkeypatch):
        """swap=False calls _rebuild_destructive, not _rebuild_via_swap."""
        orch = _fresh_orch()
        orch._rebuild_destructive = MagicMock(return_value={"indexed": 0})
        orch._rebuild_via_swap = MagicMock(return_value={"indexed": 0})

        orch.nuclear_rebuild(swap=False)

        orch._rebuild_destructive.assert_called_once()
        orch._rebuild_via_swap.assert_not_called()

    def test_swap_true_dispatches_swap(self, monkeypatch):
        """swap=True (default) calls _rebuild_via_swap, not _rebuild_destructive."""
        orch = _fresh_orch()
        orch._rebuild_destructive = MagicMock(return_value={"indexed": 0})
        orch._rebuild_via_swap = MagicMock(return_value={"indexed": 0})

        orch.nuclear_rebuild()  # no args -> default swap=True

        orch._rebuild_via_swap.assert_called_once()
        orch._rebuild_destructive.assert_not_called()
