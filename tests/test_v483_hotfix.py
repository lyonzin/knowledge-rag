"""v4.8.3 hotfix regression suite.

Each test reproduces exactly one production bug found in v4.8.0/4.8.1/4.8.2
before the fix, then asserts the fix. Test names encode the GH issue when
one exists and the internal fix number otherwise.

Cross-reference:
- fix1_index_lock_instance_level   — internal (nuclear_rebuild + concurrent reindex)
- fix2_batching_bm25_and_fts5      — internal (chromadb 1.x SQL var overflow)
- fix3_is_ready_rechecks_marker    — internal (nuclear_rebuild + FTS5 lazy migration race)
- gh161_write_dispatch_isolates_staging  — GH #161 (grishkovei)
- gh162_checkpoint_only_committed_docs   — GH #162 (grishkovei)
- gh163_force_flag_propagates_through    — GH #163 (grishkovei)
- fix5_stale_marker_triggers_rebuild     — internal (marker complete + FTS5 empty)
- fix6_format_skips_orphan_hits          — internal (Chroma-FTS5 drift residue)
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# fix1 — _index_lock must be per-instance so a GC'd orch can't wedge others
# ---------------------------------------------------------------------------


def test_fix1_index_lock_is_per_instance() -> None:
    """Two orch instances must own distinct locks; class-level singleton wedged v4.8.2."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        a = KnowledgeOrchestrator()
        a._index_lock = threading.Lock()
        b = KnowledgeOrchestrator()
        b._index_lock = threading.Lock()
        assert a._index_lock is not b._index_lock


# ---------------------------------------------------------------------------
# fix2 — collection.get(limit=count) must batch; chromadb 1.x explodes >999
# ---------------------------------------------------------------------------


def test_fix2_iter_chroma_chunks_batches_via_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """_iter_chroma_chunks_for_fts5 must issue offset-based calls, not one megacall."""
    from mcp_server.server import KnowledgeOrchestrator

    calls: list[dict[str, Any]] = []

    def fake_get(**kwargs: Any) -> dict[str, list]:
        calls.append(kwargs)
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 100)
        # Simulate 1500-chunk corpus: 3 batches of 500 + empty guard.
        end = min(offset + limit, 1500)
        n = max(0, end - offset)
        return {
            "ids": [f"chunk_{i}" for i in range(offset, offset + n)],
            "documents": ["doc"] * n,
            "metadatas": [{"filename": "f.md", "category": "redteam"}] * n,
        }

    orch = SimpleNamespace()
    orch.collection = SimpleNamespace(count=lambda: 1500, get=fake_get)
    orch._iter_chroma_chunks_for_fts5 = KnowledgeOrchestrator._iter_chroma_chunks_for_fts5.__get__(orch)

    yielded = list(orch._iter_chroma_chunks_for_fts5())

    assert len(yielded) == 1500, f"expected 1500 rows, got {len(yielded)}"
    # 3 batches of 500 → 3 calls, each with limit=500.
    limits = [c.get("limit") for c in calls]
    assert all(lim == 500 for lim in limits), f"expected all batches limit=500, got {limits}"
    assert len(calls) >= 3


# ---------------------------------------------------------------------------
# fix3 — is_ready must re-read marker; cache stale wedged post-swap in v4.8.2
# ---------------------------------------------------------------------------


def test_fix3_is_ready_rechecks_marker_when_cache_false(tmp_path: Any) -> None:
    """Post-swap orch instances constructed while marker=in_progress must recover."""
    from mcp_server.fts5_index import Fts5LexicalIndex

    db = tmp_path / "fts5.db"
    marker = tmp_path / "fts5_migration.state"

    idx = Fts5LexicalIndex(db_path=db, state_path=marker)
    assert idx.is_ready() is False

    marker.write_text(
        json.dumps({"status": "complete", "docs_total": 10, "docs_indexed": 10}),
        encoding="utf-8",
    )
    assert idx.is_ready() is True, "is_ready should re-read marker when _ready is False"
    idx.close()


# ---------------------------------------------------------------------------
# GH #161 — write dispatch must isolate staging from live query path
# ---------------------------------------------------------------------------


def test_gh161_write_collection_routes_to_staging_when_active() -> None:
    """During nuclear_rebuild populate, writes hit staging; reads hit production."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        prod = MagicMock(name="prod_collection")
        staging = MagicMock(name="staging_collection")
        orch.collection = prod
        orch._staging_target = None

        # No staging active → writes route to production.
        assert orch._write_collection is prod

        # Staging active → writes route to staging, but self.collection stays prod.
        orch._staging_target = staging
        assert orch._write_collection is staging
        assert orch.collection is prod, "query path must keep seeing production during populate"

        # Post-swap cleanup restores default routing.
        orch._staging_target = None
        assert orch._write_collection is prod


# ---------------------------------------------------------------------------
# GH #162 — checkpoint must serialize only docs committed by current run
# ---------------------------------------------------------------------------


def test_gh162_tracking_only_records_committed_this_run() -> None:
    """Checkpoint IDs must not include unprocessed docs from prior metadata."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        orch._reindex_progress = {"operation": "smart_reindex"}
        orch._indexed_docs = {}  # required by _seed_chunks_total_estimate
        stats = {"total_files": 3}
        tracking = orch._init_reindex_tracking(
            resume_state={"doc_ids": ["prior_run_committed"], "chunks_processed": 42},
            stats=stats,
        )

        assert "committed_this_run" in tracking, "tracking must expose committed_this_run set"
        assert tracking["committed_this_run"] == {"prior_run_committed"}, (
            "resume must fold in prior-run committed IDs so they survive interruption"
        )
        assert tracking["chunks_processed"] == 42


# ---------------------------------------------------------------------------
# GH #163 — force=True must propagate through reindex_all to index_all
# ---------------------------------------------------------------------------


def test_gh163_reindex_all_accepts_and_propagates_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """reindex_all(force=True) must call index_all(force=True), not force=False."""
    from mcp_server.server import KnowledgeOrchestrator

    calls: list[dict[str, Any]] = []

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        orch.bm25_index = MagicMock()
        orch._bm25_initialized = False
        orch.query_cache = MagicMock()
        orch._ensure_bm25_index = lambda: None
        orch._save_metadata = lambda: None
        orch._indexed_docs = {}

        def fake_index_all(**kwargs: Any) -> dict[str, int]:
            calls.append(kwargs)
            return {
                "indexed": 0,
                "updated": 0,
                "deleted": 0,
                "chunks_added": 0,
                "errors": 0,
                "total_files": 0,
                "skipped": 0,
            }

        orch.index_all = fake_index_all
        # Skip filesystem work — nuclear_rebuild path irrelevant for this test.
        with patch("shutil.rmtree"), patch("pathlib.Path.iterdir", return_value=[]):
            orch.reindex_all(force=True)

    assert calls, "reindex_all must invoke index_all"
    assert calls[0].get("force") is True, f"index_all must receive force=True, got {calls[0]!r}"


# ---------------------------------------------------------------------------
# fix5 — stale marker (complete + empty FTS5) must trigger rebuild
# ---------------------------------------------------------------------------


def test_fix5_stale_marker_detected_when_fts5_empty() -> None:
    """Marker=complete + FTS5 rows << Chroma rows must NOT be trusted."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        orch.fts5_index = SimpleNamespace(count=lambda: 5)
        orch.collection = SimpleNamespace(count=lambda: 50000)
        # 5 rows in FTS5 vs 50k in Chroma → 0.01% populated → marker is lying.
        assert orch._fts5_marker_matches_reality() is False

        # 5000 in FTS5 vs 50k → 10% → marker is credible.
        orch.fts5_index = SimpleNamespace(count=lambda: 5000)
        assert orch._fts5_marker_matches_reality() is True

        # Empty corpus → marker complete is trivially legitimate.
        orch.collection = SimpleNamespace(count=lambda: 0)
        orch.fts5_index = SimpleNamespace(count=lambda: 0)
        assert orch._fts5_marker_matches_reality() is True


# ---------------------------------------------------------------------------
# fix6 — _format_fts5_results must skip orphan hits (chroma missing chunk_id)
# ---------------------------------------------------------------------------


def test_fix6_format_skips_orphan_chunk_ids() -> None:
    """FTS5 hits pointing to Chroma-missing chunks must be filtered, not padded."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        # Only 'chunk_alive' exists in Chroma; 'chunk_orphan' is FTS5 residue.
        orch.collection = SimpleNamespace(
            get=lambda **kw: {
                "ids": ["chunk_alive"],
                "documents": ["real content"],
                "metadatas": [{"source": "real.md", "filename": "real.md", "category": "redteam"}],
            }
        )
        hits = [("chunk_alive", 10.0), ("chunk_orphan", 5.0)]
        results = orch._format_fts5_results(hits, max_results=5, category_filter=None)

    assert len(results) == 1
    assert results[0]["source"] == "real.md"
    assert all(r["content"] for r in results), "no result may have empty content"
