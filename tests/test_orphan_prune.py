"""Regression tests for GH #298 — orphan chunks are never pruned.

Chunks whose ``doc_id`` is missing from ``index_metadata.json`` (lost or
truncated catalogue, a preflight repair gone wrong) stay in ChromaDB and
in the FTS5 index forever: ``_prune_orphan_documents()`` iterates only
``_indexed_docs``, and ``remove_document`` resolves through the same
catalogue, so nothing can reach them. They keep appearing in search
results and are included in every BM25 rebuild.
"""

from pathlib import Path
from unittest.mock import MagicMock

from mcp_server import server as srv


class TestOrphanChunkPrune:
    def test_unknown_doc_chunks_are_pruned_on_reindex(self, isolated_orchestrator, monkeypatch) -> None:
        """given chunks whose doc_id is gone from the catalogue, when a reindex
        runs, then the chunks are deleted and counted."""
        orch = isolated_orchestrator
        docs_dir = Path(srv.config.documents_dir)
        a = docs_dir / "a.md"
        a.write_text("# alpha\n", encoding="utf-8")
        orch.index_all()
        a_id = orch._source_to_docid[str(a.resolve())]
        assert orch.collection.count() > 0

        # lose the catalogue entry and hide the file from the scan
        monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])
        temp = docs_dir / "Temp"
        temp.mkdir()
        a.rename(temp / "a.md")
        del orch._indexed_docs[a_id]
        orch._save_metadata()

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] >= 1, f"orphan chunks must be pruned: {stats}"
        assert orch.collection.count() == 0, "no chunks may survive"

    def test_no_prune_during_nuclear_rebuild(self, isolated_orchestrator) -> None:
        """given a nuclear rebuild in flight (staging collection populated on
        purpose), when the prune runs, then it leaves the staged chunks alone."""
        orch = isolated_orchestrator
        orch.collection.add(
            ids=["ghost_0"],
            documents=["ghost content"],
            metadatas=[{"doc_id": "ghostdoc"}],
        )
        orch._staging_target = MagicMock()
        stats = orch._init_index_stats()

        orch._prune_unknown_doc_chunks(stats)

        assert stats["orphan_chunks_removed"] == 0
        assert orch.collection.count() == 1, "staged chunks must survive the prune"

    def test_stats_defaults_orphan_counter_to_zero(self, isolated_orchestrator) -> None:
        """given a fresh stats dict, when initialized, then the orphan counter
        is present and zero."""
        stats = isolated_orchestrator._init_index_stats()
        assert stats["orphan_chunks_removed"] == 0
