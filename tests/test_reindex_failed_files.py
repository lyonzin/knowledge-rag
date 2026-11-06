"""Coverage for reindex failure reporting (GH #288).

When a reindex ends with errors, get_reindex_status() must expose WHICH
files failed and why — before, the paths went only to the server's stderr,
invisible to MCP clients. Groups:

1. TestFailedFilesReporting — real Chroma with an isolated corpus and the
   background reindex thread run inline, so the status flow is synchronous
   and offline.
2. TestFailureRecordingSites — the prune/repair/documents-dir error paths
   feed the same list through _record_failed_file.
3. TestFailedFilesCapAndDedupe — the _MAX_REPORTED_FAILED_FILES bound, the
   truncated flag, path dedupe and the reason bound, exercised on the
   helpers directly (no need to index 101 files).
"""

from __future__ import annotations

from unittest.mock import Mock

from mcp_server import server
from mcp_server.server import KnowledgeOrchestrator


class _InlineThread:
    """Run the background reindex inline so the status flow is deterministic."""

    def __init__(self, target, args, daemon):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class TestFailedFilesReporting:
    def test_failed_files_surface_in_last_result_of_reindex_status(self, isolated_orchestrator, monkeypatch):
        """A parse failure shows its path and short reason once the run is inactive."""
        orch = isolated_orchestrator

        # Arrange: one healthy file indexed, one new file that will fail to parse.
        good = server.config.documents_dir / "good.txt"
        good.write_text("initial document text", encoding="utf-8")
        assert orch.index_all()["errors"] == 0
        bad = server.config.documents_dir / "bad.txt"
        bad.write_text("broken document text", encoding="utf-8")
        monkeypatch.setattr(orch.parser, "parse_file", Mock(side_effect=PermissionError("access denied")))

        # Act: run the reindex through the real background path, inline.
        monkeypatch.setattr(server.threading, "Thread", _InlineThread)
        orch.start_reindex_background("incremental")

        # Assert: existing counters keep their types and the new fields carry the failure.
        status = orch.get_reindex_status()
        assert status["active"] is False
        result = status["last_result"]
        assert result["errors"] == 1
        assert result["indexed"] == 0
        assert result["skipped"] == 1
        assert result["failed_files"] == [{"path": str(bad), "reason": "PermissionError: access denied"}]
        assert result["failed_files_truncated"] is False
        assert result["errors"] == len(result["failed_files"])  # invariant below the cap

    def test_clean_run_reports_empty_failed_files(self, isolated_orchestrator):
        """A run without errors keeps the new fields present and empty."""
        orch = isolated_orchestrator

        # Arrange + Act
        good = server.config.documents_dir / "good.txt"
        good.write_text("initial document text", encoding="utf-8")
        stats = orch.index_all()

        # Assert
        assert stats["errors"] == 0
        assert stats["failed_files"] == []
        assert stats["failed_files_truncated"] is False

    def test_failed_full_rebuild_surfaces_paths_via_status(self, isolated_orchestrator, monkeypatch):
        """A failed staging populate exposes paths in both last_error and last_result."""
        orch = isolated_orchestrator

        # Arrange
        bad = server.config.documents_dir / "bad.txt"
        bad.write_text("broken document text", encoding="utf-8")
        monkeypatch.setattr(orch.parser, "parse_file", Mock(side_effect=PermissionError("access denied")))

        # Act
        monkeypatch.setattr(server.threading, "Thread", _InlineThread)
        orch.start_reindex_background("nuclear_rebuild")

        # Assert
        status = orch.get_reindex_status()
        assert status["active"] is False
        assert str(bad) in status["last_error"]
        assert "PermissionError" in status["last_error"]
        assert status["last_result"]["errors"] == 1
        assert status["last_result"]["failed_files"] == [{"path": str(bad), "reason": "PermissionError: access denied"}]


class TestFailureRecordingSites:
    def test_prune_failure_records_path(self, isolated_orchestrator, monkeypatch, tmp_path):
        """An orphan doc whose chunk removal fails is listed by its source path."""
        orch = isolated_orchestrator
        outside = tmp_path / "outside.txt"
        orch._indexed_docs["doc1"] = {"source": str(outside)}
        monkeypatch.setattr(orch, "_remove_document_chunks", Mock(side_effect=PermissionError("denied")))
        stats = KnowledgeOrchestrator._init_index_stats()

        # Act: the source is outside documents_dir, so pruning attempts removal and fails.
        orch._prune_orphan_documents([server.config.documents_dir / "present.txt"], stats)

        # Assert
        assert stats["errors"] == 1
        assert stats["failed_files"] == [{"path": str(outside), "reason": "PermissionError: denied"}]

    def test_repair_failure_records_path(self, isolated_orchestrator, monkeypatch, tmp_path):
        """A repair of an absent source that fails is listed by its path."""
        orch = isolated_orchestrator
        missing = tmp_path / "gone.txt"
        monkeypatch.setattr(orch, "_remove_repair_chunks", Mock(side_effect=PermissionError("denied")))
        stats = KnowledgeOrchestrator._init_index_stats()
        tracking = {"repair_sources": {str(missing): ["0123456789abcdef"]}}

        # Act
        orch._repair_missing_sources(stats, tracking)

        # Assert
        assert stats["errors"] == 1
        assert stats["failed_files"] == [{"path": str(missing), "reason": "PermissionError: denied"}]

    def test_documents_dir_unavailable_records_config_dir(self, isolated_orchestrator, monkeypatch, tmp_path):
        """An unavailable documents_dir is recorded with the fixed reason."""
        orch = isolated_orchestrator
        unplugged = tmp_path / "unplugged"
        monkeypatch.setattr(server.config, "documents_dir", unplugged)
        stats = KnowledgeOrchestrator._init_index_stats()

        # Act
        orch._prune_orphan_documents([], stats)

        # Assert
        assert stats["errors"] == 1
        assert stats["failed_files"] == [
            {"path": str(unplugged), "reason": KnowledgeOrchestrator._DOCUMENTS_DIR_UNAVAILABLE_REASON}
        ]


class TestFailedFilesCapAndDedupe:
    def test_cap_beyond_100_sets_truncated(self):
        """The 101st distinct failure flips failed_files_truncated and is not stored."""
        # Arrange
        stats = KnowledgeOrchestrator._init_index_stats()

        # Act
        for index in range(101):
            KnowledgeOrchestrator._record_failed_file(stats, f"/docs/f{index}.txt", f"boom {index}")

        # Assert
        assert len(stats["failed_files"]) == 100
        assert stats["failed_files"][0] == {"path": "/docs/f0.txt", "reason": "boom 0"}
        assert stats["failed_files"][-1] == {"path": "/docs/f99.txt", "reason": "boom 99"}
        assert stats["failed_files_truncated"] is True
        assert stats["errors"] == 101  # the total keeps counting beyond the cap

    def test_exactly_100_failures_are_not_truncated(self):
        """At the cap the list is complete, so nothing was hidden."""
        # Arrange
        stats = KnowledgeOrchestrator._init_index_stats()

        # Act
        for index in range(100):
            KnowledgeOrchestrator._record_failed_file(stats, f"/docs/f{index}.txt", f"boom {index}")

        # Assert
        assert len(stats["failed_files"]) == 100
        assert stats["failed_files_truncated"] is False
        assert stats["errors"] == 100

    def test_duplicate_paths_are_counted_once(self):
        """Several doc_ids sharing one source produce one entry and one error."""
        # Arrange
        stats = KnowledgeOrchestrator._init_index_stats()

        # Act
        KnowledgeOrchestrator._record_failed_file(stats, "/docs/dup.txt", "first failure")
        KnowledgeOrchestrator._record_failed_file(stats, "/docs/dup.txt", "second failure")

        # Assert
        assert stats["errors"] == 1
        assert stats["failed_files"] == [{"path": "/docs/dup.txt", "reason": "first failure"}]
        assert stats["failed_files_truncated"] is False

    def test_failure_reason_is_type_prefixed_and_bounded(self):
        """Reasons never carry tracebacks or multi-KB payloads."""
        # Arrange + Act
        reason = KnowledgeOrchestrator._failure_reason(ValueError("x" * 5000))

        # Assert
        assert reason.startswith("ValueError: ")
        assert len(reason) <= KnowledgeOrchestrator._MAX_FAILURE_REASON_CHARS
