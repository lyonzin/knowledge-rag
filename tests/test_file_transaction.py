"""File publication must preserve rollback bytes and report committed writes accurately."""

from pathlib import Path

import pytest

from mcp_server import file_transaction
from mcp_server.file_transaction import StagedTextFile, staged_text_file


def test_committed_file_survives_temporary_backup_cleanup_failure(tmp_path, monkeypatch, caplog):
    destination = tmp_path / "document.md"
    destination.write_text("old content", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_backup_cleanup(path, *args, **kwargs):
        if path.name.startswith(".rag-backup-"):
            raise PermissionError("injected temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    with staged_text_file(destination, "new content") as staged:
        staged.publish()
        staged.committed = True

    assert destination.read_text(encoding="utf-8") == "new content"
    assert "injected temporary cleanup failure" in caplog.text
    backups = list(tmp_path.glob(".rag-backup-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old content"
    assert not list(tmp_path.glob(".rag-pending-*"))


def test_failed_publication_restores_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "document.md"
    destination.write_text("old content", encoding="utf-8")
    original_replace = file_transaction.os.replace

    def fail_pending_publish(source, target):
        if Path(source).name.startswith(".rag-pending-"):
            raise PermissionError("injected publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(file_transaction.os, "replace", fail_pending_publish)
    with pytest.raises(PermissionError, match="injected publication failure"):
        with staged_text_file(destination, "new content") as staged:
            staged.publish()

    assert destination.read_text(encoding="utf-8") == "old content"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["document.md"]


def test_failed_restore_keeps_backup_for_recovery(tmp_path, monkeypatch):
    destination = tmp_path / "document.md"
    destination.write_text("old content", encoding="utf-8")
    staged = StagedTextFile(destination, "new content")
    staged.publish()
    original_replace = file_transaction.os.replace

    def fail_restore(source, target):
        if source == staged.backup:
            raise PermissionError("injected rollback failure")
        return original_replace(source, target)

    with monkeypatch.context() as local_patch:
        local_patch.setattr(file_transaction.os, "replace", fail_restore)
        with pytest.raises(PermissionError, match="injected rollback failure"):
            staged.close()

    assert staged.backup is not None
    assert staged.backup.read_text(encoding="utf-8") == "old content"
    staged.close()
    assert destination.read_text(encoding="utf-8") == "old content"


def test_failed_uncommitted_new_file_is_removed(tmp_path):
    destination = tmp_path / "document.md"
    with pytest.raises(RuntimeError, match="index failed"):
        with staged_text_file(destination, "new content") as staged:
            staged.publish()
            raise RuntimeError("index failed")

    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_committed_update_invalidates_cache_when_backup_cleanup_fails(isolated_orchestrator, monkeypatch):
    from mcp_server import server

    orchestrator = isolated_orchestrator
    destination = server.config.documents_dir / "document.txt"
    destination.write_text("old indexed content", encoding="utf-8")
    assert orchestrator.index_all()["errors"] == 0
    orchestrator.query_cache.put("saved query", 5, None, 0.3, [{"content": "old indexed content"}])
    original_unlink = Path.unlink

    def fail_backup_cleanup(path, *args, **kwargs):
        if path.name.startswith(".rag-backup-"):
            raise PermissionError("injected temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    result = orchestrator.update_document_content("document.txt", "replacement indexed content")

    assert "error" not in result
    assert result["new_chunks_added"] == 1
    assert orchestrator.collection.get(include=["documents"])["documents"] == ["replacement indexed content"]
    assert orchestrator.query_cache.get("saved query", 5, None, 0.3) is None
    assert destination.read_text(encoding="utf-8") == "replacement indexed content"
