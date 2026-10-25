"""File publication must preserve rollback bytes and report committed writes accurately."""

import json
import os
from pathlib import Path

import pytest

from mcp_server import file_transaction
from mcp_server.file_transaction import StagedTextFile, staged_text_file


def _windows_denial(winerror):
    """PermissionError as Windows raises it while another handle holds the path."""
    error = PermissionError(13, "Access is denied")
    error.winerror = winerror
    return error


def test_replace_waits_out_a_transient_windows_handle(tmp_path, monkeypatch):
    """GH #249: scanners and readers briefly hold files without FILE_SHARE_DELETE."""
    source, target = tmp_path / "state.json.tmp", tmp_path / "state.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    real_replace = os.replace
    failures = [_windows_denial(5), _windows_denial(32)]

    def held_twice(src, dst):
        if failures:
            raise failures.pop(0)
        real_replace(src, dst)

    monkeypatch.setattr(file_transaction.os, "replace", held_twice)
    monkeypatch.setattr(file_transaction, "sleep", lambda _delay: None)
    file_transaction.replace_with_retry(source, target)

    assert target.read_text(encoding="utf-8") == "new"
    assert not source.exists()


@pytest.mark.parametrize("winerror", [5, None])
def test_replace_reraises_persistent_or_non_windows_denial(tmp_path, monkeypatch, winerror):
    denial = _windows_denial(winerror)
    calls = []

    def always_denied(src, dst):
        calls.append(src)
        raise denial

    monkeypatch.setattr(file_transaction.os, "replace", always_denied)
    monkeypatch.setattr(file_transaction, "sleep", lambda _delay: None)
    with pytest.raises(PermissionError) as caught:
        file_transaction.replace_with_retry(tmp_path / "a", tmp_path / "b")

    assert caught.value is denial
    assert len(calls) == (len(file_transaction._REPLACE_RETRY_DELAYS) + 1 if winerror else 1)


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_replace_succeeds_once_a_real_windows_handle_is_released(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    source, target = tmp_path / "index_metadata.json.tmp", tmp_path / "index_metadata.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    # GENERIC_READ shared for read/write but not delete, like a scanner or open().
    handle = kernel32.CreateFileW(str(target), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle not in (None, wintypes.HANDLE(-1).value)
    released = []

    def release(_delay):
        if not released:
            assert kernel32.CloseHandle(handle), ctypes.get_last_error()
            released.append(handle)

    try:
        with pytest.raises(PermissionError):
            os.replace(source, target)
        monkeypatch.setattr(file_transaction, "sleep", release)
        file_transaction.replace_with_retry(source, target)
    finally:
        release(0)

    assert target.read_text(encoding="utf-8") == "new"


def test_reindex_state_files_survive_a_transient_windows_handle(isolated_orchestrator, monkeypatch):
    """GH #249: a held metadata or checkpoint file must not lose the write."""
    orch = isolated_orchestrator
    orch._indexed_docs = {"doc": {"source": "a.md"}}
    held = {orch._metadata_file.name, orch._checkpoint_file.name}
    real_replace = os.replace

    def held_once(src, dst):
        if Path(dst).name in held:
            held.discard(Path(dst).name)
            raise _windows_denial(5)
        real_replace(src, dst)

    monkeypatch.setattr(file_transaction.os, "replace", held_once)
    monkeypatch.setattr(file_transaction, "sleep", lambda _delay: None)
    orch._save_metadata()
    orch._write_checkpoint("smart_reindex", ["doc"], 3)

    assert not held
    assert json.loads(orch._metadata_file.read_text(encoding="utf-8")) == orch._indexed_docs
    assert json.loads(orch._checkpoint_file.read_text(encoding="utf-8"))["indexed_doc_ids"] == ["doc"]


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
