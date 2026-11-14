"""Regression tests for GH #295 — watcher must ignore excluded folders.

Before the fix, ``DocumentWatcher`` applied only the extension filter
(``_is_supported``) when scheduling a reindex; ``exclude_patterns`` was
checked later, inside the scan. Every supported file created, modified or
deleted inside an excluded folder (TEMP dirs, tool output, venvs) therefore
triggered a full ``index_all()`` pass whose only outcome was ``0 new,
N skipped``.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_server import server as srv
from mcp_server.server import DocumentWatcher


@pytest.fixture
def watcher_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "Temp").mkdir()
    monkeypatch.setattr(srv.config, "documents_dir", docs_dir)
    monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])
    monkeypatch.setattr(srv.config, "supported_formats", {".md", ".txt"})

    orch = MagicMock()
    orch._source_to_docid = {}
    orch._indexed_docs = {}
    watcher = DocumentWatcher(orchestrator_getter=lambda: orch)
    return {"tmp_path": tmp_path, "docs_dir": docs_dir, "orch": orch, "watcher": watcher}


def _event(src_path: str, dest_path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(is_directory=False, src_path=src_path, dest_path=dest_path)


def test_on_modified_in_excluded_folder_does_not_schedule(watcher_env) -> None:
    """given a modified file inside an excluded folder, when the watcher sees
    it, then no reindex is scheduled."""
    watcher = watcher_env["watcher"]
    temp_file = watcher_env["docs_dir"] / "Temp" / "scratch.md"
    temp_file.write_text("# temp\n")

    watcher.on_modified(_event(str(temp_file)))

    assert not watcher._pending_paths, "event in excluded folder must not schedule a reindex"


def test_on_created_in_excluded_folder_does_not_schedule(watcher_env) -> None:
    """given a created file inside an excluded folder, when the watcher sees
    it, then no reindex is scheduled."""
    watcher = watcher_env["watcher"]
    temp_file = watcher_env["docs_dir"] / "Temp" / "office-temp.txt"

    watcher.on_created(_event(str(temp_file)))

    assert not watcher._pending_paths, "event in excluded folder must not schedule a reindex"


def test_on_modified_outside_excluded_folder_still_schedules(watcher_env) -> None:
    """given a modified supported file outside excluded folders, when the
    watcher sees it, then a reindex is scheduled (no over-filtering)."""
    watcher = watcher_env["watcher"]
    real_file = watcher_env["docs_dir"] / "notes.md"
    real_file.write_text("# real\n")

    watcher.on_modified(_event(str(real_file)))

    assert watcher._pending_paths, "real change outside excluded folders must still schedule"


def test_on_deleted_unindexed_excluded_file_does_not_schedule(watcher_env) -> None:
    """given a deleted unindexed file in an excluded folder, when the watcher
    sees it, then no reindex is scheduled."""
    watcher = watcher_env["watcher"]

    watcher.on_deleted(_event(str(watcher_env["docs_dir"] / "Temp" / "gone.txt")))

    assert not watcher._pending_paths


def test_on_deleted_indexed_excluded_file_still_schedules(watcher_env) -> None:
    """given a deleted file that IS indexed but now lives in an excluded
    folder, when the watcher sees it, then a reindex is scheduled so the
    stale chunks get pruned."""
    watcher = watcher_env["watcher"]
    path = str((watcher_env["docs_dir"] / "Temp" / "moved-in.md").resolve())
    watcher_env["orch"]._source_to_docid = {path: "docid-1"}
    watcher_env["orch"]._indexed_docs = {"docid-1": {}}

    watcher.on_deleted(_event(path))

    assert watcher._pending_paths, "indexed file deleted from excluded folder must be pruned"


def test_on_moved_into_excluded_folder_does_not_schedule(watcher_env) -> None:
    """given a move whose destination is excluded (and source unindexed),
    when the watcher sees it, then no reindex is scheduled."""
    watcher = watcher_env["watcher"]
    src = str(watcher_env["docs_dir"] / "notes.md")
    dest = str(watcher_env["docs_dir"] / "Temp" / "notes.md")

    watcher.on_moved(_event(src, dest))

    assert not watcher._pending_paths, "move into an excluded folder must not schedule a reindex"


class TestWatcherExcludeE2E:
    """Real watchdog observer against a real orchestrator (GH #295)."""

    def test_observer_ignores_excluded_folder_but_catches_real_edits(self, isolated_orchestrator, monkeypatch) -> None:
        from watchdog.observers.polling import PollingObserver

        monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])
        monkeypatch.setattr(srv.config, "supported_formats", {".txt"})
        docs_dir = Path(srv.config.documents_dir)
        temp_dir = docs_dir / "Temp"
        temp_dir.mkdir(exist_ok=True)

        watcher = DocumentWatcher(lambda: isolated_orchestrator, debounce_seconds=0.3)
        observer = PollingObserver()
        observer.schedule(watcher, str(docs_dir), recursive=True)
        observer.start()
        try:
            (temp_dir / "scratch.txt").write_text("noise")
            time.sleep(1.0)
            assert not watcher._pending_paths, "event inside the excluded folder must not schedule a reindex"
            (docs_dir / "real.txt").write_text("# real change")
            time.sleep(1.5)
            indexed = str((docs_dir / "real.txt").resolve()) in isolated_orchestrator._source_to_docid
            assert indexed, "event outside the excluded folder must trigger a reindex that indexes the file"
        finally:
            observer.stop()
            observer.join(timeout=2)
