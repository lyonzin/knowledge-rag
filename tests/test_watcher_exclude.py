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


@pytest.mark.parametrize("pattern", ["Archive/old", "Archive/old/*", "old", "Archive/*"])
def test_event_below_an_excluded_folder_does_not_schedule(watcher_env, monkeypatch, pattern) -> None:
    """given a pattern that makes the scan skip a folder, however it is
    written, when a file changes anywhere below that folder, then no reindex
    is scheduled — "Archive/old" names the folder and matches no file path."""
    monkeypatch.setattr(srv.config, "exclude_patterns", [pattern])
    watcher = watcher_env["watcher"]
    target = watcher_env["docs_dir"] / "Archive" / "old" / "deep" / "legacy.md"
    target.parent.mkdir(parents=True)
    target.write_text("# legacy\n")

    watcher.on_created(_event(str(target)))
    watcher.on_modified(_event(str(target)))

    assert not watcher._pending_paths


def test_event_beside_an_excluded_folder_still_schedules(watcher_env, monkeypatch) -> None:
    """given the folder Archive/old is excluded, when a file changes in its
    sibling Archive/new, then a reindex is scheduled."""
    monkeypatch.setattr(srv.config, "exclude_patterns", ["Archive/old"])
    watcher = watcher_env["watcher"]
    target = watcher_env["docs_dir"] / "Archive" / "new" / "fresh.md"
    target.parent.mkdir(parents=True)
    target.write_text("# fresh\n")

    watcher.on_modified(_event(str(target)))

    assert watcher._pending_paths == {str(target)}


@pytest.mark.parametrize("destination", ["Temp/notes.md", "notes.bak", "../elsewhere/notes.md"])
def test_move_of_an_indexed_file_schedules_its_old_path(watcher_env, destination) -> None:
    """given an indexed file, when it is moved into an excluded folder, to an
    unsupported name or out of the documents folder, then its old path is
    scheduled so the chunks it left behind are pruned."""
    watcher = watcher_env["watcher"]
    source = watcher_env["docs_dir"] / "notes.md"
    watcher_env["orch"]._source_to_docid = {str(source.resolve()): "docid-1"}
    watcher_env["orch"]._indexed_docs = {"docid-1": {"source": str(source)}}

    watcher.on_moved(_event(str(source), str(watcher_env["docs_dir"] / destination)))

    assert str(source) in watcher._pending_paths


def test_move_of_an_unindexed_file_to_an_unsupported_name_does_not_schedule(watcher_env) -> None:
    """given a file that is not indexed, when it is renamed to an unsupported
    name, then nothing is scheduled."""
    watcher = watcher_env["watcher"]
    source = watcher_env["docs_dir"] / "draft.md"

    watcher.on_moved(_event(str(source), str(watcher_env["docs_dir"] / "draft.bak")))

    assert not watcher._pending_paths


def test_exception_in_a_handler_does_not_reach_the_observer(watcher_env, monkeypatch) -> None:
    """given a path that makes a handler raise (a symlink loop does, on Python
    3.11 and 3.12), when the observer dispatches the event, then the exception
    stays inside the watcher — an uncaught one ends the observer's thread."""
    from watchdog.events import FileModifiedEvent

    watcher = watcher_env["watcher"]
    monkeypatch.setattr(watcher, "_is_real_change", MagicMock(side_effect=RuntimeError("Symlink loop from 'a.md'")))
    target = watcher_env["docs_dir"] / "a.md"
    target.write_text("# a\n")

    watcher.dispatch(FileModifiedEvent(str(target)))

    assert not watcher._pending_paths


class TestWatcherExcludeE2E:
    """Real watchdog observer against a real orchestrator (GH #295)."""

    def test_observer_ignores_excluded_folder_but_catches_real_edits(
        self, isolated_orchestrator, monkeypatch, start_watcher
    ) -> None:
        monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])
        monkeypatch.setattr(srv.config, "supported_formats", {".txt"})
        docs_dir = Path(srv.config.documents_dir)
        temp_dir = docs_dir / "Temp"
        temp_dir.mkdir(exist_ok=True)
        orch = isolated_orchestrator
        scheduled: list[str] = []
        schedule = DocumentWatcher._schedule_reindex
        monkeypatch.setattr(
            DocumentWatcher, "_schedule_reindex", lambda self, path: (scheduled.append(path), schedule(self, path))
        )
        start_watcher(docs_dir)

        # The excluded file is written first: by the time the real edit below
        # is indexed, the observer has already reported both.
        (temp_dir / "scratch.txt").write_text("noise")
        real = docs_dir / "real.txt"
        real.write_text("# real change")
        deadline = time.monotonic() + 15
        while str(real.resolve()) not in orch._source_to_docid and time.monotonic() < deadline:
            time.sleep(0.05)

        assert str(real.resolve()) in orch._source_to_docid, "a real edit must be indexed"
        assert {Path(p).name for p in scheduled} == {"real.txt"}, (
            f"only the real edit may schedule a reindex, got {scheduled}"
        )
