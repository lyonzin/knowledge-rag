"""Regression tests for GH #297 — the watcher must reindex only changed paths.

``DocumentWatcher._do_reindex()`` collected the exact changed paths in
``_pending_paths`` and then used only their count, calling
``orch.index_all(force=False)`` — a full walk + metadata pass over every
indexed document for every single-file event. "Incremental" meant "don't
re-embed unchanged files", not "only look at changed files".
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_server import server as srv
from mcp_server.server import DocumentWatcher


def _wait_until(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _stats(**kwargs) -> dict:
    base = {
        "total_files": 0,
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "errors": 0,
        "chunks_added": 0,
        "chunks_removed": 0,
        "dedup_skipped": 0,
        "categories": {},
    }
    base.update(kwargs)
    return base


class _RecordingOrchestrator:
    """Records which reindex entrypoint the watcher chose."""

    def __init__(self, busy: bool = False):
        self.calls: list = []
        self.busy = busy

    def _result(self) -> dict:
        return _stats(skipped_reason="reindex_already_running") if self.busy else _stats(indexed=1)

    def index_paths(self, paths):
        self.calls.append(("paths", set(paths)))
        return self._result()

    def index_all(self, force=False):
        self.calls.append(("all", force))
        return self._result()


def _run_pass(watcher: DocumentWatcher, *paths: str) -> None:
    """One debounced pass, without waiting for the timer."""
    watcher._pending_paths.update(paths)
    watcher._do_reindex()
    if watcher._timer is not None:
        watcher._timer.cancel()


class TestWatcherDispatchesPaths:
    def test_changed_paths_are_dispatched_after_the_first_pass(self) -> None:
        """given the session already had its full scan, when files change, then
        the watcher reindexes those paths and runs no full scan."""
        orch = _RecordingOrchestrator()
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False

        _run_pass(watcher, "a.md", "b.md")

        assert orch.calls == [("paths", {"a.md", "b.md"})]

    def test_first_pass_of_a_session_is_a_full_scan(self) -> None:
        """given a server that just started, when the first change arrives,
        then the pass is a full scan — it also picks up what changed while the
        server was down — and the passes after it are path-level."""
        orch = _RecordingOrchestrator()
        watcher = DocumentWatcher(lambda: orch)

        _run_pass(watcher, "a.md")
        _run_pass(watcher, "b.md")

        assert orch.calls == [("all", False), ("paths", {"b.md"})]

    def test_full_scan_that_could_not_run_stays_due(self) -> None:
        """given the first pass found a reindex already running, when the next
        pass comes, then it is still a full scan."""
        orch = _RecordingOrchestrator(busy=True)
        watcher = DocumentWatcher(lambda: orch)

        _run_pass(watcher, "a.md")
        orch.busy = False
        _run_pass(watcher)

        assert orch.calls == [("all", False), ("all", False)]

    def test_large_batch_is_a_full_scan(self) -> None:
        """given more changed paths than a burst the observer reliably reports,
        when the pass runs, then it is a full scan, which does not depend on
        every event having arrived."""
        orch = _RecordingOrchestrator()
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False

        _run_pass(watcher, *(f"f{i}.md" for i in range(DocumentWatcher._FULL_SCAN_BATCH + 1)))
        _run_pass(watcher, "a.md")

        assert orch.calls == [("all", False), ("paths", {"a.md"})]


class TestIndexPaths:
    """index_paths() against a real orchestrator (real Chroma + real BM25)."""

    def _write(self, docs_dir: Path, name: str, text: str) -> Path:
        p = docs_dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_indexes_only_the_given_path(self, isolated_orchestrator, monkeypatch) -> None:
        """given two indexed docs, when index_paths() targets only one, then
        only that one is processed."""
        docs_dir = Path(srv.config.documents_dir)
        self._write(docs_dir, "a.md", "# a\n")
        self._write(docs_dir, "b.md", "# b\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)

        self._write(docs_dir, "a.md", "# a changed\n")
        stats = orch.index_paths([str(docs_dir / "a.md")])

        assert stats["total_files"] == 1, f"only one path may be processed: {stats}"
        assert stats["updated"] == 1, f"a.md must be re-embedded: {stats}"
        assert stats["skipped"] == 0

    def test_deleted_path_prunes_chunks(self, isolated_orchestrator, monkeypatch) -> None:
        """given an indexed doc whose file is gone, when index_paths() receives
        the path, then the chunks are removed from the index."""
        docs_dir = Path(srv.config.documents_dir)
        p = self._write(docs_dir, "gone.md", "# gone\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        assert str(p.resolve()) in orch._source_to_docid
        p.unlink()

        stats = orch.index_paths([str(p)])

        assert stats["deleted"] == 1, f"deleted doc must be pruned: {stats}"
        assert str(p.resolve()) not in orch._source_to_docid
        assert not orch._indexed_docs

    def test_directory_path_falls_back_to_full_scan(self, isolated_orchestrator, monkeypatch) -> None:
        """given a directory in the batch, when index_paths() runs, then it
        falls back to a full scan (Windows reports only the folder)."""
        docs_dir = Path(srv.config.documents_dir)
        (docs_dir / "sub").mkdir(exist_ok=True)
        self._write(docs_dir / "sub", "x.md", "# x\n")
        self._write(docs_dir, "y.md", "# y\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)

        stats = orch.index_paths([str(docs_dir / "sub")])

        assert stats["total_files"] == 2, f"full scan must cover the corpus: {stats}"

    def test_path_outside_documents_dir_is_ignored(self, isolated_orchestrator, monkeypatch, tmp_path) -> None:
        """given a path outside documents_dir, when index_paths() runs, then
        nothing is indexed and nothing errors."""
        orch = isolated_orchestrator

        stats = orch.index_paths([str(tmp_path / "elsewhere.md")])

        assert stats["total_files"] == 0
        assert stats["errors"] == 0

    def test_excluded_path_is_removed_when_indexed(self, isolated_orchestrator, monkeypatch) -> None:
        """given a doc indexed before its folder became excluded, when
        index_paths() receives it, then it is pruned instead of re-embedded."""
        docs_dir = Path(srv.config.documents_dir)
        temp = docs_dir / "Temp"
        temp.mkdir(exist_ok=True)
        p = self._write(temp, "temp.md", "# temp\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)  # no exclude yet — temp.md gets indexed
        assert str(p.resolve()) in orch._source_to_docid
        monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])

        stats = orch.index_paths([str(p)])

        assert stats["deleted"] == 1, f"indexed doc inside excluded folder must be pruned: {stats}"
        assert str(p.resolve()) not in orch._source_to_docid

    def test_skipped_path_is_not_an_error(self, isolated_orchestrator, monkeypatch) -> None:
        """given an unindexed path that no longer exists, when index_paths()
        runs, then it is skipped without errors."""
        orch = isolated_orchestrator

        stats = orch.index_paths([str(Path(srv.config.documents_dir) / "never.md")])

        assert stats["errors"] == 0
        assert stats["deleted"] == 0

    def test_deleted_folder_prunes_everything_indexed_below_it(self, isolated_orchestrator) -> None:
        """given a folder with indexed docs that was deleted as a whole, when
        index_paths() receives only the folder path, then every doc below it
        is pruned and its sibling stays indexed."""
        docs_dir = Path(srv.config.documents_dir)
        (docs_dir / "proj" / "deep").mkdir(parents=True)
        self._write(docs_dir / "proj", "a.md", "# a\n")
        self._write(docs_dir / "proj" / "deep", "b.md", "# b\n")
        keep = self._write(docs_dir, "project-notes.md", "# keep\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        assert len(orch._indexed_docs) == 3
        shutil.rmtree(docs_dir / "proj")

        stats = orch.index_paths([str(docs_dir / "proj")])

        assert stats["deleted"] == 2, f"both docs below the folder must be pruned: {stats}"
        assert list(orch._source_to_docid) == [str(keep.resolve())]
        assert orch.collection.count() == 1

    def test_unavailable_documents_dir_prunes_nothing(self, isolated_orchestrator, tmp_path) -> None:
        """given documents_dir went away (unplugged drive, unmounted share),
        when index_paths() receives an indexed path, then nothing is pruned."""
        docs_dir = Path(srv.config.documents_dir)
        p = self._write(docs_dir, "a.md", "# a\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        docs_dir.rename(tmp_path / "unplugged")

        stats = orch.index_paths([str(p)])

        assert stats["deleted"] == 0, f"an unavailable folder is not a deletion: {stats}"
        assert len(orch._indexed_docs) == 1
        assert orch.collection.count() == 1
        assert stats["errors"] == 1

    def test_unreadable_path_is_not_treated_as_deleted(self, isolated_orchestrator, monkeypatch) -> None:
        """given an indexed file that cannot be read right now, when
        index_paths() receives it, then its chunks are kept."""
        docs_dir = Path(srv.config.documents_dir)
        p = self._write(docs_dir, "locked.md", "# locked\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        real_stat = Path.stat

        def stat(self, *args, **kwargs):
            if self.name == "locked.md":
                raise PermissionError(13, "Access is denied", str(self))
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)

        stats = orch.index_paths([str(p)])

        assert stats["deleted"] == 0, f"unreadable is not evidence of deletion: {stats}"
        assert len(orch._indexed_docs) == 1

    @pytest.mark.parametrize("name", ["notes.bak", "~$report.md", ".rag-pending-x.md"])
    def test_file_the_scan_would_not_index_is_skipped(self, isolated_orchestrator, name) -> None:
        """given a path the full scan never indexes (unsupported format, Office
        owner file, staging file), when index_paths() receives it, then it is
        skipped without being indexed and without an error."""
        docs_dir = Path(srv.config.documents_dir)
        p = self._write(docs_dir, name, "# content\n")
        orch = isolated_orchestrator

        stats = orch.index_paths([str(p)])

        assert stats["errors"] == 0, f"not an error: {stats}"
        assert stats["indexed"] == 0
        assert not orch._indexed_docs

    def test_resume_checkpoint_survives_a_partial_pass(self, isolated_orchestrator) -> None:
        """given a checkpoint left by an interrupted smart reindex, when
        index_paths() handles one file, then the checkpoint is still there."""
        docs_dir = Path(srv.config.documents_dir)
        p = self._write(docs_dir, "a.md", "# a\n")
        orch = isolated_orchestrator
        orch._reindex_progress["operation"] = "smart_reindex"
        orch._checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        orch._checkpoint_file.write_text("{}", encoding="utf-8")

        orch.index_paths([str(p)])

        assert orch._checkpoint_file.exists(), "a partial pass must not clear the resume checkpoint"


class TestWatcherFolderEvents:
    """Folder deletes and moves, which do not always come with one event per file."""

    def _watcher(self, tmp_path: Path, monkeypatch, indexed: list[Path]) -> DocumentWatcher:
        monkeypatch.setattr(srv.config, "documents_dir", tmp_path)
        monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])
        monkeypatch.setattr(srv.config, "supported_formats", {".md"})
        orch = MagicMock()
        orch._source_to_docid = {str(p.resolve()): f"doc-{i}" for i, p in enumerate(indexed)}
        return DocumentWatcher(orchestrator_getter=lambda: orch)

    def test_deleted_folder_reported_as_a_file_is_scheduled(self, tmp_path, monkeypatch) -> None:
        """given Windows reports a deleted folder as a plain file event, when
        documents were indexed below it, then the folder path is scheduled."""
        watcher = self._watcher(tmp_path, monkeypatch, [tmp_path / "proj" / "a.md"])

        watcher.on_deleted(SimpleNamespace(is_directory=False, src_path=str(tmp_path / "proj")))

        assert watcher._pending_paths == {str(tmp_path / "proj")}

    def test_deleted_folder_reported_as_a_directory_is_scheduled(self, tmp_path, monkeypatch) -> None:
        """given a directory delete event, when documents were indexed below
        it, then the folder path is scheduled."""
        watcher = self._watcher(tmp_path, monkeypatch, [tmp_path / "proj" / "a.md"])

        watcher.on_deleted(SimpleNamespace(is_directory=True, src_path=str(tmp_path / "proj")))

        assert watcher._pending_paths == {str(tmp_path / "proj")}

    def test_deleted_path_with_nothing_indexed_below_is_ignored(self, tmp_path, monkeypatch) -> None:
        """given a deleted unsupported file or an empty folder, when nothing
        was indexed below it, then no reindex is scheduled."""
        watcher = self._watcher(tmp_path, monkeypatch, [tmp_path / "project-notes.md"])

        watcher.on_deleted(SimpleNamespace(is_directory=False, src_path=str(tmp_path / "proj")))
        watcher.on_deleted(SimpleNamespace(is_directory=True, src_path=str(tmp_path / "build")))
        watcher.on_deleted(SimpleNamespace(is_directory=False, src_path=str(tmp_path / "Temp" / "x.lock")))

        assert not watcher._pending_paths

    def test_moved_folder_schedules_the_old_path_and_never_a_full_scan(self, tmp_path, monkeypatch) -> None:
        """given a folder move, when documents were indexed below the old
        path, then the old path is scheduled for pruning and the new folder
        is left to the per-file events (no full scan for a folder rename)."""
        watcher = self._watcher(tmp_path, monkeypatch, [tmp_path / "old" / "a.md"])
        event = SimpleNamespace(is_directory=True, src_path=str(tmp_path / "old"), dest_path=str(tmp_path / "new"))

        watcher.on_moved(event)

        assert watcher._pending_paths == {str(tmp_path / "old")}


class TestWatcherPathsE2E:
    def test_observer_reindexes_only_the_changed_file(self, isolated_orchestrator, monkeypatch) -> None:
        from watchdog.observers.polling import PollingObserver

        monkeypatch.setattr(srv.config, "supported_formats", {".md"})
        docs_dir = Path(srv.config.documents_dir)
        a = docs_dir / "a.md"
        b = docs_dir / "b.md"
        a.write_text("# a\n", encoding="utf-8")
        b.write_text("# b\n", encoding="utf-8")
        orch = isolated_orchestrator
        orch.index_all(force=True)

        dispatched: dict = {}
        real_index_paths = orch.index_paths

        def spy_index_paths(paths):
            dispatched["paths"] = set(paths)
            return real_index_paths(paths)

        orch.index_paths = spy_index_paths  # type: ignore[method-assign]

        watcher = DocumentWatcher(lambda: orch, debounce_seconds=0.3)
        watcher._full_scan_due = False  # the session's full scan already ran
        observer = PollingObserver(timeout=0.2)
        observer.schedule(watcher, str(docs_dir), recursive=True)
        observer.start()
        try:
            a.write_text("# a changed, and longer\n", encoding="utf-8")

            def reindexed() -> bool:
                # Looked up on every poll: the entry is replaced while this thread reads.
                doc_id = orch._source_to_docid.get(str(a.resolve()))
                return orch._indexed_docs.get(doc_id, {}).get("file_size") == a.stat().st_size

            assert _wait_until(reindexed), "changed file must be re-embedded with fresh metadata"
            assert dispatched.get("paths") == {str(a.resolve())}, (
                f"watcher must dispatch only the changed path, got {dispatched}"
            )
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_observer_prunes_a_deleted_folder(self, isolated_orchestrator, monkeypatch) -> None:
        """given a folder with indexed documents, when it is deleted from disk,
        then its documents leave the index and the sibling stays."""
        from watchdog.observers.polling import PollingObserver

        monkeypatch.setattr(srv.config, "supported_formats", {".md"})
        docs_dir = Path(srv.config.documents_dir)
        (docs_dir / "proj" / "deep").mkdir(parents=True)
        (docs_dir / "proj" / "a.md").write_text("# a\n", encoding="utf-8")
        (docs_dir / "proj" / "deep" / "b.md").write_text("# b\n", encoding="utf-8")
        keep = docs_dir / "keep.md"
        keep.write_text("# keep\n", encoding="utf-8")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        assert len(orch._source_to_docid) == 3

        watcher = DocumentWatcher(lambda: orch, debounce_seconds=0.3)
        watcher._full_scan_due = False  # the session's full scan already ran
        observer = PollingObserver(timeout=0.2)
        observer.schedule(watcher, str(docs_dir), recursive=True)
        observer.start()
        try:
            shutil.rmtree(docs_dir / "proj")
            assert _wait_until(lambda: list(orch._source_to_docid) == [str(keep.resolve())]), (
                f"documents of the deleted folder must be pruned, left {list(orch._source_to_docid)}"
            )
            assert orch.collection.count() == 1
        finally:
            observer.stop()
            observer.join(timeout=2)

    def test_first_event_also_indexes_what_changed_while_the_server_was_down(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given a file edited while no server was watching, when the server
        is back and any other file changes, then the offline edit is indexed
        too — the first pass of a session is a full scan."""
        from watchdog.observers.polling import PollingObserver

        monkeypatch.setattr(srv.config, "supported_formats", {".md"})
        docs_dir = Path(srv.config.documents_dir)
        touched = docs_dir / "touched.md"
        offline = docs_dir / "offline.md"
        touched.write_text("# touched\n", encoding="utf-8")
        offline.write_text("# offline\n", encoding="utf-8")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        offline.write_text("# offline, edited while the server was down\n", encoding="utf-8")

        def indexed_size(path: Path):
            doc_id = orch._source_to_docid.get(str(path.resolve()))
            return orch._indexed_docs.get(doc_id, {}).get("file_size")

        watcher = DocumentWatcher(lambda: orch, debounce_seconds=0.3)
        observer = PollingObserver(timeout=0.2)
        observer.schedule(watcher, str(docs_dir), recursive=True)
        observer.start()
        try:
            touched.write_text("# touched, with the server running\n", encoding="utf-8")
            assert _wait_until(lambda: indexed_size(offline) == offline.stat().st_size), (
                "the edit made while the server was down must be indexed by the first pass"
            )
            assert indexed_size(touched) == touched.stat().st_size
        finally:
            observer.stop()
            observer.join(timeout=2)
