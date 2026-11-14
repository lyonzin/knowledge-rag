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
        self.failed: list = []
        self.truncated = False

    def _result(self) -> dict:
        if self.busy:
            return _stats(skipped_reason="reindex_already_running")
        return _stats(
            indexed=1,
            errors=len(self.failed),
            failed_files=[{"path": path, "reason": "PermissionError: locked"} for path in self.failed],
            failed_files_truncated=self.truncated,
        )

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

    def test_batch_at_the_limit_is_still_path_level(self) -> None:
        """given exactly as many changed paths as the limit, when the pass
        runs, then it reindexes those paths."""
        orch = _RecordingOrchestrator()
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False
        batch = {f"f{i}.md" for i in range(DocumentWatcher._FULL_SCAN_BATCH)}

        _run_pass(watcher, *batch)

        assert orch.calls == [("paths", batch)]


class TestWatcherRetriesWhatFailed:
    """In 5.0.1 every pass was a full scan, so a file that failed was tried
    again at the next event. A path-level pass has to carry it over."""

    def test_path_that_failed_is_taken_again_by_the_next_pass(self) -> None:
        """given a file that could not be indexed (locked by another program),
        when another file changes later, then the pass takes both."""
        orch = _RecordingOrchestrator()
        orch.failed = ["locked.md"]
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False

        _run_pass(watcher, "locked.md")
        orch.failed = []
        _run_pass(watcher, "other.md")
        _run_pass(watcher, "third.md")

        assert orch.calls == [
            ("paths", {"locked.md"}),
            ("paths", {"locked.md", "other.md"}),
            ("paths", {"third.md"}),
        ]

    def test_failed_path_does_not_start_a_pass_on_its_own(self) -> None:
        """given a file that fails every time (a corrupt PDF), when the pass
        ends, then nothing is pending and no timer is armed — it is retried at
        the next change, not in a loop."""
        orch = _RecordingOrchestrator()
        orch.failed = ["corrupt.pdf"]
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False

        watcher._pending_paths.add("corrupt.pdf")
        watcher._do_reindex()

        assert not watcher._pending_paths
        assert watcher._timer is None

    def test_file_that_failed_the_full_scan_is_taken_again(self) -> None:
        """given the first pass of the session, a full scan, in which one file
        failed, when the next change arrives, then that file is tried again."""
        orch = _RecordingOrchestrator()
        orch.failed = ["locked.md"]
        watcher = DocumentWatcher(lambda: orch)

        _run_pass(watcher, "a.md")
        orch.failed = []
        _run_pass(watcher, "b.md")

        assert orch.calls == [("all", False), ("paths", {"locked.md", "b.md"})]

    def test_more_failures_than_the_report_lists_make_the_next_pass_a_full_scan(self) -> None:
        """given a pass with more failed files than the report can list, when
        the next change arrives, then the pass is a full scan, which tries
        every one of them again."""
        orch = _RecordingOrchestrator()
        orch.failed = ["one.md"]
        orch.truncated = True
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False

        _run_pass(watcher, "one.md")
        orch.failed, orch.truncated = [], False
        _run_pass(watcher, "b.md")
        _run_pass(watcher, "c.md")

        assert [kind for kind, _ in orch.calls] == ["paths", "all", "paths"]


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
        # Reported as the full scan reports it, so the watcher takes the pass again.
        assert [entry["path"] for entry in stats["failed_files"]] == [str(docs_dir)]

    def test_unreadable_path_is_not_treated_as_deleted(self, isolated_orchestrator, monkeypatch) -> None:
        """given an indexed file that cannot be read right now, when
        index_paths() receives it, then its chunks are kept and the path is
        reported as failed, so it is tried again."""
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
        assert [Path(entry["path"]).name for entry in stats["failed_files"]] == ["locked.md"]

    def test_document_that_fails_to_parse_is_reported_as_failed(self, isolated_orchestrator, monkeypatch) -> None:
        """given a file whose parsing fails, when index_paths() handles it,
        then the failure is reported with its path, so it is tried again."""
        docs_dir = Path(srv.config.documents_dir)
        p = self._write(docs_dir, "broken.md", "# broken\n")
        orch = isolated_orchestrator
        monkeypatch.setattr(orch.parser, "parse_file", MagicMock(side_effect=ValueError("truncated file")))

        stats = orch.index_paths([str(p)])

        assert [Path(entry["path"]).name for entry in stats["failed_files"]] == ["broken.md"]
        assert not orch._indexed_docs

    @pytest.mark.parametrize("pattern", ["Archive/old", "Archive/old/*", "old"])
    def test_file_below_an_excluded_folder_is_not_indexed(self, isolated_orchestrator, monkeypatch, pattern) -> None:
        """given a pattern that makes the scan skip a folder, however it is
        written, when index_paths() receives a file below that folder, then
        the file is not indexed — as the full scan would not index it."""
        monkeypatch.setattr(srv.config, "exclude_patterns", [pattern])
        docs_dir = Path(srv.config.documents_dir)
        (docs_dir / "Archive" / "old" / "deep").mkdir(parents=True)
        p = self._write(docs_dir / "Archive" / "old" / "deep", "legacy.md", "# legacy\n")
        orch = isolated_orchestrator

        stats = orch.index_paths([str(p)])
        scanned = orch.index_all()

        assert stats["indexed"] == 0 and stats["errors"] == 0, stats
        assert scanned["indexed"] == 0, "the full scan agrees"
        assert not orch._indexed_docs

    def test_pending_repair_makes_the_pass_a_full_scan(self, isolated_orchestrator, monkeypatch) -> None:
        """given a rollback that left a repair pending, when index_paths()
        runs, then the pass is a full scan, the one that applies repairs."""
        docs_dir = Path(srv.config.documents_dir)
        a = self._write(docs_dir, "a.md", "# a\n")
        b = self._write(docs_dir, "b.md", "# b\n")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        b_id = orch._source_to_docid[str(b.resolve())]
        monkeypatch.setattr(orch, "_load_document_repairs", lambda: {str(b.resolve()): [b_id]})

        stats = orch.index_paths([str(a)])

        assert stats["total_files"] == 2, f"a full scan covers the corpus: {stats}"
        assert stats["updated"] == 1, f"the source under repair is reindexed: {stats}"

    def test_event_path_through_a_symlinked_documents_dir(self, isolated_orchestrator, monkeypatch, tmp_path) -> None:
        """given documents_dir is a symlink, when index_paths() receives paths
        as the observer reports them (below the link), then they are indexed,
        excluded and pruned like the same files in a full scan."""
        real = Path(srv.config.documents_dir)
        link = tmp_path / "linked-docs"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not available here")
        monkeypatch.setattr(srv.config, "documents_dir", link)
        monkeypatch.setattr(srv.config, "exclude_patterns", ["Temp/*"])
        (real / "Temp").mkdir()
        self._write(real, "a.md", "# a\n")
        self._write(real / "Temp", "x.md", "# x\n")
        orch = isolated_orchestrator

        stats = orch.index_paths([str(link / "a.md"), str(link / "Temp" / "x.md")])

        assert stats["indexed"] == 1, stats
        assert orch.index_all()["skipped"] == 1, "the full scan sees the same document"
        (real / "a.md").unlink()
        assert orch.index_paths([str(link / "a.md")])["deleted"] == 1
        assert not orch._indexed_docs

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
        orch._indexed_docs = {f"doc-{i}": {"source": str(p)} for i, p in enumerate(indexed)}
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
    """A polling observer against a real orchestrator."""

    def test_observer_reindexes_only_the_changed_file(self, isolated_orchestrator, monkeypatch, start_watcher) -> None:
        """given two indexed files, when one is edited, then the watcher hands
        only that path to index_paths() and its new version is indexed."""
        monkeypatch.setattr(srv.config, "supported_formats", {".md"})
        docs_dir = Path(srv.config.documents_dir)
        a = docs_dir / "a.md"
        b = docs_dir / "b.md"
        a.write_text("# a\n", encoding="utf-8")
        b.write_text("# b\n", encoding="utf-8")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        dispatched: list = []
        index_paths = orch.index_paths
        monkeypatch.setattr(orch, "index_paths", lambda paths: (dispatched.append(set(paths)), index_paths(paths))[1])
        start_watcher(docs_dir, full_scan_due=False)

        a.write_text("# a changed, and longer\n", encoding="utf-8")

        def reindexed() -> bool:
            # Looked up on every poll: the entry is replaced while this thread reads.
            doc_id = orch._source_to_docid.get(str(a.resolve()))
            return orch._indexed_docs.get(doc_id, {}).get("file_size") == a.stat().st_size

        assert _wait_until(reindexed), "changed file must be re-embedded with fresh metadata"
        assert dispatched and all(batch == {str(a)} for batch in dispatched), (
            f"watcher must dispatch only the changed path, got {dispatched}"
        )

    def test_observer_prunes_a_deleted_folder(self, isolated_orchestrator, monkeypatch, start_watcher) -> None:
        """given a folder with indexed documents, when it is deleted from disk,
        then its documents leave the index and the sibling stays."""
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
        start_watcher(docs_dir, full_scan_due=False)

        shutil.rmtree(docs_dir / "proj")

        assert _wait_until(lambda: list(orch._source_to_docid) == [str(keep.resolve())]), (
            f"documents of the deleted folder must be pruned, left {list(orch._source_to_docid)}"
        )
        assert orch.collection.count() == 1

    def test_folder_delete_reported_as_one_event_prunes_its_documents(self, isolated_orchestrator, monkeypatch) -> None:
        """given a folder moved out of the documents folder, which Windows
        reports as one plain delete event for the folder and nothing for the
        files in it, when the watcher handles that event, then the documents
        indexed below the folder are pruned and the sibling stays."""
        monkeypatch.setattr(srv.config, "supported_formats", {".md"})
        docs_dir = Path(srv.config.documents_dir)
        (docs_dir / "proj" / "deep").mkdir(parents=True)
        (docs_dir / "proj" / "a.md").write_text("# a\n", encoding="utf-8")
        (docs_dir / "proj" / "deep" / "b.md").write_text("# b\n", encoding="utf-8")
        keep = docs_dir / "project-notes.md"
        keep.write_text("# keep\n", encoding="utf-8")
        orch = isolated_orchestrator
        orch.index_all(force=True)
        watcher = DocumentWatcher(lambda: orch)
        watcher._full_scan_due = False
        shutil.rmtree(docs_dir / "proj")

        watcher.on_deleted(SimpleNamespace(is_directory=False, src_path=str(docs_dir / "proj")))
        _run_pass(watcher)

        assert list(orch._source_to_docid) == [str(keep.resolve())]
        assert orch.collection.count() == 1

    def test_first_event_also_indexes_what_changed_while_the_server_was_down(
        self, isolated_orchestrator, monkeypatch, start_watcher
    ) -> None:
        """given a file edited while no server was watching, when the server
        is back and any other file changes, then the offline edit is indexed
        too — the first pass of a session is a full scan."""
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

        start_watcher(docs_dir)

        touched.write_text("# touched, with the server running\n", encoding="utf-8")

        assert _wait_until(lambda: indexed_size(offline) == offline.stat().st_size), (
            "the edit made while the server was down must be indexed by the first pass"
        )
        assert _wait_until(lambda: indexed_size(touched) == touched.stat().st_size)
