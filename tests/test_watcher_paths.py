"""Regression tests for GH #297 — the watcher must reindex only changed paths.

``DocumentWatcher._do_reindex()`` collected the exact changed paths in
``_pending_paths`` and then used only their count, calling
``orch.index_all(force=False)`` — a full walk + metadata pass over every
indexed document for every single-file event. "Incremental" meant "don't
re-embed unchanged files", not "only look at changed files".
"""

from __future__ import annotations

import time
from pathlib import Path

from mcp_server import server as srv
from mcp_server.server import DocumentWatcher


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


class TestWatcherDispatchesPaths:
    def test_do_reindex_calls_index_paths_not_index_all(self, monkeypatch) -> None:
        """given pending paths, when the debounce fires, then the watcher calls
        index_paths(pending) and never a full index_all."""
        calls: dict = {}

        class FakeOrch:
            def index_paths(self, paths):
                calls["paths"] = set(paths)
                return _stats(indexed=1)

            def index_all(self, force=False):
                calls["index_all"] = True
                return _stats()

        orch = FakeOrch()
        watcher = DocumentWatcher(lambda: orch, debounce_seconds=0.05)
        watcher._schedule_reindex(str(Path("a.md")))
        watcher._schedule_reindex(str(Path("b.md")))
        time.sleep(0.3)

        assert calls.get("paths") == {str(Path("a.md")), str(Path("b.md"))}
        assert "index_all" not in calls, "watcher must not run a full scan for file events"


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
        observer = PollingObserver()
        observer.schedule(watcher, str(docs_dir), recursive=True)
        observer.start()
        try:
            a.write_text("# a changed\n", encoding="utf-8")
            time.sleep(1.5)
            assert dispatched.get("paths") == {str(a.resolve())}, (
                f"watcher must dispatch only the changed path, got {dispatched}"
            )
            a_id = orch._source_to_docid[str(a.resolve())]
            assert orch._indexed_docs[a_id]["file_size"] == a.stat().st_size, (
                "changed file must be re-embedded with fresh metadata"
            )
        finally:
            observer.stop()
            observer.join(timeout=2)
