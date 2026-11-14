"""Regression tests for GH #298 — orphan chunks are never pruned.

Chunks whose ``doc_id`` is missing from ``index_metadata.json`` (a lost or
restored catalogue) stayed in ChromaDB and in the FTS5 index forever:
the path prune iterates only the catalogue, and ``remove_document``
resolves through it too. They kept turning up in search results and in
every BM25 build.

A reindex now removes them, on the evidence the path prune accepts: the
source is gone, is excluded, or was indexed again under another id.
Chunks whose source is still there and merely lost its catalogue entry
are not orphans — the pass registers that document again.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_server import server as srv


def _write(name: str, text: str) -> Path:
    path = Path(srv.config.documents_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _forget(orch, path: Path) -> str:
    """Drop one document from the catalogue while its chunks stay in Chroma."""
    doc_id = orch._source_to_docid.pop(str(path.resolve()))
    del orch._indexed_docs[doc_id]
    orch._save_metadata()
    return doc_id


def _doc_ids_in_chroma(orch) -> set:
    return {meta["doc_id"] for meta in orch.collection.get(include=["metadatas"])["metadatas"]}


def _add_foreign_chunk(orch, source: str) -> None:
    """A chunk written by something else: its doc_id is not in the catalogue."""
    orch.collection.add(
        ids=["foreign_0"],
        documents=["foreign content"],
        embeddings=[[1.0, 1.0]],
        metadatas=[{"doc_id": "foreign", "source": source}],
    )


class TestOrphanChunksAreRemoved:
    def test_source_moved_away(self, isolated_orchestrator) -> None:
        """given chunks whose document left the catalogue and whose file was
        moved, so the path they name no longer exists, when a reindex runs,
        then they are removed and the file is indexed where it is now."""
        orch = isolated_orchestrator
        gone = _write("a.md", "# alpha\n")
        keep = _write("keep.md", "# keep\n")
        orch.index_all()
        orphan_id = _forget(orch, gone)
        moved = gone.rename(_write("elsewhere/placeholder.md", "x").with_name("a.md"))

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 1, stats
        assert orphan_id not in _doc_ids_in_chroma(orch)
        assert str(moved.resolve()) in orch._source_to_docid and str(keep.resolve()) in orch._source_to_docid

    @pytest.mark.parametrize("pattern", ["Temp/*", "Temp", "Temp/sub"])
    def test_source_still_in_place_but_left_out_by_the_scan(self, isolated_orchestrator, monkeypatch, pattern) -> None:
        """given chunks whose document left the catalogue and whose folder was
        excluded afterwards, by any pattern that makes the scan skip it, when
        a reindex runs, then they are removed."""
        orch = isolated_orchestrator
        temp = _write("Temp/sub/a.md", "# alpha\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        orphan_id = _forget(orch, temp)
        monkeypatch.setattr(srv.config, "exclude_patterns", [pattern])

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 1, stats
        assert orphan_id not in _doc_ids_in_chroma(orch)

    def test_source_outside_the_documents_folder(self, isolated_orchestrator, tmp_path) -> None:
        """given chunks that name a file outside documents_dir, when a reindex
        runs, then they are removed."""
        orch = isolated_orchestrator
        _write("keep.md", "# keep\n")
        orch.index_all()
        outside = tmp_path / "elsewhere" / "x.md"
        outside.parent.mkdir()
        outside.write_text("# x\n", encoding="utf-8")
        _add_foreign_chunk(orch, str(outside))

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 1, stats
        assert _doc_ids_in_chroma(orch) == set(orch._indexed_docs)

    @pytest.mark.skipif(os.name != "nt", reason="drive and network roots are a Windows notion")
    @pytest.mark.parametrize("source", [r"\\fileserver\share\notes.md", r"\\.\pipe\notes.md", r"Q:\notes.md"])
    def test_source_on_another_root_is_judged_without_opening_it(
        self, isolated_orchestrator, monkeypatch, source
    ) -> None:
        """given chunks that name a path on a network share, a device or
        another drive, when a reindex runs, then they are removed without the
        path being opened — resolving a share contacts its server."""
        orch = isolated_orchestrator
        _write("keep.md", "# keep\n")
        orch.index_all()
        _add_foreign_chunk(orch, source)
        is_path_within = srv.is_path_within
        touched: list = []

        def recording(base, candidate):
            touched.append(str(candidate))
            return is_path_within(base, candidate)

        monkeypatch.setattr(srv, "is_path_within", recording)

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 1, stats
        assert source not in touched

    def test_source_deleted(self, isolated_orchestrator) -> None:
        """given chunks whose document left the catalogue and whose file was
        deleted, when a reindex runs, then they are removed."""
        orch = isolated_orchestrator
        gone = _write("a.md", "# alpha\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        orphan_id = _forget(orch, gone)
        gone.unlink()

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 1, stats
        assert orphan_id not in _doc_ids_in_chroma(orch)
        assert orch.collection.count() == 1

    def test_source_edited_since_and_indexed_again(self, isolated_orchestrator) -> None:
        """given chunks of a version the catalogue forgot, when the file was
        edited and a reindex indexes the new version, then the old version's
        chunks are removed and the new one is searchable."""
        orch = isolated_orchestrator
        path = _write("a.md", "# alpha\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        old_id = _forget(orch, path)
        path.write_text("# alpha, rewritten and longer\n", encoding="utf-8")
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))

        stats = orch.index_all()

        new_id = orch._source_to_docid[str(path.resolve())]
        assert new_id != old_id
        assert stats["orphan_chunks_removed"] == 1, stats
        assert _doc_ids_in_chroma(orch) == set(orch._indexed_docs)

    def test_removed_chunks_leave_the_search_results(self, isolated_orchestrator, monkeypatch) -> None:
        """given an orphan chunk that a search returns, when a reindex removes
        it, then the same search no longer returns it."""
        monkeypatch.setattr(srv.config, "reranker_enabled", False)
        orch = isolated_orchestrator
        gone = _write("a.md", "zebra crossing notes\n")
        _write("keep.md", "unrelated keep text\n")
        _write("other.md", "another passage here\n")
        orch.index_all()
        assert any(r["filename"] == "a.md" for r in orch.query("zebra", max_results=5))
        _forget(orch, gone)
        gone.unlink()
        # A query that is not in the cache yet, so it goes through retrieval.
        assert any(r["filename"] == "a.md" for r in orch.query("crossing", max_results=5)), "orphan is still served"

        orch.index_all()

        # "zebra" was cached above: the removal must also drop cached results.
        assert not any(r["filename"] == "a.md" for r in orch.query("zebra", max_results=5))
        assert not any(r["filename"] == "a.md" for r in orch.query("crossing notes", max_results=5))

    def test_removal_marks_the_keyword_index_stale(self, isolated_orchestrator, monkeypatch) -> None:
        """given a published BM25 generation that holds an orphan chunk, when
        a reindex removes the chunk, then the generation is marked stale and
        the next keyword search no longer finds the orphan's text."""
        monkeypatch.setattr(srv.config, "reranker_enabled", False)
        orch = isolated_orchestrator
        gone = _write("a.md", "zebra crossing notes\n")
        _write("keep.md", "unrelated keep text\n")
        _write("other.md", "another passage here\n")
        orch.index_all()
        assert orch.query("zebra", max_results=5, hybrid_alpha=0.0), "the keyword index is published"
        _forget(orch, gone)
        gone.unlink()

        orch.index_all()

        assert orch._bm25_stale.is_set()
        assert orch.query("zebra", max_results=5, hybrid_alpha=0.0) == []

    def test_failure_in_the_middle_keeps_what_was_done_and_finishes_the_pass(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given the store fails after the first batch of orphan chunks was
        deleted, when the reindex ends, then that batch is counted, the
        failure is reported, and the pass still saves its catalogue."""
        orch = isolated_orchestrator
        first = _write("a.md", "# alpha\n")
        second = _write("b.md", "# beta\n")
        new = _write("new.md", "# new\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        new.unlink()
        orch.index_all()
        for path in (first, second):
            _forget(orch, path)
            path.unlink()
        new.write_text("# new\n", encoding="utf-8")
        monkeypatch.setattr(orch, "_CHROMA_BATCH_SIZE", 1)
        delete = orch.collection.delete
        calls: list = []

        def failing_second_delete(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("database is locked")
            return delete(**kwargs)

        monkeypatch.setattr(orch.collection, "delete", failing_second_delete)

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 1, stats
        assert stats["errors"] == 1 and "database is locked" in stats["failed_files"][0]["reason"], stats
        assert stats["indexed"] == 1
        saved = srv.json.loads(orch._metadata_file.read_text(encoding="utf-8"))
        assert str(new) in {entry["source"] for entry in saved.values()}, "the pass saved what it indexed"


class TestChunksThatAreNotOrphansAreKept:
    def test_file_in_place_that_the_pass_could_not_index_keeps_its_chunks(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given chunks the catalogue forgot whose file is there but fails to
        parse in this pass, when the reindex ends, then they are kept — a
        failed parse is not evidence that the file is gone."""
        orch = isolated_orchestrator
        broken = _write("broken.md", "# broken\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        orphan_id = _forget(orch, broken)
        parse_file = orch.parser.parse_file

        def failing_parse(filepath, **kwargs):
            if Path(filepath).name == "broken.md":
                raise ValueError("truncated file")
            return parse_file(filepath, **kwargs)

        monkeypatch.setattr(orch.parser, "parse_file", failing_parse)

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0, stats
        assert orphan_id in _doc_ids_in_chroma(orch)

    def test_kept_chunks_do_not_make_every_pass_read_the_whole_collection(
        self, isolated_orchestrator, monkeypatch
    ) -> None:
        """given chunks kept because their file is there but cannot be indexed,
        when the next reindex runs with nothing new, then it does not page
        through the collection again."""
        orch = isolated_orchestrator
        broken = _write("broken.md", "# broken\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        _forget(orch, broken)
        monkeypatch.setattr(orch.parser, "parse_file", MagicMock(side_effect=ValueError("truncated file")))
        orch.index_all()
        get = MagicMock(wraps=orch.collection.get)
        monkeypatch.setattr(orch.collection, "get", get)

        orch.index_all()

        paged = [call for call in get.call_args_list if call.kwargs.get("include") == ["metadatas"]]
        assert paged == []

    def test_relative_source_keeps_its_chunks(self, isolated_orchestrator) -> None:
        """given chunks that name their file by a relative path, which says
        nothing about where it is, when a reindex runs, then they are kept."""
        orch = isolated_orchestrator
        _write("keep.md", "# keep\n")
        orch.index_all()
        _add_foreign_chunk(orch, "notes.md")

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0, stats
        assert orch.collection.count() == 2

    def test_chunk_without_a_source_is_kept(self, isolated_orchestrator) -> None:
        """given chunks with an unknown doc_id and no source, when a reindex
        runs, then they are kept: nothing can be checked."""
        orch = isolated_orchestrator
        _write("keep.md", "# keep\n")
        orch.index_all()
        _add_foreign_chunk(orch, "")

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0, stats
        assert orch.collection.count() == 2

    def test_lost_catalogue_with_the_files_in_place_deletes_nothing(self, isolated_orchestrator, monkeypatch) -> None:
        """given the whole catalogue was lost while every file is unchanged,
        when a reindex runs, then no chunk is deleted at any point and the
        catalogue is rebuilt."""
        orch = isolated_orchestrator
        _write("a.md", "# alpha\n")
        _write("b.md", "# beta\n")
        orch.index_all()
        before = _doc_ids_in_chroma(orch)
        orch._indexed_docs.clear()
        orch._source_to_docid.clear()
        orch._save_metadata()
        deleted: list = []
        delete = orch.collection.delete
        monkeypatch.setattr(orch.collection, "delete", lambda **kwargs: (deleted.append(kwargs), delete(**kwargs)))

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0, stats
        assert deleted == [], "a lost catalogue must not delete the chunks of files that are still there"
        assert _doc_ids_in_chroma(orch) == before == set(orch._indexed_docs)

    def test_unreadable_source_keeps_its_chunks(self, isolated_orchestrator, monkeypatch) -> None:
        """given chunks the catalogue forgot whose file cannot be read right
        now, when a reindex runs, then they are kept."""
        orch = isolated_orchestrator
        locked = _write("locked.md", "# locked\n")
        _write("keep.md", "# keep\n")
        orch.index_all()
        orphan_id = _forget(orch, locked)
        real_stat = Path.stat

        def stat(self, *args, **kwargs):
            if self.name == "locked.md":
                raise PermissionError(13, "Access is denied", str(self))
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0, stats
        assert orphan_id in _doc_ids_in_chroma(orch)

    def test_empty_scan_deletes_nothing(self, isolated_orchestrator) -> None:
        """given the documents folder came back empty (a share that is not
        mounted), when a reindex runs, then no chunk is deleted."""
        orch = isolated_orchestrator
        path = _write("a.md", "# alpha\n")
        orch.index_all()
        _forget(orch, path)
        path.unlink()

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0, stats
        assert orch.collection.count() == 1

    def test_healthy_index_reports_zero(self, isolated_orchestrator) -> None:
        """given a healthy index, when a reindex runs, then nothing is removed."""
        orch = isolated_orchestrator
        _write("a.md", "# alpha\n")
        orch.index_all()

        stats = orch.index_all()

        assert stats["orphan_chunks_removed"] == 0
        assert orch.collection.count() == 1

    def test_full_rebuild_of_a_healthy_index_removes_nothing(
        self, isolated_orchestrator, monkeypatch, tmp_path
    ) -> None:
        """given a healthy index, when a full rebuild runs through the staging
        collection, then every document is still there."""
        monkeypatch.setattr(srv.config, "chroma_dir", tmp_path / "chroma")
        orch = isolated_orchestrator
        _write("a.md", "# alpha\n")
        _write("b.md", "# beta\n")
        orch.index_all()

        stats = orch.nuclear_rebuild()

        assert stats["orphan_chunks_removed"] == 0
        assert orch.collection.count() == 2
