"""Regression tests for GH #296 — "folder/*" exclude patterns must prune the walk.

``DocumentParser.iter_files()`` prunes excluded directories with
``_should_exclude(Path(root) / d, ...)``. For a pattern like ``Temp/*`` the
directory's own relative path is ``Temp``, and ``fnmatch("Temp", "Temp/*")``
is False — so the walk descended into the whole excluded tree and filtered
every file one by one. Files were still excluded (correctness), but every
scan paid the cost of walking venvs, model caches and TEMP folders in full.
"""

from pathlib import Path

import pytest

import mcp_server.ingestion as ingestion
from mcp_server.ingestion import DocumentParser


def test_folder_star_pattern_prunes_the_walk(tmp_path, monkeypatch) -> None:
    """given an excluded Big/ tree with nested dirs, when iter_files() scans,
    then the walk never enters Big/ and only non-excluded files are yielded."""
    docs = tmp_path / "docs"
    (docs / "Big" / "nested" / "deep").mkdir(parents=True)
    (docs / "Big" / "a.md").write_text("# a")
    (docs / "Big" / "nested" / "b.md").write_text("# b")
    (docs / "keep.md").write_text("# keep")

    monkeypatch.setattr(ingestion.config, "documents_dir", docs)
    monkeypatch.setattr(ingestion.config, "exclude_patterns", ["Big/*"])
    monkeypatch.setattr(ingestion.config, "supported_formats", {".md"})

    visited: list[str] = []
    real_walk = ingestion.walk_document_paths

    def spy(directory):
        for root, dirs, files in real_walk(directory):
            visited.append(str(Path(root).relative_to(directory)))
            yield root, dirs, files

    monkeypatch.setattr(ingestion, "walk_document_paths", spy)

    files = [str(f.relative_to(docs)) for f in DocumentParser().iter_files()]

    assert files == ["keep.md"], f"excluded files must not be yielded, got {files}"
    assert all(not v.startswith("Big") for v in visited), (
        f"walk must not descend into the excluded folder, visited {visited}"
    )


@pytest.mark.parametrize(
    ("pattern", "filename"),
    [
        ("notes*/*", "notes.md"),
        ("_*/*", "_index.md"),
        ("*/*", "readme.md"),
        ("drafts*/*", "drafts-2026.md"),
    ],
)
def test_folder_pattern_does_not_exclude_a_root_file_with_a_matching_name(tmp_path, pattern, filename) -> None:
    """given a "<glob>/*" pattern whose glob also matches a file name, when the
    file sits at the root, then it stays indexed — the pattern only covers
    what is inside matching folders."""
    assert not DocumentParser._should_exclude(tmp_path / filename, tmp_path, [pattern])


def test_folder_pattern_keeps_root_files_in_the_scan(tmp_path, monkeypatch) -> None:
    """given the pattern "_*/*", when iter_files() scans, then the folder
    _private/ is pruned and the root file _index.md is still yielded."""
    docs = tmp_path / "docs"
    (docs / "_private").mkdir(parents=True)
    (docs / "_private" / "secret.md").write_text("# secret")
    (docs / "_index.md").write_text("# index")

    monkeypatch.setattr(ingestion.config, "documents_dir", docs)
    monkeypatch.setattr(ingestion.config, "exclude_patterns", ["_*/*"])
    monkeypatch.setattr(ingestion.config, "supported_formats", {".md"})

    files = sorted(str(f.relative_to(docs)) for f in DocumentParser().iter_files())

    assert files == ["_index.md"]


def test_partial_folder_pattern_does_not_prune_the_folder(tmp_path, monkeypatch) -> None:
    """given a pattern that excludes only some files of a folder, when
    iter_files() scans, then the other files in that folder are still yielded."""
    docs = tmp_path / "docs"
    (docs / "Temp").mkdir(parents=True)
    (docs / "Temp" / "scratch.tmp.md").write_text("# scratch")
    (docs / "Temp" / "keep.md").write_text("# keep")

    monkeypatch.setattr(ingestion.config, "documents_dir", docs)
    monkeypatch.setattr(ingestion.config, "exclude_patterns", ["Temp/*.tmp.md"])
    monkeypatch.setattr(ingestion.config, "supported_formats", {".md"})

    files = [f.relative_to(docs).as_posix() for f in DocumentParser().iter_files()]

    assert files == ["Temp/keep.md"]


def test_nested_folder_pattern_prunes_only_that_folder(tmp_path, monkeypatch) -> None:
    """given "docs/drafts/*", when iter_files() scans, then docs/drafts is
    pruned and its sibling docs/final is still walked."""
    base = tmp_path / "base"
    (base / "docs" / "drafts" / "old").mkdir(parents=True)
    (base / "docs" / "final").mkdir(parents=True)
    (base / "docs" / "drafts" / "old" / "a.md").write_text("# a")
    (base / "docs" / "final" / "b.md").write_text("# b")

    monkeypatch.setattr(ingestion.config, "documents_dir", base)
    monkeypatch.setattr(ingestion.config, "exclude_patterns", ["docs/drafts/*"])
    monkeypatch.setattr(ingestion.config, "supported_formats", {".md"})

    visited: list[str] = []
    real_walk = ingestion.walk_document_paths

    def spy(directory):
        for root, dirs, files in real_walk(directory):
            visited.append(Path(root).relative_to(directory).as_posix())
            yield root, dirs, files

    monkeypatch.setattr(ingestion, "walk_document_paths", spy)

    files = [f.relative_to(base).as_posix() for f in DocumentParser().iter_files()]

    assert files == ["docs/final/b.md"]
    assert not any(v.startswith("docs/drafts") for v in visited), f"visited {visited}"
