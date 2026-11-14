"""Regression tests for GH #296 — "folder/*" exclude patterns must prune the walk.

``DocumentParser.iter_files()`` prunes excluded directories with
``_should_exclude(Path(root) / d, ...)``. For a pattern like ``Temp/*`` the
directory's own relative path is ``Temp``, and ``fnmatch("Temp", "Temp/*")``
is False — so the walk descended into the whole excluded tree and filtered
every file one by one. Files were still excluded (correctness), but every
scan paid the cost of walking venvs, model caches and TEMP folders in full.
"""

from pathlib import Path

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
