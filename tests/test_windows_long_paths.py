"""Real Windows regressions: long-path I/O must preserve public source identity."""

import hashlib
import os
from pathlib import Path

import pytest

from mcp_server import server
from mcp_server.filesystem import resolve_path
from mcp_server.ingestion import DocumentParser
from mcp_server.security import PathEscapeError, validate_path_within

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Requires Win32 extended-length file I/O")


def extended(path):
    return Path("\\\\?\\" + str(path))


def create_long_source(base, *, deep_directory=False):
    directory = base
    target_length = 285 if deep_directory else 225
    while len(str(directory)) < target_length:
        directory /= "nested-" + "x" * 36
    name = "source-ação-" + "y" * 55 + ".md"
    source = directory / name
    extended(directory).mkdir(parents=True, exist_ok=True)
    extended(source).write_text("# Long Windows source\n\nUnique retrieval content.\n", encoding="utf-8")
    assert len(str(source)) > 260
    return source


def test_long_source_parses_without_changing_source_or_identifier(tmp_path):
    source = create_long_source(tmp_path)
    status = extended(source).stat()
    expected_id = hashlib.sha256(f"{source}:{status.st_mtime}:{status.st_size}".encode()).hexdigest()[:16]

    document = DocumentParser().parse_file(source)

    assert document.source == source
    assert document.id == expected_id
    assert document.content.startswith("# Long Windows source")


def test_discovery_reaches_long_directories_and_keeps_normal_paths(tmp_path, monkeypatch):
    source = create_long_source(tmp_path, deep_directory=True)
    monkeypatch.setattr(server.config, "documents_dir", tmp_path)
    monkeypatch.setattr(server.config, "supported_formats", [".md"])
    monkeypatch.setattr(server.config, "exclude_patterns", [])

    discovered = list(DocumentParser().iter_files())

    assert discovered == [source]
    assert not str(discovered[0]).startswith("\\\\?\\")


def test_long_source_indexes_skips_unchanged_and_replaces_on_force(isolated_orchestrator, monkeypatch):
    orchestrator = isolated_orchestrator
    source = create_long_source(server.config.documents_dir)
    first = orchestrator.index_all(force=True)
    original_ids = orchestrator.collection.get()["ids"]
    parser = orchestrator.parser.parse_file
    monkeypatch.setattr(orchestrator.parser, "parse_file", lambda path: pytest.fail("Unchanged source was parsed"))

    unchanged = orchestrator.index_all()
    monkeypatch.setattr(orchestrator.parser, "parse_file", parser)
    forced = orchestrator.index_all(force=True)

    assert first["indexed"] == 1 and first["errors"] == 0
    assert unchanged["skipped"] == 1 and unchanged["errors"] == 0
    assert forced["updated"] == 1 and forced["errors"] == 0
    assert orchestrator.collection.get()["ids"] == original_ids
    assert {entry["source"] for entry in orchestrator.collection.get()["metadatas"]} == {str(source)}


def test_long_source_still_rejects_ntfs_alternate_stream(tmp_path):
    source = create_long_source(tmp_path)
    with pytest.raises(PathEscapeError, match="alternate data stream"):
        validate_path_within(tmp_path, str(source) + ":hidden")


def test_long_symlink_cannot_escape_containment_or_discovery(tmp_path, monkeypatch):
    base = tmp_path / "documents"
    source = create_long_source(base, deep_directory=True)
    outside = tmp_path / "private.md"
    outside.write_text("outside source must not be read", encoding="utf-8")
    link = source.parent / "outside.md"
    try:
        extended(link).symlink_to(outside)
    except OSError as error:
        if error.winerror == 1314:
            pytest.skip("Windows symlinks require Developer Mode or the symbolic-link privilege")
        raise
    monkeypatch.setattr(server.config, "documents_dir", base)
    monkeypatch.setattr(server.config, "supported_formats", [".md"])
    monkeypatch.setattr(server.config, "exclude_patterns", [])

    with pytest.raises(PathEscapeError, match="escapes"):
        validate_path_within(base, link)
    assert list(DocumentParser().iter_files()) == [source]


@pytest.mark.parametrize("directory_link", [False, True])
def test_short_alias_to_long_target_keeps_normal_contained_identity(tmp_path, directory_link):
    source = create_long_source(tmp_path, deep_directory=True)
    target = source.parent if directory_link else source
    alias = tmp_path / ("alias" if directory_link else "alias.md")
    try:
        extended(alias).symlink_to(extended(target), target_is_directory=directory_link)
    except OSError as error:
        if error.winerror == 1314:
            pytest.skip("Windows symlinks require Developer Mode or the symbolic-link privilege")
        raise

    assert resolve_path(alias) == target
    assert validate_path_within(tmp_path, alias) == target
