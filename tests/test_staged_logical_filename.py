"""Transactional parsing preserves exact filenames without exposing staging."""

import pytest

from mcp_server import server
from mcp_server.file_transaction import staged_text_file
from mcp_server.ingestion import DocumentParser


@pytest.mark.parametrize("filename", ["Dockerfile", "Makefile", "Tiltfile", "control.py"])
def test_extensionless_crud_uses_logical_format_before_publication(isolated_orchestrator, monkeypatch, filename):
    orch = isolated_orchestrator
    destination = server.config.documents_dir / filename
    parse = orch.parser.parse_file
    observed = []

    def inspect_staging(path, **kwargs):
        observed.append(path)
        assert path != destination and path.name.startswith(".rag-pending-")
        assert not server.DocumentWatcher._is_supported(str(path))
        assert path not in list(orch.parser.iter_files())
        assert destination.read_text() == "Original content" if destination.exists() else len(observed) == 1
        parsed = parse(path, **kwargs)
        if filename == "Tiltfile":
            assert parsed.metadata["language"] == "starlark"
        return parsed

    monkeypatch.setattr(orch.parser, "parse_file", inspect_staging)
    assert orch.add_document_from_content("Original content", filename, "general")["chunks_added"] == 1
    assert orch.update_document_content(filename, "Replacement content")["new_chunks_added"] == 1
    assert destination.read_text() == "Replacement content"
    assert orch.collection.get(include=["documents"])["documents"] == ["Replacement content"]
    assert orch._source_to_docid[str(destination)] == orch.parser._generate_id(destination)
    if filename == "Tiltfile":
        assert parse(destination).metadata["language"] == "starlark"
    assert not list(destination.parent.glob(".rag-pending-*"))


@pytest.mark.parametrize("filename", ["Dockerfile", "Makefile", "Tiltfile"])
def test_extensionless_failed_commit_restores_original(isolated_orchestrator, monkeypatch, filename):
    orch = isolated_orchestrator
    orch.add_document_from_content("Original content", filename, "general")

    def fail_save():
        raise OSError("metadata unavailable")

    monkeypatch.setattr(orch, "_save_metadata", fail_save)
    with pytest.raises(OSError, match="metadata unavailable"):
        orch.update_document_content(filename, "Replacement content")
    assert (server.config.documents_dir / filename).read_text() == "Original content"
    assert orch.collection.get(include=["documents"])["documents"] == ["Original content"]
    assert not list(server.config.documents_dir.glob(".rag-pending-*"))


@pytest.mark.parametrize("filename", ["../Dockerfile", "..\\Dockerfile", "/Dockerfile", "folder/Makefile"])
def test_logical_filename_cannot_redirect_reads(tmp_path, filename):
    physical = tmp_path / ".rag-pending-example"
    physical.write_text("Physical staged content", encoding="utf-8")
    with pytest.raises(ValueError, match="logical filename"):
        DocumentParser().parse_file(physical, logical_filename=filename)


def test_logical_filename_cannot_change_existing_suffix(tmp_path):
    physical = tmp_path / "native.pdf"
    physical.write_text("This must never be parsed as plain text", encoding="utf-8")
    with pytest.raises(ValueError, match="suffix"):
        DocumentParser().parse_file(physical, logical_filename="pretend.txt")


def test_unknown_logical_filename_remains_unsupported(tmp_path):
    with staged_text_file(tmp_path / "UnknownBuildfile", "synthetic text") as staged:
        with pytest.raises(ValueError, match="Unsupported format"):
            DocumentParser().parse_file(staged.path, logical_filename="UnknownBuildfile")
        assert not (tmp_path / "UnknownBuildfile").exists()


def test_legacy_parser_positional_contract_is_unchanged(tmp_path):
    source = tmp_path / "Dockerfile"
    source.write_text("FROM synthetic", encoding="utf-8")
    parsed = DocumentParser().parse_file(source)
    assert parsed is not None and parsed.format == "Dockerfile"
