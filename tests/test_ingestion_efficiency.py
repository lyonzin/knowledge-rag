"""Regression coverage for bounded parsing and lossless chunking."""

from unittest.mock import MagicMock

import pytest

from mcp_server.ingestion import DocumentParser


@pytest.mark.parametrize("length", [1, 199, 200, 201, 500, 1000, 1800])
def test_chunking_does_not_emit_overlap_only_tail(length):
    chunks = DocumentParser(chunk_size=1000, chunk_overlap=200)._chunk_text("x" * length, {})
    assert chunks[-1].end_char == length
    assert len(chunks) == (1 if length <= 1000 else 2)


def test_zero_overlap_is_honored():
    chunks = DocumentParser(chunk_size=100, chunk_overlap=0)._chunk_text("x" * 250, {})
    assert [chunk.start_char for chunk in chunks] == [0, 100, 200]


def test_markdown_code_placeholder_cannot_replace_literal_text():
    text = "## One\n" + "a" * 120 + "\n__CODE_BLOCK_0__\n## Two\n```\n## Code header\n```\n" + "b" * 120
    chunks = DocumentParser()._chunk_markdown(text, {})
    joined = "\n".join(chunk.content for chunk in chunks)
    assert "__CODE_BLOCK_0__" in joined
    assert joined.count("## Code header") == 1


def test_xlsx_always_closes_workbook_on_parser_failure(tmp_path, monkeypatch):
    import mcp_server.ingestion as ingestion

    path = tmp_path / "broken.xlsx"
    path.touch()
    workbook = MagicMock()
    workbook.sheetnames = ["one"]
    workbook.__getitem__.return_value.iter_rows.side_effect = RuntimeError("invalid XML")
    monkeypatch.setattr(ingestion, "HAS_XLSX", True)
    monkeypatch.setattr(ingestion.openpyxl, "load_workbook", lambda *args, **kwargs: workbook)
    with pytest.raises(RuntimeError, match="invalid XML"):
        DocumentParser()._parse_xlsx(path)
    workbook.close.assert_called_once()
