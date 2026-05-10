"""Locale and encoding versatility tests.

Pillar 4: Versatility — guarantee the parser does not crash or silently
drop content when faced with realistic non-ASCII inputs that 70+ enterprise
users routinely throw at us:

- BOM-prefixed UTF-8 (Windows Notepad / Excel CSV exports)
- Filenames with accents and spaces (Brazilian Portuguese, French, German)
- Mixed-script content (Cyrillic, Greek, CJK)
- LF / CRLF line endings on the same input

Excludes here are deliberate:
- We do NOT test embedding behavior on non-English text — that belongs to
  the multilingual feature roadmap, not a regression test.
- We do NOT exercise the FastEmbed model — these tests run offline.
"""

from __future__ import annotations

import pytest

from mcp_server.ingestion import DocumentParser

# ---------------------------------------------------------------------------
# UTF-8 BOM handling
# ---------------------------------------------------------------------------


def test_markdown_with_utf8_bom_does_not_crash(tmp_path):
    """Windows-saved markdown carries a BOM. Parser must at least not crash.

    Caveat: the BOM currently leaks into the first chunk content; that is
    suboptimal but not a regression of any prior release. The strict
    assertion lives below as xfail to track the followup.
    """
    bom = b"\xef\xbb\xbf"
    path = tmp_path / "with_bom.md"
    path.write_bytes(bom + b"# Title\n\n## Section\n\nContent body.\n")

    parser = DocumentParser()
    doc = parser.parse_file(path)
    assert doc is not None
    assert doc.chunks, "parser produced no chunks for BOM-prefixed markdown"


@pytest.mark.xfail(
    reason=(
        "Parser does not strip leading UTF-8 BOM from chunk content. "
        "Tracked as a followup quality improvement; not a regression of any prior release."
    ),
    strict=False,
)
def test_markdown_strips_utf8_bom_from_chunk(tmp_path):
    """When the followup fix lands: the BOM must NOT appear inside chunk text."""
    bom = b"\xef\xbb\xbf"
    path = tmp_path / "with_bom.md"
    path.write_bytes(bom + b"# Title\n\n## Section\n\nContent body.\n")

    parser = DocumentParser()
    doc = parser.parse_file(path)
    assert doc is not None
    first_chunk_text = doc.chunks[0].content
    assert "﻿" not in first_chunk_text


def test_csv_with_utf8_bom_parses(tmp_path):
    """Excel-exported CSVs always include a BOM. Parser must accept."""
    bom = "﻿"
    path = tmp_path / "with_bom.csv"
    path.write_text(bom + "Name,Role\nAlice,Admin\nBob,User\n", encoding="utf-8")

    parser = DocumentParser()
    doc = parser.parse_file(path)
    assert doc is not None
    assert doc.chunks


# ---------------------------------------------------------------------------
# Non-ASCII filenames and paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "documento_em_portugues.md",
        "rapport_français.md",
        "Bericht_größe.md",
        "файл.md",  # Cyrillic
        "file with spaces.md",
        "file-with-emoji-🚀.md",
    ],
)
def test_non_ascii_filename_parses(tmp_path, filename):
    """Filenames with accents, spaces, and unicode must not break the parser."""
    path = tmp_path / filename
    try:
        path.write_text("# Heading\n\nBody.\n", encoding="utf-8")
    except OSError as exc:
        pytest.skip(f"Filesystem rejects this filename ({exc}) — host-specific limitation, not a parser bug")

    parser = DocumentParser()
    doc = parser.parse_file(path)
    assert doc is not None
    assert doc.chunks


def test_path_with_accented_subdirectory(tmp_path):
    """Subdirectory names with accents must not affect chunk extraction."""
    subdir = tmp_path / "categoría_segurança"
    subdir.mkdir()
    path = subdir / "test.md"
    path.write_text("# Title\n\nContent.\n", encoding="utf-8")

    parser = DocumentParser()
    doc = parser.parse_file(path)
    assert doc is not None
    assert doc.chunks


# ---------------------------------------------------------------------------
# Mixed-script content
# ---------------------------------------------------------------------------


def test_markdown_with_mixed_scripts(tmp_path):
    """Real-world docs frequently mix English with other scripts."""
    content = """# Multi-script document

## Portuguese
Atenção: este é um teste com acentos e cedilha (ção, ã, ñ).

## Russian (Cyrillic)
Это тестовый документ.

## Greek
Αυτό είναι ένα δοκιμαστικό έγγραφο.

## CJK
これは日本語のテストドキュメントです。
这是中文测试文档。

## Mixed inline
The token "São Paulo" should survive chunk boundaries cleanly.
"""
    path = tmp_path / "mixed.md"
    path.write_text(content, encoding="utf-8")

    parser = DocumentParser()
    doc = parser.parse_file(path)
    assert doc is not None
    assert doc.chunks
    # Reassemble all chunks and verify no character was silently mangled
    full = "".join(c.content for c in doc.chunks)
    assert "São Paulo" in full
    assert "Это" in full
    assert "Αυτό" in full
    assert "日本語" in full


# ---------------------------------------------------------------------------
# Line ending tolerance
# ---------------------------------------------------------------------------


def test_crlf_and_lf_produce_same_chunks(tmp_path):
    """Windows (CRLF) and Unix (LF) line endings must yield equivalent chunks."""
    body = "# Title\n\n## Section A\n\nFirst paragraph.\n\n## Section B\n\nSecond paragraph.\n"

    lf_path = tmp_path / "lf.md"
    lf_path.write_bytes(body.encode("utf-8"))

    crlf_path = tmp_path / "crlf.md"
    crlf_path.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))

    parser = DocumentParser()
    lf_doc = parser.parse_file(lf_path)
    crlf_doc = parser.parse_file(crlf_path)

    assert lf_doc is not None and crlf_doc is not None
    assert len(lf_doc.chunks) == len(crlf_doc.chunks), (
        "CRLF and LF inputs produced different chunk counts — "
        "Windows users would see different indexing results from Linux users"
    )
