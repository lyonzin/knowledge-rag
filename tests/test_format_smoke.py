"""Format smoke matrix — every supported parser produces at least one chunk.

Pillar 4: Versatility — guarantee that the text-based formats we
ship parsers for keep producing chunks across every release.

Binary formats (PDF, DOCX, XLSX, PPTX) are exercised by their dedicated
tests in test_ingestion.py — they need real binary fixtures the parser
can stream, which is more involved than a smoke check.

Each fixture in tests/fixtures/formats/ becomes one parametrized test
run. Adding a fixture for a new format here automatically extends the
matrix without touching the test code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server.ingestion import DocumentParser

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "formats"

# Formats expected to parse and yield at least one chunk on the smoke fixture
EXPECTED_FORMATS = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".csv",
    ".c",
    ".h",
    ".cpp",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".xml",
    ".ipynb",
    ".go",
    ".rs",
    ".yaml",
    ".yml",
    ".hujson",
    ".cue",
    ".proto",
    ".rego",
    ".kt",
    ".sql",
    ".sh",
    ".jq",
}

# Extensionless files dispatched by exact filename instead of suffix
EXPECTED_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Tiltfile",
}


@pytest.fixture(
    params=sorted(p for p in FIXTURES_DIR.iterdir() if p.suffix in EXPECTED_FORMATS or p.name in EXPECTED_FILENAMES),
    ids=lambda p: p.suffix or p.name,
)
def fixture_path(request):
    """Yield each format fixture so each format gets its own test run."""
    return request.param


def test_format_parses_to_at_least_one_chunk(fixture_path):
    """Every supported text format must yield ``chunks`` after parsing."""
    parser = DocumentParser()
    doc = parser.parse_file(fixture_path)
    assert doc is not None, f"{fixture_path.name}: parse returned None on smoke fixture"
    assert doc.chunks, f"{fixture_path.name}: parser produced zero chunks"
    # Every chunk should carry non-empty content
    for idx, chunk in enumerate(doc.chunks):
        assert chunk.content.strip(), f"{fixture_path.name}: chunk #{idx} content is empty/whitespace"


def test_all_text_formats_have_a_smoke_fixture():
    """Releasing without a smoke fixture for every supported text format is a regression."""
    files = [p for p in FIXTURES_DIR.iterdir() if p.is_file()]
    present_suffixes = {p.suffix for p in files}
    present_names = {p.name for p in files}
    missing = sorted(EXPECTED_FORMATS - present_suffixes) + sorted(EXPECTED_FILENAMES - present_names)
    assert not missing, (
        f"Missing smoke fixture(s) under tests/fixtures/formats/ for: {missing}. "
        f"Add one short example file per missing format to keep the matrix complete."
    )


def test_format_metadata_attached(fixture_path):
    """Parsed document must carry its source path and detected format (suffix or filename)."""
    parser = DocumentParser()
    doc = parser.parse_file(fixture_path)
    assert doc is not None
    assert doc.source == fixture_path
    expected = fixture_path.suffix or fixture_path.name
    assert doc.format == expected, f"{fixture_path.name}: doc.format='{doc.format}' != '{expected}'"
