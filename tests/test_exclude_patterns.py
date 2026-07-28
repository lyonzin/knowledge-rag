"""Tests for exclude_patterns feature in document ingestion."""

from pathlib import Path

import pytest

from mcp_server.ingestion import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


class TestShouldExclude:
    """Unit tests for DocumentParser._should_exclude()."""

    def test_empty_patterns_excludes_nothing(self):
        path = Path("docs/readme.md")
        assert DocumentParser._should_exclude(path, Path("docs"), []) is False

    def test_matches_simple_name(self):
        """Pattern 'node_modules' should match at any depth."""
        base = Path("/project")
        assert DocumentParser._should_exclude(Path("/project/node_modules/pkg/index.js"), base, ["node_modules"])
        assert DocumentParser._should_exclude(Path("/project/src/node_modules/pkg.js"), base, ["node_modules"])

    def test_matches_dotfile_pattern(self):
        """Pattern '.git' should match .git directory."""
        base = Path("/project")
        assert DocumentParser._should_exclude(Path("/project/.git/config"), base, [".git"])
        assert DocumentParser._should_exclude(Path("/project/sub/.git/HEAD"), base, [".git"])

    def test_matches_glob_extension(self):
        """Pattern '*.tmp' should match files by extension."""
        base = Path("/project")
        assert DocumentParser._should_exclude(Path("/project/notes.tmp"), base, ["*.tmp"])

    def test_no_match_returns_false(self):
        base = Path("/project")
        assert DocumentParser._should_exclude(Path("/project/src/main.py"), base, ["node_modules", ".git"]) is False

    def test_matches_nested_glob(self):
        """Pattern 'drafts/*' should match files inside drafts/."""
        base = Path("/project")
        assert DocumentParser._should_exclude(Path("/project/drafts/wip.md"), base, ["drafts/*"])

    def test_multiple_patterns_any_match(self):
        """Should return True if ANY pattern matches."""
        base = Path("/project")
        patterns = ["node_modules", ".venv", "__pycache__"]
        assert DocumentParser._should_exclude(Path("/project/.venv/lib/site.py"), base, patterns)
        assert DocumentParser._should_exclude(Path("/project/__pycache__/mod.pyc"), base, patterns)

    def test_venv_pattern(self):
        base = Path("/project")
        assert DocumentParser._should_exclude(Path("/project/.venv/lib/python3.11/site.py"), base, [".venv"])
        assert DocumentParser._should_exclude(Path("/project/venv/bin/python"), base, ["venv"]) is True


class TestParseDirectoryExclusion:
    """Integration tests for exclude_patterns in parse_directory()."""

    def test_excludes_files_by_pattern(self, parser, tmp_path):
        """Files matching exclude patterns should not be parsed."""
        from unittest.mock import patch

        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "keep.md").write_text("# Keep this", encoding="utf-8")
        (docs / "skip.tmp").write_text("Skip this", encoding="utf-8")

        with patch("mcp_server.ingestion.config") as mock_cfg:
            mock_cfg.documents_dir = docs
            mock_cfg.supported_formats = [".md", ".tmp"]
            mock_cfg.exclude_patterns = ["*.tmp"]
            mock_cfg.chunk_size = 1000
            mock_cfg.chunk_overlap = 200
            mock_cfg.category_mappings = {}
            mock_cfg.keyword_routes = {}
            # A3.7 + A3.8 gating flags — both default OFF; explicit here so
            # patch("config") does not leak a MagicMock (truthy) into the
            # chunker selection branch in DocumentParser.parse_file.
            mock_cfg.parent_document_enabled = False
            mock_cfg.code_aware_chunking = False

            result = parser.parse_directory(docs)
            sources = [str(d.source.name) for d in result]
            assert "keep.md" in sources
            assert "skip.tmp" not in sources

    def test_excludes_directories(self, parser, tmp_path):
        """Directories matching patterns should not be traversed at all."""
        from unittest.mock import patch

        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "good.md").write_text("# Good", encoding="utf-8")

        # Create excluded directory with a file
        excluded = docs / "node_modules" / "pkg"
        excluded.mkdir(parents=True)
        (excluded / "readme.md").write_text("# Should not appear", encoding="utf-8")

        with patch("mcp_server.ingestion.config") as mock_cfg:
            mock_cfg.documents_dir = docs
            mock_cfg.supported_formats = [".md"]
            mock_cfg.exclude_patterns = ["node_modules"]
            mock_cfg.chunk_size = 1000
            mock_cfg.chunk_overlap = 200
            mock_cfg.category_mappings = {}
            mock_cfg.keyword_routes = {}
            # A3.7 + A3.8 gating flags — both default OFF; explicit here so
            # patch("config") does not leak a MagicMock (truthy) into the
            # chunker selection branch in DocumentParser.parse_file.
            mock_cfg.parent_document_enabled = False
            mock_cfg.code_aware_chunking = False

            result = parser.parse_directory(docs)
            sources = [str(d.source.name) for d in result]
            assert "good.md" in sources
            assert "readme.md" not in sources

    def test_no_patterns_indexes_everything(self, parser, tmp_path):
        """Empty exclude_patterns should not change behavior."""
        from unittest.mock import patch

        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "a.md").write_text("# A", encoding="utf-8")
        (docs / "b.md").write_text("# B", encoding="utf-8")

        with patch("mcp_server.ingestion.config") as mock_cfg:
            mock_cfg.documents_dir = docs
            mock_cfg.supported_formats = [".md"]
            mock_cfg.exclude_patterns = []
            mock_cfg.chunk_size = 1000
            mock_cfg.chunk_overlap = 200
            mock_cfg.category_mappings = {}
            mock_cfg.keyword_routes = {}
            # A3.7 + A3.8 gating flags — both default OFF; explicit here so
            # patch("config") does not leak a MagicMock (truthy) into the
            # chunker selection branch in DocumentParser.parse_file.
            mock_cfg.parent_document_enabled = False
            mock_cfg.code_aware_chunking = False

            result = parser.parse_directory(docs)
            assert len(result) == 2
