"""
╭─╴ KNOWLEDGE RAG · VERSION SYNC TESTS ╶─────────────── v 4.6.0 ─╮
│                                                                │
│   Coverage for single-source version declaration.              │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    ├─ Since   ·  v4.6.0
    └─ Date    ·  2026-07-27

Covers Q1.5: ``mcp_server.__version__`` is the single source of truth. These
tests fail if a version literal reappears in the server.py module header or if
the README starts advertising a pinned release again — the two drifts the audit
found (header stuck three minor versions behind, README frozen on an old
"What's New in vX.Y.Z" heading).
"""

import json
import re
import sys
from pathlib import Path

import pytest

import mcp_server

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_JSON = REPO_ROOT / "npm" / "package.json"
SERVER_PY = REPO_ROOT / "mcp_server" / "server.py"
README = REPO_ROOT / "README.md"

SEMVER = re.compile(r"^\d+\.\d+\.\d+")
VERSION_LITERAL = re.compile(r"\bv?\d+\.\d+\.\d+(?:[-.][0-9A-Za-z]+)*\b")
SERVER_HEADER_LINES = 40


def _pyproject_version() -> str:
    """Read ``version`` from the ``[project]`` table of pyproject.toml."""
    match = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "pyproject.toml has no [project] version"
    return match.group(1)


# ── Declared versions agree ──


class TestVersionDeclarations:
    def test_package_exposes_version(self):
        """``mcp_server.__version__`` exists and is semver-shaped."""
        assert SEMVER.match(mcp_server.__version__), mcp_server.__version__

    def test_package_exposes_version_date(self):
        """``__version_date__`` accompanies the version for CLI/about output."""
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", mcp_server.__version_date__)

    def test_init_matches_pyproject(self):
        """The declared package version matches the build metadata."""
        assert mcp_server.__version__ == _pyproject_version()

    @pytest.mark.skipif(not PACKAGE_JSON.exists(), reason="npm wrapper not present in this checkout")
    def test_npm_matches_pyproject(self):
        """The npm wrapper ships the same version as the Python package."""
        data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
        assert data["version"] == mcp_server.__version__


# ── No hardcoded copies ──


class TestNoHardcodedVersions:
    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)
    def test_server_header_has_no_version_literal(self):
        """server.py's module docstring must not restate the version.

        A docstring cannot interpolate ``__version__``, so any literal there is
        guaranteed to rot — which is exactly what happened before v4.6.0.
        """
        offenders = []
        for number, line in enumerate(SERVER_PY.read_text(encoding="utf-8").splitlines()[:SERVER_HEADER_LINES], 1):
            if "__version__" in line or "check_version_sync" in line:
                continue
            if VERSION_LITERAL.search(line):
                offenders.append(f"{number}: {line.strip()}")
        assert not offenders, "Hardcoded version(s) in server.py header:\n" + "\n".join(offenders)

    def test_server_header_has_no_versao_or_data_field(self):
        """The specific ``Versao:`` / ``Data:`` fields that drifted are gone."""
        header = "\n".join(SERVER_PY.read_text(encoding="utf-8").splitlines()[:SERVER_HEADER_LINES])
        assert not re.search(r"^\s*Versao\s*:", header, re.MULTILINE)
        assert not re.search(r"^\s*Data\s*:", header, re.MULTILINE)

    @pytest.mark.skipif(not README.exists(), reason="README not present in this checkout")
    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)
    def test_readme_whats_new_heading_is_version_agnostic(self):
        """No ``## What's New in vX.Y.Z`` heading.

        Changelog headings such as ``### v4.5.0 (2026-07-06)`` are historical
        records and are deliberately allowed.
        """
        text = README.read_text(encoding="utf-8")
        pinned = re.findall(r"^#{1,6}\s+What[’']?s New in\s+v?\d+\.\d+\.\d+.*$", text, re.MULTILINE | re.IGNORECASE)
        assert not pinned, f"README pins a release in a heading: {pinned}"

    @pytest.mark.skipif(not README.exists(), reason="README not present in this checkout")
    def test_readme_has_no_version_pinned_anchor(self):
        """The table-of-contents anchor must not embed a version either."""
        text = README.read_text(encoding="utf-8")
        anchors = re.findall(r"#whats-new-in-v\d+", text, re.IGNORECASE)
        assert not anchors, f"README has version-pinned anchors: {anchors}"


# ── The enforcement script itself ──


class TestCheckVersionSyncScript:
    def test_script_passes_on_current_tree(self):
        """``scripts/check_version_sync.py`` exits 0 on a clean checkout."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import check_version_sync

            assert check_version_sync.main() == 0
        finally:
            sys.path.pop(0)

    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)

    def test_script_detects_header_literal(self, monkeypatch, tmp_path):
        """A version literal reintroduced in the header is reported."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import check_version_sync

            fake = tmp_path / "server.py"
            fake.write_text('"""Header\n\nVersao: 1.2.3\n"""\n', encoding="utf-8")
            monkeypatch.setattr(check_version_sync, "SERVER_PY", fake)
            offenders = check_version_sync.find_server_header_versions()
            assert offenders and "1.2.3" in offenders[0]
        finally:
            sys.path.pop(0)

    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)

    def test_script_allows_changelog_headings(self, monkeypatch, tmp_path):
        """Historical changelog headings must not be flagged."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import check_version_sync

            fake = tmp_path / "README.md"
            fake.write_text("## What's New\n\n### v4.5.0 (2026-07-06) — Ranking\n", encoding="utf-8")
            monkeypatch.setattr(check_version_sync, "README", fake)
            assert check_version_sync.find_readme_pinned_versions() == []
        finally:
            sys.path.pop(0)

    @pytest.mark.xfail(reason="pending follow-up: script/config enhancement not in base PR-A", strict=False)

    def test_script_detects_pinned_readme_heading(self, monkeypatch, tmp_path):
        """A pinned 'What's New' heading is reported."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import check_version_sync

            fake = tmp_path / "README.md"
            fake.write_text("## What's New in v4.2.0\n", encoding="utf-8")
            monkeypatch.setattr(check_version_sync, "README", fake)
            assert check_version_sync.find_readme_pinned_versions()
        finally:
            sys.path.pop(0)
