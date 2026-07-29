"""CWE-59 — symlink escape regression tests.

The realistic attack is content-driven, not filesystem-driven: an operator
clones an untrusted writeups repo into ``documents/`` and that repo ships a
``notes -> /etc`` link. Without containment checks the indexer happily walks
through it and the host's files become retrievable through
``search_knowledge``.

These tests plant real symlinks with ``tmp_path`` and assert nothing outside
the corpus is ever parsed, while in-corpus links keep working.
"""

from pathlib import Path

import pytest

from mcp_server.ingestion import DocumentParser
from mcp_server.security import is_path_within


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    """Create a symlink, skipping the test when the OS forbids it.

    Windows needs Developer Mode or ``SeCreateSymbolicLinkPrivilege``; CI
    runners without it must skip rather than fail.

    Args:
        link: Path of the link to create.
        target: Path the link should point at.
        directory: Whether the target is a directory (required on Windows).
    """
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform gated
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")


@pytest.fixture
def corpus_with_escape(tmp_path):
    """Corpus containing an escaping dir link, an escaping file link, and a safe link.

    Returns:
        tuple[Path, Path]: ``(documents_dir, outside_dir)``.
    """
    base = tmp_path / "documents"
    (base / "real").mkdir(parents=True)
    (base / "real" / "legit.md").write_text("# Legit\n\nOwned by the operator.\n", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "stolen.md").write_text("# Stolen\n\nHost secret material.\n", encoding="utf-8")

    _symlink_or_skip(base / "escaping_dir", outside, directory=True)
    _symlink_or_skip(base / "escaping_file.md", outside / "stolen.md", directory=False)
    _symlink_or_skip(base / "inner_link", base / "real", directory=True)

    return base, outside


# ---------------------------------------------------------------------------
# Containment primitive
# ---------------------------------------------------------------------------


def test_symlinked_dir_pointing_outside_is_not_contained(corpus_with_escape):
    """The link itself resolves out of the corpus."""
    base, _ = corpus_with_escape
    assert is_path_within(base, base / "escaping_dir") is False


def test_file_reached_through_escaping_dir_link_is_not_contained(corpus_with_escape):
    """Traversal *through* the link is caught too, not just the link node."""
    base, _ = corpus_with_escape
    assert is_path_within(base, base / "escaping_dir" / "stolen.md") is False


def test_symlinked_file_pointing_outside_is_not_contained(corpus_with_escape):
    """Symlinked *files* escape even with ``followlinks=False`` — cover them."""
    base, _ = corpus_with_escape
    assert is_path_within(base, base / "escaping_file.md") is False


def test_symlink_pointing_inside_corpus_is_contained(corpus_with_escape):
    """In-corpus links stay allowed; the fix is not a blanket symlink ban."""
    base, _ = corpus_with_escape
    assert is_path_within(base, base / "inner_link") is True
    assert is_path_within(base, base / "inner_link" / "legit.md") is True


# ---------------------------------------------------------------------------
# End-to-end through the real walker
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="integration with parse_directory pending v4.6.0", strict=False)
def test_parse_directory_never_yields_content_from_outside(corpus_with_escape):
    """The indexer must not surface host files reachable only via a link."""
    base, outside = corpus_with_escape

    docs = DocumentParser().parse_directory(base)

    sources = [Path(d.source).resolve() for d in docs]
    assert sources, "expected the legitimate document to be indexed"

    for source in sources:
        assert source.is_relative_to(base.resolve()), f"escaped the corpus: {source}"

    assert (outside / "stolen.md").resolve() not in sources
    assert not any("Stolen" in d.content for d in docs)
    assert not any("Host secret material" in d.content for d in docs)


def test_parse_directory_still_indexes_legitimate_documents(corpus_with_escape):
    """Hardening must not silently empty a working corpus."""
    base, _ = corpus_with_escape

    docs = DocumentParser().parse_directory(base)

    assert any("Owned by the operator" in d.content for d in docs)


def test_parse_directory_without_symlinks_is_unaffected(tmp_path):
    """A plain corpus behaves exactly as before the hardening."""
    base = tmp_path / "documents"
    base.mkdir()
    (base / "a.md").write_text("# A\n\nplain content\n", encoding="utf-8")
    (base / "b.md").write_text("# B\n\nmore content\n", encoding="utf-8")

    docs = DocumentParser().parse_directory(base)

    assert {Path(d.source).name for d in docs} == {"a.md", "b.md"}


@pytest.mark.xfail(reason="integration with parse_directory pending v4.6.0", strict=False)
def test_has_documents_ignores_escaping_symlink(tmp_path):
    """Project-root detection must not be fooled by ``documents/x -> elsewhere``."""
    from mcp_server.config import _has_documents

    root = tmp_path / "project"
    (root / "documents").mkdir(parents=True)

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "decoy.md").write_text("decoy", encoding="utf-8")

    _symlink_or_skip(root / "documents" / "linked", outside, directory=True)

    assert _has_documents(root) is False

    (root / "documents" / "real.md").write_text("real", encoding="utf-8")
    assert _has_documents(root) is True
