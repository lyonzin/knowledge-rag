"""CWE-22 — path traversal regression tests.

Proves that ``validate_path_within`` refuses every escape shape an MCP client
could send through ``add_document`` / ``update_document`` / ``remove_document``
/ ``get_document``, while still accepting the legitimate relative and absolute
forms those tools are documented to take.
"""

import os
from pathlib import Path

import pytest

from mcp_server.security import PathEscapeError, is_path_within, validate_path_within


@pytest.fixture
def corpus(tmp_path):
    """A documents_dir with one real file and a sibling directory outside it."""
    base = tmp_path / "documents"
    (base / "security").mkdir(parents=True)
    (base / "security" / "notes.md").write_text("inside", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("SECRET", encoding="utf-8")

    return base


# ---------------------------------------------------------------------------
# Escapes that must be refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "..",
        "../",
        "../secret.md",
        "../outside/secret.md",
        "../../etc/passwd",
        "../../../../../../etc/shadow",
        "security/../../outside/secret.md",
        "security/./.././../outside/secret.md",
        "./../../outside/secret.md",
        "a/b/c/../../../../outside/secret.md",
    ],
)
def test_dotdot_escape_is_blocked(corpus, hostile):
    """Relative traversal in any shape resolves outside and is rejected."""
    with pytest.raises(PathEscapeError):
        validate_path_within(corpus, hostile)


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "/root/.ssh/id_rsa",
        "//etc/passwd",
        "\\Windows\\win.ini",
    ],
)
def test_rooted_path_is_blocked(corpus, hostile):
    """A rooted path never lands inside the corpus on either platform.

    ``pathlib`` lets an absolute right-hand operand replace the base (POSIX) or
    replace the root while keeping the drive (Windows). Both outcomes fall
    outside ``documents_dir`` and are rejected — absolute input is never
    silently honoured.
    """
    with pytest.raises(PathEscapeError):
        validate_path_within(corpus, hostile)


def test_absolute_path_outside_corpus_is_blocked(corpus, tmp_path):
    """A fully-qualified path to a real file outside the corpus is rejected."""
    with pytest.raises(PathEscapeError):
        validate_path_within(corpus, tmp_path / "outside" / "secret.md")


def test_null_byte_is_blocked_before_any_syscall(corpus):
    """NUL truncation cannot be used to smuggle a second path past a check."""
    with pytest.raises(PathEscapeError, match="NUL byte"):
        validate_path_within(corpus, "notes.md\x00../../../etc/passwd")


def test_null_byte_alone_is_blocked(corpus):
    """Even a bare embedded NUL is refused rather than passed to the OS."""
    with pytest.raises(PathEscapeError, match="NUL byte"):
        validate_path_within(corpus, "a\x00b.md")


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
def test_ntfs_alternate_data_stream_is_blocked(corpus):
    """``notes.md:hidden`` would write an invisible stream inside the corpus."""
    with pytest.raises(PathEscapeError, match="alternate data stream"):
        validate_path_within(corpus, "security/notes.md:hidden")


@pytest.mark.parametrize(
    "hostile",
    [
        "..%2f..%2fetc%2fpasswd",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..／..／etc",  # fullwidth solidus
        "../../etc/passwd",  # explicit-codepoint dots
    ],
)
def test_encoded_and_unicode_variants_never_escape(corpus, hostile):
    """Percent-encoding and lookalike glyphs must not become separators.

    ``pathlib`` performs no URL decoding and no Unicode confusable folding, so
    these either stay inert filenames *inside* the corpus or resolve out and
    get rejected. Either outcome is safe; what must never happen is resolving
    outside while being reported as contained.
    """
    try:
        resolved = validate_path_within(corpus, hostile)
    except PathEscapeError:
        return
    assert resolved.is_relative_to(corpus.resolve())


def test_escape_message_is_stable(corpus):
    """The documented error text is part of the contract for callers."""
    with pytest.raises(PathEscapeError, match="Path escapes documents_dir"):
        validate_path_within(corpus, "../../etc/passwd")


def test_path_escape_error_is_a_value_error():
    """Existing ``except ValueError`` handlers keep working unchanged."""
    assert issubclass(PathEscapeError, ValueError)


# ---------------------------------------------------------------------------
# Legitimate paths that must keep working (backwards compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legit",
    [
        "security/notes.md",
        "security/new-file.md",
        "brand/new/nested/doc.md",
        "notes.md",
        "./security/notes.md",
        "security/sub/../notes.md",
    ],
)
def test_relative_paths_inside_corpus_are_allowed(corpus, legit):
    """Relative paths — including harmless internal ``..`` — stay allowed."""
    resolved = validate_path_within(corpus, legit)
    assert resolved.is_relative_to(corpus.resolve())


def test_absolute_path_inside_corpus_is_allowed(corpus):
    """``list_documents()`` returns absolute sources; they must round-trip."""
    absolute = corpus / "security" / "notes.md"
    assert validate_path_within(corpus, absolute) == absolute.resolve()


def test_accepts_path_objects_and_strings_identically(corpus):
    """Callers pass either type; both must resolve to the same result."""
    as_str = validate_path_within(corpus, "security/notes.md")
    as_path = validate_path_within(corpus, Path("security/notes.md"))
    assert as_str == as_path


def test_base_need_not_exist_yet(tmp_path):
    """First-run bootstrap validates paths before documents_dir is created."""
    base = tmp_path / "not-created-yet"
    assert validate_path_within(base, "a/b.md").is_relative_to(base.resolve())


# ---------------------------------------------------------------------------
# Non-raising variant
# ---------------------------------------------------------------------------


def test_is_path_within_mirrors_validate(corpus):
    """The boolean helper agrees with the raising one on both verdicts."""
    assert is_path_within(corpus, "security/notes.md") is True
    assert is_path_within(corpus, "../../etc/passwd") is False
    assert is_path_within(corpus, "a\x00b") is False


# ---------------------------------------------------------------------------
# Wiring — every orchestrator entry point that takes a client-supplied path
# ---------------------------------------------------------------------------
#
# These drive the real (unbound) methods with a stub ``self`` so the actual
# validation code runs. Mocking the orchestrator wholesale would prove nothing.

ESCAPE = "../../outside/secret.md"


@pytest.fixture
def orch(corpus, monkeypatch):
    """Stub orchestrator whose ``documents_dir`` is the temp corpus."""
    from unittest.mock import MagicMock

    from mcp_server import server as server_module

    monkeypatch.setattr(server_module.config, "documents_dir", corpus)

    stub = MagicMock()
    stub._source_to_docid = {}
    stub._indexed_docs = {}
    return stub


def test_add_document_rejects_escape(orch, corpus, tmp_path):
    """Writing outside the corpus must fail and touch nothing on disk."""
    from mcp_server.server import KnowledgeOrchestrator

    result = KnowledgeOrchestrator.add_document_from_content(orch, "pwned", ESCAPE, "general")

    assert "error" in result
    assert (tmp_path / "outside" / "secret.md").read_text(encoding="utf-8") == "SECRET"
    orch._index_new_file.assert_not_called()


def test_add_document_from_file_rejects_escape(orch, corpus, tmp_path):
    """The CLI ingest path shares the same guard."""
    from mcp_server.server import KnowledgeOrchestrator

    source = tmp_path / "payload.md"
    source.write_text("payload", encoding="utf-8")

    result = KnowledgeOrchestrator.add_document_from_file(orch, source, ESCAPE, "general")

    assert "error" in result
    orch._index_new_file.assert_not_called()


def test_update_document_rejects_escape(orch, tmp_path):
    """An unchecked path here overwrites any writable file on the host."""
    from mcp_server.server import KnowledgeOrchestrator

    victim = tmp_path / "outside" / "secret.md"
    result = KnowledgeOrchestrator.update_document_content(orch, str(victim), "overwritten")

    assert "error" in result
    assert victim.read_text(encoding="utf-8") == "SECRET"


def test_remove_document_rejects_escape(orch, tmp_path):
    """With ``delete_file=True`` this would be arbitrary file deletion."""
    from mcp_server.server import KnowledgeOrchestrator

    victim = tmp_path / "outside" / "secret.md"
    result = KnowledgeOrchestrator.remove_document_by_path(orch, str(victim), delete_file=True)

    assert "error" in result
    assert victim.exists()


def test_get_document_rejects_escape(orch, tmp_path):
    """The arbitrary-read primitive: any supported suffix, anywhere on disk."""
    from mcp_server.server import KnowledgeOrchestrator

    secret = tmp_path / "outside" / "creds.json"
    secret.write_text('{"api_key": "sk-leak"}', encoding="utf-8")

    assert KnowledgeOrchestrator.get_document(orch, str(secret)) is None
    orch.parser.parse_file.assert_not_called()


def test_search_similar_rejects_escape(orch, tmp_path):
    """Returns empty rather than probing index state for outside paths."""
    from mcp_server.server import KnowledgeOrchestrator

    victim = tmp_path / "outside" / "secret.md"
    assert KnowledgeOrchestrator.search_similar(orch, str(victim)) == []


def test_get_document_still_serves_in_corpus_files(orch, corpus):
    """Backwards compatibility: legitimate reads keep working."""
    from mcp_server.server import KnowledgeOrchestrator

    orch.parser.parse_file.return_value = None  # parse result is not under test

    KnowledgeOrchestrator.get_document(orch, "security/notes.md")

    orch.parser.parse_file.assert_called_once()
    called_with = orch.parser.parse_file.call_args[0][0]
    assert Path(called_with) == (corpus / "security" / "notes.md").resolve()
