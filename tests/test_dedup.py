"""Regression tests for chunk dedup integrity (#90, #91).

#90: Moving an indexed file must not lose chunks on reindex.
#91: Removing a document must not delete chunks shared with another document.

These tests use a real ChromaDB (tmp_path) with mocked embeddings to exercise
the actual indexing pipeline without requiring model downloads.
"""

from typing import List
from unittest.mock import patch

import pytest


class _FakeEmbeddings:
    """Minimal embedding function that satisfies ChromaDB's full interface."""

    _dim = 384
    is_legacy = False

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in input]

    @staticmethod
    def name() -> str:
        return "fake-test-embeddings"

    @staticmethod
    def build_from_config(config):  # noqa: ARG004
        return _FakeEmbeddings()

    @staticmethod
    def get_config():
        return {}

    @staticmethod
    def validate_config_update(old_config, new_config):  # noqa: ARG004
        pass


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    """Isolated RAG environment: real ChromaDB, mocked embeddings, tmp dirs."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    chroma_dir = data_dir / "chroma_db"
    chroma_dir.mkdir()
    models_dir = tmp_path / "models_cache"
    models_dir.mkdir()

    import mcp_server.config as cfg

    monkeypatch.setattr(cfg.config, "documents_dir", docs_dir)
    monkeypatch.setattr(cfg.config, "data_dir", data_dir)
    monkeypatch.setattr(cfg.config, "chroma_dir", chroma_dir)
    monkeypatch.setattr(cfg.config, "models_cache_dir", models_dir)
    monkeypatch.setattr(cfg.config, "transport", "stdio")

    with patch("mcp_server.server.FastEmbedEmbeddings", _FakeEmbeddings):
        from mcp_server.server import KnowledgeOrchestrator

        orch = KnowledgeOrchestrator()
        yield orch, docs_dir


CONTENT_A = """\
# Document Alpha

## Overview

This document covers advanced SQL injection bypass techniques including
UNION-based, blind boolean, and time-based attacks against web applications.

## Methodology

Use parameterized queries and input validation as primary defenses.
"""

CONTENT_B = """\
# Document Beta

## Overview

Cross-site scripting attack vectors for reflected, stored, and DOM-based
XSS in modern single-page applications with CSP bypass techniques.

## Methodology

Use parameterized queries and input validation as primary defenses.
"""


class TestFileMovePreservesChunks:
    """#90: Moving an indexed file must re-index at the new path."""

    def test_move_file_chunks_preserved(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "alpha.md").write_text(CONTENT_A, encoding="utf-8")
        stats1 = orch.index_all(force=True)
        assert stats1["chunks_added"] > 0
        chunks_before = orch.collection.count()

        sub = docs_dir / "sub"
        sub.mkdir()
        (docs_dir / "alpha.md").rename(sub / "alpha.md")

        stats2 = orch.index_all(force=False)
        chunks_after = orch.collection.count()

        assert chunks_after >= chunks_before, (
            f"Chunks dropped from {chunks_before} to {chunks_after} after move. Orphan cleanup raced dedup (issue #90)."
        )
        assert stats2["deleted"] >= 1

    def test_move_file_searchable_after_reindex(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "alpha.md").write_text(CONTENT_A, encoding="utf-8")
        orch.index_all(force=True)
        chunks_before = len(orch.collection.get(where={"filename": "alpha.md"}, include=[])["ids"])

        sub = docs_dir / "sub"
        sub.mkdir()
        (docs_dir / "alpha.md").rename(sub / "alpha.md")
        orch.index_all(force=False)

        chunks_after = len(orch.collection.get(where={"filename": "alpha.md"}, include=[])["ids"])
        assert chunks_after == chunks_before, (
            f"Chunks went from {chunks_before} to {chunks_after} after move+reindex. "
            "File content should still be fully searchable."
        )


class TestCrossDocDedup:
    """#91: Removing one doc must not delete chunks from another doc."""

    def test_shared_content_survives_removal(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "alpha.md").write_text(CONTENT_A, encoding="utf-8")
        (docs_dir / "beta.md").write_text(CONTENT_B, encoding="utf-8")
        orch.index_all(force=True)

        beta_chunks_before = orch.collection.get(where={"filename": "beta.md"}, include=[])
        assert len(beta_chunks_before["ids"]) > 0

        (docs_dir / "alpha.md").unlink()
        orch.index_all(force=False)

        beta_chunks_after = orch.collection.get(where={"filename": "beta.md"}, include=[])
        assert len(beta_chunks_after["ids"]) == len(beta_chunks_before["ids"]), (
            f"Beta chunks dropped from {len(beta_chunks_before['ids'])} to "
            f"{len(beta_chunks_after['ids'])} after removing alpha. "
            "Cross-document dedup coupling (issue #91)."
        )

    def test_identical_files_both_indexed(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "copy1.md").write_text(CONTENT_A, encoding="utf-8")
        (docs_dir / "copy2.md").write_text(CONTENT_A, encoding="utf-8")
        orch.index_all(force=True)

        c1 = orch.collection.get(where={"filename": "copy1.md"}, include=[])
        c2 = orch.collection.get(where={"filename": "copy2.md"}, include=[])

        assert len(c1["ids"]) > 0, "copy1 has no chunks"
        assert len(c2["ids"]) > 0, "copy2 has no chunks — global dedup suppressed it"
        assert len(c1["ids"]) == len(c2["ids"]), "Identical files should have the same number of chunks each."

    def test_remove_one_copy_other_intact(self, rag_env):
        orch, docs_dir = rag_env

        (docs_dir / "copy1.md").write_text(CONTENT_A, encoding="utf-8")
        (docs_dir / "copy2.md").write_text(CONTENT_A, encoding="utf-8")
        orch.index_all(force=True)

        count_before = len(orch.collection.get(where={"filename": "copy2.md"}, include=[])["ids"])

        (docs_dir / "copy1.md").unlink()
        orch.index_all(force=False)

        count_after = len(orch.collection.get(where={"filename": "copy2.md"}, include=[])["ids"])
        assert count_after == count_before, (
            f"copy2 chunks dropped from {count_before} to {count_after} "
            "after removing copy1 — cross-doc dedup coupling."
        )
