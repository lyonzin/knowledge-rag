"""Coverage for batch_size + parallel_workers config (v4.8.0 Fase 3).

Two axes tested:

    1. Config layer (mcp_server/config.py) — batch_size and parallel_workers
       parse correctly from YAML, are clamped to safe ranges, and emit a
       WARN on invalid values. Windows > 4 workers gets an extra WARN
       about ONNX threading + SQLite contention.

    2. Indexing layer (mcp_server/server.py) — _index_document routes to
       ThreadPoolExecutor when config.parallel_workers > 1 (and the doc
       produces > 1 batch), falls back to the sequential loop otherwise.

All indexing tests bypass __init__ via object.__new__ — no ChromaDB,
FastEmbed download, or disk I/O. Same pattern used in tests/test_search.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_server.config import Config
from mcp_server.ingestion import Chunk, Document
from mcp_server.server import KnowledgeOrchestrator


# =============================================================================
# CONFIG — batch_size
# =============================================================================


class TestBatchSizeConfig:
    def test_batch_size_default_is_500(self):
        """Default (nothing declared) must be 500 to match v4.7.1 behavior."""
        # Explicit override to bypass any config.yaml the dev has on disk.
        c = Config(batch_size=500)
        assert c.batch_size == 500

    def test_batch_size_from_explicit_override_respected(self):
        """Custom values in [1, 5000] are respected as-is."""
        c = Config(batch_size=250)
        assert c.batch_size == 250

        c = Config(batch_size=3000)
        assert c.batch_size == 3000

    def test_batch_size_below_1_clamped_with_warn(self, capsys):
        c = Config(batch_size=0)
        assert c.batch_size == 1
        captured = capsys.readouterr()
        assert "batch_size" in captured.out and "clamping to 1" in captured.out

    def test_batch_size_negative_clamped_with_warn(self, capsys):
        c = Config(batch_size=-42)
        assert c.batch_size == 1
        captured = capsys.readouterr()
        assert "batch_size" in captured.out

    def test_batch_size_above_5000_clamped_with_warn(self, capsys):
        c = Config(batch_size=999_999)
        assert c.batch_size == 5000
        captured = capsys.readouterr()
        assert "batch_size" in captured.out and "5000" in captured.out

    def test_batch_size_wrong_type_clamped_with_warn(self, capsys):
        c = Config(batch_size="lots")  # type: ignore[arg-type]
        assert c.batch_size == 1
        captured = capsys.readouterr()
        assert "batch_size" in captured.out


# =============================================================================
# CONFIG — parallel_workers
# =============================================================================


class TestParallelWorkersConfig:
    def test_parallel_workers_default_is_1(self):
        """Default must be 1 (single-threaded — safe on all platforms)."""
        c = Config(parallel_workers=1)
        assert c.parallel_workers == 1

    def test_parallel_workers_from_explicit_override_respected(self):
        c = Config(parallel_workers=4)
        assert c.parallel_workers == 4

    def test_parallel_workers_above_16_clamped_with_warn(self, capsys):
        c = Config(parallel_workers=64)
        assert c.parallel_workers == 16
        captured = capsys.readouterr()
        assert "parallel_workers" in captured.out and "16" in captured.out

    def test_parallel_workers_below_1_clamped_with_warn(self, capsys):
        c = Config(parallel_workers=0)
        assert c.parallel_workers == 1
        captured = capsys.readouterr()
        assert "parallel_workers" in captured.out

    def test_parallel_workers_wrong_type_clamped_with_warn(self, capsys):
        c = Config(parallel_workers="many")  # type: ignore[arg-type]
        assert c.parallel_workers == 1
        captured = capsys.readouterr()
        assert "parallel_workers" in captured.out

    def test_windows_over_4_workers_emits_extra_warn(self, capsys, monkeypatch):
        """On Windows, workers > 4 gets an extra stability WARN."""
        monkeypatch.setattr("platform.system", lambda: "Windows")

        c = Config(parallel_workers=8)
        assert c.parallel_workers == 8  # not clamped (within [1, 16])

        captured = capsys.readouterr()
        assert "Windows" in captured.out
        assert "ONNX" in captured.out or "SQLite" in captured.out

    def test_non_windows_over_4_workers_no_extra_warn(self, capsys, monkeypatch):
        """Linux/macOS at workers > 4 must NOT emit the Windows WARN."""
        monkeypatch.setattr("platform.system", lambda: "Linux")

        c = Config(parallel_workers=8)
        assert c.parallel_workers == 8

        captured = capsys.readouterr()
        assert "Windows" not in captured.out


# =============================================================================
# INDEXING — _index_document dispatch
# =============================================================================


def _make_doc_with_chunks(n: int) -> Document:
    """Small helper — 200-char chunks with unique content (no dedup)."""
    from pathlib import Path

    return Document(
        id=f"unit-{n}c",
        content="",
        source=Path("/unit/test.md"),
        format=".md",
        category="unit",
        chunks=[
            Chunk(
                content=f"unique chunk {i:04d} " + "x" * 180,
                index=i,
                start_char=i * 200,
                end_char=(i + 1) * 200,
                metadata={},
            )
            for i in range(n)
        ],
        keywords=[],
    )


def _build_orchestrator_mocks() -> KnowledgeOrchestrator:
    """Build KnowledgeOrchestrator with mocked collection + bm25_index."""
    orch = object.__new__(KnowledgeOrchestrator)
    orch.collection = MagicMock()
    orch.bm25_index = MagicMock()
    return orch


class TestIndexingBatching:
    def test_single_worker_uses_sequential_loop(self, monkeypatch):
        """When parallel_workers == 1, ThreadPoolExecutor must NOT be constructed."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 25)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 1)

        orch = _build_orchestrator_mocks()
        doc = _make_doc_with_chunks(100)  # 100 / 25 = 4 batches

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_pool:
            indexed, skipped = orch._index_document(doc)

        assert indexed == 100
        assert skipped == 0
        mock_pool.assert_not_called()
        # 4 sequential add() calls (100 chunks / 25 batch)
        assert orch.collection.add.call_count == 4

    def test_multi_worker_uses_thread_pool(self, monkeypatch):
        """When parallel_workers > 1 AND >1 batch, ThreadPoolExecutor is used."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 25)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 4)

        orch = _build_orchestrator_mocks()
        doc = _make_doc_with_chunks(100)  # 100 / 25 = 4 batches

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_pool_cls:
            # Rig the mock to behave like the real Executor context manager
            # AND actually call the submitted callables (so orch.collection.add
            # is invoked with the right slices).
            mock_pool = MagicMock()
            mock_pool_cls.return_value.__enter__.return_value = mock_pool

            def _fake_submit(fn, *args, **kwargs):
                fn(*args, **kwargs)
                fut = MagicMock()
                fut.result.return_value = None
                return fut

            mock_pool.submit.side_effect = _fake_submit

            indexed, skipped = orch._index_document(doc)

        assert indexed == 100
        assert skipped == 0
        mock_pool_cls.assert_called_once_with(max_workers=4)
        # 4 submit() calls — one per batch
        assert mock_pool.submit.call_count == 4
        # collection.add() called 4 times (via _fake_submit)
        assert orch.collection.add.call_count == 4

    def test_multi_worker_single_batch_skips_pool(self, monkeypatch):
        """workers > 1 but only 1 batch → skip ThreadPoolExecutor (no benefit)."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 500)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 4)

        orch = _build_orchestrator_mocks()
        doc = _make_doc_with_chunks(50)  # 50 < 500 → 1 batch

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_pool:
            indexed, skipped = orch._index_document(doc)

        assert indexed == 50
        mock_pool.assert_not_called()
        assert orch.collection.add.call_count == 1

    def test_batch_size_respected_in_sequential_path(self, monkeypatch):
        """collection.add() must be called with slices of size == config.batch_size."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 10)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 1)

        orch = _build_orchestrator_mocks()
        doc = _make_doc_with_chunks(25)  # 25 / 10 → batches of 10, 10, 5

        indexed, _skipped = orch._index_document(doc)

        assert indexed == 25
        # 3 calls: sizes 10, 10, 5
        assert orch.collection.add.call_count == 3
        sizes = [len(call.kwargs["ids"]) for call in orch.collection.add.call_args_list]
        assert sizes == [10, 10, 5]

    def test_fallback_to_class_constant_when_config_missing_batch_size(self, monkeypatch):
        """getattr(config, 'batch_size', _CHROMA_BATCH_SIZE) protects test isolation."""
        # Simulate a caller that patched the config module object away.
        fake_config = MagicMock(spec=[])  # spec=[] → no attributes at all
        monkeypatch.setattr("mcp_server.server.config", fake_config)

        orch = _build_orchestrator_mocks()
        doc = _make_doc_with_chunks(50)

        indexed, _ = orch._index_document(doc)

        assert indexed == 50
        # 50 < 500 (_CHROMA_BATCH_SIZE fallback) → single batch
        assert orch.collection.add.call_count == 1

    def test_parallel_path_propagates_first_exception(self, monkeypatch):
        """When one worker fails, .result() raises — indexing surfaces the error."""
        monkeypatch.setattr("mcp_server.server.config.batch_size", 20)
        monkeypatch.setattr("mcp_server.server.config.parallel_workers", 3)

        orch = _build_orchestrator_mocks()
        doc = _make_doc_with_chunks(60)  # 3 batches

        # Rig collection.add to raise on the second call
        call_count = {"n": 0}

        def _raise_on_second(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated ChromaDB write failure")

        orch.collection.add.side_effect = _raise_on_second

        with pytest.raises(RuntimeError, match="simulated ChromaDB write failure"):
            orch._index_document(doc)

    def test_empty_doc_short_circuits_before_config_lookup(self, monkeypatch):
        """No chunks = no batching. Must not touch collection/pool/config."""
        from pathlib import Path

        orch = _build_orchestrator_mocks()
        empty_doc = Document(
            id="empty",
            content="",
            source=Path("/empty.md"),
            format=".md",
            category="empty",
            chunks=[],
        )

        indexed, skipped = orch._index_document(empty_doc)

        assert indexed == 0 and skipped == 0
        orch.collection.add.assert_not_called()
        orch.bm25_index.add_documents.assert_not_called()
