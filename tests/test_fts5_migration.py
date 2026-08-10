"""Task 05 tests: FTS5 lazy migration lifecycle + CRUD sync (ADR-008).

Covers the assigned catalog:
- Unit — TestMigrationLifecycle: UT-043 .. UT-048
- Unit — TestCRUDSync: UT-049, UT-050, UT-051, UT-052
- Integration — TestMigrationIntegration: IT-021, IT-022, IT-023
- Integration — TestCRUDSyncIntegration: IT-CRUD-001 .. IT-CRUD-004

The tests exercise ``Fts5LexicalIndex`` directly (real SQLite, real FTS5
tables, real marker files) so the SQLite locking behaviour Windows CI is
sensitive to actually runs. Orchestrator-level hooks use a lightweight
stub (``_build_sync_orch``) that mounts the real ``_fts5_sync_add`` +
``_fts5_sync_remove_by_doc_id`` bound methods so the CRUD-sync
integration path is unmediated.
"""

from __future__ import annotations

import sqlite3
import threading
import types
from typing import Iterable, List, Tuple
from unittest.mock import patch

import pytest

from mcp_server.fts5_index import (
    Fts5LexicalIndex,
    Fts5MigrationState,
)

Row = Tuple[str, str, str, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(tmp_path, *, marker_status: str | None = None, docs_indexed: int = 0) -> Fts5LexicalIndex:
    db_path = tmp_path / "fts5_index.db"
    state_path = tmp_path / "fts5_migration.state"
    if marker_status is not None:
        Fts5MigrationState(state_path).write(
            {
                "status": marker_status,
                "docs_total": 100,
                "docs_indexed": docs_indexed,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": None,
                "error": None,
            }
        )
    return Fts5LexicalIndex(db_path=db_path, state_path=state_path)


def _gen_rows(n: int, prefix: str = "chunk") -> List[Row]:
    return [(f"{prefix}_{i:04d}", f"content mentioning CVE-2021-{i:04d}", f"f{i}.md", "security") for i in range(n)]


def _iter_factory(rows: List[Row]):
    def _factory() -> Iterable[Row]:
        return iter(rows)

    return _factory


# ===========================================================================
# TestMigrationLifecycle — UT-043 .. UT-048
# ===========================================================================


class TestMigrationLifecycle:
    def test_ut043_migration_populates_and_marks_complete(self, tmp_path):
        """UT-043: background worker populates every row and flips status to complete."""
        index = _make_index(tmp_path)
        try:
            rows = _gen_rows(250)
            thread = index.start_migration_background(_iter_factory(rows), docs_total=len(rows))
            thread.join(timeout=15.0)
            assert not thread.is_alive(), "migration thread did not finish"
            state = index.state.read()
            assert state["status"] == "complete"
            assert state["docs_indexed"] == 250
            assert index.is_ready() is True
            # Every row visible via SQL count
            with index._fts5_lock:  # noqa: SLF001
                total = index._conn.execute("SELECT COUNT(*) FROM fts5_documents").fetchone()[0]  # noqa: SLF001
            assert total == 250
        finally:
            index.close()

    def test_ut044_checkpoint_written_per_batch(self, tmp_path):
        """UT-044: docs_indexed advances by 100 per batch — verified via progress hook."""
        index = _make_index(tmp_path)
        try:
            checkpoints: list[int] = []

            def on_progress(done: int, total: int) -> None:
                checkpoints.append(done)

            thread = index.start_migration_background(
                _iter_factory(_gen_rows(250)), docs_total=250, on_progress=on_progress
            )
            thread.join(timeout=15.0)
            # Callback fires after each 100-batch flush plus once for the tail.
            assert checkpoints[:2] == [100, 200]
            assert checkpoints[-1] == 250
        finally:
            index.close()

    def test_ut045_progress_log_every_10_percent(self, tmp_path, capsys):
        """UT-045/UT-048: [FTS5] migration progress log fires per >=10% jump."""
        index = _make_index(tmp_path)
        try:
            thread = index.start_migration_background(_iter_factory(_gen_rows(1000)), docs_total=1000)
            thread.join(timeout=20.0)
            out = capsys.readouterr().out
            # 10 batches of 100 = 10 progress checkpoints, each crossing a 10% boundary.
            progress_lines = [line for line in out.splitlines() if "migration progress" in line]
            assert len(progress_lines) >= 5, f"expected >=5 progress lines, got {progress_lines}"
            assert "migration complete" in out
        finally:
            index.close()

    def test_ut046_migration_failure_marks_state_failed(self, tmp_path):
        """UT-046: an exception inside the worker persists status=failed + error message."""
        index = _make_index(tmp_path)
        try:

            def _boom():
                raise sqlite3.OperationalError("disk I/O error")

            def _bad_iter() -> Iterable[Row]:
                yield from _gen_rows(50)
                _boom()

            def _factory():
                return _bad_iter()

            thread = index.start_migration_background(_factory, docs_total=50)
            thread.join(timeout=10.0)
            state = index.state.read()
            assert state["status"] == "failed"
            assert "OperationalError" in (state.get("error") or "")
            assert index.is_ready() is False
        finally:
            index.close()

    def test_ut047_killed_daemon_state_retained_for_resume(self, tmp_path):
        """UT-047: after ``in_progress`` checkpoint, restart resumes from docs_indexed."""
        # Simulate a partial run by writing an in_progress marker directly.
        Fts5MigrationState(tmp_path / "fts5_migration.state").write(
            {
                "status": "in_progress",
                "docs_total": 200,
                "docs_indexed": 100,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": None,
                "error": None,
            }
        )
        index = Fts5LexicalIndex(
            db_path=tmp_path / "fts5_index.db",
            state_path=tmp_path / "fts5_migration.state",
        )
        try:
            assert index.is_ready() is False
            rows = _gen_rows(200)
            # Resume from checkpoint — worker should skip first 100 rows.
            thread = index.start_migration_background(_iter_factory(rows), docs_total=200, resume_from=100)
            thread.join(timeout=10.0)
            state = index.state.read()
            assert state["status"] == "complete"
            with index._fts5_lock:  # noqa: SLF001
                total = index._conn.execute("SELECT COUNT(*) FROM fts5_documents").fetchone()[0]  # noqa: SLF001
            assert total == 100, "resume must skip the 100 already-indexed rows"
        finally:
            index.close()

    def test_ut048_no_progress_log_when_docs_total_zero(self, tmp_path, capsys):
        """UT-048 boundary: docs_total=0 must not divide-by-zero in the log helper."""
        index = _make_index(tmp_path)
        try:
            thread = index.start_migration_background(_iter_factory([]), docs_total=0)
            thread.join(timeout=5.0)
            out = capsys.readouterr().out
            assert "migration complete" in out
            assert "migration progress" not in out
        finally:
            index.close()


# ===========================================================================
# TestCRUDSync — UT-049, UT-050, UT-051, UT-052
# ===========================================================================


class TestCRUDSync:
    def test_ut049_add_document_visible_in_search(self, tmp_path):
        """UT-049: add_document inserts a row + subsequent search hits it."""
        index = _make_index(tmp_path, marker_status="complete", docs_indexed=0)
        try:
            index.add_document("chunk_1", "Content mentioning CVE-2021-4034", "file.md", "security")
            hits = index.search("CVE-2021-4034", top_k=5)
            ids = [chunk_id for chunk_id, _ in hits]
            assert "chunk_1" in ids
        finally:
            index.close()

    def test_ut050_remove_document_clears_from_search(self, tmp_path):
        """UT-050: remove_document deletes the row + search no longer returns it."""
        index = _make_index(tmp_path, marker_status="complete", docs_indexed=0)
        try:
            index.add_document("chunk_1", "Content mentioning CVE-2021-4034", "file.md", "security")
            assert index.search("CVE-2021-4034", top_k=5)
            index.remove_document("chunk_1")
            assert index.search("CVE-2021-4034", top_k=5) == []
        finally:
            index.close()

    def test_ut051_add_document_error_caught_and_metric_incremented(self, tmp_path):
        """UT-051: orchestrator CRUD hook swallows FTS5 error + bumps error counter."""
        from mcp_server.metrics import FAST_PATH_ERRORS_TOTAL

        orch, capture = _build_sync_orch(tmp_path)
        # Force the FTS5 write to raise OperationalError.
        capture.raise_on_add = sqlite3.OperationalError("disk full")

        before = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')
        # Must NOT raise even though the FTS5 write blows up.
        orch._fts5_sync_add(  # noqa: SLF001 — bound helper under test
            ["chunk_a"], ["content"], [{"filename": "f.md", "category": "sec"}]
        )
        after = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')
        assert after > before

    def test_ut052_concurrent_add_document_distinct_chunk_ids(self, tmp_path):
        """UT-052: two threads adding distinct chunk_ids both persist (RLock serializes)."""
        index = _make_index(tmp_path, marker_status="complete", docs_indexed=0)
        try:
            errors: list[BaseException] = []

            def worker(chunk_id: str, content: str) -> None:
                try:
                    index.add_document(chunk_id, content, "f.md", "security")
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t1 = threading.Thread(target=worker, args=("chunk_A", "CVE-2021-1111 alpha"))
            t2 = threading.Thread(target=worker, args=("chunk_B", "CVE-2021-2222 beta"))
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            assert errors == []
            ids_alpha = {cid for cid, _ in index.search("CVE-2021-1111", top_k=5)}
            ids_beta = {cid for cid, _ in index.search("CVE-2021-2222", top_k=5)}
            assert "chunk_A" in ids_alpha
            assert "chunk_B" in ids_beta
        finally:
            index.close()


# ===========================================================================
# TestMigrationIntegration — IT-021, IT-022, IT-023
# ===========================================================================


class TestMigrationIntegration:
    def test_it021_migration_from_missing_state_populates_and_flips_ready(self, tmp_path):
        """IT-021: no marker file → migration runs → ready flips + rows visible."""
        rows = _gen_rows(120)
        index = Fts5LexicalIndex(
            db_path=tmp_path / "fts5_index.db",
            state_path=tmp_path / "fts5_migration.state",
        )
        try:
            assert index.is_ready() is False
            thread = index.start_migration_background(_iter_factory(rows), docs_total=len(rows))
            thread.join(timeout=15.0)
            assert index.is_ready() is True
            hits = index.search("CVE-2021-0000", top_k=5)
            assert hits, "post-migration search must return the seeded chunk"
        finally:
            index.close()

    def test_it022_migration_failure_recovers_permanently_via_marker(self, tmp_path):
        """IT-022: failed marker + reopen → still not ready + status stays 'failed'."""
        index = _make_index(tmp_path)
        try:

            def _iter():
                yield from _gen_rows(10)
                raise RuntimeError("simulated corpus fault")

            thread = index.start_migration_background(lambda: _iter(), docs_total=10)
            thread.join(timeout=5.0)
            state = index.state.read()
            assert state["status"] == "failed"
        finally:
            index.close()

        # Reopen — the marker survives across process boundaries.
        reopened = Fts5LexicalIndex(
            db_path=tmp_path / "fts5_index.db",
            state_path=tmp_path / "fts5_migration.state",
        )
        try:
            assert reopened.is_ready() is False
            assert (reopened.state.read() or {}).get("status") == "failed"
        finally:
            reopened.close()

    def test_it023_migration_resume_from_checkpoint_not_zero(self, tmp_path):
        """IT-023: in_progress marker with docs_indexed=40 → resume skips first 40 rows."""
        Fts5MigrationState(tmp_path / "fts5_migration.state").write(
            {
                "status": "in_progress",
                "docs_total": 100,
                "docs_indexed": 40,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": None,
                "error": None,
            }
        )
        index = Fts5LexicalIndex(
            db_path=tmp_path / "fts5_index.db",
            state_path=tmp_path / "fts5_migration.state",
        )
        try:
            rows = _gen_rows(100)
            thread = index.start_migration_background(_iter_factory(rows), docs_total=100, resume_from=40)
            thread.join(timeout=10.0)
            with index._fts5_lock:  # noqa: SLF001
                total = index._conn.execute("SELECT COUNT(*) FROM fts5_documents").fetchone()[0]  # noqa: SLF001
            assert total == 60, "resume must skip the first 40 rows"
            assert index.is_ready() is True
        finally:
            index.close()


# ===========================================================================
# TestCRUDSyncIntegration — IT-CRUD-001 .. IT-CRUD-004
# ===========================================================================


class TestCRUDSyncIntegration:
    def test_it_crud_001_add_update_remove_reflected_in_search(self, tmp_path):
        """IT-CRUD-001: add → search → update → search → remove → search sequence."""
        orch, capture = _build_sync_orch(tmp_path)
        capture.raise_on_add = None

        orch._fts5_sync_add(  # noqa: SLF001
            ["chunk_1"], ["CVE-2021-4034 exploit primer"], [{"filename": "a.md", "category": "sec"}]
        )
        assert capture.index.search("CVE-2021-4034", top_k=5)

        # Update: remove then add same chunk_id with new content.
        capture.index.update_document("chunk_1", "New content mentioning CVE-2999-9999", "b.md", "sec")
        assert capture.index.search("CVE-2999-9999", top_k=5)
        assert capture.index.search("CVE-2021-4034", top_k=5) == []

        capture.index.remove_document("chunk_1")
        assert capture.index.search("CVE-2999-9999", top_k=5) == []

    def test_it_crud_002_concurrent_add_document_via_helper(self, tmp_path):
        """IT-CRUD-002: 5 threads driving _fts5_sync_add — all rows visible after."""
        orch, capture = _build_sync_orch(tmp_path)
        capture.raise_on_add = None
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                orch._fts5_sync_add(  # noqa: SLF001
                    [f"chunk_{i}"],
                    [f"content mentioning CVE-2021-{i:04d}"],
                    [{"filename": f"{i}.md", "category": "sec"}],
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert errors == []
        with capture.index._fts5_lock:  # noqa: SLF001
            total = capture.index._conn.execute("SELECT COUNT(*) FROM fts5_documents").fetchone()[0]  # noqa: SLF001
        assert total == 5

    def test_it_crud_003_disk_full_error_swallowed_no_raise(self, tmp_path):
        """IT-CRUD-003: SQLite OperationalError never propagates to the CRUD tool."""
        from mcp_server.metrics import FAST_PATH_ERRORS_TOTAL

        orch, capture = _build_sync_orch(tmp_path)
        capture.raise_on_add = sqlite3.OperationalError("disk full")
        before = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')

        # No exception should escape.
        orch._fts5_sync_add(  # noqa: SLF001
            ["chunk_z"], ["content"], [{"filename": "z.md", "category": "sec"}]
        )
        after = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')
        assert after > before

    def test_it_crud_004_feature_off_skips_fts5_calls(self, tmp_path):
        """IT-CRUD-004: config.fts5_enabled=False → zero add_document calls made."""
        orch, capture = _build_sync_orch(tmp_path, fts5_enabled=False)
        capture.raise_on_add = RuntimeError("should never be invoked")

        # Even with a booby-trapped index, no exception because the hook exits early.
        orch._fts5_sync_add(  # noqa: SLF001
            ["chunk_x"], ["content"], [{"filename": "x.md", "category": "sec"}]
        )
        assert capture.add_calls == 0


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _CaptureIndex:
    """Wrapper around a real ``Fts5LexicalIndex`` that can inject failures."""

    def __init__(self, index: Fts5LexicalIndex) -> None:
        self.index = index
        self.raise_on_add: BaseException | None = None
        self.add_calls = 0

    def add_document(self, chunk_id: str, content: str, filename: str, category: str) -> None:
        self.add_calls += 1
        if self.raise_on_add is not None:
            raise self.raise_on_add
        self.index.add_document(chunk_id, content, filename, category)

    def remove_document(self, chunk_id: str) -> None:
        self.index.remove_document(chunk_id)


def _build_sync_orch(tmp_path, *, fts5_enabled: bool = True):
    """Stub orchestrator with the real CRUD-sync helpers bound to a capture index."""
    import mcp_server.server as srv
    from mcp_server.server import KnowledgeOrchestrator

    Fts5MigrationState(tmp_path / "fts5_migration.state").write(
        {
            "status": "complete",
            "docs_total": 0,
            "docs_indexed": 0,
            "started_at": "2026-08-07T12:00:00Z",
            "completed_at": "2026-08-07T12:00:00Z",
            "error": None,
        }
    )
    real_index = Fts5LexicalIndex(
        db_path=tmp_path / "fts5_index.db",
        state_path=tmp_path / "fts5_migration.state",
    )
    capture = _CaptureIndex(real_index)

    orch = object.__new__(KnowledgeOrchestrator)
    orch.fts5_index = capture  # helpers accept the duck-typed wrapper
    orch.collection = types.SimpleNamespace(get=lambda where, include: {"ids": []})

    # Bind the real hooks so behaviour under test is production-shaped.
    orch._fts5_sync_add = types.MethodType(  # noqa: SLF001
        KnowledgeOrchestrator._fts5_sync_add, orch
    )
    orch._fts5_sync_remove_by_doc_id = types.MethodType(  # noqa: SLF001
        KnowledgeOrchestrator._fts5_sync_remove_by_doc_id, orch
    )

    # Patch config toggles for the duration of the test.
    patcher = patch.object(srv.config, "fts5_enabled", fts5_enabled)
    patcher.start()

    def _teardown():
        patcher.stop()
        real_index.close()

    # Register teardown via a finalizer bound to the orchestrator stub.
    orch.__teardown__ = _teardown  # noqa: SLF001
    return orch, capture


@pytest.fixture(autouse=True)
def _cleanup_sync_orch(request):
    """Run any ``__teardown__`` recorded by ``_build_sync_orch``."""
    yield
    for name, obj in list(request.node.__dict__.items()):
        if hasattr(obj, "__teardown__"):
            try:
                obj.__teardown__()
            except Exception:  # noqa: BLE001
                pass


def _metric_value(name: str, label: str) -> float:
    """Read a labelled counter from the shared metrics collector."""
    from mcp_server.metrics import get_metrics

    prefix = f"{name}{{{label}}} "
    for line in get_metrics().exposition().split("\n"):
        if line.startswith(prefix):
            return float(line.split()[-1])
    return 0.0
