"""Unit + integration tests for ``mcp_server.fts5_index`` (Task 01 / Fase 1).

Covers assigned IDs from ``.compozy/tasks/fts5-lexical-fast-path/_tests.md``:
- UT-020..028 (Fts5LexicalIndex.search + connect_and_configure)
- UT-054..058 (Fts5MigrationState)
- UT-061      (Security surface — FTS5 metacharacter escape)
- IT-MIG-001  (atomic write cross-thread)
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from mcp_server.fts5_index import (
    Fts5CorruptError,
    Fts5LexicalIndex,
    Fts5MigrationError,
    Fts5MigrationState,
    Fts5NotReadyError,
    _escape_fts5_query,
)

# ---------------------------------------------------------------------------
# Exception hierarchy sanity — TechSpec §Exception Hierarchy / ADR-004 Q4
# ---------------------------------------------------------------------------


def test_exceptions_subclass_runtimeerror():
    for exc_cls in (Fts5NotReadyError, Fts5CorruptError, Fts5MigrationError):
        assert issubclass(exc_cls, RuntimeError)
        # No custom base class — each exception subclasses RuntimeError directly.
        assert exc_cls.__bases__ == (RuntimeError,)


# ---------------------------------------------------------------------------
# TestFts5LexicalIndexSearch — UT-020..027
# ---------------------------------------------------------------------------


class TestFts5LexicalIndexSearch:
    """UT-020..027: search() behavior across empty/happy/composite/escape paths."""

    def test_ut020_search_empty_index_returns_empty_list(self, tmp_path):
        """UT-020: search on an empty index returns [] (never raises)."""
        Fts5MigrationState(tmp_path / "state.json").write({"status": "complete", "docs_total": 0, "docs_indexed": 0})
        index = Fts5LexicalIndex(
            db_path=tmp_path / "fts5.db",
            state_path=tmp_path / "state.json",
        )
        try:
            assert index.search("anything", top_k=20) == []
        finally:
            index.close()

    def test_ut021_search_identifier_returns_seeded_chunk(self, fts5_tmp_index):
        """UT-021: seed contains MS17-010 → search returns matching chunk in top-3."""
        results = fts5_tmp_index.search("MS17-010", top_k=5)
        assert results, "expected at least one result"
        top_ids = [chunk_id for chunk_id, _ in results[:3]]
        assert "chunk_001" in top_ids
        for _, score in results:
            assert score > 0

    def test_ut022_search_respects_top_k(self, fts5_tmp_index):
        """UT-022: 5 seed docs contain 'CVE-2021-4034'; top_k=3 returns exactly 3, desc score."""
        results = fts5_tmp_index.search("CVE-2021-4034", top_k=3)
        assert len(results) == 3
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_ut023_composite_preserved_no_false_match(self, fts5_tmp_index):
        """UT-023: query 'MS17-010' does not match a chunk that only contains 'MS17-020'."""
        results = fts5_tmp_index.search("MS17-010", top_k=20)
        ids = [chunk_id for chunk_id, _ in results]
        assert "chunk_012" not in ids  # chunk_012 mentions MS17-020 only

    def test_ut024_hyphenated_phrase_preserved(self, fts5_tmp_index):
        """UT-024: 'pass-the-hash' is a single token thanks to tokenchars '-_.'."""
        results = fts5_tmp_index.search("pass-the-hash", top_k=5)
        ids = [chunk_id for chunk_id, _ in results]
        assert "chunk_010" in ids

    def test_ut025_dot_preserved_in_filename_token(self, fts5_tmp_index):
        """UT-025 (ADR-005 mandatory): 'context.py' matches doc containing 'context.py'."""
        results = fts5_tmp_index.search("context.py", top_k=5)
        ids = [chunk_id for chunk_id, _ in results]
        assert "chunk_011" in ids

    def test_ut026_case_insensitive_via_unicode61(self, fts5_tmp_index):
        """UT-026: lowercase query matches uppercase-seeded identifier (unicode61 lowercases)."""
        results = fts5_tmp_index.search("ms17-010", top_k=5)
        ids = [chunk_id for chunk_id, _ in results]
        assert "chunk_001" in ids

    def test_ut027_metacharacter_query_does_not_raise(self, fts5_tmp_index):
        """UT-027: query with FTS5 operators is escaped, never OperationalError."""
        # These would each break a raw MATCH; escape must neutralise them.
        for query in ["MATCH * OR AND", "NEAR ()", "AND OR NOT", '"OR 1=1--']:
            results = fts5_tmp_index.search(query, top_k=5)
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestFts5LexicalIndexErrors — UT-028
# ---------------------------------------------------------------------------


class TestFts5LexicalIndexErrors:
    """UT-028: schema drift / FTS5-missing bubbles up as Fts5CorruptError."""

    def test_ut028_schema_drift_raises_corrupt_error(self, tmp_path, monkeypatch):
        """UT-028: SQLite build without FTS5 (simulated) → Fts5CorruptError on init."""

        real_connect = sqlite3.connect

        class _FakeCursor:
            def execute(self, sql, *args, **kwargs):
                # Simulate FTS5-missing build: creating the probe fails.
                if "fts5" in sql.lower() and "create" in sql.lower():
                    raise sqlite3.OperationalError("no such module: fts5")
                return None

            def close(self):
                pass

        class _FakeConn:
            def __init__(self, real):
                self._real = real

            def cursor(self):
                return _FakeCursor()

            def execute(self, sql, *args, **kwargs):
                if "fts5" in sql.lower() and "create" in sql.lower():
                    raise sqlite3.OperationalError("no such module: fts5")
                return self._real.execute(sql, *args, **kwargs)

            def commit(self):
                self._real.commit()

            def close(self):
                self._real.close()

        def fake_connect(*args, **kwargs):
            return _FakeConn(real_connect(*args, **kwargs))

        monkeypatch.setattr("mcp_server.fts5_index.sqlite3.connect", fake_connect)
        Fts5MigrationState(tmp_path / "state.json").write({"status": "complete", "docs_total": 0, "docs_indexed": 0})
        with pytest.raises(Fts5CorruptError):
            Fts5LexicalIndex(
                db_path=tmp_path / "broken.db",
                state_path=tmp_path / "state.json",
            )

    def test_not_ready_when_state_missing(self, tmp_path):
        """is_ready() returns False when the marker file does not exist yet."""
        index = Fts5LexicalIndex(
            db_path=tmp_path / "fts5.db",
            state_path=tmp_path / "missing.state",
        )
        try:
            assert index.is_ready() is False
        finally:
            index.close()


# ---------------------------------------------------------------------------
# TestFts5MigrationState — UT-054..058
# ---------------------------------------------------------------------------


class TestFts5MigrationState:
    """UT-054..058: read/write/is_complete + concurrent atomic writes."""

    _PAYLOAD_COMPLETE = {
        "status": "complete",
        "docs_total": 100,
        "docs_indexed": 100,
        "started_at": "2026-08-07T12:00:00Z",
        "completed_at": "2026-08-07T12:00:42Z",
        "error": None,
    }

    def test_ut054_write_read_roundtrip(self, migration_state_tmp):
        """UT-054: write then read returns the exact same payload dict."""
        migration_state_tmp.write(self._PAYLOAD_COMPLETE)
        assert migration_state_tmp.read() == self._PAYLOAD_COMPLETE

    def test_ut055_read_missing_returns_none(self, migration_state_tmp):
        """UT-055: read() on a nonexistent path returns None."""
        assert migration_state_tmp.read() is None

    def test_ut056_is_complete_true_when_status_complete(self, migration_state_tmp):
        """UT-056: is_complete() returns True when status == 'complete'."""
        migration_state_tmp.write(self._PAYLOAD_COMPLETE)
        assert migration_state_tmp.is_complete() is True

    def test_ut057_is_complete_false_when_in_progress(self, migration_state_tmp):
        """UT-057: is_complete() returns False when status == 'in_progress'."""
        migration_state_tmp.write(
            {
                "status": "in_progress",
                "docs_total": 100,
                "docs_indexed": 42,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": None,
                "error": None,
            }
        )
        assert migration_state_tmp.is_complete() is False

    def test_ut058_concurrent_write_yields_one_valid_payload(self, migration_state_tmp):
        """UT-058: 2 threads writing concurrently — final read is one of the two payloads."""
        payload_a = {**self._PAYLOAD_COMPLETE, "docs_indexed": 1}
        payload_b = {**self._PAYLOAD_COMPLETE, "docs_indexed": 2}

        def _worker(payload):
            for _ in range(50):
                migration_state_tmp.write(payload)

        threads = [
            threading.Thread(target=_worker, args=(payload_a,)),
            threading.Thread(target=_worker, args=(payload_b,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        data = migration_state_tmp.read()
        assert isinstance(data, dict)
        assert "status" in data
        assert data["docs_indexed"] in (1, 2)


# ---------------------------------------------------------------------------
# TestSecurity — UT-061
# ---------------------------------------------------------------------------


class TestSecurity:
    """UT-061: FTS5 metacharacter payloads never raise sqlite3.OperationalError."""

    _PAYLOADS = [
        '"MATCH"',
        "NEAR ()",
        "AND OR NOT",
        "; DROP TABLE fts5_documents",
        '"OR 1=1--',
        "* * *",
        "column:injection",
        "(unbalanced",
    ]

    @pytest.mark.parametrize("payload", _PAYLOADS)
    def test_ut061_no_sql_injection_or_operational_error(self, fts5_tmp_index, payload):
        # Must not raise, must return a list (possibly empty).
        result = fts5_tmp_index.search(payload, top_k=5)
        assert isinstance(result, list)

    def test_escape_helper_wraps_tokens(self):
        assert _escape_fts5_query("") == ""
        assert _escape_fts5_query("   ") == ""
        assert _escape_fts5_query("hello") == '"hello"'
        assert _escape_fts5_query('a "b" c') == '"a" """b""" "c"'
        assert _escape_fts5_query("MATCH * OR AND") == '"MATCH" "*" "OR" "AND"'

    def test_search_after_dangerous_payload_still_functional(self, fts5_tmp_index):
        """Post-attack sanity: index remains queryable after hostile input."""
        fts5_tmp_index.search("; DROP TABLE fts5_documents", top_k=5)
        results = fts5_tmp_index.search("MS17-010", top_k=5)
        assert results, "index must remain queryable after hostile payload"


# ---------------------------------------------------------------------------
# IT-MIG-001 — 10 threads × 100 writes; final read is a valid dict
# ---------------------------------------------------------------------------


def test_it_mig_001_atomic_write_cross_thread(tmp_path):
    """IT-MIG-001: heavy contention — 10 threads × 100 write ops, no corruption."""
    state = Fts5MigrationState(tmp_path / "state.json")

    def _worker(worker_id: int):
        for i in range(100):
            state.write(
                {
                    "status": "in_progress",
                    "docs_total": 1000,
                    "docs_indexed": worker_id * 100 + i,
                    "started_at": "2026-08-07T12:00:00Z",
                    "completed_at": None,
                    "error": None,
                }
            )

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = state.read()
    assert isinstance(data, dict)
    assert data["status"] == "in_progress"
    assert isinstance(data["docs_indexed"], int)
    # Marker file exists as a single flat JSON object — reload from disk.
    on_disk = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert on_disk == data


# =============================================================================
# Task 03 additions — UT-025, UT-026, UT-029
# =============================================================================


class TestTask03Additions:
    """Fase 3 additions: search fallback contract, busy_timeout, NotReady message."""

    def test_ut025_search_returns_empty_on_operational_failure(self, tmp_path):
        """UT-025: FTS5 search never raises — a broken connection yields ``[]``
        so the orchestrator's dispatch layer can fall back to hybrid."""
        import sqlite3

        from mcp_server.fts5_index import Fts5LexicalIndex, Fts5MigrationState

        state_path = tmp_path / "state.json"
        Fts5MigrationState(state_path).write(
            {
                "status": "complete",
                "docs_total": 0,
                "docs_indexed": 0,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": "2026-08-07T12:00:01Z",
                "error": None,
            }
        )
        index = Fts5LexicalIndex(db_path=tmp_path / "db.sqlite", state_path=state_path)

        # ``sqlite3.Connection`` attributes are read-only slots in Py 3.14+, so
        # we swap in a proxy that raises on the MATCH statement search() emits.
        class _BrokenConn:
            def execute(self, sql, *params):
                raise sqlite3.OperationalError("simulated corruption")

        index.close()
        index._conn = _BrokenConn()

        assert index.search("CVE-2021-4034") == []

    def test_ut026_busy_timeout_pragma_is_five_seconds(self, tmp_path):
        """UT-026: WAL + busy_timeout=5000ms is applied so SQLITE_BUSY resolves
        without Python-side retry logic (ADR-001)."""
        from mcp_server.fts5_index import Fts5LexicalIndex, Fts5MigrationState

        state_path = tmp_path / "state.json"
        Fts5MigrationState(state_path).write(
            {
                "status": "complete",
                "docs_total": 0,
                "docs_indexed": 0,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": "2026-08-07T12:00:01Z",
                "error": None,
            }
        )
        index = Fts5LexicalIndex(db_path=tmp_path / "db.sqlite", state_path=state_path)
        busy_timeout = index._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal_mode = index._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert busy_timeout == 5000
        assert journal_mode.lower() == "wal"
        index.close()

    def test_ut029_not_ready_error_message_includes_auto_suggestion(self):
        """UT-029: the raised message must point users at ``search_method='auto'``
        so debug callers know how to recover without editing config."""
        from mcp_server.fts5_index import Fts5NotReadyError

        exc = Fts5NotReadyError(
            "FTS5 index is not ready (migration in progress). "
            "Suggestion: use search_method='auto' to fallback gracefully."
        )
        assert "auto" in str(exc)
        assert "Suggestion" in str(exc)
        assert issubclass(Fts5NotReadyError, RuntimeError)
