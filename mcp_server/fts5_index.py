"""FTS5 lexical index encapsulation for the knowledge-rag fast-path feature.

Isolated module — instantiated only when ``config.fts5_enabled`` is true and
wired into ``KnowledgeOrchestrator`` (Task 03). Storage layout, tokenizer,
and PRAGMAs are pinned by ADR-001 and ADR-005.

References:
- ADR-001: ``<data_dir>/fts5_index.db`` with WAL + busy_timeout=5000ms
- ADR-004: exception hierarchy — each error subclasses ``RuntimeError`` directly
- ADR-005: tokenizer ``unicode61 remove_diacritics 2 tokenchars '-_.'``
- TechSpec §Impl.Design.Core Interfaces #2, #3
- TechSpec §Security Surface — prepared statements + FTS5 metacharacter escape
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

ChunkRow = Tuple[str, str, str, str]
ChunkIterFactory = Callable[[], Iterable[ChunkRow]]
ProgressCallback = Callable[[int, int], None]

_FTS5_TOKENIZER = "unicode61 remove_diacritics 2 tokenchars '-_.'"

_FTS5_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS fts5_documents USING fts5(
    chunk_id UNINDEXED,
    content,
    filename,
    category,
    tokenize = "{_FTS5_TOKENIZER}"
);
"""

# Metacharacters that carry syntactic meaning in an FTS5 MATCH expression.
# Wrapping each user-provided token in double quotes turns it into a literal
# phrase token, which neutralises operators (AND/OR/NOT/NEAR), prefix wildcard
# (``*``), column filters (``:``), grouping (``(``, ``)``) and injected
# quotes. See TechSpec §Security Surface.
_FTS5_TOKEN_SPLIT = re.compile(r"\s+")


def _escape_fts5_query(query: str) -> str:
    """Return an FTS5 MATCH expression that treats ``query`` as literal tokens.

    Doubles embedded quotes (FTS5 escape convention) and wraps every
    whitespace-separated fragment in double quotes so operators cannot leak.
    Empty input becomes an empty string — callers must skip the search.
    """
    stripped = query.strip()
    if not stripped:
        return ""
    parts = [p for p in _FTS5_TOKEN_SPLIT.split(stripped) if p]
    quoted = ['"' + p.replace('"', '""') + '"' for p in parts]
    return " ".join(quoted)


class Fts5NotReadyError(RuntimeError):
    """Raised when the fast-path is invoked before the FTS5 index is ready.

    Message includes a suggestion to fall back to ``search_method='auto'``
    (PRD OQ-4) so debug users can recover without editing config.
    """


class Fts5CorruptError(RuntimeError):
    """Raised when the FTS5 database file cannot be opened or is malformed."""


class Fts5MigrationError(RuntimeError):
    """Raised when the initial FTS5 rebuild fails. See marker file for cause."""


class Fts5MigrationState:
    """Atomic JSON marker for the FTS5 rebuild lifecycle (PRD OQ-5, Q3).

    Schema::

        {
            "status": "complete" | "in_progress" | "failed",
            "docs_total": int,
            "docs_indexed": int,
            "started_at": ISO8601 str,
            "completed_at": ISO8601 str | None,
            "error": str | None,
        }

    Writes are cross-platform atomic: a NamedTemporaryFile in the same
    directory is fsynced and swapped in via ``os.replace``.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Optional[dict]:
        """Return the persisted payload, or ``None`` if the file is missing.

        Silently returns ``None`` on JSON decode failure — callers treat a
        corrupt marker the same as a missing one and rebuild from scratch.
        """
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def write(self, payload: dict) -> None:
        """Persist ``payload`` atomically (tempfile + fsync + os.replace)."""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        tmp_dir = self._path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(prefix=".fts5_state.", suffix=".tmp", dir=str(tmp_dir))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    json.dump(payload, tmp)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                os.replace(tmp_name, self._path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def is_complete(self) -> bool:
        data = self.read()
        return bool(data) and data.get("status") == "complete"


class Fts5LexicalIndex:
    """SQLite FTS5 wrapper — search, connection lifecycle, readiness flag.

    Task 05 will extend this class with ``add_document``/``remove_document``/
    ``update_document``/``start_migration_background``. For Fase 1 only
    read-side + lifecycle primitives ship.
    """

    def __init__(self, db_path: Path, state_path: Path) -> None:
        self._db_path = Path(db_path)
        self._state_path = Path(state_path)
        # Q2 (TechSpec): dedicated RLock, independent of BM25 build lock.
        self._fts5_lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ready: bool = False
        self._connect_and_configure()
        self._migration_state = Fts5MigrationState(self._state_path)
        self._ready = self._migration_state.is_complete()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def state(self) -> Fts5MigrationState:
        return self._migration_state

    def is_ready(self) -> bool:
        return self._ready

    def _connect_and_configure(self) -> None:
        """Open the SQLite connection and apply ADR-001 PRAGMAs + schema."""
        try:
            in_memory = str(self._db_path) == ":memory:"
            if not in_memory:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=5.0,
            )
            self._apply_pragmas(in_memory=in_memory)
            self._verify_fts5_available()
            # _FTS5_SCHEMA is a hardcoded module constant with the tokenizer
            # literal interpolated at import time from another module constant
            # — zero user input reaches this execute() call, zero injection
            # surface. Semgrep rules flag the f-string source pattern
            # regardless of taint origin, so inline suppression is justified.
            self._conn.execute(_FTS5_SCHEMA)  # nosem
            self._conn.commit()
        except sqlite3.DatabaseError as exc:
            raise Fts5CorruptError(f"Failed to open FTS5 index at {self._db_path}: {exc}") from exc

    def _apply_pragmas(self, *, in_memory: bool) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        if not in_memory:
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    def _verify_fts5_available(self) -> None:
        """Confirm the SQLite build exposes FTS5. Raises ``Fts5CorruptError``."""
        assert self._conn is not None
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
            self._conn.execute("DROP TABLE IF EXISTS _fts5_probe")
        except sqlite3.OperationalError as exc:
            raise Fts5CorruptError(f"SQLite build lacks FTS5 support or schema drift detected: {exc}") from exc

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return ``[(chunk_id, score)]`` sorted by best rank.

        Mirrors ``BM25Index.search`` signature (score is positive; higher is
        better — we invert FTS5's negative bm25 rank). Never raises for
        malformed input: metacharacter tokens are quoted so a well-formed
        MATCH is always issued or an empty list is returned.
        """
        if self._conn is None:
            return []
        if top_k <= 0:
            return []
        escaped = _escape_fts5_query(query)
        if not escaped:
            return []
        sql = (
            "SELECT chunk_id, bm25(fts5_documents) AS rank "
            "FROM fts5_documents WHERE fts5_documents MATCH ? "
            "ORDER BY rank LIMIT ?"
        )
        try:
            with self._fts5_lock:
                cur = self._conn.execute(sql, (escaped, int(top_k)))
                rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Malformed MATCH survived escaping (defensive) — treat as no hits.
            return []
        # FTS5 bm25() returns lower-is-better; invert so callers see the same
        # "higher-is-better" contract BM25Index exposes.
        return [(str(chunk_id), -float(rank)) for chunk_id, rank in rows]

    def close(self) -> None:
        """Release the SQLite connection. Safe to call multiple times."""
        with self._fts5_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -----------------------------------------------------------------
    # CRUD sync (Task 05, ADR-008). SQL nativo incremental — diverge do
    # BM25 full-rebuild pattern porque FTS5 tem INSERT/DELETE O(1) e o
    # full rebuild custaria segundos por mutation em corpus 3865 docs.
    # Todos os writes acquire ``_fts5_lock`` (RLock, Q2 do TechSpec) e
    # sao serializados pelo WAL SQLite (ADR-001).
    # -----------------------------------------------------------------

    def add_document(self, chunk_id: str, content: str, filename: str, category: str) -> None:
        """Insert one chunk row via ``INSERT`` (ADR-008)."""
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            self._conn.execute(
                "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                (chunk_id, content, filename, category),
            )
            self._conn.commit()

    def remove_document(self, chunk_id: str) -> None:
        """Delete every row matching ``chunk_id`` (ADR-008)."""
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            self._conn.execute(
                "DELETE FROM fts5_documents WHERE chunk_id = ?",
                (chunk_id,),
            )
            self._conn.commit()

    def update_document(self, chunk_id: str, content: str, filename: str, category: str) -> None:
        """DELETE + INSERT atomico — FTS5 nao tem UPDATE efficient em virtual table."""
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            self._conn.execute("DELETE FROM fts5_documents WHERE chunk_id = ?", (chunk_id,))
            self._conn.execute(
                "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                (chunk_id, content, filename, category),
            )
            self._conn.commit()

    # -----------------------------------------------------------------
    # Migration lifecycle (Task 05). ``start_migration_background``
    # dispara uma thread nao-daemon (join graceful no shutdown), que
    # persiste checkpoint a cada 100 docs no marker file. Retomavel:
    # se marker mostra ``in_progress`` + ``docs_indexed=N``, o worker
    # skipa as primeiras N rows do iterator e continua do batch N+1.
    # -----------------------------------------------------------------

    def start_migration_background(
        self,
        chunk_iter_factory: ChunkIterFactory,
        docs_total: int,
        *,
        resume_from: int = 0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> threading.Thread:
        """Launch the migration daemon thread. Returns the started thread."""
        thread = threading.Thread(
            target=self._migration_worker,
            args=(chunk_iter_factory, docs_total, resume_from, on_progress),
            name="fts5-migration",
            daemon=False,
        )
        thread.start()
        return thread

    def _migration_worker(
        self,
        chunk_iter_factory: ChunkIterFactory,
        docs_total: int,
        resume_from: int,
        on_progress: Optional[ProgressCallback],
    ) -> None:
        started_at = datetime.now(timezone.utc).isoformat()
        docs_indexed = int(resume_from)
        self._write_state("in_progress", docs_total, docs_indexed, started_at, None, None)
        try:
            docs_indexed = self._run_migration_batches(
                chunk_iter_factory, docs_total, docs_indexed, started_at, on_progress
            )
        except Exception as exc:  # noqa: BLE001 — one place to record every failure
            self._write_state(
                "failed",
                docs_total,
                docs_indexed,
                started_at,
                None,
                f"{exc.__class__.__name__}: {exc}",
            )
            print(f"[FTS5] migration failed at {docs_indexed}/{docs_total}: {exc}")
            return
        completed_at = datetime.now(timezone.utc).isoformat()
        self._write_state("complete", docs_total, docs_indexed, started_at, completed_at, None)
        with self._fts5_lock:
            self._ready = True
        print(f"[FTS5] migration complete: {docs_indexed} docs indexed")

    def _run_migration_batches(
        self,
        chunk_iter_factory: ChunkIterFactory,
        docs_total: int,
        docs_indexed: int,
        started_at: str,
        on_progress: Optional[ProgressCallback],
    ) -> int:
        """Consume the iterator batch-by-batch. Returns the final ``docs_indexed``."""
        resume_from = docs_indexed
        seen = 0
        batch: List[ChunkRow] = []
        last_percent_logged = -10
        for row in chunk_iter_factory():
            if seen < resume_from:
                seen += 1
                continue
            seen += 1
            batch.append(row)
            if len(batch) >= 100:
                self._populate_batch(batch)
                docs_indexed += len(batch)
                batch = []
                self._write_state("in_progress", docs_total, docs_indexed, started_at, None, None)
                if on_progress is not None:
                    on_progress(docs_indexed, docs_total)
                last_percent_logged = self._maybe_log_progress(docs_indexed, docs_total, last_percent_logged)
        if batch:
            self._populate_batch(batch)
            docs_indexed += len(batch)
            if on_progress is not None:
                on_progress(docs_indexed, docs_total)
        return docs_indexed

    @staticmethod
    def _maybe_log_progress(docs_indexed: int, docs_total: int, last_percent_logged: int) -> int:
        """Emit an INFO log line every 10% of progress. Returns the new watermark."""
        if docs_total <= 0:
            return last_percent_logged
        percent = int(100 * docs_indexed / docs_total)
        if percent >= last_percent_logged + 10:
            print(f"[FTS5] migration progress: {percent}% ({docs_indexed}/{docs_total})")
            return percent
        return last_percent_logged

    def _populate_batch(self, rows: Sequence[ChunkRow]) -> None:
        """Insert a batch under ``_fts5_lock``. Raises on SQL failure."""
        if self._conn is None:
            raise Fts5MigrationError("FTS5 connection is closed during migration")
        with self._fts5_lock:
            self._conn.executemany(
                "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def _write_state(
        self,
        status: str,
        docs_total: int,
        docs_indexed: int,
        started_at: str,
        completed_at: Optional[str],
        error: Optional[str],
    ) -> None:
        """Persist the marker file with the canonical schema (TechSpec §Q3)."""
        self._migration_state.write(
            {
                "status": status,
                "docs_total": int(docs_total),
                "docs_indexed": int(docs_indexed),
                "started_at": started_at,
                "completed_at": completed_at,
                "error": error,
            }
        )
