"""Startup preflight checks for persistent ChromaDB state.

GH #216 (v4.9.1): probe retries + inconclusive-on-timeout + atomic rename.
The previous single-attempt quarantine turned any transient failure
(file locks after client restart, slow disk, cold GPU init) into a
full-corpus reindex loop on Windows stdio MCP clients. The reporter's
forensic analysis on issue #216 documented the failure mode and the
fixes below were validated on their environment before merging.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import BASE_DIR, config

PROBE_ATTEMPTS = 3
PROBE_RETRY_DELAY_SECONDS = 5
PROBE_TIMEOUT_SECONDS = 30


def _log(msg: str) -> None:
    """Emit preflight message to stderr and append to data/preflight.log.

    Persisting to a file matters because stdio MCP clients hide the server's
    stderr in their own logs — without the file, users could only ever see
    the truncated exit code, not the actual Chroma stack trace.
    """
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, file=sys.stderr)
    try:
        log_file = config.data_dir / "preflight.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _backup_active_index(reason: str) -> Path:
    """Atomically move active ChromaDB state aside so the server can rebuild.

    Uses os.rename (atomic on same volume) instead of shutil.move — the
    fallback copy+rmtree path in shutil.move leaks a partial copy in
    backups/ AND a partially deleted live chroma_db/ when another process
    holds a segment file open (WinError 32).
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = config.data_dir / "backups" / f"auto-repair-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    if config.chroma_dir.exists():
        os.rename(str(config.chroma_dir), str(backup_dir / f"chroma_db.{reason}"))

    metadata_file = config.data_dir / "index_metadata.json"
    if metadata_file.exists():
        os.rename(str(metadata_file), str(backup_dir / f"index_metadata.{reason}.json"))

    return backup_dir


def _probe_chroma(timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    """Check Chroma in a child process so native crashes do not kill MCP startup."""
    code = r"""
import chromadb

from mcp_server.config import config

if not config.chroma_dir.exists():
    print("missing")
    raise SystemExit(0)

client = chromadb.PersistentClient(path=str(config.chroma_dir))
collection = client.get_or_create_collection(name=config.collection_name)
print(collection.count())
"""
    env = os.environ.copy()
    env.setdefault("KNOWLEDGE_RAG_DIR", str(BASE_DIR))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BASE_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )


def run_preflight(timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True when active Chroma state was moved aside for repair.

    Runs the probe up to PROBE_ATTEMPTS times so transient failures
    (file locks lingering from a previous PID, slow disk, cold GPU init)
    do not trigger a destructive quarantine. TimeoutExpired is treated
    as inconclusive — the index is kept and the user can debug from
    preflight.log without losing the corpus. Only PROBE_ATTEMPTS
    consecutive non-zero exits quarantine.

    If os.rename fails while quarantining (e.g. another process still
    holds a segment file open on Windows) the index is left untouched
    and the process exits with EX_TEMPFAIL (75) instead of crashing
    halfway with a partial move.
    """
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            result = _probe_chroma(timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired:
            _log(
                f"[PREFLIGHT] attempt {attempt}/{PROBE_ATTEMPTS}: probe timed out "
                f"after {timeout_seconds}s — inconclusive, keeping index"
            )
            return False
        if result.returncode == 0:
            if attempt > 1:
                _log(f"[PREFLIGHT] attempt {attempt}: OK (recovered from transient failure)")
            return False
        stderr_tail = (result.stderr or "").strip()[-1500:]
        _log(
            f"[PREFLIGHT] attempt {attempt}/{PROBE_ATTEMPTS}: probe failed with code {result.returncode}\n{stderr_tail}"
        )
        last_result = result
        if attempt < PROBE_ATTEMPTS:
            time.sleep(PROBE_RETRY_DELAY_SECONDS)

    assert last_result is not None
    reason = "segfault" if last_result.returncode in (-11, 139) else "failed"
    _log(f"[RECOVERY] preflight failed {PROBE_ATTEMPTS}x (last exit code {last_result.returncode}); moving index aside")
    try:
        backup_dir = _backup_active_index(reason)
    except OSError as exc:
        _log(
            f"[PREFLIGHT] could not move index aside ({exc}); leaving it untouched "
            f"to avoid partial corruption. Retry after freeing file handles."
        )
        raise SystemExit(75) from exc
    _log(f"[RECOVERY] moved active index to {backup_dir}")
    return True
