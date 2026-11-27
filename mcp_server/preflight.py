"""Startup preflight checks for persistent ChromaDB state.

GH #216 (v4.9.1): probe retries + inconclusive-on-timeout + atomic rename.
The previous single-attempt quarantine turned any transient failure
(file locks after client restart, slow disk, cold GPU init) into a
full-corpus reindex loop on Windows stdio MCP clients. The reporter's
forensic analysis on issue #216 documented the failure mode and the
fixes below were validated on their environment before merging. The probe
holds the access guard of ``_index_access`` while it reads; the repair half
lives in ``_preflight_repair``.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from . import _index_access, _preflight_probe, _writer_role, instance_lock
from ._preflight_probe import _classify, _index_id, _Outcome, _Target
from ._preflight_repair import _log, move_index_aside
from .config import _get, config

PROBE_ATTEMPTS = 3
PROBE_RETRY_DELAY_SECONDS = 5
PROBE_TIMEOUT_SECONDS = 30
# Cloud-sync folders break ChromaDB (HNSW mmap) and SQLite WAL/SHM:
# the provider opens handles on live files to upload them, racing with
# the engine's own mmap/WAL writes → native access violation (0xC0000005
# on Windows, SIGBUS on macOS/Linux) → corrupted segments → auto-repair
# wipes the index. Match on any path segment to catch OneDrive, iCloud,
# Dropbox, Google Drive, Box, pCloud, Mega, Nextcloud, Sync.com, MEGAsync.
_CLOUD_SYNC_PATTERNS = re.compile(
    r"(?:^|[\\/])("
    r"OneDrive(?:[ -][^\\/]*)?"
    r"|iCloud(?:[ ]Drive)?|com~apple~CloudDocs"
    r"|Dropbox(?:[ -][^\\/]*)?"
    r"|Google[ ]Drive|GoogleDrive|My[ ]Drive"
    r"|Box(?:[ ]Sync)?"
    r"|pCloudDrive|pCloud"
    r"|MEGA(?:sync)?"
    r"|Nextcloud|ownCloud"
    r"|Sync|Syncthing"
    r")(?:[\\/]|$)"
)


def _detect_cloud_sync_provider(path: Path) -> str | None:
    """Return the matched provider segment when ``path`` resolves under a
    known cloud-sync folder, otherwise ``None``. Case-insensitive.
    """
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    match = _CLOUD_SYNC_PATTERNS.search(resolved) or _CLOUD_SYNC_PATTERNS.search(resolved.replace("/", "\\"))
    if match:
        return match.group(1)
    # Case-insensitive fallback for odd-cased folders.
    match_ci = _CLOUD_SYNC_PATTERNS.search(resolved, re.IGNORECASE) if hasattr(re, "search") else None
    return match_ci.group(1) if match_ci else None


def _check_cloud_sync() -> None:
    """Abort startup when ``data_dir`` lives under a cloud-sync folder,
    unless ``paths.allow_cloud_sync: true`` is set as an opt-in override.

    Cloud sync corrupts ChromaDB HNSW segments (mmap) and SQLite WAL —
    see _CLOUD_SYNC_PATTERNS docstring for the failure mode.
    """
    provider = _detect_cloud_sync_provider(config.data_dir)
    if not provider:
        return
    allow = bool(_get("paths", "allow_cloud_sync", False))
    if allow:
        _log(
            f"[PREFLIGHT] WARNING: data_dir is under cloud sync ({provider}): {config.data_dir}. "
            f"Index corruption is likely. Running anyway because paths.allow_cloud_sync=true.",
            config.data_dir,
        )
        return
    _log(
        f"[PREFLIGHT] FATAL: data_dir is under cloud sync ({provider}): {config.data_dir}\n"
        f"  Cloud sync corrupts ChromaDB (HNSW mmap) and SQLite WAL — the engine will crash with\n"
        f"  access violations (0xC0000005 on Windows, SIGBUS on *nix) and auto-repair will wipe\n"
        f"  the index.\n"
        f"  Fix: move paths.data_dir to a local path (e.g. %LOCALAPPDATA%\\knowledge-rag\\data on\n"
        f"  Windows, ~/.local/share/knowledge-rag/data on Linux/macOS), or set paths.allow_cloud_sync=true\n"
        f"  to acknowledge the risk and continue.",
        config.data_dir,
    )
    raise SystemExit(78)  # EX_CONFIG


# -I ignores cwd and PYTHONPATH. Verify the exact sibling module before importing anything that can read the index.
_PROBE_MODULE = Path(_preflight_probe.__file__).resolve()
_PROBE_PROLOGUE = r"""
import sys
from importlib.util import find_spec
from pathlib import Path

origin = Path(sys.argv[1])
sys.path.insert(0, str(origin.parent.parent))
spec = find_spec("mcp_server._preflight_probe")
if spec is None or not spec.origin or Path(spec.origin).resolve() != origin:
    raise ImportError(f"preflight module is not from {origin}")
from mcp_server._preflight_probe import _enter, _read_index

target = _enter(sys.argv[2])
"""
_PROBE_BODY = "_read_index(target)\n"


def _capture_target() -> _Target:
    """One parent snapshot, shared by every child, the final guard and both renames."""
    configured_data = config.data_dir.absolute()
    data_dir = configured_data.resolve()  # a data_dir alias is safe: the guard follows it too
    chroma_dir = config.chroma_dir.absolute()
    if chroma_dir.is_relative_to(configured_data):
        chroma_dir = data_dir / chroma_dir.relative_to(configured_data)
    return _Target(data_dir, chroma_dir, config.collection_name, _index_id(chroma_dir))


def _probe_command(target: _Target) -> list[str]:
    return [sys.executable, "-I", "-c", _PROBE_PROLOGUE + _PROBE_BODY, str(_PROBE_MODULE), target.wire()]


def _probe_chroma(target: _Target, timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    """Isolate native crashes and never let a probe consume the stdio client's requests."""
    if _index_access.is_held():
        raise RuntimeError("this process holds the index access guard, so its probe would find the index busy")
    return subprocess.run(
        _probe_command(target),
        cwd=_PROBE_MODULE.parent,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


def run_preflight(timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True only for a confirmed repair. Timeout keeps the index (#216).

    An unverifiable probe raises, not a healthy result. Exit 75 means a busy guard or
    a blocked move; exit 78 rejects a cloud-sync directory. Borrow an eligible caller's
    writer guard without releasing it, or hold our own for this call only.
    """
    _check_cloud_sync()
    target = _capture_target()
    if _index_access.is_held():
        _log("[PREFLIGHT] this process serves the index; the standalone check did not run", target.data_dir)
        raise SystemExit(_index_access.BUSY_EXIT_CODE)
    target.data_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as owned:
        try:
            owned.enter_context(instance_lock.standalone_guard(target.data_dir / instance_lock.LOCK_FILENAME))
        except instance_lock.AlreadyRunningError as exc:
            _log(f"[PREFLIGHT] a knowledge-rag server owns {target.data_dir}; the check did not run", target.data_dir)
            raise SystemExit(_index_access.BUSY_EXIT_CODE) from exc
        return _probe_and_repair_for_call(target, timeout_seconds)


def _probe_and_repair_for_call(target: _Target, timeout_seconds: int) -> bool:
    """The standalone caller holds the writer guard until this decision has finished."""
    try:
        failure = _probe_until_conclusive(target, timeout_seconds)
        with _index_access.held_for_call(target.access_path):
            return _repair_if_damaged(target, failure)
    except _index_access.IndexAccessBusy as exc:
        _log(f"[PREFLIGHT] {exc}", target.data_dir)
        raise SystemExit(_index_access.BUSY_EXIT_CODE) from exc


def _probe_until_conclusive(target: _Target, timeout_seconds: int) -> _Outcome | None:
    """Require independent, matching read failures. Neither bootstrap nor stderr is evidence."""
    failures: list[_Outcome] = []
    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            outcome = _probe_inside(target, timeout_seconds)
        except subprocess.TimeoutExpired:
            _log(f"[PREFLIGHT] probe timed out after {timeout_seconds}s — inconclusive, keeping index", target.data_dir)
            return None
        if outcome is _Outcome.VERIFIED:
            if failures:
                _log(f"[PREFLIGHT] attempt {attempt}: OK (recovered from transient failure)", target.data_dir)
            return None
        failures.append(outcome)
        if attempt < PROBE_ATTEMPTS:
            time.sleep(PROBE_RETRY_DELAY_SECONDS)
    if len(failures) >= 2 and len(set(failures)) == 1 and failures[0] in (_Outcome.HNSW, _Outcome.NATIVE):
        return failures[0]
    raise RuntimeError(
        f"Could not verify collection {target.collection_name!r} in {target.chroma_dir}; "
        "index left in place (see preflight.log)."
    )


def _probe_inside(target: _Target, timeout_seconds: int) -> _Outcome:
    """A busy child never entered; wait separately so it cannot confirm or disprove a read failure."""
    deadline = time.monotonic() + _index_access.WAIT_SECONDS
    while True:
        done = _probe_chroma(target, timeout_seconds=timeout_seconds)
        outcome = _classify(target, done)
        if outcome is not _Outcome.BUSY:
            if outcome is not _Outcome.VERIFIED:
                _log(
                    f"[PREFLIGHT] probe {outcome.value}, exit {done.returncode}: {(done.stderr or '').strip()[-1500:]}",
                    target.data_dir,
                )
            return outcome
        if time.monotonic() >= deadline:
            raise _index_access.busy(_index_access.WAIT_SECONDS)
        _log(
            f"[PREFLIGHT] another process is inside the index; probing again in {PROBE_RETRY_DELAY_SECONDS}s",
            target.data_dir,
        )
        time.sleep(PROBE_RETRY_DELAY_SECONDS)


def _probe_and_repair(timeout_seconds: int = PROBE_TIMEOUT_SECONDS) -> bool:
    """Server path: retain access for the subsequent opening; refuse unverifiable startup."""
    target = _capture_target()
    failure = _probe_until_conclusive(target, timeout_seconds)
    _index_access.hold(path=target.access_path)
    with _writer_role.startup_repair_guard():
        return _repair_if_damaged(target, failure)


def _repair_if_damaged(target: _Target, failure: _Outcome | None) -> bool:
    target.check_identity()  # held by the parent, after all children have exited
    if failure is None:
        return False
    if failure not in (_Outcome.HNSW, _Outcome.NATIVE) or None in target.index_id:
        raise RuntimeError("Index repair requires confirmed damage")
    _log(f"[RECOVERY] repeated {failure.value} during collection.count(); moving index aside", target.data_dir)
    move_index_aside(target, failure.value)
    return True
