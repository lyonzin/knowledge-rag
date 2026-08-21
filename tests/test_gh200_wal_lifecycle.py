"""GH #200 regression suite — SQLite WAL lifecycle.

Bug (reported by @admincheg on v4.8.5): in HTTP/SSE mode the server created
``chromadb.PersistentClient`` and *then* opened a second ``sqlite3`` connection
to force ``PRAGMA journal_mode=WAL``. The Chroma Rust binding keeps a live sqlx
pool, so switching journal mode underneath it left stale ``-wal``/``-shm``
handles and the first ``Collection.add()`` failed during compaction with
``SQLITE_NOTADB`` (error 26, "file is not a database").

Root cause is ordering + lifecycle, confirmed independently by upstream
chroma-core/chroma#7040 ("configure WAL before starting Chroma, not underneath
live connections"). The fix moves the WAL switch to *before* the client opens
the DB, makes it idempotent, and skips WAL on network filesystems where it is
unsafe.

These tests lock the ordering, the idempotency, and the network-FS guard so a
future refactor cannot quietly reintroduce the post-client toggle.
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

from mcp_server import server as server_module

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_chroma_db(dir_path: Path) -> Path:
    """Create a minimal real ``chroma.sqlite3`` in DELETE journal mode."""
    dir_path.mkdir(parents=True, exist_ok=True)
    db = dir_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
    conn.commit()
    conn.close()
    return db


def _journal_mode(db: Path) -> str:
    conn = sqlite3.connect(str(db))
    try:
        return str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
    finally:
        conn.close()


def _new_orch():
    """Bare KnowledgeOrchestrator instance without running the heavy __init__."""
    return server_module.KnowledgeOrchestrator.__new__(server_module.KnowledgeOrchestrator)


# ---------------------------------------------------------------------------
# ordering — the core of the bug
# ---------------------------------------------------------------------------


def test_gh200_wal_toggled_before_client_opens(monkeypatch):
    """HTTP transport: WAL switch must happen BEFORE PersistentClient opens."""
    order: list[str] = []
    monkeypatch.setattr(server_module.config, "transport", "streamable-http")
    monkeypatch.setattr(server_module, "_enable_wal_mode", lambda d: order.append("wal"))

    def fake_client(path):
        order.append("client")
        return object()

    monkeypatch.setattr(server_module.chromadb, "PersistentClient", fake_client)

    client = _new_orch()._init_chroma_client()

    assert order == ["wal", "client"], f"WAL must precede the Rust client, got {order}"
    assert client is not None


def test_gh200_stdio_keeps_default_journal_mode(monkeypatch):
    """stdio is single-process: never touch journal mode, only open the client."""
    order: list[str] = []
    monkeypatch.setattr(server_module.config, "transport", "stdio")
    monkeypatch.setattr(server_module, "_enable_wal_mode", lambda d: order.append("wal"))
    monkeypatch.setattr(
        server_module.chromadb, "PersistentClient", lambda path: order.append("client") or object()
    )

    _new_orch()._init_chroma_client()

    assert order == ["client"], f"stdio must not toggle WAL, got {order}"


def test_gh200_init_toggles_wal_before_opening_client():
    """Source-level guard: the ordering can't be silently flipped by a refactor."""
    src = inspect.getsource(server_module.KnowledgeOrchestrator._init_chroma_client)
    # Match the actual calls (with "(") so docstring mentions don't skew the order.
    wal_pos = src.find("_enable_wal_mode(")
    client_pos = src.find("chromadb.PersistentClient(")
    assert wal_pos != -1 and client_pos != -1
    assert wal_pos < client_pos, "WAL toggle must appear before PersistentClient in _init_chroma_client"


def test_gh200_init_does_not_toggle_wal_after_client():
    """__init__ must delegate to _init_chroma_client, never re-toggle post-client."""
    src = inspect.getsource(server_module.KnowledgeOrchestrator.__init__)
    assert "_enable_wal_mode" not in src, "WAL toggle must live in _init_chroma_client, not in __init__"
    assert "_init_chroma_client" in src, "__init__ must build the client via _init_chroma_client"


# ---------------------------------------------------------------------------
# _enable_wal_mode behaviour
# ---------------------------------------------------------------------------


def test_gh200_enable_wal_sets_wal_on_local_db(tmp_path, monkeypatch):
    """A local DELETE-mode DB is switched to WAL (and it sticks in the header)."""
    monkeypatch.setattr(server_module, "_is_network_filesystem", lambda p: False)
    db = _make_chroma_db(tmp_path)
    assert _journal_mode(db) != "wal"  # starts in delete

    server_module._enable_wal_mode(tmp_path)

    assert _journal_mode(db) == "wal"


def test_gh200_enable_wal_is_idempotent(tmp_path, monkeypatch, capsys):
    """A DB already in WAL is not re-toggled; the switch only announces once."""
    monkeypatch.setattr(server_module, "_is_network_filesystem", lambda p: False)
    db = _make_chroma_db(tmp_path)

    server_module._enable_wal_mode(tmp_path)
    first = capsys.readouterr()
    server_module._enable_wal_mode(tmp_path)
    second = capsys.readouterr()

    assert _journal_mode(db) == "wal"
    assert "WAL mode enabled" in (first.out + first.err)
    assert "WAL mode enabled" not in (second.out + second.err), "second call must be a no-op"


def test_gh200_enable_wal_noop_when_db_absent(tmp_path, monkeypatch):
    """Missing chroma.sqlite3 returns before probing the filesystem — no crash."""
    probed = {"net": False}

    def spy(_path):
        probed["net"] = True
        return False

    monkeypatch.setattr(server_module, "_is_network_filesystem", spy)
    server_module._enable_wal_mode(tmp_path)  # no DB created

    assert probed["net"] is False, "absent DB must early-return before the FS probe"


def test_gh200_wal_skipped_on_network_filesystem(tmp_path, monkeypatch, capsys):
    """On NFS/SMB/CIFS the DB is left in its default mode (WAL would corrupt)."""
    db = _make_chroma_db(tmp_path)
    monkeypatch.setattr(server_module, "_is_network_filesystem", lambda p: True)

    server_module._enable_wal_mode(tmp_path)

    captured = capsys.readouterr()
    assert _journal_mode(db) != "wal", "WAL must not be enabled on a network filesystem"
    assert "network filesystem" in (captured.err + captured.out)


def test_gh200_enable_wal_swallows_sqlite_errors(tmp_path, monkeypatch, capsys):
    """A locked/broken DB must warn, never crash startup."""
    monkeypatch.setattr(server_module, "_is_network_filesystem", lambda p: False)
    _make_chroma_db(tmp_path)

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", boom)
    server_module._enable_wal_mode(tmp_path)  # must not raise

    captured = capsys.readouterr()
    assert "Could not enable WAL mode" in (captured.err + captured.out)


# ---------------------------------------------------------------------------
# network filesystem detection
# ---------------------------------------------------------------------------


_MOUNTS = (
    "proc /proc proc rw 0 0\n"
    "/dev/sda1 / ext4 rw 0 0\n"
    "server:/export /home/user/.kr/data nfs4 rw 0 0\n"
    "//nas/share /mnt/share cifs rw 0 0\n"
)


def test_gh200_network_match_detects_nfs():
    assert server_module._network_fstype_match("/home/user/.kr/data/chromadb", _MOUNTS) is True


def test_gh200_network_match_detects_cifs():
    assert server_module._network_fstype_match("/mnt/share/db", _MOUNTS) is True


def test_gh200_network_match_local_ext4_is_not_network():
    assert server_module._network_fstype_match("/home/user/proj/data", _MOUNTS) is False


def test_gh200_network_match_longest_mount_wins():
    """A network mount nested under the local root must win for paths under it."""
    assert server_module._network_fstype_match("/home/user/.kr/data", _MOUNTS) is True
    assert server_module._network_fstype_match("/var/lib/x", _MOUNTS) is False


def test_gh200_windows_unc_path_is_network():
    assert server_module._is_network_filesystem(Path(r"\\nas\share\kr")) is True


def test_gh200_missing_proc_mounts_falls_open_to_local(monkeypatch):
    """No /proc/mounts (macOS, containers) → treated as local, not blocked."""
    real_exists = Path.exists

    def fake_exists(self):
        if str(self) == "/proc/mounts":
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert server_module._is_network_filesystem(Path("/home/user/.kr/data")) is False
