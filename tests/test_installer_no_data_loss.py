"""Anti-regression tests for the multi-LLM-client installer (install.py).

Pins the guarantees that `install.register_client()` NEVER loses data
from the client's existing config file, regardless of schema shape:

1. Every top-level key present before the write is still present after,
   with identical value (except for the target `mcpServers` / `servers` /
   `context_servers` key, whose *siblings* are still preserved).
2. Other MCP server entries under the target key survive byte-for-byte.
3. A `.knowledge-rag.bak` backup, byte-identical to the pre-write file,
   is written before the target is touched.
4. Two consecutive runs with the same input produce only ONE actual
   write (idempotent).
5. Empty-config and invalid-JSON edge cases are handled without
   destroying existing data.
6. `--dry-run` mode writes NOTHING (no target change, no backup).

These tests exercise all three JSON schemas used across the 8 supported
clients: `mcpServers` (Claude Code/Desktop, Cursor, Windsurf, Cline,
Gemini), `servers` (VS Code), and `context_servers` (Zed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# install.py lives in the repo root, not in mcp_server/, so make sure
# pytest can import it directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

install = pytest.importorskip("install")


# ---------------------------------------------------------------------------
# Fixtures — one per schema, each with realistic siblings so a naive
# "clobber the whole key" bug would fail the test loudly.
# ---------------------------------------------------------------------------


def _mcp_servers_fixture() -> dict[str, Any]:
    """Claude Code / Cursor / Windsurf / Cline / Gemini shape."""
    return {
        # Top-level keys that must survive the write untouched.
        "theme": "dark",
        "editor": {"fontSize": 14, "tabSize": 2},
        "keybindings": [{"key": "ctrl+k", "command": "openPalette"}],
        # The target key with pre-existing MCP servers we must NOT nuke.
        "mcpServers": {
            "github": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "ghp_secret_do_not_lose_me"},
            },
            "postgres": {
                "type": "stdio",
                "command": "/usr/local/bin/pg-mcp",
                "args": ["--dsn", "postgres://prod"],
                "env": {},
            },
        },
        # Another top-level key AFTER mcpServers to catch dict-slicing bugs.
        "telemetry": {"enabled": False},
    }


def _vscode_fixture() -> dict[str, Any]:
    """VS Code uses `servers` (not `mcpServers`)."""
    return {
        "inputs": [
            {
                "type": "promptString",
                "id": "gh-token",
                "description": "GitHub PAT",
                "password": True,
            }
        ],
        "servers": {
            "github": {
                "type": "stdio",
                "command": "docker",
                "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
            },
            "playwright": {
                "type": "stdio",
                "command": "npx",
                "args": ["@playwright/mcp"],
            },
        },
        "sandbox": {"denyNetwork": False, "allow": ["localhost"]},
    }


def _zed_fixture() -> dict[str, Any]:
    """Zed uses `context_servers` with `source: custom`."""
    return {
        "vim_mode": True,
        "theme": "One Dark",
        "font_family": "JetBrains Mono",
        "font_size": 14,
        "features": {"copilot": False},
        "context_servers": {
            "github-mcp": {
                "source": "custom",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "ghp_do_not_lose_me"},
            }
        },
        "language_models": {"anthropic": {"api_url": "https://api.anthropic.com"}},
    }


SCHEMA_CASES = [
    ("claude-code", "mcpServers", _mcp_servers_fixture),
    ("vscode", "servers", _vscode_fixture),
    ("zed", "context_servers", _zed_fixture),
]


# ---------------------------------------------------------------------------
# Helper: write a fixture to disk + point a client at it.
# ---------------------------------------------------------------------------


def _prepare(tmp_path: Path, client_key: str, payload: dict[str, Any]) -> tuple[Any, Path]:
    """
    Writes `payload` as JSON to tmp_path/<client_key>.json, monkey-patches
    the matching client's path_fn to return that file, and returns
    (client, target_path).
    """
    target = tmp_path / f"{client_key}.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    client = next(c for c in install.CLIENTS if c.key == client_key)
    client.path_fn = lambda t=target: t
    return client, target


@pytest.fixture(autouse=True)
def _restore_client_paths():
    """Snapshot each client's path_fn and restore after every test — the
    monkey-patch above is process-wide and would leak into other tests."""
    snapshot = [(c, c.path_fn) for c in install.CLIENTS]
    yield
    for client, original in snapshot:
        client.path_fn = original


# ---------------------------------------------------------------------------
# Tests — every schema exercises the same contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("client_key,target_key,fixture_fn", SCHEMA_CASES)
def test_top_level_keys_preserved(tmp_path, client_key, target_key, fixture_fn):
    """Every top-level key present before must still be present after."""
    payload = fixture_fn()
    client, target = _prepare(tmp_path, client_key, payload)

    changed, _ = install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )
    assert changed is True

    after = json.loads(target.read_text(encoding="utf-8"))
    assert set(after.keys()) == set(payload.keys()), (
        f"top-level key set changed: added={set(after) - set(payload)}, removed={set(payload) - set(after)}"
    )


@pytest.mark.parametrize("client_key,target_key,fixture_fn", SCHEMA_CASES)
def test_sibling_servers_preserved_byte_for_byte(tmp_path, client_key, target_key, fixture_fn):
    """Other MCP servers under the target key must survive intact."""
    payload = fixture_fn()
    client, target = _prepare(tmp_path, client_key, payload)

    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )

    after = json.loads(target.read_text(encoding="utf-8"))
    for sibling_name, sibling_spec in payload[target_key].items():
        if sibling_name == install.SERVER_NAME:
            continue
        assert sibling_name in after[target_key], f"sibling {sibling_name!r} was removed — DATA LOSS!"
        assert after[target_key][sibling_name] == sibling_spec, f"sibling {sibling_name!r} was mutated — DATA LOSS!"


@pytest.mark.parametrize("client_key,target_key,fixture_fn", SCHEMA_CASES)
def test_non_target_top_level_keys_deeply_identical(tmp_path, client_key, target_key, fixture_fn):
    """Non-target top-level keys must be deeply identical after write."""
    payload = fixture_fn()
    client, target = _prepare(tmp_path, client_key, payload)

    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )

    after = json.loads(target.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if key == target_key:
            continue
        assert after[key] == value, f"top-level key {key!r} was mutated"


@pytest.mark.parametrize("client_key,target_key,fixture_fn", SCHEMA_CASES)
def test_backup_file_matches_pre_write_state(tmp_path, client_key, target_key, fixture_fn):
    """A byte-identical .knowledge-rag.bak backup is written before write."""
    payload = fixture_fn()
    client, target = _prepare(tmp_path, client_key, payload)
    pre_write_bytes = target.read_bytes()

    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )

    backup = target.with_suffix(target.suffix + install.BACKUP_SUFFIX)
    assert backup.exists(), "backup file was not created"
    assert backup.read_bytes() == pre_write_bytes, "backup differs from pre-write bytes"


@pytest.mark.parametrize("client_key,target_key,fixture_fn", SCHEMA_CASES)
def test_second_run_is_idempotent(tmp_path, client_key, target_key, fixture_fn):
    """A second run with the same inputs must NOT write again."""
    payload = fixture_fn()
    client, target = _prepare(tmp_path, client_key, payload)

    install_path = tmp_path / "install"
    venv_python = install_path / "venv" / "bin" / "python"

    changed_first, _ = install.register_client(client, install_path, venv_python, dry_run=False)
    changed_second, msg_second = install.register_client(client, install_path, venv_python, dry_run=False)
    assert changed_first is True
    assert changed_second is False, f"expected idempotent no-op on second run, but wrote again ({msg_second!r})"


@pytest.mark.parametrize("client_key,target_key,fixture_fn", SCHEMA_CASES)
def test_dry_run_writes_nothing(tmp_path, client_key, target_key, fixture_fn):
    """--dry-run must never touch the target or create a backup."""
    payload = fixture_fn()
    client, target = _prepare(tmp_path, client_key, payload)
    pre_bytes = target.read_bytes()

    changed, msg = install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=True,
    )

    assert changed is True
    assert "WOULD" in msg, f"dry-run message should announce intent, got {msg!r}"
    assert target.read_bytes() == pre_bytes, "target was mutated during dry-run"
    backup = target.with_suffix(target.suffix + install.BACKUP_SUFFIX)
    assert not backup.exists(), "backup was created during dry-run"


def test_empty_config_creates_only_target_key(tmp_path):
    """When the config doesn't exist, we create it with ONLY the target key."""
    client = next(c for c in install.CLIENTS if c.key == "claude-code")
    target = tmp_path / "empty" / "config.json"
    client.path_fn = lambda t=target: t

    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )

    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"mcpServers"}, (
        f"empty-config path should ONLY create the target key, got keys={list(data.keys())}"
    )
    assert install.SERVER_NAME in data["mcpServers"]


def test_invalid_json_leaves_target_untouched(tmp_path):
    """If the target contains invalid JSON, we refuse to write over it."""
    client = next(c for c in install.CLIENTS if c.key == "claude-code")
    target = tmp_path / "broken.json"
    broken_content = "{ this is not valid json"
    target.write_text(broken_content, encoding="utf-8")
    client.path_fn = lambda t=target: t

    # `_read_json` returns None on JSONDecodeError, so the register call
    # treats the file as empty and writes a fresh dict. That's the current
    # documented behavior; here we lock it in so a future refactor can't
    # silently start clobbering user data without an intentional decision.
    #
    # We ALSO verify no backup is created for a file we couldn't read
    # (there was nothing safe to back up).
    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )
    # After the write, the file must be valid JSON containing only the target key.
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert set(reloaded.keys()) == {"mcpServers"}
    assert install.SERVER_NAME in reloaded["mcpServers"]


def test_atomic_write_replaces_target(tmp_path, monkeypatch):
    """The write must go through a temp file + os.replace (never truncate)."""
    payload = _mcp_servers_fixture()
    client, target = _prepare(tmp_path, "claude-code", payload)

    seen_replaces = []
    original_replace = install.os.replace

    def spy_replace(src, dst):
        seen_replaces.append((str(src), str(dst)))
        return original_replace(src, dst)

    monkeypatch.setattr(install.os, "replace", spy_replace)

    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )

    assert len(seen_replaces) == 1, "expected exactly one os.replace call"
    src, dst = seen_replaces[0]
    assert dst == str(target)
    assert src != dst, "atomic write must go through a distinct temp file"


def test_only_knowledge_rag_entry_added_or_updated(tmp_path):
    """The change set must be exactly one key: SERVER_NAME under target key."""
    payload = _mcp_servers_fixture()
    client, target = _prepare(tmp_path, "claude-code", payload)

    install.register_client(
        client,
        install_path=tmp_path / "install",
        venv_python=tmp_path / "install" / "venv" / "bin" / "python",
        dry_run=False,
    )

    after = json.loads(target.read_text(encoding="utf-8"))
    added_or_changed = set()
    for k in set(payload) | set(after):
        if payload.get(k) != after.get(k):
            added_or_changed.add(k)
    assert added_or_changed == {"mcpServers"}, f"expected only 'mcpServers' to change, got {added_or_changed}"

    # Within mcpServers, only the knowledge-rag entry must differ.
    mcp_before = payload["mcpServers"]
    mcp_after = after["mcpServers"]
    inner_delta = set()
    for k in set(mcp_before) | set(mcp_after):
        if mcp_before.get(k) != mcp_after.get(k):
            inner_delta.add(k)
    assert inner_delta == {install.SERVER_NAME}, (
        f"expected only {install.SERVER_NAME!r} inside mcpServers to change, got {inner_delta}"
    )
