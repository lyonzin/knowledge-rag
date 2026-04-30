"""Startup preflight checks for persistent ChromaDB state."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import BASE_DIR, config


def _backup_active_index(reason: str) -> Path:
    """Move active ChromaDB state aside so the server can rebuild cleanly."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = config.data_dir / "backups" / f"auto-repair-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    if config.chroma_dir.exists():
        shutil.move(str(config.chroma_dir), str(backup_dir / f"chroma_db.{reason}"))

    metadata_file = config.data_dir / "index_metadata.json"
    if metadata_file.exists():
        shutil.move(str(metadata_file), str(backup_dir / f"index_metadata.{reason}.json"))

    return backup_dir


def _probe_chroma(timeout_seconds: int = 30) -> subprocess.CompletedProcess[str]:
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


def run_preflight(timeout_seconds: int = 30) -> bool:
    """Return True when active Chroma state was moved aside for repair."""
    result = _probe_chroma(timeout_seconds=timeout_seconds)
    if result.returncode == 0:
        return False

    reason = "segfault" if result.returncode in (-11, 139) else "failed"
    backup_dir = _backup_active_index(reason)
    print(
        f"[RECOVERY] Chroma preflight failed with code {result.returncode}; "
        f"moved active index to {backup_dir}",
        file=sys.stderr,
    )
    if result.stderr:
        print(result.stderr[-2000:], file=sys.stderr)
    return True
