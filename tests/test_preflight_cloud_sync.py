"""Cloud-sync detection in the startup preflight (it used to be a __main__ self-check).

A data_dir under a sync client corrupts ChromaDB (HNSW mmap, SQLite WAL), so
startup exits 78 there; run_preflight() still runs that check for its callers,
while the server's owner runs it once, before taking the writer lock.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from unittest.mock import MagicMock

import pytest

CLOUD_SYNC_EXIT_CODE = 78  # EX_CONFIG
CASES = [
    (r"C:\Users\x\OneDrive\Documentos\knowledge-rag\data", "OneDrive"),
    (r"C:\Users\x\OneDrive - Contoso\knowledge-rag\data", "OneDrive - Contoso"),
    (r"/Users/x/Library/Mobile Documents/com~apple~CloudDocs/kr", "com~apple~CloudDocs"),
    (r"/Users/x/iCloud Drive/kr", "iCloud Drive"),
    (r"/home/x/Dropbox/kr/data", "Dropbox"),
    (r"C:\Users\x\Google Drive\kr", "Google Drive"),
    (r"/home/x/GoogleDrive/kr", "GoogleDrive"),
    (r"C:\Users\x\pCloudDrive\kr", "pCloudDrive"),
    (r"/home/x/MEGAsync/kr", "MEGAsync"),
    (r"/home/x/Nextcloud/kr", "Nextcloud"),
    (r"C:\knowledge-rag\data", None),
    (r"/var/lib/knowledge-rag/data", None),
    (r"C:\Users\x\AppData\Local\knowledge-rag\data", None),
]


@pytest.mark.parametrize("raw, provider", CASES)
def test_given_a_data_dir_path_when_detecting_cloud_sync_then_the_provider_segment_is_named(raw, provider):
    from mcp_server import preflight

    if os.name != "nt" and PureWindowsPath(raw).drive:
        pytest.skip("a drive path is relative outside Windows: it would pick up the working directory")

    assert preflight._detect_cloud_sync_provider(Path(raw)) == provider


def test_given_a_cloud_synced_data_dir_when_run_preflight_runs_then_it_exits_78_before_any_probe(tmp_path, monkeypatch):
    from mcp_server import preflight

    monkeypatch.setattr(preflight.config, "data_dir", tmp_path / "OneDrive" / "data")
    monkeypatch.setattr(preflight, "_get", lambda section, key, default: default)  # no allow_cloud_sync
    probe = MagicMock()
    monkeypatch.setattr(preflight, "_probe_chroma", probe)

    with pytest.raises(SystemExit) as exited:
        preflight.run_preflight()

    assert (exited.value.code, probe.call_count) == (CLOUD_SYNC_EXIT_CODE, 0)
