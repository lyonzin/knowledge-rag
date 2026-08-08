"""Shared fixtures for knowledge-rag tests.

Mocks embeddings and ChromaDB to avoid model downloads in CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Workaround: defense-in-depth for Windows pytest atexit interpreter-shutdown
# races
# ---------------------------------------------------------------------------
# Two flake sources have surfaced on Windows GHA runners with pytest 9.0.3:
#
#   1) pytest's own ``_pytest.pathlib.cleanup_numbered_dir`` atexit callback
#      runs ``root.glob("garbage-*")`` without try/except. Concurrent FS
#      access (Defender, Search Indexer, parallel runner) can raise OSError
#      during interpreter shutdown -> exit code 1 with "Exception ignored
#      in atexit callback" even though all tests passed.
#
#   2) Background HuggingFace download threads (huggingface_hub uses
#      ``concurrent.futures.ThreadPoolExecutor`` for parallel snapshot
#      fetches) can outlive pytest's stdout. A late warning emit then trips
#      "ValueError: I/O operation on closed file" -> non-zero exit. The
#      primary fix for this is ``HF_HUB_OFFLINE=1`` in the CI workflow;
#      we also wrap pathlib cleanup here so any pytest-side race is contained.
#
# We patch ``_pytest.pathlib.cleanup_numbered_dir`` BEFORE the first
# ``tmp_path`` fixture runs (which is when ``make_numbered_dir_with_cleanup``
# calls ``atexit.register(cleanup_numbered_dir, ...)`` and binds the global
# lookup). ``conftest.py`` is imported at collection time so the atexit
# callback registered later is our safe wrapper.
#
# Track upstream pytest-dev/pytest#7491 family. Remove this block once
# pytest ships a real fix.
try:
    import _pytest.pathlib as _pp

    # Public attribute on this module so the regression test in
    # test_pytest_atexit_patch.py can verify the wrapper is in place
    # without poking at closure cells.
    _ORIGINAL_CLEANUP_NUMBERED_DIR = _pp.cleanup_numbered_dir

    def _safe_cleanup_numbered_dir(*args, **kwargs):
        """OSError-safe wrapper around pytest's atexit tmp_path cleanup."""
        try:
            return _ORIGINAL_CLEANUP_NUMBERED_DIR(*args, **kwargs)
        except OSError:
            # Race during interpreter shutdown — leftovers will be removed on
            # the next pytest run via the same cleanup mechanism.
            return None

    _pp.cleanup_numbered_dir = _safe_cleanup_numbered_dir
except Exception:
    # Never let the workaround itself break test collection.
    _ORIGINAL_CLEANUP_NUMBERED_DIR = None


@pytest.fixture
def mock_embedding():
    """Mock FastEmbed to avoid model download in CI."""
    with patch("mcp_server.server.FastEmbedEmbeddings") as mock:
        instance = MagicMock()
        instance.__call__ = MagicMock(return_value=[[0.1] * 384])
        instance.name.return_value = "mock-embed"
        instance.embed_documents.return_value = [[0.1] * 384]
        instance.embed_query.return_value = [[0.1] * 384]
        instance._dim = 384
        mock.return_value = instance
        yield instance


@pytest.fixture
def sample_markdown(tmp_path):
    """Create a sample markdown file for testing."""
    content = """# Test Document

## Section One

This section covers SQL injection bypass techniques including UNION-based attacks.

## Section Two

Cross-site scripting (XSS) payloads for reflected and DOM-based attacks.

## Section Three

Linux SUID exploitation and kernel privilege escalation methods.
"""
    f = tmp_path / "test.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_markdown_with_code(tmp_path):
    """Markdown with code blocks containing # comments."""
    content = """# Main Title

## Real Section

Some content here.

```bash
# This is a comment inside code block
echo "hello"
# Another comment
```

## Another Section

More content after code block.
"""
    f = tmp_path / "test_code.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    content = "Name,Role,Score\nAlice,Admin,95\nBob,User,80\n"
    f = tmp_path / "test.csv"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_json(tmp_path):
    """Create a sample JSON file."""
    content = '{"key": "value", "items": [1, 2, 3]}'
    f = tmp_path / "test.json"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_text(tmp_path):
    """Create a sample text file."""
    content = "Line one.\nLine two.\nLine three.\n"
    f = tmp_path / "test.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_python(tmp_path):
    """Create a sample Python file."""
    content = '''"""Module docstring."""

def hello(name: str) -> str:
    """Say hello."""
    return f"Hello {name}"

class Greeter:
    pass
'''
    f = tmp_path / "test.py"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_c(tmp_path):
    """Create a sample C source file."""
    content = """/**
 * Module documentation.
 */
#include <stdio.h>
#include "helper.h"

struct Config {
    int value;
};

int main(int argc, char *argv[]) {
    return 0;
}

void helper_func(int x) {
    printf("%d", x);
}
"""
    f = tmp_path / "test.c"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_cpp(tmp_path):
    """Create a sample C++ source file."""
    content = """/**
 * C++ module documentation.
 */
#include <iostream>
#include <vector>

class Engine {
public:
    void start();
};

struct Config {
    int value;
};

int main() {
    return 0;
}
"""
    f = tmp_path / "test.cpp"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_js(tmp_path):
    """Create a sample JavaScript file."""
    content = """/**
 * JavaScript module.
 */
import React from 'react';
import { useState } from 'react';

function fetchData(url) {
    return fetch(url);
}

class DataService {
    constructor() {}
}

export function processItems(items) {
    return items.map(i => i.id);
}
"""
    f = tmp_path / "test.js"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_ts(tmp_path):
    """Create a sample TypeScript file."""
    content = """/**
 * TypeScript module.
 */
import { Router } from 'express';
import type { Config } from './types';

interface AppConfig {
    port: number;
}

enum Status {
    Active,
    Inactive,
}

type Result = {
    data: string;
};

export function createRouter(): Router {
    return Router();
}

export class ApiController {
    handle() {}
}
"""
    f = tmp_path / "test.ts"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def fts5_tmp_index(tmp_path):
    """FTS5 index backed by a temp SQLite file, populated with 20 sample chunks.

    Uses a real file (not ``:memory:``) so the WAL PRAGMA applied by
    ``Fts5LexicalIndex._connect_and_configure`` exercises the same code path
    as production. Yields the live ``Fts5LexicalIndex`` and closes it on
    teardown.
    """
    from mcp_server.fts5_index import Fts5LexicalIndex, Fts5MigrationState

    db_path = tmp_path / "fts5_index.db"
    state_path = tmp_path / "fts5_migration.state"
    Fts5MigrationState(state_path).write(
        {
            "status": "complete",
            "docs_total": 20,
            "docs_indexed": 20,
            "started_at": "2026-08-07T12:00:00Z",
            "completed_at": "2026-08-07T12:00:01Z",
            "error": None,
        }
    )
    index = Fts5LexicalIndex(db_path=db_path, state_path=state_path)
    seeds = [
        ("chunk_001", "Report references MDR-AD002 remediation", "mdr.md", "security"),
        ("chunk_002", "Advisory covers CVE-2021-4034 Pwnkit", "pwnkit.md", "security"),
        ("chunk_003", "CVE-2021-4034 exploitation walkthrough", "cve.md", "security"),
        ("chunk_004", "CVE-2021-4034 patch analysis", "patch.md", "security"),
        ("chunk_005", "Additional CVE-2021-4034 IoCs", "iocs.md", "security"),
        ("chunk_006", "Baseline for CVE-2021-4034 telemetry", "telemetry.md", "security"),
        ("chunk_007", "MITRE T1078.001 default accounts", "mitre.md", "attck"),
        ("chunk_008", "CWE-79 cross-site scripting notes", "cwe.md", "security"),
        ("chunk_009", "H1-P4-XXX-1234 disclosure summary", "h1.md", "bugbounty"),
        ("chunk_010", "pass-the-hash lateral movement playbook", "pth.md", "adversary"),
        ("chunk_011", "context.py loads YAML configuration", "context.md", "development"),
        ("chunk_012", "Only MDR-AD003 mentioned here", "mdr3.md", "security"),
        ("chunk_013", "OAuth token refresh workflow", "oauth.md", "development"),
        ("chunk_014", "Prose about incident response process", "ir.md", "general"),
        ("chunk_015", "General ransomware hardening guide", "ransom.md", "security"),
        ("chunk_016", "How does OAuth token refresh work", "oauth2.md", "development"),
        ("chunk_017", "Zero trust architecture overview", "zt.md", "security"),
        ("chunk_018", "Blue team detection engineering", "bt.md", "security"),
        ("chunk_019", "Threat hunting hypothesis backlog", "hunt.md", "security"),
        ("chunk_020", "Retro on MDR incident closure", "retro.md", "security"),
    ]
    with index._fts5_lock:  # noqa: SLF001 — test helper for direct seeding
        for chunk_id, content, filename, category in seeds:
            index._conn.execute(  # noqa: SLF001
                "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                (chunk_id, content, filename, category),
            )
        index._conn.commit()  # noqa: SLF001
    try:
        yield index
    finally:
        index.close()


@pytest.fixture
def migration_state_tmp(tmp_path):
    """Empty ``Fts5MigrationState`` anchored at ``tmp_path/fts5_migration.state``."""
    from mcp_server.fts5_index import Fts5MigrationState

    return Fts5MigrationState(tmp_path / "fts5_migration.state")


@pytest.fixture
def sample_lexical_queries():
    """Canonical lexical query set used across FTS5/router tests."""
    return ["MDR-AD002", "CVE-2021-4034", "T1078.001", "CWE-79", "H1-P4-XXX-1234"]


@pytest.fixture
def mcp_client_test():
    """In-process MCP client harness for e2e tests (Task 03, IT-020/E2E-*).

    A real MCP subprocess spawn is too heavy for CI. This fixture routes
    calls straight to the module-level tool functions and parses the JSON
    envelope so tests can assert on the returned dict — same call surface
    the production MCP handshake exposes, minus the protocol framing.
    """
    import json as _json

    class _Client:
        def call(self, tool_name: str, **kwargs):
            from mcp_server import server

            fn = getattr(server, tool_name)
            raw = fn(**kwargs)
            return _json.loads(raw) if isinstance(raw, str) else raw

    return _Client()


@pytest.fixture
def sample_xml(tmp_path):
    """Create a sample XML file."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
</project>
"""
    f = tmp_path / "test.xml"
    f.write_text(content, encoding="utf-8")
    return f
