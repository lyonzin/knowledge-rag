"""Run the real stdio server and writer claim over a fake index that logs each write.

Every write the index receives appends "<pid> <tool>" to <data_dir>/writes.log,
and building the index appends "<pid> open", so a test can tell which process
wrote and which ones opened the index (only owners may). Two holds avoid
timing guesses: with KNOWLEDGE_RAG_TEST_HOLD_PREFLIGHT=1 the startup check
waits for <data_dir>/release-preflight, and with KNOWLEDGE_RAG_TEST_HOLD_EXIT=1
the process lives on after its transport ended, as a daemon thread still
writing during shutdown would, until <data_dir>/release-exit appears. With
KNOWLEDGE_RAG_TEST_HARMLESS_PROBE=1 the real preflight runs, prologue and all,
with a probe body that imports no Chroma (support/harmless_probe.py) and the
timeout in KNOWLEDGE_RAG_TEST_PROBE_TIMEOUT.
"""

import os
import time
from functools import partial
from unittest.mock import MagicMock

import harmless_probe  # a sibling module: this script's folder is on sys.path

from mcp_server import logging_config, preflight, server

WRITES_LOG = "writes.log"
RELEASE_PREFLIGHT_FILE = "release-preflight"
RELEASE_EXIT_FILE = "release-exit"
HOLD_PREFLIGHT_ENV = "KNOWLEDGE_RAG_TEST_HOLD_PREFLIGHT"
HOLD_EXIT_ENV = "KNOWLEDGE_RAG_TEST_HOLD_EXIT"
HARMLESS_PROBE_ENV = "KNOWLEDGE_RAG_TEST_HARMLESS_PROBE"
PROBE_TIMEOUT_ENV = "KNOWLEDGE_RAG_TEST_PROBE_TIMEOUT"
OPEN_ENTRY = "open"
_POLL_SECONDS = 0.05
_HOLD_LIMIT_SECONDS = 60


def _log(entry):
    with (server.config.data_dir / WRITES_LOG).open("a", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()} {entry}\n")


class _LoggedIndex:
    """The KnowledgeOrchestrator surface these tools use, without Chroma or a model."""

    def __init__(self):
        self.collection = MagicMock()
        self.collection.count.return_value = 1  # populated: the startup indexes nothing
        self.query_cache = MagicMock()
        self.query_cache.stats.return_value = {"hit_rate": "0%"}
        _log(OPEN_ENTRY)

    def _check_dimension_mismatch(self):
        return False

    def _start_bm25_background_build(self, thread_name):
        """The BM25 prewarm (GH #321: a background build) has nothing to build here."""

    def query(self, query, **options):
        return [{"content": f"hit for {query}", "source": "notes/hit.md", "score": 1.0}]

    def add_document_from_content(self, content, filepath, category):
        _log("add_document")
        return {"filepath": filepath}

    def update_document_content(self, filepath, content):
        _log("update_document")
        return {"filepath": filepath}

    def remove_document_by_path(self, filepath, delete_file=False):
        _log("remove_document")
        return {"filepath": filepath}

    def add_from_url(self, url, category, title):
        _log("add_from_url")
        return {"url": url}

    def start_reindex_background(self, mode, resume_state=None):
        _log("reindex_documents")
        return {"status": "started", "operation": mode}


def _wait_for(name, hold_env):
    """Wait for the test's release file while the hold is on."""
    release = server.config.data_dir / name
    deadline = time.monotonic() + _HOLD_LIMIT_SECONDS
    while os.environ.get(hold_env) == "1" and not release.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"{release} never appeared")
        time.sleep(_POLL_SECONDS)


def _held_preflight():
    _wait_for(RELEASE_PREFLIGHT_FILE, HOLD_PREFLIGHT_ENV)
    return False


@server.mcp.tool()
def server_pid() -> str:
    """This process's PID: on Windows a venv python.exe is a launcher with a PID of its own."""
    return str(os.getpid())


def _use_preflight():
    """The real preflight with a harmless probe body, or the held stand-in."""
    if os.environ.get(HARMLESS_PROBE_ENV) != "1":
        preflight._probe_and_repair = _held_preflight
        return
    preflight._PROBE_BODY = harmless_probe.BODY
    timeout = float(os.environ.get(PROBE_TIMEOUT_ENV, preflight.PROBE_TIMEOUT_SECONDS))
    preflight._probe_and_repair = partial(preflight._probe_and_repair, timeout_seconds=timeout)


def main():
    server.KnowledgeOrchestrator = _LoggedIndex
    server.config.transport = "stdio"
    preflight._check_cloud_sync = lambda: None
    _use_preflight()
    logging_config.setup_logging = lambda **kwargs: None
    server.main()
    _wait_for(RELEASE_EXIT_FILE, HOLD_EXIT_ENV)


if __name__ == "__main__":
    main()
