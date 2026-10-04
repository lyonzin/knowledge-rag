"""Run the real stdio transport without a model or persistent database."""

import os
import threading
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

from mcp_server import instance_lock, logging_config, preflight, server
from mcp_server.fts5_index import Fts5LexicalIndex
from mcp_server.ingestion import DocumentParser

_emit = threading.Event()
_done = threading.Event()
_errors = []


def _diagnostics_worker():
    try:
        if not _emit.wait(timeout=25):
            raise RuntimeError("No post-handshake diagnostic request")
        Fts5LexicalIndex._maybe_log_progress(10, 100, 0)
        path = Path(os.environ["KNOWLEDGE_RAG_DIR"]) / "empty.md"
        path.write_text("", encoding="utf-8")
        DocumentParser().parse_file(path)
        server.FastEmbedEmbeddings._print_gpu_banner(None, "forced-cpu")
    except Exception as exc:
        _errors.append(str(exc))
    finally:
        _done.set()


def _start_background(mode):
    threading.Thread(target=_diagnostics_worker, daemon=True).start()
    return {"status": "started", "operation": mode}


@server.mcp.tool()
def audit_stdout() -> str:
    """Emit diagnostics after initialize while the transport is active."""
    _emit.set()
    if not _done.wait(timeout=5) or _errors:
        raise RuntimeError(f"Background diagnostic failure: {_errors}")
    return "diagnostics complete"


def main():
    orchestrator = MagicMock()
    orchestrator._check_dimension_mismatch.return_value = False
    orchestrator.collection.count.return_value = 0
    orchestrator.start_reindex_background.side_effect = _start_background
    server.get_orchestrator = lambda: orchestrator
    server.config.transport = "stdio"
    instance_lock.single_instance_lock = nullcontext
    preflight.run_preflight = lambda: False
    logging_config.setup_logging = lambda **kwargs: None
    server.main()


if __name__ == "__main__":
    main()
