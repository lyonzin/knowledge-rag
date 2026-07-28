"""
╭─╴ DOCUMENT WATCHER ╶───────────────────────────────────────────╮
│                                                                │
│   Watchdog-based filesystem observer with accumulate-mode      │
│   debounce so bulk file copies never starve the reindex        │
│   trigger. Extracted verbatim from server.py in the A2.1       │
│   refactor.                                                    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler

from ..config import config
from ..telemetry import get_tracer

log = logging.getLogger(__name__)


class DocumentWatcher(FileSystemEventHandler):
    """Watches documents directory and triggers reindex on changes.

    Uses accumulate-mode debounce: collects changed paths during a silence
    window instead of resetting the timer on every file event.  This prevents
    bulk file copies (1000+ files) from starving the reindex trigger.
    """

    def __init__(self, orchestrator_getter, debounce_seconds: float = 10.0):
        self._get_orchestrator = orchestrator_getter
        self._debounce = debounce_seconds
        self._lock = threading.Lock()
        self._pending_paths: set = set()
        self._timer = None
        self._reindex_lock = threading.Lock()

    def _schedule_reindex(self, path: str):
        """Accumulate-mode debounce: collect paths, fire once after silence."""
        with self._lock:
            self._pending_paths.add(path)
            if self._timer is None or not self._timer.is_alive():
                self._timer = threading.Timer(self._debounce, self._do_reindex)
                self._timer.daemon = True
                self._timer.start()

    def _do_reindex(self):
        """Perform incremental reindex in background (serialized)."""
        if not self._reindex_lock.acquire(blocking=False):
            log.info("[WATCHER] Reindex already in progress, skipping")
            return
        try:
            with self._lock:
                count = len(self._pending_paths)
                self._pending_paths.clear()
            if count == 0:
                return
            log.info("[WATCHER] %d file(s) changed, starting incremental reindex...", count)
            # A2.6 — watcher-initiated reindexes get their own root span so an
            # APM trace shows the trigger origin, not just an anonymous call.
            with get_tracer().start_as_current_span(
                "knowledge_rag.watcher.reindex", attributes={"watcher.changed_files": count}
            ):
                orch = self._get_orchestrator()
                stats = orch.index_all(force=False)
            changed = stats.get("indexed", 0) + stats.get("updated", 0) + stats.get("deleted", 0)
            if changed > 0:
                log.info(
                    "[WATCHER] Auto-reindexed: %d new, %d updated, %d deleted",
                    stats["indexed"],
                    stats["updated"],
                    stats["deleted"],
                )
        except Exception as e:
            log.exception("[WATCHER] Reindex failed: %s", e)
        finally:
            self._reindex_lock.release()

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix in config.supported_formats:
            self._schedule_reindex(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        src_supported = Path(event.src_path).suffix in config.supported_formats
        dest_supported = Path(event.dest_path).suffix in config.supported_formats
        if src_supported or dest_supported:
            self._schedule_reindex(event.dest_path)
