"""``add_from_url`` gives every page its own file (GH #262).

The file name comes from the first 60 characters of the page title, so two
pages of one series ("... - Part 1", "... - Part 2") used to share a file and
the second silently replaced the first on disk and in the index. A page with
no ``<title>`` at a URL ending in "/" got the hidden name ".md", shared by every such page.

``requests`` is replaced by a stub, so no test touches the network.
"""

import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_server import server

SERIES = "Kubernetes Security Best Practices for Production Clusters"
SHARED_NAME = "kubernetes-security-best-practices-for-production-clusters--.md"
# The provenance marker drops quotes from the URL, so the same-URL check must compare that escaped form.
PART_1 = "https://blog.example.test/o'brien/k8s-security-part-1"
PART_2 = "https://blog.example.test/o'brien/k8s-security-part-2"
TITLES = {PART_1: f"{SERIES} - Part 1", PART_2: f"{SERIES} - Part 2"}


def _page(title, body):
    """Return a minimal HTML page; a falsy title leaves out the <title> tag."""
    head = f"<head><title>{title}</title></head>" if title else ""
    return f"<html>{head}<body><p>{body}</p></body></html>"


def _serve(monkeypatch, pages):
    """Make ``requests.get(url)`` return ``pages[url]``, read at call time so a test can revise a page."""
    monkeypatch.setitem(sys.modules, "requests", MagicMock(get=lambda url, **_: MagicMock(text=pages[url])))
    return pages


def _serve_series(monkeypatch):
    """Serve two pages whose titles only differ after the 60th character."""
    return _serve(
        monkeypatch,
        {
            PART_1: _page(TITLES[PART_1], "Rotate the etcd encryption keys every quarter."),
            PART_2: _page(TITLES[PART_2], "Pin admission webhooks to a dedicated namespace."),
        },
    )


class _AnnouncingLock:
    """Index lock that counts the threads arriving at it, before they wait."""

    def __init__(self, lock):
        self.lock, self.arrivals = lock, threading.Semaphore(0)

    def __enter__(self):
        self.arrivals.release()
        return self.lock.__enter__()

    def __exit__(self, *exc_info):
        return self.lock.__exit__(*exc_info)


def test_pages_whose_titles_share_60_chars_keep_separate_files(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    _serve_series(monkeypatch)

    first = orch.add_from_url(PART_1, "general")
    second = orch.add_from_url(PART_2, "general")

    assert Path(first["filepath"]).name == SHARED_NAME
    assert first["filepath"] != second["filepath"]
    assert {doc["source"] for doc in orch.list_documents()} == {first["filepath"], second["filepath"]}
    assert "Rotate the etcd" in Path(first["filepath"]).read_text(encoding="utf-8")
    assert any("Rotate the etcd" in text for text in orch.collection.get(include=["documents"])["documents"])


@pytest.mark.parametrize("url", [PART_1, PART_2], ids=["plain-name", "suffixed-name"])
def test_re_adding_a_url_refreshes_its_own_file(isolated_orchestrator, monkeypatch, url):
    orch = isolated_orchestrator
    pages = _serve_series(monkeypatch)
    paths = {page: orch.add_from_url(page, "general")["filepath"] for page in (PART_1, PART_2)}
    pages[url] = _page(TITLES[url], "Revised guidance.")

    refreshed = orch.add_from_url(url, "general")

    assert refreshed["filepath"] == paths[url]
    assert len(list(server.config.documents_dir.rglob("*.md"))) == 2
    assert len(orch.list_documents()) == 2
    assert "Revised guidance." in Path(paths[url]).read_text(encoding="utf-8")


def test_a_file_without_provenance_is_never_overwritten(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    _serve_series(monkeypatch)
    note = server.config.documents_dir / "general" / SHARED_NAME
    note.parent.mkdir()
    note.write_text("# My own notes\n", encoding="utf-8")

    result = orch.add_from_url(PART_1, "general")

    assert Path(result["filepath"]).name != SHARED_NAME
    assert note.read_text(encoding="utf-8") == "# My own notes\n"


def test_a_file_that_cannot_be_read_is_never_overwritten(isolated_orchestrator, monkeypatch):
    """A sharing violation or an offline cloud placeholder leaves the file's owner unproven."""
    orch = isolated_orchestrator
    _serve_series(monkeypatch)
    held = server.config.documents_dir / "general" / SHARED_NAME
    held.parent.mkdir()
    held.write_text("# Locked by another program\n", encoding="utf-8")
    open_path = Path.open

    def deny_held(path, *args, **kwargs):
        if path.name == SHARED_NAME:
            raise PermissionError(13, "The process cannot access the file", str(path))
        return open_path(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", deny_held)
        result = orch.add_from_url(PART_1, "general")

    suffix = hashlib.sha256(PART_1.encode()).hexdigest()[:8]
    assert Path(result["filepath"]).name == f"{Path(SHARED_NAME).stem}-{suffix}.md"
    assert held.read_text(encoding="utf-8") == "# Locked by another program\n"


def test_untitled_pages_at_trailing_slash_urls_get_valid_distinct_files(isolated_orchestrator, monkeypatch):
    orch = isolated_orchestrator
    guide, faq = "https://docs.example.test/guide/", "https://docs.example.test/faq/"
    _serve(
        monkeypatch,
        {
            guide: _page(None, "Install the agent before the collector."),
            faq: _page(None, "Restart the collector after every upgrade."),
        },
    )

    first = orch.add_from_url(guide, "general")
    second = orch.add_from_url(faq, "general")

    assert Path(first["filepath"]).name == "page.md"
    assert first["filepath"] != second["filepath"]
    assert {doc["source"] for doc in orch.list_documents()} == {first["filepath"], second["filepath"]}


def test_parallel_adds_cannot_both_claim_the_free_name(isolated_orchestrator, monkeypatch):
    """Both name checks must not run before either write, e.g. while a reindex holds the index lock."""
    orch = isolated_orchestrator
    _serve_series(monkeypatch)
    held = orch._index_lock
    orch._index_lock = gate = _AnnouncingLock(held)

    with ThreadPoolExecutor(max_workers=2) as pool:
        with held:
            futures = [pool.submit(orch.add_from_url, url, "general") for url in (PART_1, PART_2)]
            assert gate.arrivals.acquire(timeout=5) and gate.arrivals.acquire(timeout=5)
        paths = {future.result(timeout=30)["filepath"] for future in futures}

    assert len(paths) == 2
    assert {doc["source"] for doc in orch.list_documents()} == paths
