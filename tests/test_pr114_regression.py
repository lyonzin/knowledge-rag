"""Anti-regression tests for issue #114.

Pins the contract that ``_ensure_bm25_index`` only marks
``_bm25_initialized = True`` after a build that actually produced an index.
An empty-collection bootup or a build-time exception must leave the flag
``False`` so a later call retries once content is present.

Bug history (issue #114): prior behavior set the flag unconditionally at
the end of the guarded block. If the server booted against an empty
ChromaDB (fresh install, or metadata-mapping loss after a partial
reindex), the flag trapped in ``True`` with an empty inverted index.
Subsequent ``add_document`` calls populate ``bm25_index.corpus`` but do
not rebuild the inverted index — ``_ensure_bm25_index`` is the only path
that does, and it short-circuited on the stale flag. All keyword-only
searches returned zero results; hybrid searches returned only semantic
hits with ``bm25_rank: null``.
"""

import threading
from unittest.mock import MagicMock

from mcp_server.server import BM25Index, KnowledgeOrchestrator


def _make_orchestrator_with_mock_chroma(count_value=0, get_return=None):
    """Build a KnowledgeOrchestrator with mocked ChromaDB, real BM25Index.

    Uses ``__new__`` to skip ``__init__`` — no filesystem, no embedding model,
    no Chroma persistence touched. Only wires the attributes ``_ensure_bm25_index``
    actually reads.
    """
    o = KnowledgeOrchestrator.__new__(KnowledgeOrchestrator)
    o.collection = MagicMock()
    o.collection.count.return_value = count_value
    if get_return is not None:
        o.collection.get.return_value = get_return
    o.bm25_index = BM25Index()  # real, not mocked — we assert against its state
    o._bm25_initialized = False
    o._bm25_build_lock = threading.Lock()
    return o


class TestEmptyBootupDoesNotTrapGuardFlag:
    """Empty collection at boot must NOT mark BM25 initialized."""

    def test_empty_collection_leaves_flag_false(self):
        # Simulate a fresh install: ChromaDB is empty, no documents yet.
        o = _make_orchestrator_with_mock_chroma(count_value=0)

        o._ensure_bm25_index()

        # Contract: flag stays False so a later add_document + search can
        # trigger a real build. The pre-fix behavior set this to True
        # unconditionally, trapping BM25 in an uninitialized state forever.
        assert o._bm25_initialized is False, (
            "empty-collection bootup must leave _bm25_initialized=False so "
            "the guard retries once documents are added"
        )
        assert o.bm25_index._index_built is False
        assert len(o.bm25_index) == 0

    def test_populated_collection_marks_flag_true(self):
        # After documents exist, a call should build and mark initialized.
        o = _make_orchestrator_with_mock_chroma(
            count_value=2,
            get_return={
                "ids": ["doc1_0", "doc2_0"],
                "documents": [
                    "kerberoast is a technique for extracting AS-REP hashes",
                    "adcs certificate templates enable ESC1 domain escalation",
                ],
            },
        )

        o._ensure_bm25_index()

        assert o._bm25_initialized is True
        assert o.bm25_index._index_built is True
        assert len(o.bm25_index) == 2

    def test_recovery_after_empty_then_populated(self):
        """Full bug-repro path: empty boot -> collection populated -> retry works."""
        # Step 1: empty boot. Flag stays False (post-fix). Pre-fix: became True
        # and trapped BM25 forever.
        o = _make_orchestrator_with_mock_chroma(count_value=0)
        o._ensure_bm25_index()
        assert o._bm25_initialized is False

        # Step 2: caller adds documents. Simulate by pointing count/get to
        # non-empty state — models a real add_document flow adding chunks
        # to ChromaDB.
        o.collection.count.return_value = 1
        o.collection.get.return_value = {
            "ids": ["kerb_0"],
            "documents": ["kerberoast asreproast delegation"],
        }

        # Step 3: next call must retry and succeed. Pre-fix: guard was True,
        # this returned instantly with no build (BM25 dead).
        o._ensure_bm25_index()

        assert o._bm25_initialized is True
        assert o.bm25_index._index_built is True

        # Step 4: BM25 search actually returns hits. This is what a real
        # search_knowledge() call would exercise.
        hits = o.bm25_index.search("kerberoast", top_k=5)
        assert len(hits) == 1, "BM25 must return the indexed document on keyword match"
        assert hits[0][0] == "kerb_0"

    def test_empty_ids_from_get_leaves_flag_false(self):
        # Rare: count > 0 but get() returns empty ids/documents (dedup edge
        # case, race with removal). Must not mark initialized.
        o = _make_orchestrator_with_mock_chroma(
            count_value=5,
            get_return={"ids": [], "documents": []},
        )

        o._ensure_bm25_index()

        assert o._bm25_initialized is False
        assert o.bm25_index._index_built is False

    def test_exception_during_build_leaves_flag_false(self, monkeypatch):
        # If build_index() raises (memory pressure, corrupted state), the
        # WARN is logged and the flag must NOT be set — retry must be allowed.
        o = _make_orchestrator_with_mock_chroma(
            count_value=1,
            get_return={
                "ids": ["a_0"],
                "documents": ["some content"],
            },
        )

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated OOM during BM25 build")

        monkeypatch.setattr(o.bm25_index, "build_index", _raise)

        o._ensure_bm25_index()

        assert o._bm25_initialized is False, (
            "an exception during build_index() must leave the flag False "
            "so the caller can retry after resolving the underlying issue"
        )
