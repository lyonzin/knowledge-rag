"""
╭─╴ MMR TESTS ╶──────────────────────────────────────────────────╮
│                                                                │
│   Unit + regression coverage for the embedding-based MMR       │
│   reranker. Verifies classic MMR properties (identical vectors │
│   collapse, orthogonal vectors preserve order, lambda knob     │
│   behaves), input hardening, and end-to-end integration        │
│   through KnowledgeOrchestrator._apply_mmr.                    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from mcp_server.retrieval.mmr import apply_mmr

# ---------------------------------------------------------------------------
# apply_mmr — pure unit tests
# ---------------------------------------------------------------------------


class TestApplyMmrEdgeCases:
    """Boundary and error handling for the pure function."""

    def test_empty_candidates_returns_empty(self):
        assert apply_mmr([1.0, 0.0], [], [], [], top_k=5) == []

    def test_top_k_zero_returns_empty(self):
        assert apply_mmr([1.0], ["a"], [[1.0]], [0.9], top_k=0) == []

    def test_top_k_larger_than_pool_caps_at_pool_size(self):
        picks = apply_mmr(
            [1.0, 0.0],
            ["a", "b"],
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 0.5],
            top_k=10,
            lambda_param=0.5,
        )
        assert len(picks) == 2

    def test_single_candidate_short_circuits(self):
        picks = apply_mmr(
            [1.0, 0.0],
            ["only"],
            [[1.0, 0.0]],
            [0.42],
            top_k=5,
        )
        assert picks == [("only", 0.42)]

    def test_top_k_one_returns_first_pick(self):
        # When top_k=1 we short-circuit to the caller's #1 candidate to keep
        # the reranker's top choice visible.
        picks = apply_mmr(
            [1.0, 0.0],
            ["reranker_first", "b", "c"],
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            [0.99, 0.80, 0.10],
            top_k=1,
            lambda_param=0.3,
        )
        assert picks == [("reranker_first", 0.99)]

    def test_score_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_mmr(
                [1.0, 0.0],
                ["a", "b"],
                [[1.0, 0.0], [0.0, 1.0]],
                [0.9],  # only one score for two candidates
                top_k=2,
            )

    def test_embedding_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_mmr(
                [1.0, 0.0, 0.0],  # 3-D query
                ["a", "b"],
                [[1.0, 0.0], [0.0, 1.0]],  # 2-D candidates
                [0.9, 0.8],
                top_k=2,
            )

    def test_embedding_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_mmr(
                [1.0, 0.0],
                ["a", "b", "c"],
                [[1.0, 0.0]],  # only one embedding for three ids
                [0.9, 0.8, 0.7],
                top_k=2,
            )

    def test_lambda_out_of_range_raises(self):
        with pytest.raises(ValueError):
            apply_mmr(
                [1.0, 0.0],
                ["a", "b"],
                [[1.0, 0.0], [0.0, 1.0]],
                [0.9, 0.5],
                top_k=2,
                lambda_param=1.5,
            )
        with pytest.raises(ValueError):
            apply_mmr(
                [1.0, 0.0],
                ["a", "b"],
                [[1.0, 0.0], [0.0, 1.0]],
                [0.9, 0.5],
                top_k=2,
                lambda_param=-0.1,
            )

    def test_preserves_original_scores_in_output(self):
        picks = apply_mmr(
            [1.0, 0.0, 0.0],
            ["x", "y"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [42.0, 3.14],
            top_k=2,
            lambda_param=0.5,
        )
        # Scores round-trip untouched, regardless of order.
        by_id = dict(picks)
        assert by_id["x"] == pytest.approx(42.0)
        assert by_id["y"] == pytest.approx(3.14)


class TestApplyMmrClassicalBehavior:
    """MMR properties that the caller downstream relies on."""

    def test_identical_vectors_pick_order_matches_input(self):
        # Three copies of the same document — MMR has no diversity signal to
        # go on. The first pick is candidate 0 (contract). Subsequent picks
        # tie on MMR score, so numpy argmax picks the lowest remaining index:
        # order should be [a, b, c].
        v = [1.0, 0.0, 0.0]
        picks = apply_mmr(
            v,
            ["a", "b", "c"],
            [v, v, v],
            [0.9, 0.8, 0.7],
            top_k=3,
            lambda_param=0.5,
        )
        assert [pid for pid, _ in picks] == ["a", "b", "c"]

    def test_orthogonal_vectors_yield_full_diversity(self):
        # Three mutually orthogonal candidates — every pair has cosine 0, so
        # the diversity penalty is always 0 and the ordering is driven purely
        # by relevance-to-query. Candidate ``a`` has the highest cosine to
        # the query (1.0), then ``b`` (0.9-ish query projection = 0), then
        # ``c``. Because b and c have equal cosine to Q, argmax's tiebreaker
        # (lowest index) picks b then c.
        q = [1.0, 0.0, 0.0]
        picks = apply_mmr(
            q,
            ["a", "b", "c"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.9, 0.8, 0.7],
            top_k=3,
            lambda_param=0.7,
        )
        assert [pid for pid, _ in picks] == ["a", "b", "c"]

    def test_lambda_one_orders_by_pure_relevance(self):
        # With lambda=1 the diversity penalty drops out, so the ranking is
        # cosine-to-query descending. First pick is always index 0 (contract);
        # remaining picks come out in decreasing cosine similarity to Q.
        # Candidate cosine values to Q=[1,0,0]:  a=1.0, b=0.5, c=0.2, d=0.8.
        # First pick: a (contract). Remaining order by relevance: d, b, c.
        q = [1.0, 0.0, 0.0]
        picks = apply_mmr(
            q,
            ["a", "b", "c", "d"],
            [
                [1.0, 0.0, 0.0],  # cos = 1.0
                [0.5, 0.5, 0.0],  # cos = 0.707
                [0.2, 0.9, 0.0],  # cos ≈ 0.217
                [0.8, 0.2, 0.0],  # cos ≈ 0.970
            ],
            [1.0, 0.5, 0.2, 0.8],
            top_k=4,
            lambda_param=1.0,
        )
        assert [pid for pid, _ in picks] == ["a", "d", "b", "c"]

    def test_lambda_zero_maximizes_diversity(self):
        # With lambda=0 the relevance term drops out: the algorithm greedily
        # picks candidates that minimise the maximum cosine similarity to any
        # already-selected candidate. Setup: A is picked first (contract);
        # then the algorithm should prefer the orthogonal C over the almost-A
        # neighbour B, even though B has the higher relevance score.
        picks = apply_mmr(
            [1.0, 0.0, 0.0],
            ["A", "B", "C"],
            [
                [1.0, 0.0, 0.0],  # A
                [0.99, 0.10, 0.0],  # B — very close to A
                [0.0, 1.0, 0.0],  # C — orthogonal to A
            ],
            [1.0, 0.9, 0.1],
            top_k=2,
            lambda_param=0.0,
        )
        assert [pid for pid, _ in picks] == ["A", "C"]

    def test_lambda_zero_beats_lambda_one_on_dense_cluster(self):
        # A cluster of near-duplicates around A plus one truly different item.
        # lambda=1 (pure relevance) keeps the duplicates; lambda=0 (pure
        # diversity) drops one duplicate in favour of the different item.
        ids = ["a", "a2", "a3", "z"]
        embs = [
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.98, 0.10, 0.0],
            [0.0, 0.0, 1.0],
        ]
        scores = [1.0, 0.95, 0.90, 0.10]

        relevance_only = apply_mmr([1.0, 0.0, 0.0], ids, embs, scores, top_k=2, lambda_param=1.0)
        diversity_only = apply_mmr([1.0, 0.0, 0.0], ids, embs, scores, top_k=2, lambda_param=0.0)

        assert [pid for pid, _ in relevance_only] == ["a", "a2"]
        assert [pid for pid, _ in diversity_only] == ["a", "z"]

    def test_first_pick_always_candidate_zero(self):
        # Even with a top candidate whose cosine to the query is not the
        # highest, the caller's #1 pick is preserved. Here `a` has cosine 0.6
        # while `b` has cosine 1.0 — but `a` is at index 0, so MMR keeps it.
        picks = apply_mmr(
            [1.0, 0.0, 0.0],
            ["a", "b", "c"],
            [
                [0.6, 0.8, 0.0],  # cos ≈ 0.6
                [1.0, 0.0, 0.0],  # cos = 1.0
                [0.0, 1.0, 0.0],  # cos = 0
            ],
            [0.9, 0.8, 0.5],
            top_k=3,
            lambda_param=0.7,
        )
        assert picks[0][0] == "a"

    def test_defensive_normalization_handles_unnormalized_input(self):
        # Same directions as the orthogonal test, but with different magnitudes.
        # After defensive L2 normalization the algorithm should produce the
        # same ordering.
        q = [7.0, 0.0, 0.0]  # non-unit
        picks = apply_mmr(
            q,
            ["a", "b", "c"],
            [[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]],
            [0.9, 0.8, 0.7],
            top_k=3,
            lambda_param=0.7,
        )
        assert [pid for pid, _ in picks] == ["a", "b", "c"]

    def test_all_zero_candidate_does_not_crash(self):
        # A degenerate zero-vector candidate must not divide-by-zero the
        # normalization step. It contributes cosine 0 everywhere.
        picks = apply_mmr(
            [1.0, 0.0, 0.0],
            ["a", "zero", "c"],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [1.0, 0.5, 0.7],
            top_k=3,
            lambda_param=0.7,
        )
        assert len(picks) == 3
        assert picks[0][0] == "a"

    def test_accepts_numpy_arrays(self):
        # ChromaDB returns numpy arrays; the API must accept them directly.
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        picks = apply_mmr(q, ["a", "b", "c"], embs, [0.9, 0.6, 0.3], top_k=3)
        assert [pid for pid, _ in picks] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Orchestrator._apply_mmr — integration through the real method
# ---------------------------------------------------------------------------


def _make_orchestrator_stub():
    """Return a minimally-wired KnowledgeOrchestrator for _apply_mmr tests.

    Bypasses __init__ (which touches ChromaDB, FastEmbed and the filesystem)
    and wires only the attributes _apply_mmr needs.
    """
    from mcp_server.retrieval.orchestrator import KnowledgeOrchestrator

    orch = KnowledgeOrchestrator.__new__(KnowledgeOrchestrator)
    orch.collection = MagicMock()
    orch.embed_fn = MagicMock()
    orch._current_query_text = None
    return orch


def _mk_data(document: str, score: float) -> dict:
    return {
        "document": document,
        "metadata": {"source": f"{document}.md", "filename": f"{document}.md"},
        "reranker_score": score,
        "rrf_score": score / 2.0,
        "distance": 1.0 - score,
        "semantic_rank": 1,
        "bm25_rank": 1,
    }


class TestOrchestratorApplyMmrIntegration:
    """End-to-end path through the instance method (with mocked collection)."""

    def test_uses_embedding_mmr_when_query_and_embeddings_available(self):
        orch = _make_orchestrator_stub()

        # Cluster of near-duplicates around A plus a distinct doc Z.
        results = [
            ("A", _mk_data("A", 1.0)),
            ("A2", _mk_data("A2", 0.95)),
            ("A3", _mk_data("A3", 0.90)),
            ("Z", _mk_data("Z", 0.10)),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.05, 0.0],
                [0.98, 0.10, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        orch.collection.get.return_value = {
            "ids": ["A", "A2", "A3", "Z"],
            "embeddings": embeddings,
        }
        orch.embed_fn.embed_query.return_value = [[1.0, 0.0, 0.0]]
        orch._current_query_text = "attack surface"

        picked = orch._apply_mmr(results, top_k=2, lambda_param=0.0)

        # Pure diversity should keep A (first pick) and prefer Z over the
        # near-duplicate cluster A2/A3.
        assert [pid for pid, _ in picked] == ["A", "Z"]
        # The Jaccard implementation on this same input would keep an
        # A-cluster neighbour, since the doc text "A" and "A2" share zero
        # tokens (Jaccard sim 0) — verifying we are no longer on that path.

    def test_fallback_to_jaccard_when_query_text_missing(self):
        orch = _make_orchestrator_stub()
        # No query text stashed — MMR must not crash; it degrades.
        orch._current_query_text = None

        results = [
            ("A", _mk_data("alpha beta gamma", 1.0)),
            ("B", _mk_data("alpha beta gamma", 0.9)),  # duplicate of A
            ("C", _mk_data("delta epsilon", 0.5)),
        ]

        picked = orch._apply_mmr(results, top_k=2, lambda_param=0.5)

        # Jaccard path: A picked first (highest score), then C (Jaccard 0 vs A)
        # is more diverse than B (Jaccard 1 vs A).
        assert [pid for pid, _ in picked] == ["A", "C"]
        # And we never touched ChromaDB or the embedder.
        orch.collection.get.assert_not_called()
        orch.embed_fn.embed_query.assert_not_called()

    def test_fallback_when_chroma_returns_no_embeddings(self):
        orch = _make_orchestrator_stub()
        orch._current_query_text = "anything"
        orch.collection.get.return_value = {"ids": [], "embeddings": None}

        results = [
            ("A", _mk_data("hello world", 1.0)),
            ("B", _mk_data("something else", 0.5)),
        ]
        picked = orch._apply_mmr(results, top_k=1, lambda_param=0.7)

        # top_k=1 always yields the caller's #1 candidate.
        assert picked == [("A", results[0][1])]

    def test_fallback_when_embed_query_raises(self):
        orch = _make_orchestrator_stub()
        orch._current_query_text = "anything"
        orch.collection.get.return_value = {
            "ids": ["A", "B"],
            "embeddings": np.eye(2, dtype=np.float32),
        }
        orch.embed_fn.embed_query.side_effect = RuntimeError("model down")

        results = [
            ("A", _mk_data("shared token", 1.0)),
            ("B", _mk_data("shared token", 0.5)),
        ]
        picked = orch._apply_mmr(results, top_k=2, lambda_param=0.5)

        # Even under embed failure we come back with something.
        assert len(picked) == 2
        assert picked[0][0] == "A"

    def test_short_pool_returns_input_untouched(self):
        orch = _make_orchestrator_stub()
        results = [
            ("A", _mk_data("one", 1.0)),
            ("B", _mk_data("two", 0.5)),
        ]
        # top_k >= len(results) — no work to do, no ChromaDB call.
        picked = orch._apply_mmr(results, top_k=5, lambda_param=0.7)
        assert picked == results
        orch.collection.get.assert_not_called()


# ---------------------------------------------------------------------------
# Regression: MMR real produces less redundant chunks than Jaccard
# ---------------------------------------------------------------------------


class TestMmrRegressionVsJaccard:
    """Fixed corpus fixture: the new MMR must not increase redundancy.

    The 'redundancy' metric here is the average pairwise cosine similarity
    between the top-k picks — lower is more diverse. The corpus is crafted
    so the query lives inside a dense semantic cluster (near-duplicates in
    embedding space) but the chunks share almost no surface tokens (so
    Jaccard cannot detect the redundancy). This is the exact failure mode
    A2.5 is designed to fix.
    """

    def _avg_pairwise_cosine(self, ids, id_to_emb):
        vecs = np.array([id_to_emb[i] for i in ids], dtype=np.float32)
        # Already unit-norm by construction below.
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(np.dot(vecs[i], vecs[j])))
        if not sims:
            return 0.0
        return sum(sims) / len(sims)

    def test_embedding_mmr_beats_jaccard_on_semantic_cluster(self):
        orch = _make_orchestrator_stub()

        # Corpus: 4 near-duplicates in embedding space with disjoint surface
        # tokens, plus 2 truly diverse chunks.
        chunks = [
            ("dup1", "authentication bypass jwt algorithm", [1.00, 0.02, 0.02]),
            ("dup2", "login token cryptographic weakness", [0.99, 0.05, 0.04]),
            ("dup3", "credential signing symmetric key", [0.98, 0.03, 0.06]),
            ("dup4", "session cookie hmac collision", [0.97, 0.06, 0.05]),
            ("div1", "kubernetes rbac privilege escalation", [0.10, 1.00, 0.10]),
            ("div2", "supply chain sbom slsa attestation", [0.10, 0.10, 1.00]),
        ]

        # Normalize embeddings so cosine == dot product.
        def _norm(v):
            v = np.array(v, dtype=np.float32)
            n = np.linalg.norm(v)
            return (v / n).tolist() if n > 0 else v.tolist()

        results = []
        embeddings_row = []
        ids = []
        id_to_emb = {}
        for i, (cid, doc, emb) in enumerate(chunks):
            score = 1.0 - i * 0.05  # descending reranker score
            results.append((cid, _mk_data(doc, score)))
            embeddings_row.append(_norm(emb))
            ids.append(cid)
            id_to_emb[cid] = _norm(emb)

        query_embedding = _norm([1.0, 0.02, 0.02])
        embeddings = np.array(embeddings_row, dtype=np.float32)

        orch.collection.get.return_value = {"ids": ids, "embeddings": embeddings}
        orch.embed_fn.embed_query.return_value = [query_embedding]
        orch._current_query_text = "authentication bypass jwt"

        # Both pipelines run with the same lambda. A moderately diversity-
        # biased setting (0.3) is where the difference between "sees semantic
        # redundancy" (embeddings) and "sees only token overlap" (Jaccard)
        # actually surfaces — at lambda >= 0.5 relevance dominates in both.
        embedding_picks = orch._apply_mmr(list(results), top_k=3, lambda_param=0.3)
        jaccard_picks = orch._apply_mmr_jaccard(list(results), top_k=3, lambda_param=0.3)

        emb_ids = [pid for pid, _ in embedding_picks]
        jac_ids = [pid for pid, _ in jaccard_picks]

        emb_redundancy = self._avg_pairwise_cosine(emb_ids, id_to_emb)
        jac_redundancy = self._avg_pairwise_cosine(jac_ids, id_to_emb)

        # Both keep the reranker's top pick at position 0.
        assert emb_ids[0] == "dup1"
        assert jac_ids[0] == "dup1"

        # The new pipeline must be at least as diverse — that is the whole
        # point of A2.5.
        assert emb_redundancy < jac_redundancy, (
            f"Embedding MMR did not reduce redundancy vs Jaccard: "
            f"emb_ids={emb_ids} ({emb_redundancy:.3f}) vs "
            f"jac_ids={jac_ids} ({jac_redundancy:.3f})"
        )

        # And it must actually surface at least one of the truly diverse chunks.
        assert set(emb_ids) & {"div1", "div2"}, f"Embedding MMR did not surface any diverse chunk: {emb_ids}"


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_mmr_defaults_are_sane(self):
        from mcp_server.config import Config

        cfg = Config()
        assert cfg.mmr_enabled is True
        assert 0.0 <= cfg.mmr_lambda <= 1.0
        # Contract with the operational docs: default is relevance-biased.
        assert math.isclose(cfg.mmr_lambda, 0.7, rel_tol=1e-6)
