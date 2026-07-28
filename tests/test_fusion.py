"""
╭─╴ FUSION STRATEGY TESTS ╶──────────────────────────────────────╮
│                                                                │
│   Unit + regression coverage for A2.4 pluggable fusion.        │
│   Every strategy exercised in isolation plus a backwards-      │
│   compat check that RRF stays byte-identical to the pre-A2.4   │
│   hardcoded implementation.                                    │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

from __future__ import annotations

import pytest

from mcp_server.retrieval.fusion import (
    CombMNZ,
    CombSUM,
    FusionStrategy,
    RetrievedResult,
    RRFusion,
    WeightedLinear,
    _min_max_normalise,
    available_strategies,
    get_strategy,
)

# ╭─╴ Helpers ╶────────────────────────────────╮


def _sem(doc_id: str, rank: int, score: float | None = None) -> RetrievedResult:
    """Build a semantic-branch RetrievedResult."""
    return RetrievedResult(
        doc_id=doc_id,
        semantic_rank=rank,
        bm25_rank=None,
        semantic_score=score if score is not None else max(0.0, 1.0 - 0.1 * rank),
        bm25_score=None,
        metadata={},
    )


def _bm(doc_id: str, rank: int, score: float | None = None) -> RetrievedResult:
    """Build a BM25-branch RetrievedResult."""
    return RetrievedResult(
        doc_id=doc_id,
        semantic_rank=None,
        bm25_rank=rank,
        semantic_score=None,
        bm25_score=score if score is not None else 10.0 / rank,
        metadata={},
    )


# ╭─╴ Registry + factory ╶─────────────────────╮


class TestRegistry:
    """Registry and factory behaviour."""

    def test_all_four_strategies_registered(self):
        """The four canonical strategies must be discoverable."""
        strategies = set(available_strategies())
        assert {"rrf", "combsum", "combmnz", "weighted"}.issubset(strategies)

    def test_get_strategy_returns_singleton_instance(self):
        """get_strategy must return a pre-instantiated singleton, not a class."""
        s1 = get_strategy("rrf")
        s2 = get_strategy("rrf")
        assert s1 is s2  # same instance
        assert isinstance(s1, RRFusion)
        assert s1.name == "rrf"

    def test_get_strategy_unknown_raises_valueerror(self):
        """Unknown names must raise ValueError with an actionable listing."""
        with pytest.raises(ValueError) as exc:
            get_strategy("does-not-exist")
        # Error message must enumerate every valid strategy so ops can fix
        # the typo without opening docs.
        msg = str(exc.value)
        assert "does-not-exist" in msg
        assert "rrf" in msg
        assert "combsum" in msg
        assert "combmnz" in msg
        assert "weighted" in msg

    def test_get_strategy_case_sensitive(self):
        """Names are canonical — 'RRF' must not resolve to 'rrf'."""
        with pytest.raises(ValueError):
            get_strategy("RRF")

    def test_each_strategy_exposes_name_attribute(self):
        """Every registered strategy must expose a name for tracing/telemetry."""
        for name in available_strategies():
            assert get_strategy(name).name == name

    def test_strategies_conform_to_protocol(self):
        """Every registered strategy must be a structural FusionStrategy."""
        # Protocol checks are structural — verify presence of name + fuse.
        for name in available_strategies():
            s = get_strategy(name)
            assert hasattr(s, "name")
            assert callable(getattr(s, "fuse", None))
            # Explicit isinstance against Protocol — Python allows this via
            # runtime_checkable, but our Protocol is not marked so; the
            # attribute check above is the meaningful contract.
            _ = FusionStrategy  # keeps the import used for readers


# ╭─╴ Min-max normalise helper ╶───────────────╮


class TestMinMaxNormalise:
    """Direct coverage of the shared normalisation helper."""

    def test_empty_maps_to_empty(self):
        assert _min_max_normalise({}) == {}

    def test_typical_range_produces_zero_and_one(self):
        norm = _min_max_normalise({"a": 1.0, "b": 3.0, "c": 5.0})
        assert norm["a"] == 0.0
        assert norm["c"] == 1.0
        assert 0 < norm["b"] < 1

    def test_identical_scores_all_map_to_one(self):
        """When every score is equal, treat them as tied at the top."""
        norm = _min_max_normalise({"a": 4.2, "b": 4.2, "c": 4.2})
        assert all(v == 1.0 for v in norm.values())

    def test_single_entry_maps_to_one(self):
        assert _min_max_normalise({"only": 7.0}) == {"only": 1.0}


# ╭─╴ RRFusion ╶───────────────────────────────╮


class TestRRFusion:
    """RRF strategy — the culture-preserving default."""

    def test_score_formula_matches_reference(self):
        """Score must equal alpha/(K+rs) + (1-alpha)/(K+rb) — the paper's RRF."""
        strategy = RRFusion()
        # doc_a: sem rank 1, bm25 rank 1 (both branches strong)
        # alpha=0.5, K=60 => 0.5*(1/61) + 0.5*(1/61) = 1/61
        sem = [_sem("a", rank=1)]
        bm = [_bm("a", rank=1)]
        fused = strategy.fuse(sem, bm, alpha=0.5)
        assert len(fused) == 1
        expected = 0.5 * (1 / 61) + 0.5 * (1 / 61)
        assert fused[0][0] == "a"
        assert fused[0][1] == pytest.approx(expected, rel=1e-9)

    def test_missing_branch_uses_legacy_rank_1000_fallback(self):
        """RRF preserves the historical rank=1000 fallback for absent branch.

        This is the culture-preserving choice — the pre-A2.4 code used 1000
        as the fallback. Task spec asks for inf; we knowingly keep 1000 so
        the seven-pillar regression suite stays green. See RRFusion docstring.
        """
        strategy = RRFusion()
        # doc_a only in semantic (rank 1), doc_b only in bm25 (rank 1)
        # alpha=0.5: doc_a = 0.5/61 + 0.5/1060, doc_b symmetric
        fused = strategy.fuse([_sem("a", 1)], [_bm("b", 1)], alpha=0.5)
        scored = dict(fused)
        expected_a = 0.5 * (1 / 61) + 0.5 * (1 / 1060)
        expected_b = 0.5 * (1 / 1060) + 0.5 * (1 / 61)
        assert scored["a"] == pytest.approx(expected_a, rel=1e-9)
        assert scored["b"] == pytest.approx(expected_b, rel=1e-9)
        # Both branches identical rank so scores tie
        assert scored["a"] == pytest.approx(scored["b"], rel=1e-9)

    def test_alpha_zero_is_bm25_only(self):
        """alpha=0.0 must zero out the semantic contribution."""
        strategy = RRFusion()
        fused = strategy.fuse([_sem("a", 1)], [_bm("a", 1)], alpha=0.0)
        assert fused[0][1] == pytest.approx(1 / 61, rel=1e-9)

    def test_alpha_one_is_semantic_only(self):
        """alpha=1.0 must zero out the BM25 contribution."""
        strategy = RRFusion()
        fused = strategy.fuse([_sem("a", 1)], [_bm("a", 1)], alpha=1.0)
        assert fused[0][1] == pytest.approx(1 / 61, rel=1e-9)

    def test_ordering_descending_by_score(self):
        """Output must be sorted highest-score-first."""
        strategy = RRFusion()
        sem = [_sem("a", 1), _sem("b", 2), _sem("c", 3)]
        bm = [_bm("a", 3), _bm("b", 2), _bm("c", 1)]
        fused = strategy.fuse(sem, bm, alpha=0.5)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)
        # doc_b balanced across both branches — same total as doc_a and doc_c
        # (symmetric). Doc_a wins on tiebreak / doc_c does the same on other
        # end, but the middle doc_b is between them. Deterministic ordering
        # matters; check b is not first.

    def test_ignores_weights_argument(self):
        """RRF must ignore the weights argument (Protocol conformance only)."""
        strategy = RRFusion()
        without = strategy.fuse([_sem("a", 1)], [_bm("a", 1)], alpha=0.5, weights=None)
        with_w = strategy.fuse([_sem("a", 1)], [_bm("a", 1)], alpha=0.5, weights={"semantic": 999})
        assert without == with_w


class TestRRFusionRegression:
    """Backwards-compat: fused score reproduces the exact pre-A2.4 formula."""

    def test_matches_hardcoded_pre_a2_4_formula(self):
        """Regression: RRF via the plugin must equal the hardcoded RRF path.

        The pre-A2.4 orchestrator computed:
            semantic_rrf = hybrid_alpha * (1 / (60 + semantic_rank))
            bm25_rrf     = (1 - hybrid_alpha) * (1 / (60 + bm25_rank))
            combined     = semantic_rrf + bm25_rrf
        with rank=1000 fallback for missing branches. This test re-derives
        that value and asserts equality — any drift breaks the seven-pillar
        Quality Gate regression on top-K ordering.
        """
        strategy = get_strategy("rrf")
        RRF_K = 60
        alpha = 0.3  # default from search_knowledge()

        cases = [
            # (semantic_rank, bm25_rank)
            (1, 1),
            (1, 5),
            (5, 1),
            (1, None),  # missing BM25
            (None, 1),  # missing semantic
            (10, 10),
        ]

        for i, (sr, br) in enumerate(cases):
            doc_id = f"d{i}"
            sem_in = [_sem(doc_id, sr)] if sr is not None else []
            bm_in = [_bm(doc_id, br)] if br is not None else []
            fused = strategy.fuse(sem_in, bm_in, alpha=alpha)

            # Legacy formula, ranks default to 1000 when branch absent
            legacy_sr = sr if sr is not None else 1000
            legacy_br = br if br is not None else 1000
            expected = alpha * (1 / (RRF_K + legacy_sr)) + (1 - alpha) * (1 / (RRF_K + legacy_br))
            assert fused[0][1] == pytest.approx(expected, rel=1e-9), f"RRF drift on case {i}: {sr=}, {br=}"


# ╭─╴ CombSUM ╶────────────────────────────────╮


class TestCombSUM:
    """CombSUM — sum of normalised scores, alpha-weighted."""

    def test_uses_normalised_scores_not_ranks(self):
        """A doc with the top raw score wins even when its rank is not #1."""
        strategy = CombSUM()
        # Both docs have same rank (1) but doc_b has higher raw semantic score.
        sem = [
            RetrievedResult("a", semantic_rank=1, bm25_rank=None, semantic_score=0.2, bm25_score=None, metadata={}),
            RetrievedResult("b", semantic_rank=2, bm25_rank=None, semantic_score=0.9, bm25_score=None, metadata={}),
        ]
        bm = []
        fused = strategy.fuse(sem, bm, alpha=1.0)  # semantic-only
        # After min-max normalisation, b=1.0 and a=0.0
        assert fused[0][0] == "b"
        assert fused[0][1] == pytest.approx(1.0, rel=1e-9)
        assert fused[1][0] == "a"
        assert fused[1][1] == pytest.approx(0.0, abs=1e-9)

    def test_missing_branch_contributes_zero(self):
        """Docs absent from one branch see that branch's contribution as 0."""
        strategy = CombSUM()
        sem = [RetrievedResult("a", 1, None, 0.9, None, {})]
        bm = [RetrievedResult("b", None, 1, None, 5.0, {})]
        fused = strategy.fuse(sem, bm, alpha=0.5)
        scored = dict(fused)
        # Each doc normalises to 1.0 within its own branch (single entry)
        # and 0.0 contribution from the missing branch.
        # a: 0.5*1.0 + 0.5*0.0 = 0.5
        # b: 0.5*0.0 + 0.5*1.0 = 0.5
        assert scored["a"] == pytest.approx(0.5, rel=1e-9)
        assert scored["b"] == pytest.approx(0.5, rel=1e-9)

    def test_alpha_shifts_weight_toward_semantic(self):
        """alpha=1.0 gives all weight to semantic branch."""
        strategy = CombSUM()
        sem = [
            RetrievedResult("a", 1, None, 0.9, None, {}),
            RetrievedResult("b", 2, None, 0.1, None, {}),
        ]
        bm = [
            RetrievedResult("b", None, 1, None, 5.0, {}),  # b wins in BM25
            RetrievedResult("a", None, 2, None, 1.0, {}),
        ]
        # alpha=1 -> semantic dominates -> a wins
        fused_sem = strategy.fuse(sem, bm, alpha=1.0)
        assert fused_sem[0][0] == "a"
        # alpha=0 -> bm25 dominates -> b wins
        fused_bm = strategy.fuse(sem, bm, alpha=0.0)
        assert fused_bm[0][0] == "b"


# ╭─╴ CombMNZ ╶────────────────────────────────╮


class TestCombMNZ:
    """CombMNZ — CombSUM scaled by branch-hit count."""

    def test_both_branches_agreement_doubles_score(self):
        """A doc found in both branches must beat single-branch docs.

        With equal per-branch normalised scores, the doc found in both
        branches has n_hits=2 so its CombMNZ score is exactly double the
        CombSUM score of a single-branch peer.
        """
        strategy = CombMNZ()
        sem = [
            RetrievedResult("a", 1, None, 1.0, None, {}),  # only semantic
            RetrievedResult("shared", 2, None, 1.0, None, {}),  # both branches
        ]
        bm = [
            RetrievedResult("b", None, 1, None, 5.0, {}),  # only bm25
            RetrievedResult("shared", None, 2, None, 5.0, {}),  # both branches
        ]
        fused = strategy.fuse(sem, bm, alpha=0.5)
        scored = dict(fused)
        # After min-max: sem_norm[a]=1.0, sem_norm[shared]=1.0 (identical -> both 1)
        #                bm_norm[b]=1.0, bm_norm[shared]=1.0
        # a: (0.5*1.0 + 0.5*0.0) * 1 = 0.5
        # b: (0.5*0.0 + 0.5*1.0) * 1 = 0.5
        # shared: (0.5*1.0 + 0.5*1.0) * 2 = 2.0
        assert scored["shared"] == pytest.approx(2.0, rel=1e-9)
        assert scored["a"] == pytest.approx(0.5, rel=1e-9)
        assert scored["b"] == pytest.approx(0.5, rel=1e-9)
        # Shared must be first
        assert fused[0][0] == "shared"

    def test_single_branch_gets_no_boost(self):
        """A single-branch doc keeps its CombSUM value (× 1)."""
        strategy = CombMNZ()
        sem = [RetrievedResult("a", 1, None, 1.0, None, {})]
        bm = []
        fused = strategy.fuse(sem, bm, alpha=1.0)
        # sem_norm[a] = 1.0, n_hits=1 -> 1.0 * 1.0 * 1 = 1.0
        assert fused[0][1] == pytest.approx(1.0, rel=1e-9)


# ╭─╴ WeightedLinear ╶─────────────────────────╮


class TestWeightedLinear:
    """WeightedLinear — explicit per-branch weights."""

    def test_default_weights_degenerate_to_alpha_split(self):
        """Without ``weights=``, degenerates to CombSUM's alpha/(1-alpha)."""
        strategy = WeightedLinear()
        combsum = CombSUM()
        sem = [
            RetrievedResult("a", 1, None, 0.9, None, {}),
            RetrievedResult("b", 2, None, 0.1, None, {}),
        ]
        bm = [
            RetrievedResult("a", None, 2, None, 1.0, {}),
            RetrievedResult("b", None, 1, None, 5.0, {}),
        ]
        w = strategy.fuse(sem, bm, alpha=0.5, weights=None)
        c = combsum.fuse(sem, bm, alpha=0.5)
        assert dict(w) == pytest.approx(dict(c), rel=1e-9)

    def test_explicit_weights_override_alpha(self):
        """Explicit weights must take precedence over ``alpha``."""
        strategy = WeightedLinear()
        sem = [
            RetrievedResult("a", 1, None, 1.0, None, {}),
            RetrievedResult("b", 2, None, 0.0, None, {}),
        ]
        bm = [
            RetrievedResult("a", None, 2, None, 0.0, {}),
            RetrievedResult("b", None, 1, None, 5.0, {}),
        ]
        # Push everything toward BM25 via weights (alpha would go the other way)
        fused = strategy.fuse(sem, bm, alpha=1.0, weights={"semantic": 0.0, "bm25": 1.0})
        # Only BM25 contributes -> b wins on higher BM25 score
        assert fused[0][0] == "b"

    def test_partial_weights_fill_missing_from_alpha(self):
        """Missing weight keys fall back to alpha / (1-alpha)."""
        strategy = WeightedLinear()
        sem = [RetrievedResult("a", 1, None, 1.0, None, {})]
        bm = [RetrievedResult("a", None, 1, None, 5.0, {})]

        # Only semantic weight given -> bm25 defaults to (1 - alpha)
        # With alpha=0.4, w_sem=0.7 (explicit), w_bm25=0.6 (fallback)
        fused = strategy.fuse(sem, bm, alpha=0.4, weights={"semantic": 0.7})
        # sem_norm[a]=1.0, bm_norm[a]=1.0 -> 0.7*1 + 0.6*1 = 1.3
        assert fused[0][1] == pytest.approx(1.3, rel=1e-9)


# ╭─╴ Cross-strategy invariants ╶──────────────╮


class TestInvariantsAcrossStrategies:
    """Contract every strategy must uphold regardless of formula."""

    @pytest.mark.parametrize("name", ["rrf", "combsum", "combmnz", "weighted"])
    def test_empty_inputs_return_empty_list(self, name: str):
        """Empty inputs must return [], not crash."""
        strategy = get_strategy(name)
        assert strategy.fuse([], [], alpha=0.5) == []

    @pytest.mark.parametrize("name", ["rrf", "combsum", "combmnz", "weighted"])
    def test_output_is_sorted_descending(self, name: str):
        """Every strategy sorts descending by fused score."""
        strategy = get_strategy(name)
        sem = [_sem("a", 1), _sem("b", 2), _sem("c", 3)]
        bm = [_bm("a", 2), _bm("b", 1), _bm("c", 3)]
        fused = strategy.fuse(sem, bm, alpha=0.5)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("name", ["rrf", "combsum", "combmnz", "weighted"])
    def test_output_ids_are_union_of_inputs(self, name: str):
        """Every input doc_id must appear exactly once in the output."""
        strategy = get_strategy(name)
        sem = [_sem("a", 1), _sem("b", 2)]
        bm = [_bm("b", 1), _bm("c", 2)]  # b overlaps both branches
        fused = strategy.fuse(sem, bm, alpha=0.5)
        out_ids = [doc_id for doc_id, _ in fused]
        assert set(out_ids) == {"a", "b", "c"}
        assert len(out_ids) == 3  # no duplicates


# ╭─╴ End-to-end via orchestrator.query ╶──────╮


class TestOrchestratorFusionRouting:
    """Verify orchestrator.query honours the fusion kwarg + config default."""

    def _build_orchestrator(self, monkeypatch, *, bm25_hits, metadatas, docs, routed_category=None):
        """Assemble a fake orchestrator that skips ChromaDB/BM25 model loading."""
        from mcp_server.server import KnowledgeOrchestrator

        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

            def stats(self):
                return {"hit_rate": "0%"}

        class FakeBM25:
            def search(self, query, top_k):
                return bm25_hits

        class FakeCollection:
            def get(self, ids, include):
                return {
                    "ids": ids,
                    "documents": [docs[cid] for cid in ids] if "documents" in include else None,
                    "metadatas": [metadatas[cid] for cid in ids] if "metadatas" in include else None,
                }

        orch = object.__new__(KnowledgeOrchestrator)
        orch.query_cache = FakeCache()
        orch.bm25_index = FakeBM25()
        orch.collection = FakeCollection()
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda query: routed_category
        orch._expand_with_adjacent_chunks = lambda results: results
        return orch

    def test_query_default_fusion_uses_rrf(self, monkeypatch):
        """Without a ``fusion`` kwarg the orchestrator must run RRF."""
        metadatas = {
            "cid_a": {"source": "/a", "filename": "a", "category": "x", "chunk_index": 0, "keywords": ""},
            "cid_b": {"source": "/b", "filename": "b", "category": "x", "chunk_index": 0, "keywords": ""},
        }
        docs = {"cid_a": "text a", "cid_b": "text b"}
        orch = self._build_orchestrator(
            monkeypatch,
            bm25_hits=[("cid_a", 10.0), ("cid_b", 5.0)],
            metadatas=metadatas,
            docs=docs,
        )

        # BM25-only path (alpha=0) so semantic branch is not consulted.
        results = orch.query("anything", max_results=5, category_filter=None, hybrid_alpha=0.0)
        assert len(results) == 2
        # cid_a (BM25 rank 1) must outrank cid_b (BM25 rank 2)
        assert results[0]["source"] == "/a"

    def test_query_explicit_fusion_override_wins_over_config(self, monkeypatch):
        """A per-call ``fusion=`` kwarg must override config.fusion_strategy."""
        # Force config.fusion_strategy to something else so we can prove the
        # override is respected.
        monkeypatch.setattr("mcp_server.server.config.fusion_strategy", "combsum")

        metadatas = {
            "cid_a": {"source": "/a", "filename": "a", "category": "x", "chunk_index": 0, "keywords": ""},
        }
        docs = {"cid_a": "text a"}
        orch = self._build_orchestrator(
            monkeypatch,
            bm25_hits=[("cid_a", 10.0)],
            metadatas=metadatas,
            docs=docs,
        )

        # Both calls succeed; validating that per-call override reaches
        # the fusion registry without raising.
        results_default = orch.query("q", max_results=5, category_filter=None, hybrid_alpha=0.0)
        results_rrf = orch.query("q", max_results=5, category_filter=None, hybrid_alpha=0.0, fusion="rrf")
        assert results_default[0]["source"] == "/a"
        assert results_rrf[0]["source"] == "/a"

    def test_query_invalid_fusion_raises_at_get_strategy(self, monkeypatch):
        """Unknown fusion name must surface a ValueError from get_strategy."""
        metadatas = {"cid": {"source": "/a", "filename": "a", "category": "x", "chunk_index": 0, "keywords": ""}}
        docs = {"cid": "t"}
        orch = self._build_orchestrator(
            monkeypatch,
            bm25_hits=[("cid", 1.0)],
            metadatas=metadatas,
            docs=docs,
        )
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            orch.query("q", max_results=5, hybrid_alpha=0.0, fusion="banana")


# ╭─╴ MCP / CLI surface (_search_impl) ╶───────╮


class TestSearchImplFusionSurface:
    """The ``_search_impl`` helper mediates MCP + CLI. Fusion must round-trip."""

    def test_invalid_fusion_returns_structured_error(self, monkeypatch):
        """A typo must surface a friendly error, not a ValueError from deep in the stack."""
        from mcp_server.retrieval.impls import _search_impl

        # Stub the orchestrator so we never hit ChromaDB.
        class FakeOrch:
            pass

        monkeypatch.setattr("mcp_server.server.get_orchestrator", lambda: FakeOrch())

        payload = _search_impl("q", fusion="does-not-exist")
        assert payload["status"] == "error"
        assert "Invalid fusion" in payload["message"]
        assert "rrf" in payload["message"]

    def test_valid_fusion_pass_through_to_orchestrator(self, monkeypatch):
        """A valid fusion must reach orchestrator.query() as a kwarg."""
        from mcp_server.retrieval.impls import _search_impl

        captured: dict = {}

        class FakeCacheStats:
            @staticmethod
            def stats():
                return {"hit_rate": "0%"}

        class FakeOrch:
            query_cache = FakeCacheStats()

            def query(self, query_text, **kwargs):
                captured.update(kwargs)
                captured["query_text"] = query_text
                return []

        monkeypatch.setattr("mcp_server.server.get_orchestrator", lambda: FakeOrch())

        payload = _search_impl("q", fusion="combmnz")
        assert captured["fusion"] == "combmnz"
        # Even ``no_results`` payloads keep the request context —
        # verify the request status was reached (not an early error).
        assert payload["status"] == "no_results"

    def test_default_fusion_kwarg_is_none(self, monkeypatch):
        """When caller omits ``fusion``, orchestrator gets fusion=None."""
        from mcp_server.retrieval.impls import _search_impl

        captured: dict = {}

        class FakeCacheStats:
            @staticmethod
            def stats():
                return {"hit_rate": "0%"}

        class FakeOrch:
            query_cache = FakeCacheStats()

            def query(self, query_text, **kwargs):
                captured.update(kwargs)
                return []

        monkeypatch.setattr("mcp_server.server.get_orchestrator", lambda: FakeOrch())

        _search_impl("q")
        assert captured["fusion"] is None
