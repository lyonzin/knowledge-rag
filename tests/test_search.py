"""Tests for search pipeline components (no model/DB required)."""

from mcp_server.config import _merge_query_expansion_sources
from mcp_server.server import BM25Index, KnowledgeOrchestrator, QueryCache

# ── BM25 Query Expansion ──


class TestQueryExpansion:
    def setup_method(self):
        self.bm25 = BM25Index()

    def test_sqli_expands(self):
        """sqli must expand to sql injection."""
        expanded = self.bm25.expand_query("sqli")
        assert "sql injection" in expanded

    def test_privesc_expands(self):
        """privesc must expand to privilege escalation."""
        expanded = self.bm25.expand_query("privesc")
        assert "privilege escalation" in expanded

    def test_amsi_expands(self):
        """amsi must expand to antimalware scan interface."""
        expanded = self.bm25.expand_query("amsi")
        assert "antimalware" in expanded

    def test_cve_alias_printnightmare(self):
        """printnightmare must expand to CVE-2021-34527."""
        expanded = self.bm25.expand_query("printnightmare")
        assert "cve-2021-34527" in expanded

    def test_cve_alias_eternalblue(self):
        """eternalblue must expand to ms17-010."""
        expanded = self.bm25.expand_query("eternalblue")
        assert "ms17-010" in expanded

    def test_no_expansion_unknown(self):
        """Unknown terms return unchanged."""
        expanded = self.bm25.expand_query("xyzunknownterm")
        assert expanded == "xyzunknownterm"

    def test_bigram_expansion(self):
        """Two-word terms must expand."""
        expanded = self.bm25.expand_query("reverse shell")
        assert "revshell" in expanded

    def test_legacy_directional_expansion_still_works(self, monkeypatch):
        """Legacy directional mappings must still expand from the left-hand key."""
        monkeypatch.setattr(
            "mcp_server.server.config.query_expansions",
            {"tb": ["triple barrier", "trip_barr"]},
        )

        expanded = self.bm25.expand_query("tb")

        assert "triple barrier" in expanded
        assert "trip_barr" in expanded

    def test_group_expansion_is_symmetric(self, monkeypatch):
        """Any term from a group must expand to the rest of the group."""
        merged = _merge_query_expansion_sources({}, [["triple barrier", "tb", "trip_barr"]])
        monkeypatch.setattr("mcp_server.server.config.query_expansions", merged)

        expanded = self.bm25.expand_query("tb")

        assert "triple barrier" in expanded
        assert "trip_barr" in expanded

    def test_group_bigram_expansion(self, monkeypatch):
        """Multi-word group members must match via full query and bigrams."""
        merged = _merge_query_expansion_sources({}, [["triple barrier", "tb", "trip_barr"]])
        monkeypatch.setattr("mcp_server.server.config.query_expansions", merged)

        expanded = self.bm25.expand_query("triple barrier")

        assert "tb" in expanded
        assert "trip_barr" in expanded

    def test_mixed_expansion_sources_merge_cleanly(self, monkeypatch):
        """Legacy and grouped expansions must combine without losing entries."""
        merged = _merge_query_expansion_sources(
            {"pf": ["profit factor", "profit-factor"]},
            [["profit factor", "pf", "profit_factor"]],
        )
        monkeypatch.setattr("mcp_server.server.config.query_expansions", merged)

        expanded = self.bm25.expand_query("pf")

        assert "profit factor" in expanded
        assert "profit-factor" in expanded
        assert "profit_factor" in expanded


# ── BM25 Search ──


class TestBM25Search:
    def test_search_empty_index(self):
        """Search on empty index returns empty."""
        bm25 = BM25Index()
        results = bm25.search("test query")
        assert results == []

    def test_search_with_data(self):
        """Search returns ranked results."""
        bm25 = BM25Index()
        bm25.add_documents(
            ["doc1", "doc2", "doc3"],
            ["SQL injection bypass techniques", "XSS reflected attack", "SQL injection UNION based"],
        )
        bm25.build_index()
        results = bm25.search("SQL injection")
        assert len(results) >= 1
        # doc1 or doc3 should rank highest (both mention SQL injection)
        top_ids = [r[0] for r in results[:2]]
        assert "doc1" in top_ids or "doc3" in top_ids

    def test_search_empty_query(self):
        """Empty query returns empty."""
        bm25 = BM25Index()
        bm25.add_documents(["doc1"], ["some content"])
        bm25.build_index()
        results = bm25.search("")
        assert results == []


class TestHybridCategoryFilter:
    def test_bm25_results_respect_category_filter(self, monkeypatch):
        """BM25-only results must not bypass an explicit category filter."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return [("chunk_report", 10.0), ("chunk_code", 9.0)]

        class FakeCollection:
            _docs = {
                "chunk_report": "report content",
                "chunk_code": "code content",
            }
            _metadatas = {
                "chunk_report": {
                    "source": "/docs/report.md",
                    "filename": "report.md",
                    "category": "reports",
                    "chunk_index": 0,
                    "keywords": "",
                },
                "chunk_code": {
                    "source": "/src/code.py",
                    "filename": "code.py",
                    "category": "code",
                    "chunk_index": 0,
                    "keywords": "",
                },
            }

            def get(self, ids, include):
                return {
                    "ids": ids,
                    "documents": [self._docs[chunk_id] for chunk_id in ids] if "documents" in include else None,
                    "metadatas": [self._metadatas[chunk_id] for chunk_id in ids] if "metadatas" in include else None,
                }

        orchestrator = object.__new__(KnowledgeOrchestrator)
        orchestrator.query_cache = FakeCache()
        orchestrator.bm25_index = FakeBM25()
        orchestrator.collection = FakeCollection()
        orchestrator._ensure_bm25_index = lambda: None
        orchestrator._route_by_keywords = lambda query: None
        orchestrator._expand_with_adjacent_chunks = lambda results: results

        results = orchestrator.query("anything", max_results=5, category_filter="reports", hybrid_alpha=0.0)

        assert [result["source"] for result in results] == ["/docs/report.md"]
        assert {result["category"] for result in results} == {"reports"}


class TestKeywordRoutingBehavior:
    """When the user omits an explicit category_filter, keyword auto-routing must NOT
    restrict the search to a single category. The router is informational only:
    it can populate the ``routed_by`` metadata field, but must not act as a hard
    where-filter on either BM25 or semantic candidates.

    Regression: prior to this fix, ``_route_by_keywords()`` could pick an
    under-populated category (e.g. ``redteam`` with 2 docs) and hide the relevant
    material sitting in a larger category (e.g. ``security`` with thousands of docs).
    """

    METADATAS = {
        "chunk_redteam_generic": {
            "source": "/docs/redteam/rtfm.pdf",
            "filename": "rtfm.pdf",
            "category": "redteam",
            "chunk_index": 0,
            "keywords": "",
        },
        "chunk_security_esc1": {
            "source": "/docs/security/pentest-everything/adcs/esc1.md",
            "filename": "esc1.md",
            "category": "security",
            "chunk_index": 0,
            "keywords": "",
        },
    }
    DOCS = {
        "chunk_redteam_generic": "generic redteam content",
        "chunk_security_esc1": "ESC1 vulnerable template EKU Client Authentication",
    }

    def _build_orchestrator(self, monkeypatch, *, routed_category, bm25_hits):
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        metadatas = self.METADATAS
        docs = self.DOCS

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

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

        orchestrator = object.__new__(KnowledgeOrchestrator)
        orchestrator.query_cache = FakeCache()
        orchestrator.bm25_index = FakeBM25()
        orchestrator.collection = FakeCollection()
        orchestrator._ensure_bm25_index = lambda: None
        orchestrator._route_by_keywords = lambda query: routed_category
        orchestrator._expand_with_adjacent_chunks = lambda results: results
        return orchestrator

    def test_routed_category_does_not_restrict_bm25_when_no_explicit_filter(self, monkeypatch):
        """BM25 candidates from other categories must survive when user omits category_filter."""
        orchestrator = self._build_orchestrator(
            monkeypatch,
            routed_category="redteam",  # router picks an under-populated category
            bm25_hits=[("chunk_redteam_generic", 10.0), ("chunk_security_esc1", 9.5)],
        )

        results = orchestrator.query("ESC1 ADCS", max_results=5, category_filter=None, hybrid_alpha=0.0)

        categories_seen = {r["category"] for r in results}
        assert "security" in categories_seen, (
            "routed_category='redteam' must not hide docs from other categories when the user omitted category_filter"
        )
        # routed_by remains populated as informational telemetry (public API unchanged).
        assert {r["routed_by"] for r in results} == {"redteam"}

    def test_routed_category_does_not_restrict_semantic_when_no_explicit_filter(self, monkeypatch):
        """The semantic branch must be called with where=None when user omits category_filter."""
        orchestrator = self._build_orchestrator(
            monkeypatch,
            routed_category="redteam",
            bm25_hits=[],
        )

        captured_where: list = []

        def fake_query(query_texts, n_results, where, include):
            captured_where.append(where)
            return {
                "ids": [["chunk_redteam_generic", "chunk_security_esc1"]],
                "distances": [[0.10, 0.20]],
                "documents": [[self.DOCS["chunk_redteam_generic"], self.DOCS["chunk_security_esc1"]]],
                "metadatas": [[self.METADATAS["chunk_redteam_generic"], self.METADATAS["chunk_security_esc1"]]],
            }

        orchestrator.collection.query = fake_query

        _ = orchestrator.query("ESC1 ADCS", max_results=5, category_filter=None, hybrid_alpha=1.0)

        assert captured_where == [None], (
            f"semantic where_filter must be None when user omitted category_filter, got {captured_where}"
        )

    def test_explicit_category_filter_still_overrides_routing(self, monkeypatch):
        """Explicit category_filter must take effect regardless of the router (preserves #109)."""
        orchestrator = self._build_orchestrator(
            monkeypatch,
            routed_category="redteam",  # router would pick redteam...
            bm25_hits=[("chunk_redteam_generic", 10.0), ("chunk_security_esc1", 9.5)],
        )

        results = orchestrator.query("ESC1 ADCS", max_results=5, category_filter="security", hybrid_alpha=0.0)

        # ...but user asked explicitly for `security` — must win.
        assert [r["source"] for r in results] == ["/docs/security/pentest-everything/adcs/esc1.md"]
        assert {r["category"] for r in results} == {"security"}


# ── Query Cache ──


class TestQueryCache:
    def test_cache_miss(self):
        """First query is always a miss."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        result = cache.get("test", 5, None, 0.3)
        assert result is None

    def test_cache_hit(self):
        """Cached query returns stored result."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.put("test", 5, None, 0.3, [{"content": "result"}])
        result = cache.get("test", 5, None, 0.3)
        assert result is not None
        assert result[0]["content"] == "result"

    def test_cache_different_params(self):
        """Different params = different cache entries."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.put("test", 5, None, 0.3, ["result_a"])
        cache.put("test", 5, None, 0.7, ["result_b"])
        assert cache.get("test", 5, None, 0.3) == ["result_a"]
        assert cache.get("test", 5, None, 0.7) == ["result_b"]

    def test_cache_invalidate(self):
        """Invalidate clears all entries."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.put("test", 5, None, 0.3, ["result"])
        cache.invalidate()
        assert cache.get("test", 5, None, 0.3) is None

    def test_cache_stats(self):
        """Stats track hits and misses."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.get("miss", 5, None, 0.3)  # miss
        cache.put("hit", 5, None, 0.3, ["data"])
        cache.get("hit", 5, None, 0.3)  # hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_eviction(self):
        """LRU eviction when max_size reached."""
        cache = QueryCache(max_size=2, ttl_seconds=300)
        cache.put("a", 5, None, 0.3, ["a"])
        cache.put("b", 5, None, 0.3, ["b"])
        cache.put("c", 5, None, 0.3, ["c"])  # should evict "a"
        assert cache.get("a", 5, None, 0.3) is None
        assert cache.get("b", 5, None, 0.3) is not None


# ── Keyword Routing ──


class TestKeywordRouting:
    def test_routing_detects_redteam(self):
        """Security terms route to redteam."""
        # Test the static method logic without instantiating orchestrator
        import re

        from mcp_server.config import config

        query = "mimikatz credential dump"
        query_lower = query.lower()
        matches = {}
        for category, keywords in config.keyword_routes.items():
            count = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if " " in kw_lower:
                    if kw_lower in query_lower:
                        count += 1
                else:
                    if re.search(r"\b" + re.escape(kw_lower) + r"\b", query_lower):
                        count += 1
            if count > 0:
                matches[category] = count

        assert "redteam" in matches

    def test_word_boundary_prevents_false_positive(self):
        """'api' must NOT match inside 'RAPID'."""
        import re

        assert not re.search(r"\bapi\b", "rapid deployment")
        assert re.search(r"\bapi\b", "api endpoint")


# ── Path-aware ranking ──


class TestPathAwareRanking:
    def test_path_match_can_lift_keyword_result(self, monkeypatch):
        """Path and filename matches provide a small generic ranking signal."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return [("chunk_generic", 10.0), ("chunk_target", 9.0)]

        class FakeCollection:
            def get(self, ids, include):
                chunk_id = ids[0]
                documents = {
                    "chunk_generic": "same keyword content",
                    "chunk_target": "same keyword content",
                }
                metadatas = {
                    "chunk_generic": {
                        "source": "/docs/notes/general.md",
                        "filename": "general.md",
                        "category": "docs",
                        "chunk_index": 0,
                    },
                    "chunk_target": {
                        "source": "/docs/reports/api-security.md",
                        "filename": "api-security.md",
                        "category": "docs",
                        "chunk_index": 0,
                    },
                }
                return {"documents": [documents[chunk_id]], "metadatas": [metadatas[chunk_id]]}

        orchestrator = object.__new__(KnowledgeOrchestrator)
        orchestrator.query_cache = FakeCache()
        orchestrator.bm25_index = FakeBM25()
        orchestrator.collection = FakeCollection()
        orchestrator._ensure_bm25_index = lambda: None
        orchestrator._route_by_keywords = lambda query: None
        orchestrator._expand_with_adjacent_chunks = lambda results: results

        results = orchestrator.query("api security", max_results=2, hybrid_alpha=0.0)

        assert results[0]["source"] == "/docs/reports/api-security.md"


# ── Candidate pool math (v4.8.0 Fase 3) ──


class TestDoSemanticCandidateMath:
    """Pin the ``_do_semantic`` candidate count against future clamping regressions.

    Before v4.8.0 Fase 3 the line
        ``n_candidates = min(max_results * 3, config.max_results)``
    was silently capped at 20 because ``config.max_results`` defaulted to
    20 — while BM25 pulled up to ``max_results * 20 = 400`` candidates.
    Semantic starved on hybrid mode without any log signal.

    The Fase 3 fix raised the ``max_results`` default from 20 to 100, so
    the ``min(...)`` now yields ``max_results * 3`` for typical callers
    (``max_results * 3 = 15 << 100``). This regression pin protects
    against future PRs that "helpfully" restore the 20-cap.
    """

    def test_semantic_asks_chromadb_for_3x_max_results(self, monkeypatch):
        """With max_results=5 and config.max_results=100 → 15 candidates (not 20)."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)
        monkeypatch.setattr("mcp_server.server.config.max_results", 100)

        captured = {}

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return []

        class FakeCollection:
            def query(self, query_texts, n_results, where, include):
                captured["n_results"] = n_results
                return {
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                }

            def get(self, ids, include):
                return {"documents": [], "metadatas": []}

        orch = object.__new__(KnowledgeOrchestrator)
        orch.query_cache = FakeCache()
        orch.bm25_index = FakeBM25()
        orch.collection = FakeCollection()
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda query: None
        orch._expand_with_adjacent_chunks = lambda results: results

        # hybrid_alpha=1.0 forces the semantic-only branch (skips BM25).
        _ = orch.query("test query", max_results=5, hybrid_alpha=1.0)

        # 5 * 3 = 15; min(15, 100) = 15. NOT clamped at 20 anymore.
        assert captured["n_results"] == 15, (
            f"Expected 15 candidates (max_results=5 * 3, capped at "
            f"config.max_results=100), got {captured['n_results']}. "
            f"If this fails, someone probably reverted the v4.8.0 Fase 3 "
            f"default bump — check config.max_results and the min(...) "
            f"expression in server.py::_do_semantic."
        )

    def test_semantic_pool_is_bounded_by_config_max_results(self, monkeypatch):
        """When max_results * 3 exceeds config.max_results, the config value wins."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)
        # Cap intentionally small so max_results * 3 > cap.
        monkeypatch.setattr("mcp_server.server.config.max_results", 50)

        captured = {}

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return []

        class FakeCollection:
            def query(self, query_texts, n_results, where, include):
                captured["n_results"] = n_results
                return {
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                }

            def get(self, ids, include):
                return {"documents": [], "metadatas": []}

        orch = object.__new__(KnowledgeOrchestrator)
        orch.query_cache = FakeCache()
        orch.bm25_index = FakeBM25()
        orch.collection = FakeCollection()
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda query: None
        orch._expand_with_adjacent_chunks = lambda results: results

        # 20 * 3 = 60, but config.max_results = 50 → min() clamps to 50
        _ = orch.query("test query", max_results=20, hybrid_alpha=1.0)

        assert captured["n_results"] == 50


# =============================================================================
# FTS5 Fast-Path Dispatch — Task 03 (ADR-002, ADR-003, ADR-006)
# =============================================================================


class _FakeFts5:
    """Controllable ``Fts5LexicalIndex`` substitute for dispatch tests."""

    def __init__(self, hits=(), ready=True, raises=None):
        self._hits = list(hits)
        self._ready = ready
        self._raises = raises
        self.search_calls = []

    def is_ready(self):
        return self._ready

    def search(self, query, top_k=20):
        self.search_calls.append((query, top_k))
        if self._raises is not None:
            raise self._raises
        return list(self._hits)


class _FakeRouter:
    """Controllable ``QueryRouter`` substitute — fixed or callable decision."""

    def __init__(self, decision):
        self._decision = decision
        self.classify_calls = []

    def classify(self, query):
        self.classify_calls.append(query)
        if callable(self._decision):
            return self._decision(query)
        return self._decision


def _build_dispatch_orch(
    monkeypatch,
    *,
    fts5_enabled=True,
    fts5_index=None,
    query_router=None,
    fts5_min_hits=3,
    fake_docs=None,
    fake_metadatas=None,
):
    """Bare-bones orchestrator sufficient for exercising query() dispatch."""
    import mcp_server.server as srv

    monkeypatch.setattr(srv.config, "fts5_enabled", fts5_enabled)
    monkeypatch.setattr(srv.config, "fts5_min_hits", fts5_min_hits)
    monkeypatch.setattr(srv.config, "fts5_rerank_enabled", False)
    monkeypatch.setattr(srv.config, "reranker_enabled", False)
    monkeypatch.setattr(srv.config, "default_results", 5)
    monkeypatch.setattr(srv.config, "max_results", 50)

    docs = fake_docs or {}
    metadatas = fake_metadatas or {}

    class _FakeCollection:
        def get(self, ids, include):
            return {
                "ids": list(ids),
                "documents": [docs.get(cid, "") for cid in ids] if "documents" in include else None,
                "metadatas": [metadatas.get(cid, {}) for cid in ids] if "metadatas" in include else None,
            }

        def query(self, query_texts, n_results, where, include):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        def count(self):
            return len(docs)

    class _FakeBM25:
        def search(self, query, top_k):
            return []

    orch = object.__new__(KnowledgeOrchestrator)
    orch.query_cache = QueryCache(max_size=32, ttl_seconds=300)
    orch.bm25_index = _FakeBM25()
    orch.collection = _FakeCollection()
    orch.fts5_index = fts5_index
    orch.query_router = query_router
    orch._ensure_bm25_index = lambda: None
    orch._route_by_keywords = lambda q: None
    orch._expand_with_adjacent_chunks = lambda r: r
    orch._apply_mmr = lambda results, top_k, lambda_param=0.7: results[:top_k]
    return orch


class TestFtsDispatch:
    """IT-001..IT-009 — dispatch decisions in ``KnowledgeOrchestrator.query()``."""

    def test_it001_feature_off_never_dispatches_fts5(self, monkeypatch):
        """IT-001: fts5_enabled=False → hybrid pipeline only, FTS5 untouched."""
        fts5 = _FakeFts5(hits=[("chunk_1", 5.0)], ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_enabled=False, fts5_index=fts5, query_router=router)

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert results == []
        assert fts5.search_calls == [], "FTS5 must never be invoked when the feature is off"
        assert router.classify_calls == [], "router must never run when the feature is off"

    def test_it002_lexical_query_dispatches_fts5(self, monkeypatch):
        """IT-002: enabled + lexical + ready + hit → search_method='fts5' in results."""
        hits = [("chunk_1", 5.0), ("chunk_2", 4.0), ("chunk_3", 3.0)]
        docs = {"chunk_1": "H1-P4-XXX-1234 disclosure summary", "chunk_2": "other", "chunk_3": "more"}
        metas = {
            cid: {
                "source": f"/{cid}",
                "filename": f"{cid}.md",
                "category": "bugbounty",
                "chunk_index": 0,
                "keywords": "",
            }
            for cid in docs
        }
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(
            monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs, fake_metadatas=metas
        )

        results = orch.query("H1-P4-XXX-1234", search_method="auto")

        assert results, "expected at least one FTS5 result"
        assert results[0]["search_method"] == "fts5"
        assert fts5.search_calls, "FTS5 search was not invoked"

    def test_it003_parametrized_lexical_queries_all_dispatch(self, monkeypatch, sample_lexical_queries):
        """IT-003: each canonical lexical query dispatches to FTS5 when hits exist."""
        for q in sample_lexical_queries:
            hits = [(f"chunk_{i}", 10 - i) for i in range(3)]
            docs = {f"chunk_{i}": f"doc mentioning {q}" for i in range(3)}
            metas = {
                f"chunk_{i}": {
                    "source": f"/{i}",
                    "filename": f"{i}.md",
                    "category": "security",
                    "chunk_index": 0,
                    "keywords": "",
                }
                for i in range(3)
            }
            fts5 = _FakeFts5(hits=hits, ready=True)
            router = _FakeRouter("lexical")
            orch = _build_dispatch_orch(
                monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs, fake_metadatas=metas
            )
            results = orch.query(q, search_method="auto")
            assert results, f"no results for canonical lexical query {q!r}"
            assert results[0]["search_method"] == "fts5"

    def test_it004_zero_docs_corpus_returns_empty(self, monkeypatch):
        """IT-004: FTS5 empty + hybrid empty → orch.query returns []."""
        fts5 = _FakeFts5(hits=[], ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        results = orch.query("MDR-AD002", search_method="auto")

        assert results == []

    def test_it005_semantic_query_skips_fts5(self, monkeypatch):
        """IT-005: router says semantic → FTS5 is never consulted."""
        fts5 = _FakeFts5(hits=[("chunk_1", 5.0)], ready=True)
        router = _FakeRouter("semantic")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        results = orch.query("nuclei", search_method="auto")

        assert fts5.search_calls == []
        assert results == []

    def test_it006_low_hits_triggers_fallback_metric(self, monkeypatch):
        """IT-006: lexical + 0 fts5 hits + min_hits=3 → fallback_total{reason=low_hits} +1."""
        from mcp_server.metrics import FAST_PATH_FALLBACK_TOTAL, get_metrics

        fts5 = _FakeFts5(hits=[], ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        needle = f'{FAST_PATH_FALLBACK_TOTAL}{{reason="low_hits"}}'
        before = get_metrics().exposition().count(needle)
        results = orch.query("T-800", search_method="auto")
        after = get_metrics().exposition().count(needle)

        assert results == []
        assert after > before, f"{needle} was not incremented"

    def test_it007_high_min_hits_causes_fallback(self, monkeypatch):
        """IT-007: min_hits=100 + 10 fts5 hits → fallback (result count below threshold)."""
        hits = [(f"chunk_{i}", 10 - i) for i in range(10)]
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_min_hits=100, fts5_index=fts5, query_router=router)

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert results == []

    def test_it008_fts5_error_triggers_fallback_and_error_metric(self, monkeypatch):
        """IT-008: fts5.search raises OperationalError → fallback + errors_total{OperationalError}."""
        import sqlite3

        from mcp_server.metrics import FAST_PATH_ERRORS_TOTAL, get_metrics

        fts5 = _FakeFts5(raises=sqlite3.OperationalError("readonly"))
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        exposition_before = get_metrics().exposition()
        orch.query("CVE-2021-4034", search_method="auto")
        exposition_after = get_metrics().exposition()

        assert 'error_class="OperationalError"' in exposition_after
        assert exposition_after.count(FAST_PATH_ERRORS_TOTAL) > exposition_before.count(FAST_PATH_ERRORS_TOTAL)

    def test_it009_fts5_databaseerror_falls_back(self, monkeypatch):
        """IT-009: fts5.search raises DatabaseError → fallback + error metric labeled correctly."""
        import sqlite3

        from mcp_server.metrics import get_metrics

        fts5 = _FakeFts5(raises=sqlite3.DatabaseError("file missing"))
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        orch.query("CVE-2021-4034", search_method="auto")
        exposition = get_metrics().exposition()

        assert 'error_class="DatabaseError"' in exposition


class TestConfigToggle:
    """IT-015..IT-018 — config-driven feature toggle + custom pattern classification."""

    def test_it015_fresh_install_default_off(self, monkeypatch):
        """IT-015: fresh YAML has no search section → fts5_enabled defaults False."""
        from mcp_server import config as config_module

        monkeypatch.setattr(config_module, "_yaml", {})
        cfg = config_module.Config()
        assert cfg.fts5_enabled is False

    def test_it016_legacy_config_unchanged(self, monkeypatch):
        """IT-016: v4.8.1 config sans lexical_fast_path → fts5_enabled stays False."""
        from mcp_server import config as config_module

        monkeypatch.setattr(config_module, "_yaml", {"search": {"hybrid_alpha": 0.3}})
        cfg = config_module.Config()
        assert cfg.fts5_enabled is False

    def test_it017_yaml_enables_fts5(self, monkeypatch):
        """IT-017: enabled: true in YAML flips fts5_enabled."""
        from mcp_server import config as config_module

        monkeypatch.setattr(config_module, "_yaml", {"search": {"lexical_fast_path": {"enabled": True}}})
        cfg = config_module.Config()
        assert cfg.fts5_enabled is True

    def test_it018_custom_pattern_classifies_query(self):
        """IT-018: custom PROJ pattern classifies matching query as lexical."""
        from mcp_server.query_router import QueryRouter

        router = QueryRouter([r"PROJ-\d{3,5}"])
        assert router.classify("PROJ-12345") == "lexical"
        assert router.classify("random prose") == "semantic"


class TestMetricsScrape:
    """IT-019 — exposition after mixed lexical + semantic query traffic."""

    def test_it019_mixed_queries_produce_expected_counters(self, monkeypatch):
        from mcp_server.metrics import FAST_PATH_HITS_TOTAL, get_metrics

        hits = [(f"chunk_{i}", 10 - i) for i in range(5)]
        docs = {f"chunk_{i}": f"CVE-2021-4034 content {i}" for i in range(5)}
        metas = {
            f"chunk_{i}": {
                "source": f"/{i}",
                "filename": f"{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(5)
        }
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter(lambda q: "lexical" if "CVE" in q else "semantic")
        orch = _build_dispatch_orch(
            monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs, fake_metadatas=metas
        )

        for _ in range(3):
            orch.query("CVE-2021-4034", search_method="auto")
            orch.query_cache.invalidate()
        for _ in range(2):
            orch.query("how does OAuth token refresh work", search_method="auto")
            orch.query_cache.invalidate()

        exposition = get_metrics().exposition()
        assert FAST_PATH_HITS_TOTAL in exposition
        assert 'path="fts5"' in exposition
        assert 'path="hybrid"' in exposition


class TestQueryCacheKey:
    """UT-059, UT-060 — 5th-param backward compat + collision-free keys per path."""

    def test_ut059_make_key_default_matches_explicit_auto(self):
        cache = QueryCache()
        without_arg = cache._make_key("q", 5, None, 0.3)
        with_auto = cache._make_key("q", 5, None, 0.3, "auto")
        assert without_arg == with_auto

    def test_ut060_search_method_variants_produce_distinct_keys(self):
        cache = QueryCache()
        keys = {cache._make_key("q", 5, None, 0.3, sm) for sm in ("auto", "hybrid", "fts5")}
        assert len(keys) == 3
