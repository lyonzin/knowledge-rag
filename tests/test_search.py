"""Tests for search pipeline components (no model/DB required)."""

from mcp_server.config import _merge_query_expansion_sources
from mcp_server.server import BM25Index, QueryCache

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
