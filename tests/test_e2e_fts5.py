"""E2E-001..E2E-008 (minus E2E-006 — deferred to Task 05 with migration flow).

Exercises the ``search_knowledge`` MCP tool wrapper via the in-process
``mcp_client_test`` fixture. Assertions target the JSON envelope structure
the real MCP client would observe — status, error message shape, and the
``search_method`` field of every result item.
"""

from __future__ import annotations


def _install_dispatch_orch(monkeypatch, fts5, router, docs=None, metas=None):
    import mcp_server.server as srv
    from tests.test_search import _build_dispatch_orch

    orch = _build_dispatch_orch(
        monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs or {}, fake_metadatas=metas or {}
    )
    monkeypatch.setattr(srv, "get_orchestrator", lambda: orch)
    return orch


def _sample_data(prefix):
    docs = {f"chunk_{i}": f"{prefix} match #{i}" for i in range(5)}
    metas = {
        f"chunk_{i}": {
            "source": f"/{prefix}/{i}",
            "filename": f"{i}.md",
            "category": "security",
            "chunk_index": 0,
            "keywords": "",
        }
        for i in range(5)
    }
    return docs, metas


class TestE2EFts5:
    def test_e2e001_bug_bounty_happy_path_sequence(self, monkeypatch, mcp_client_test):
        """E2E-001: H1 → CWE-79 → dalfox produces ['fts5','fts5','hybrid'] in order."""
        from tests.test_search import _FakeFts5, _FakeRouter

        hits = [(f"chunk_{i}", 10 - i) for i in range(5)]
        fts5 = _FakeFts5(hits=hits, ready=True)
        # Router classifies lexical when query contains uppercase-dash-digits,
        # otherwise semantic.
        router = _FakeRouter(lambda q: "lexical" if any(c.isupper() and "-" in q for c in q) else "semantic")
        docs, metas = _sample_data("bug")
        _install_dispatch_orch(monkeypatch, fts5, router, docs, metas)

        methods = []
        for q in ("H1-P4-XXX-1234", "CWE-79", "dalfox"):
            payload = mcp_client_test.call("search_knowledge", query=q)
            if payload.get("status") == "success" and payload.get("results"):
                methods.append(payload["results"][0]["search_method"])
            else:
                methods.append("no_results")

        # First two are lexical → FTS5. dalfox is prose → hybrid (empty pipeline
        # → no_results in the fake environment). Assertion is on the DECISION.
        assert methods[0] == "fts5"
        assert methods[1] == "fts5"
        assert methods[2] in ("no_results",)  # semantic path yields nothing here

    def test_e2e002_ambiguous_tool_name_goes_hybrid(self, monkeypatch, mcp_client_test):
        """E2E-002: query 'nuclei' → router says semantic → hybrid path."""
        from tests.test_search import _FakeFts5, _FakeRouter

        fts5 = _FakeFts5(hits=[("chunk_1", 1.0)], ready=True)
        router = _FakeRouter("semantic")
        _install_dispatch_orch(monkeypatch, fts5, router)

        payload = mcp_client_test.call("search_knowledge", query="nuclei")

        assert fts5.search_calls == []
        assert payload.get("status") in ("no_results", "success")

    def test_e2e003_soc_cve_search_lands_on_fts5(self, monkeypatch, mcp_client_test):
        """E2E-003: CVE-2021-4034 exact match → search_method='fts5' + top-1 matches."""
        import mcp_server.server as srv
        from tests.test_search import _FakeFts5, _FakeRouter

        hits = [("chunk_0", 10.0)]
        docs = {"chunk_0": "CVE-2021-4034 Pwnkit local privilege escalation"}
        metas = {
            "chunk_0": {
                "source": "/pwnkit.md",
                "filename": "pwnkit.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
        }
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter("lexical")
        _install_dispatch_orch(monkeypatch, fts5, router, docs, metas)
        # Single seeded chunk — lower min_hits so the fast-path is not
        # short-circuited by the default threshold of 3.
        monkeypatch.setattr(srv.config, "fts5_min_hits", 1)

        payload = mcp_client_test.call("search_knowledge", query="CVE-2021-4034")

        assert payload["status"] == "success"
        assert payload["results"][0]["search_method"] == "fts5"
        assert "CVE-2021-4034" in payload["results"][0]["content"]

    def test_e2e004_fallback_low_hits(self, monkeypatch, mcp_client_test):
        """E2E-004: min_hits=5 + only 2 fts5 hits → fallback → hybrid label."""
        import mcp_server.server as srv
        from mcp_server.metrics import FAST_PATH_FALLBACK_TOTAL
        from tests.conftest import _get_metric_value
        from tests.test_search import _FakeFts5, _FakeRouter

        fts5 = _FakeFts5(hits=[("chunk_0", 5.0), ("chunk_1", 4.0)], ready=True)
        router = _FakeRouter("lexical")
        _install_dispatch_orch(monkeypatch, fts5, router)
        monkeypatch.setattr(srv.config, "fts5_min_hits", 5)

        needle = f'{FAST_PATH_FALLBACK_TOTAL}{{reason="low_hits"}}'
        before = _get_metric_value(needle)
        mcp_client_test.call("search_knowledge", query="MDR-AD999")
        after = _get_metric_value(needle)

        assert after > before, f"{needle} was not incremented"

    def test_e2e005_kill_switch_hybrid_override_bypasses_fts5(self, monkeypatch, mcp_client_test):
        """E2E-005: search_method='hybrid' override forces hybrid even for lexical query."""
        from tests.test_search import _FakeFts5, _FakeRouter

        fts5 = _FakeFts5(hits=[("chunk_1", 5.0)], ready=True)
        router = _FakeRouter("lexical")
        _install_dispatch_orch(monkeypatch, fts5, router)

        mcp_client_test.call("search_knowledge", query="CVE-2021-4034", search_method="hybrid")

        assert fts5.search_calls == [], "hybrid override must not consult FTS5"
        assert router.classify_calls == [], "hybrid override must skip the router"

    def test_e2e007_zero_impact_when_feature_off(self, monkeypatch, mcp_client_test):
        """E2E-007: fresh install (feature OFF) — MCP tool returns valid JSON, no FTS5 traffic."""
        import mcp_server.server as srv
        from tests.test_search import _FakeFts5, _FakeRouter

        fts5 = _FakeFts5(hits=[("chunk_1", 5.0)], ready=True)
        router = _FakeRouter("lexical")
        _install_dispatch_orch(monkeypatch, fts5, router)
        monkeypatch.setattr(srv.config, "fts5_enabled", False)

        payload = mcp_client_test.call("search_knowledge", query="CVE-2021-4034")

        assert payload.get("status") in ("no_results", "success")
        assert fts5.search_calls == []

    def test_e2e008_legacy_mcp_client_calls_never_break(self, monkeypatch, mcp_client_test):
        """E2E-008: 20 queries WITHOUT ``search_method`` all return well-formed JSON."""
        from tests.test_search import _FakeFts5, _FakeRouter

        fts5 = _FakeFts5(hits=[], ready=True)
        router = _FakeRouter("semantic")
        _install_dispatch_orch(monkeypatch, fts5, router)

        for i in range(20):
            payload = mcp_client_test.call("search_knowledge", query=f"query {i}")
            assert isinstance(payload, dict)
            assert "status" in payload
