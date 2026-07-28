"""
╭─╴ SHARED TOOL IMPLEMENTATIONS ╶────────────────────────────────╮
│                                                                │
│   Plain-Python implementations for MCP tools that both the     │
│   ``@mcp.tool`` wrappers AND the standalone CLI dispatch call. │
│                                                                │
│   Extracted verbatim from server.py in the A2.1 refactor.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

The ``*_impl`` helpers hold the actual logic and return plain dicts. The
``@mcp.tool()`` wrappers are thin JSON serializers on top of them, and
``mcp_server.cli`` calls the same helpers directly so the CLI ``--json``
output has byte-identical shape to the MCP tool responses without paying
the MCP protocol / rate-limit / metrics overhead.

Late-bind pattern:
    ``get_orchestrator`` is resolved through ``mcp_server.server`` at call
    time so tests that patch ``mcp_server.server.get_orchestrator`` (e.g.
    ``tests/test_tools.py``, ``tests/test_truncation.py``) keep working.
"""

from typing import Any, Dict, Optional

from ..config import config
from ..stopwords import filter_query_stopwords
from .truncation import _make_snippet, _truncate_results_safe, _truncation_note


def _get_orchestrator():
    """Late-bind ``get_orchestrator`` from the server module.

    Tests patch ``mcp_server.server.get_orchestrator``; late lookup keeps
    those patches effective after the A2.1 module split.
    """
    from mcp_server import server as _srv

    return _srv.get_orchestrator()


def _search_impl(
    query: str,
    max_results: int = 5,
    category: Optional[str] = None,
    hybrid_alpha: float = 0.3,
    min_score: float = 0.0,
    snippet_mode: bool = True,
    fusion: Optional[str] = None,
    query_rewrite: Optional[bool] = None,
    self_query: Optional[bool] = None,
    hyde: Optional[int] = None,
    multi_query: Optional[int] = None,
    adaptive: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run a hybrid search and return the structured payload shared by MCP and CLI.

    Validates and clamps every argument before delegating to the orchestrator,
    so callers may pass raw user input.

    Args:
        query: Search query text
        max_results: Maximum results to return (clamped to ``config.max_results``)
        category: Optional category filter (validated against the known routes)
        hybrid_alpha: Semantic/keyword balance, clamped to 0.0–1.0
        min_score: Minimum normalized relevance score, clamped to 0.0–1.0
        snippet_mode: Truncate content to ~500 chars and add ``content_length``
        fusion: Optional per-request fusion strategy override (``rrf`` /
            ``combsum`` / ``combmnz`` / ``weighted``). ``None`` (default)
            falls back to ``config.fusion_strategy``.
        query_rewrite: Optional per-call override for the A3.2 LLM query
            rewriting feature. ``True`` forces the rewrite, ``False`` skips
            it for this call, and ``None`` (default) inherits from
            ``config.query_rewrite_enabled``. Failure to reach the LLM is
            fail-open — retrieval never breaks.
        self_query: Optional per-call override for the A3.5 LLM self-query
            filter-extraction feature. ``True`` forces LLM-based extraction
            of category + source + date filters from the natural-language
            query, ``False`` skips it, and ``None`` (default) inherits from
            ``config.self_query_enabled``. Only fires when ``category`` was
            NOT passed explicitly — user intent always wins. Fail-open.
        hyde: Optional per-request override for the A3.3 HyDE feature.
            Positive int forces HyDE with that many hypothetical
            passages averaged into the query embedding; ``0`` disables
            HyDE for this call; ``None`` (default) inherits from
            ``config.hyde_enabled`` + ``config.hyde_num_hypotheses``.
            Clamped to ``[0, 10]`` to cap per-query LLM cost. Failure to
            reach the LLM or embed the passages is fail-open — the
            semantic branch reverts to the raw query embedding and
            retrieval continues.
        multi_query: Optional per-request override for the A3.4 multi-query
            fan-out. Integer ``N > 1`` forces N-way retrieval (original +
            LLM-generated variations) with top-level RRF fusion. ``0`` /
            ``1`` disables fan-out for this call. ``None`` (default)
            inherits from ``config.multi_query_enabled`` +
            ``config.multi_query_n``. Clamped to ``[0, 10]`` to cap
            per-query LLM + retrieval cost. Fail-open on any LLM error —
            single-query retrieval kept.
        adaptive: Optional per-request override for the A3.9 LLM adaptive
            retrieval router. ``True`` asks the LLM to pick a strategy
            (``simple`` / ``hybrid`` / ``multi_hop`` / ``code`` /
            ``filter``) and fill in defaults for whichever of
            ``hybrid_alpha`` / ``self_query`` / ``hyde`` / ``multi_query``
            the caller left unset. ``False`` disables the router for this
            call regardless of config. ``None`` (default) inherits from
            ``config.adaptive_retrieval_enabled``. User-explicit values
            for the router-controlled params ALWAYS win. Fail-open to
            the ``hybrid`` strategy (no-op) on any LLM error.

    Returns:
        dict: ``status`` is one of ``success`` / ``no_results`` / ``error``.
        On success the payload also carries ``query``, ``hybrid_alpha``,
        ``fusion``, ``hyde``, ``result_count``, ``filtered_by_score``,
        ``cache_hit_rate`` and ``results``.
    """
    if not query or not query.strip():
        return {"status": "error", "message": "Query cannot be empty"}

    max_results = max(1, min(max_results or 5, config.max_results))
    hybrid_alpha = max(0.0, min(hybrid_alpha if hybrid_alpha is not None else 0.3, 1.0))
    min_score = max(0.0, min(min_score if min_score is not None else 0.0, 1.0))

    valid_categories = list(config.keyword_routes.keys()) + list(set(config.category_mappings.values())) + ["general"]
    if category and category not in valid_categories:
        return {"status": "error", "message": f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}"}

    # A2.4 — validate fusion name up-front so a typo surfaces before we do any
    # retrieval work. Import is late so tests that patch fusion internals still
    # take effect.
    if fusion is not None:
        from .fusion import available_strategies

        if fusion not in available_strategies():
            return {
                "status": "error",
                "message": f"Invalid fusion '{fusion}'. Valid: {', '.join(available_strategies())}",
            }

    # A3.3 — clamp HyDE hypothesis count. Non-integers, negatives and
    # values above 10 are silently coerced so a stray CLI/tool arg
    # cannot rack up an LLM bill. ``None`` passes through untouched so
    # the orchestrator inherits from config.
    hyde_effective: Optional[int]
    if hyde is None:
        hyde_effective = None
    else:
        try:
            hyde_int = int(hyde)
        except (TypeError, ValueError):
            hyde_int = 0
        if hyde_int < 0:
            hyde_int = 0
        elif hyde_int > 10:
            hyde_int = 10
        hyde_effective = hyde_int

    # A3.4 — clamp multi-query fan-out N. Same coercion contract as
    # ``hyde``: bad types collapse to 0 (disabled), negatives to 0,
    # values above 10 to 10. ``None`` passes through so the orchestrator
    # inherits from config.
    multi_query_effective: Optional[int]
    if multi_query is None:
        multi_query_effective = None
    else:
        try:
            mq_int = int(multi_query)
        except (TypeError, ValueError):
            mq_int = 0
        if mq_int < 0:
            mq_int = 0
        elif mq_int > 10:
            mq_int = 10
        multi_query_effective = mq_int

    orchestrator = _get_orchestrator()
    results = orchestrator.query(
        query.strip(),
        max_results=max_results,
        category_filter=category,
        hybrid_alpha=hybrid_alpha,
        fusion=fusion,
        query_rewrite=query_rewrite,
        self_query=self_query,
        hyde=hyde_effective,
        multi_query=multi_query_effective,
        adaptive=adaptive,
    )

    if not results:
        return {"status": "no_results", "query": query, "message": "No relevant documents found."}

    total_before_filter = len(results)
    if min_score > 0.0:
        results = [r for r in results if r.get("score", 0) >= min_score]

    if snippet_mode:
        for r in results:
            full_len = len(r.get("content", ""))
            r["content"] = _make_snippet(r["content"])
            r["content_length"] = full_len

    matched = len(results)
    results, was_truncated, _ = _truncate_results_safe(results)

    # A3.3 — surface the effective HyDE ``n_hypos`` for observability.
    # Resolves the same way as the orchestrator: explicit per-request
    # override wins, else config default, else 0 (feature off).
    if hyde_effective is not None:
        hyde_reported = hyde_effective
    elif config.hyde_enabled:
        hyde_reported = int(config.hyde_num_hypotheses)
    else:
        hyde_reported = 0

    # A3.4 — surface the effective multi-query fan-out N. Same
    # resolution as HyDE. NOTE: a non-1 value here does NOT guarantee
    # the fan-out actually ran — the feature is fail-open and may have
    # collapsed to single-query (no LLM provider, LLM error, ...).
    if multi_query_effective is not None:
        multi_query_reported = multi_query_effective
    elif config.multi_query_enabled:
        multi_query_reported = int(config.multi_query_n)
    else:
        multi_query_reported = 1

    # A3.9 — surface the effective adaptive-router toggle. Explicit per
    # request override wins, else config default, else False. NOTE: a
    # ``True`` here does NOT guarantee a strategy was actually chosen —
    # the router fails open to ``"hybrid"`` (a no-op) when no LLM is
    # configured or the LLM misfired. Consumers looking to confirm
    # routing actually happened should check the ``query.adaptive_strategy``
    # OpenTelemetry attribute instead.
    if adaptive is not None:
        adaptive_reported = bool(adaptive)
    else:
        adaptive_reported = bool(getattr(config, "adaptive_retrieval_enabled", False))

    payload: Dict[str, Any] = {
        "status": "success",
        "query": query,
        # Query actually fed to BM25 after stopword removal. Surfaced so a
        # surprising ranking can be debugged without re-running the pipeline.
        "bm25_query": filter_query_stopwords(query.strip(), config.stopword_languages),
        "hybrid_alpha": hybrid_alpha,
        # A2.4 — surface the effective fusion strategy so the response is
        # self-describing. Falls back to the config default when the caller
        # did not supply a per-request override.
        "fusion": fusion or config.fusion_strategy,
        # A3.3 — number of hypothetical passages HyDE was asked to
        # generate. ``0`` means HyDE was disabled for this call.
        # NOTE: a non-zero value here does NOT guarantee HyDE actually
        # ran — the feature is fail-open and may have reverted to the
        # raw embedding path (missing LLM provider, network error, ...).
        "hyde": hyde_reported,
        # A3.4 — total number of query variations the multi-query fan-out
        # was asked to run (original + LLM paraphrases). ``1`` means
        # fan-out was disabled for this call. Same fail-open caveat as
        # ``hyde``: a value > 1 here does NOT guarantee the fan-out
        # actually ran — the LLM may have failed and single-query
        # retrieval kicked in silently.
        "multi_query": multi_query_reported,
        # A3.9 — whether adaptive routing was requested for this call.
        # ``True`` means the router had a chance to influence params;
        # it does NOT guarantee the router actually picked a non-hybrid
        # strategy or that the LLM succeeded. See the OTel span
        # ``query.adaptive_strategy`` attribute for the actual outcome.
        "adaptive": adaptive_reported,
        "result_count": len(results),
        "filtered_by_score": total_before_filter - matched,
        "cache_hit_rate": orchestrator.query_cache.stats()["hit_rate"],
        "truncated": was_truncated,
        "results": results,
    }
    if was_truncated:
        payload["shown"] = len(results)
        payload["total_matched"] = matched
        payload["note"] = _truncation_note(len(results), matched)
    return payload


def _stats_impl() -> Dict[str, Any]:
    """
    Collect index statistics shared by MCP ``get_index_stats`` and CLI ``stats``.

    Returns:
        dict: ``{"status": "success", "stats": {...}}`` with document/chunk counts,
        embedding + reranker model names, chunking parameters, query cache stats
        and background reindex progress.
    """
    return {"status": "success", "stats": _get_orchestrator().get_stats()}


def _list_impl(category: Optional[str] = None) -> Dict[str, Any]:
    """
    List indexed documents, shared by MCP ``list_documents`` and CLI ``list``.

    Args:
        category: Optional category filter; ``None`` lists every indexed document

    Returns:
        dict: ``{"status": "success", "filter": ..., "count": ..., "documents": [...]}``
    """
    docs = _get_orchestrator().list_documents(category=category)
    return {"status": "success", "filter": category or "all", "count": len(docs), "documents": docs}
