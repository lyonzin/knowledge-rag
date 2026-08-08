"""Static regex-based router that classifies queries as lexical or semantic.

Implements ADR-002: heuristic regex classifier (no LLM, no ML). Patterns are
pre-compiled at construction time so ``classify()`` stays microsecond-fast and
never becomes a bottleneck on the FTS5 fast-path.

Phase 2 of the FTS5 lexical fast-path feature. This module is intentionally
isolated: nothing under ``mcp_server/`` imports it yet. Task 03 wires the
router into ``KnowledgeOrchestrator`` behind ``config.fts5_enabled``.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence

QueryClass = Literal["lexical", "semantic"]


class QueryRouter:
    """Classify a query as ``lexical`` (dispatch FTS5) or ``semantic`` (hybrid).

    Patterns are matched in YAML-declared order — first match wins (OQ-2 of
    the PRD). Empty queries always classify as ``semantic`` (safe default: the
    hybrid path already handles trivial input gracefully).
    """

    def __init__(self, patterns: Sequence[str]) -> None:
        """Compile every pattern eagerly.

        A malformed pattern raises ``re.error`` at construction so a broken
        config fails fast at daemon startup rather than at the first query.
        """
        self._patterns: list[re.Pattern[str]] = [re.compile(p) for p in patterns]

    def classify(self, query: str) -> QueryClass:
        """Return ``lexical`` if any pattern matches, else ``semantic``.

        First-match-wins: iteration stops at the first pattern that hits, so
        ordering of ``search.lexical_fast_path.patterns`` in YAML determines
        precedence. Uses ``re.Pattern.search`` (unanchored) so identifiers can
        appear anywhere in the query text.
        """
        if not query:
            return "semantic"
        for pattern in self._patterns:
            if pattern.search(query):
                return "lexical"
        return "semantic"
