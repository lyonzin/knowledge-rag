"""Preserve both retrieval channels within a bounded cross-encoder budget."""

from __future__ import annotations

from typing import Any

Candidate = tuple[str, dict[str, Any]]


def select_rerank_candidates(candidates: list[Candidate], limit: int, results: int) -> list[Candidate]:
    """Reserve at most one result-page per channel, then fill by fused score.

    Weighted RRF can rank every lexical candidate above every semantic-only
    candidate (for example alpha=.3, k=60 and 30 candidates). Truncating that
    union before reranking silently removes one retrieval channel. Reserving
    a bounded head from each channel lets the cross-encoder compare them while
    preserving weighted fusion for the remaining budget. Missing channels and
    single-channel searches retain their original ordering.
    """
    ordered = sorted(candidates, key=lambda row: (-row[1]["rrf_score"], row[0]))
    if len(ordered) <= limit:
        return ordered
    quota = min(results, limit // 2)
    selected: set[str] = set()
    for channel in ("semantic_rank", "bm25_rank"):
        ranked = [row for row in ordered if row[1].get(channel) is not None]
        ranked.sort(key=lambda row: (row[1][channel], row[0]))
        selected.update(identity for identity, _ in ranked[:quota])
    for identity, _ in ordered:
        if len(selected) >= limit:
            break
        selected.add(identity)
    return [row for row in ordered if row[0] in selected]
