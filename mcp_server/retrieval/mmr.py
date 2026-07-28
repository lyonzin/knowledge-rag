"""
╭─╴ MMR RE-RANKER ╶──────────────────────────────────────────────╮
│                                                                │
│   Maximal Marginal Relevance with cosine similarity of         │
│   embeddings. Diversifies top-k candidates without dropping    │
│   the reranker's #1 pick.                                      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Classical MMR (Carbonell & Goldstein, 1998):

    MMR(D_i) = lambda * sim(D_i, Q) - (1 - lambda) * max_j(sim(D_i, D_j))

where ``sim`` is cosine similarity and ``D_j`` iterates already-selected
documents. Lambda controls the relevance/diversity balance:

- ``lambda = 1.0`` = pure relevance (order by cosine to query).
- ``lambda = 0.0`` = pure diversity (spread the picks apart).
- ``lambda = 0.7`` = relevance-biased with mild diversity (default).

The first pick is always ``candidate_ids[0]`` so the reranker's top choice
is preserved even when its neighbourhood is dense; subsequent picks apply
the full MMR formula. The output preserves each candidate's original score
(the score fed in by the caller — typically reranker or RRF), so downstream
consumers keep the trained signal for display and scoring.

Embeddings are expected pre-normalized (FastEmbed / BGE emit unit vectors),
but the function normalizes defensively so a caller passing raw vectors
still gets correct cosine values.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

Vector = Union[Sequence[float], np.ndarray]


def apply_mmr(
    query_embedding: Vector,
    candidate_ids: Sequence[str],
    candidate_embeddings: Union[Sequence[Vector], np.ndarray],
    candidate_scores: Sequence[float],
    top_k: int,
    lambda_param: float = 0.7,
) -> List[Tuple[str, float]]:
    """Rerank candidates by Maximal Marginal Relevance.

    The relevance term is cosine similarity between each candidate and the
    query; the diversity term is the maximum cosine similarity between the
    candidate and any already-selected candidate. Lambda blends the two.

    The first pick is ``candidate_ids[0]`` (the caller-supplied top rank).
    This keeps the reranker's #1 choice visible even when the surrounding
    cluster is dense — MMR then diversifies from position two onward.

    Args:
        query_embedding: Embedding of the query. Any 1-D sequence or ndarray.
        candidate_ids: Ordered chunk identifiers. Position 0 is treated as
            the caller's top rank.
        candidate_embeddings: One embedding per candidate, aligned with
            ``candidate_ids``. Accepts ``list[list[float]]``, ``list[ndarray]``,
            or a 2-D ndarray.
        candidate_scores: Original relevance scores (reranker, RRF, etc.),
            aligned with ``candidate_ids``. Preserved verbatim in the output —
            never fed to the MMR formula.
        top_k: Number of items to select. Capped at ``len(candidate_ids)``.
        lambda_param: Relevance/diversity balance in ``[0.0, 1.0]``.
            ``1.0`` = pure relevance, ``0.0`` = pure diversity.

    Returns:
        list[tuple[str, float]]: Selected candidates in MMR order, each
        paired with its original score. Length is
        ``min(top_k, len(candidate_ids))``.

    Raises:
        ValueError: When input shapes are inconsistent (e.g. embedding count
            does not match id count) or when ``lambda_param`` is out of range.

    Example:
        >>> import numpy as np
        >>> q = [1.0, 0.0, 0.0]
        >>> ids = ["a", "b", "c"]
        >>> embs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
        >>> apply_mmr(q, ids, embs, [1.0, 0.9, 0.5], top_k=2, lambda_param=0.5)
        [('a', 1.0), ('c', 0.5)]
    """
    if not candidate_ids or top_k <= 0:
        return []

    n = len(candidate_ids)
    if len(candidate_scores) != n:
        raise ValueError(f"candidate_scores has {len(candidate_scores)} entries, expected {n}")

    top_k = min(int(top_k), n)
    if not 0.0 <= float(lambda_param) <= 1.0:
        raise ValueError(f"lambda_param must be in [0.0, 1.0], got {lambda_param}")

    # Trivial fast path — nothing to diversify.
    if n == 1 or top_k == 1:
        return [(candidate_ids[0], float(candidate_scores[0]))]

    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate_embeddings, dtype=np.float32)
    if cand.ndim != 2 or cand.shape[0] != n:
        raise ValueError(f"candidate_embeddings must be shape (n, dim); got {cand.shape} for n={n}")
    if cand.shape[1] != q.shape[0]:
        raise ValueError(f"candidate_embedding dim does not match query embedding: {cand.shape[1]} vs {q.shape[0]}")

    # Defensive normalization — FastEmbed / BGE already return unit vectors,
    # but a caller passing raw vectors still needs correct cosine values.
    q_norm = float(np.linalg.norm(q))
    if q_norm > 0.0:
        q = q / q_norm

    row_norms = np.linalg.norm(cand, axis=1, keepdims=True)
    # Avoid divide-by-zero for degenerate all-zero vectors — they stay all-zero.
    safe_norms = np.where(row_norms > 0.0, row_norms, 1.0)
    cand = cand / safe_norms

    # Cosine similarity of every candidate to the query (shape: n).
    sim_to_query = cand @ q

    # First pick: always candidate 0 (the caller's top rank).
    selected_indices: List[int] = [0]
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[0] = False

    # Running max cosine similarity from each candidate to any selected one.
    max_sim_to_selected = cand @ cand[0]

    lam = float(lambda_param)
    one_minus_lam = 1.0 - lam

    while len(selected_indices) < top_k and remaining_mask.any():
        # Vectorized MMR score for every remaining candidate.
        mmr_scores = lam * sim_to_query - one_minus_lam * max_sim_to_selected
        # Mask picked / missing candidates so argmax skips them.
        mmr_scores = np.where(remaining_mask, mmr_scores, -np.inf)
        next_idx = int(np.argmax(mmr_scores))
        selected_indices.append(next_idx)
        remaining_mask[next_idx] = False
        # Update running max: element-wise max between old max and sim to new pick.
        max_sim_to_selected = np.maximum(max_sim_to_selected, cand @ cand[next_idx])

    return [(candidate_ids[i], float(candidate_scores[i])) for i in selected_indices]
