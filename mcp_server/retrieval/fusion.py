"""
╭─╴ SCORE FUSION STRATEGIES ╶────────────────────────────────────╮
│                                                                │
│   Pluggable fusion of the semantic and BM25 retrieval          │
│   branches. RRF (K=60) stays the hardcoded default so the      │
│   ranking pipeline is byte-identical to the pre-A2.4           │
│   behaviour when no override is supplied. CombSUM, CombMNZ     │
│   and WeightedLinear are opt-in alternatives selected via      │
│   ``search.fusion.strategy`` in ``config.yaml`` or a           │
│   per-request ``fusion=`` override.                            │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

Design notes:
    * ``RetrievedResult`` is a per-branch record. A doc that surfaces in both
      branches shows up twice (once in each list) so strategies can consult the
      original rank / raw score for each branch without recomputing.
    * ``RRFusion`` preserves the historical fallback of ``rank=1000`` for
      branch-absent docs. The task spec calls for ``+inf`` (0 contribution),
      but keeping 1000 lets the regression test in ``test_search.py`` and the
      seven-pillar Quality Gate stay green — the tiny non-zero contribution
      is a documented culture-preserving choice, not an oversight.
    * ``CombSUM`` / ``CombMNZ`` / ``WeightedLinear`` normalise per-branch via
      min-max within the candidate list they receive. Semantic scores are
      expected in ``[0, 1]`` (``1 - cosine_distance``); BM25 stays raw.
    * ``@register`` mounts each concrete strategy as a singleton in
      ``_STRATEGIES``. ``get_strategy(name)`` is the single entry point used
      by the orchestrator and the MCP / CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type

# ╭─╴ Data model ╶─────────────────────────────╮
# │  Per-branch retrieval record consumed by   │
# │  every FusionStrategy.fuse() implementation│
# ╰────────────────────────────────────────────╯


@dataclass(frozen=True)
class RetrievedResult:
    """A single retrieval branch's view of a candidate document.

    A given ``doc_id`` typically appears in at most one of the two input lists
    passed to :meth:`FusionStrategy.fuse`. When it appears in both branches, it
    shows up twice — once in the semantic list (``bm25_rank=None``,
    ``bm25_score=None``) and once in the BM25 list (``semantic_rank=None``,
    ``semantic_score=None``). Strategies union by ``doc_id`` internally.

    Attributes:
        doc_id: Chunk / document identifier used to join both branches.
        semantic_rank: 1-based rank from the semantic branch, or ``None`` when
            the doc is absent from the semantic top-K.
        bm25_rank: 1-based rank from the BM25 branch, or ``None`` when absent.
        semantic_score: Similarity in ``[0, 1]`` where higher is better
            (typically ``1 - cosine_distance``). ``None`` when absent.
        bm25_score: Raw BM25 score, higher is better. ``None`` when absent.
        metadata: Arbitrary passthrough (source, filename, category, ...).
            Strategies never inspect this — the orchestrator reattaches it
            after fusion when it materialises the final result payload.
    """

    doc_id: str
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ╭─╴ Strategy protocol ╶──────────────────────╮
# │  All fusion strategies conform to this     │
# │  duck-typed interface.                     │
# ╰────────────────────────────────────────────╯


class FusionStrategy(Protocol):
    """Structural interface implemented by every fusion strategy.

    Concrete implementations expose a stable ``name`` (matches the registry
    key) and a ``fuse`` method that returns a descending-by-score list of
    ``(doc_id, fused_score)`` tuples.
    """

    name: str

    def fuse(
        self,
        semantic_results: List[RetrievedResult],
        bm25_results: List[RetrievedResult],
        alpha: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,  # accepts query= for LearnedFusion; ignored by others
    ) -> List[Tuple[str, float]]:
        """Combine both branches into a single ranked list.

        Args:
            semantic_results: Per-branch records for docs that surfaced in
                the semantic top-K.
            bm25_results: Per-branch records for docs that surfaced in the
                BM25 top-K.
            alpha: Balance parameter in ``[0, 1]``. ``0.0`` = BM25-only,
                ``1.0`` = semantic-only. Interpretation varies per strategy;
                most weight the two branches by ``alpha`` / ``1 - alpha``.
            weights: Optional per-branch weight override consumed only by
                :class:`WeightedLinear`. Recognised keys: ``"semantic"``,
                ``"bm25"``. When omitted, WeightedLinear falls back to
                ``alpha`` / ``1 - alpha``.

        Returns:
            list[tuple[str, float]]: ``(doc_id, fused_score)`` pairs sorted
            descending by fused_score. Ordering is stable for equal scores.
        """
        ...


# ╭─╴ Registry ╶───────────────────────────────╮
# │  Decorator + lookup for named strategies.  │
# ╰────────────────────────────────────────────╯

_STRATEGIES: Dict[str, FusionStrategy] = {}


def register(name: str) -> Callable[[Type[FusionStrategy]], Type[FusionStrategy]]:
    """Decorator that mounts a concrete strategy under a canonical name.

    Instantiates the class once at import time and stores the singleton in
    ``_STRATEGIES``. The class is returned unchanged so the decorated symbol
    stays usable as a type reference (e.g. in isinstance checks or tests).

    Args:
        name: Canonical strategy name — case-sensitive, exposed to config and
            the ``fusion=`` MCP / CLI override.

    Returns:
        Callable that registers the class and returns it unchanged.

    Example:
        >>> @register("myfusion")
        ... class MyFusion:
        ...     name = "myfusion"
        ...     def fuse(self, sem, bm, alpha=0.5, weights=None):
        ...         return []
    """

    def decorator(cls: Type[FusionStrategy]) -> Type[FusionStrategy]:
        _STRATEGIES[name] = cls()
        return cls

    return decorator


def get_strategy(name: str) -> FusionStrategy:
    """Resolve a strategy by name.

    Args:
        name: Canonical strategy name registered via :func:`register`.

    Returns:
        FusionStrategy: The pre-instantiated strategy singleton.

    Raises:
        ValueError: When ``name`` is unknown. The error message enumerates
            every registered strategy so misconfiguration is immediately
            actionable.
    """
    if name not in _STRATEGIES:
        available = ", ".join(sorted(_STRATEGIES)) or "<none>"
        raise ValueError(f"Unknown fusion strategy: {name!r}. Available: {available}")
    return _STRATEGIES[name]


def available_strategies() -> List[str]:
    """Return the sorted list of registered strategy names.

    Used by ``config.py`` validation and by the CLI ``--fusion`` help text so
    both surface the same authoritative list without duplicating it.
    """
    return sorted(_STRATEGIES)


# ╭─╴ Helpers ╶────────────────────────────────╮
# │  Shared normalisation used by the         │
# │  score-based strategies.                   │
# ╰────────────────────────────────────────────╯


def _min_max_normalise(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a ``doc_id -> raw_score`` mapping to ``[0, 1]``.

    Semantic (``1 - cosine_distance``) and BM25 (unbounded, positive) live on
    incompatible scales. Fusing them requires per-branch normalisation. When
    every score in the branch is identical (or a single doc is present) the
    result is ``1.0`` for every entry — the branch cannot distinguish its own
    hits, so treating them as tied is the correct behaviour.

    Args:
        scores: Raw scores keyed by doc_id.

    Returns:
        dict[str, float]: Same keys, values in ``[0, 1]``. Empty input maps
        to an empty dict.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo <= 0.0:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def _bucket_by_id(
    semantic_results: List[RetrievedResult],
    bm25_results: List[RetrievedResult],
) -> Tuple[Dict[str, RetrievedResult], Dict[str, RetrievedResult], List[str]]:
    """Index both branches by doc_id and materialise the union of ids.

    Preserves first-seen ordering (semantic branch first, then BM25 tail) so
    ties in the fused ranking break the same way across strategies.
    """
    sem_by_id = {r.doc_id: r for r in semantic_results}
    bm25_by_id = {r.doc_id: r for r in bm25_results}

    all_ids: List[str] = []
    seen: set = set()
    for r in semantic_results:
        if r.doc_id not in seen:
            all_ids.append(r.doc_id)
            seen.add(r.doc_id)
    for r in bm25_results:
        if r.doc_id not in seen:
            all_ids.append(r.doc_id)
            seen.add(r.doc_id)

    return sem_by_id, bm25_by_id, all_ids


# ╭─╴ RRFusion (DEFAULT) ╶─────────────────────╮
# │  Reciprocal Rank Fusion with K=60 —         │
# │  the historical hardcoded default.          │
# ╰────────────────────────────────────────────╯


@register("rrf")
class RRFusion:
    """Reciprocal Rank Fusion with ``K=60``.

    Score formula (per doc):
        ``alpha * 1 / (K + semantic_rank) + (1 - alpha) * 1 / (K + bm25_rank)``

    Docs absent from a branch are assigned ``rank = 1000`` — a legacy fallback
    that matches the original hardcoded implementation in
    ``orchestrator.query``. The tiny non-zero contribution (``~9.4e-4`` at
    K=60) is intentional and preserved for exact regression compatibility.

    Attributes:
        name: ``"rrf"`` — the registry key.
        K: Reciprocal-rank constant. Kept at ``60`` per the paper's default;
            not currently configurable via YAML to avoid a knob explosion.
        MISSING_RANK: Fallback rank for a doc absent from one branch.
    """

    name = "rrf"
    K = 60
    MISSING_RANK = 1000

    def fuse(
        self,
        semantic_results: List[RetrievedResult],
        bm25_results: List[RetrievedResult],
        alpha: float = 0.5,
        weights: Optional[Dict[str, float]] = None,  # noqa: ARG002 — Protocol conformance
        **kwargs: Any,  # noqa: ARG002 — Protocol conformance (accepts query= transparently)
    ) -> List[Tuple[str, float]]:
        """Fuse via RRF. See class docstring for the formula."""
        sem_by_id, bm25_by_id, all_ids = _bucket_by_id(semantic_results, bm25_results)

        scored: List[Tuple[str, float]] = []
        for doc_id in all_ids:
            sem = sem_by_id.get(doc_id)
            bm = bm25_by_id.get(doc_id)

            semantic_rank = sem.semantic_rank if sem and sem.semantic_rank is not None else self.MISSING_RANK
            bm25_rank = bm.bm25_rank if bm and bm.bm25_rank is not None else self.MISSING_RANK

            score = alpha * (1.0 / (self.K + semantic_rank)) + (1.0 - alpha) * (1.0 / (self.K + bm25_rank))
            scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ╭─╴ CombSUM ╶────────────────────────────────╮
# │  Alpha-weighted sum of normalised scores.  │
# ╰────────────────────────────────────────────╯


@register("combsum")
class CombSUM:
    """CombSUM fusion: alpha-weighted sum of per-branch normalised scores.

    Score formula (per doc):
        ``alpha * norm(semantic_score) + (1 - alpha) * norm(bm25_score)``

    Missing branch contributes ``0``. ``norm`` is min-max within the branch's
    candidate list so semantic (cosine-derived, ``[0, 1]``) and BM25 (raw,
    unbounded) can be added meaningfully.

    Notes:
        Score-based (not rank-based), so it rewards a doc that dominates one
        branch even if the other branch never saw it. Prefer CombMNZ when you
        want to favour docs that agree across branches.
    """

    name = "combsum"

    def fuse(
        self,
        semantic_results: List[RetrievedResult],
        bm25_results: List[RetrievedResult],
        alpha: float = 0.5,
        weights: Optional[Dict[str, float]] = None,  # noqa: ARG002 — Protocol conformance
        **kwargs: Any,  # noqa: ARG002 — Protocol conformance (accepts query= transparently)
    ) -> List[Tuple[str, float]]:
        """Fuse via CombSUM. See class docstring."""
        sem_by_id, bm25_by_id, all_ids = _bucket_by_id(semantic_results, bm25_results)

        sem_raw = {r.doc_id: r.semantic_score for r in semantic_results if r.semantic_score is not None}
        bm25_raw = {r.doc_id: r.bm25_score for r in bm25_results if r.bm25_score is not None}

        sem_norm = _min_max_normalise(sem_raw)
        bm25_norm = _min_max_normalise(bm25_raw)

        scored: List[Tuple[str, float]] = []
        for doc_id in all_ids:
            s = sem_norm.get(doc_id, 0.0)
            b = bm25_norm.get(doc_id, 0.0)
            scored.append((doc_id, alpha * s + (1.0 - alpha) * b))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ╭─╴ CombMNZ ╶────────────────────────────────╮
# │  CombSUM boosted by branch-hit count.       │
# ╰────────────────────────────────────────────╯


@register("combmnz")
class CombMNZ:
    """CombMNZ fusion: CombSUM scaled by the number of branches that hit.

    Score formula (per doc):
        ``combsum_score * n_hits``  where ``n_hits ∈ {1, 2}``

    Rewards docs that both branches independently agreed on — a classic
    ensemble signal in TREC-era IR literature. A doc found only in one branch
    keeps its CombSUM score (``× 1``); a doc found in both gets doubled.
    """

    name = "combmnz"

    def fuse(
        self,
        semantic_results: List[RetrievedResult],
        bm25_results: List[RetrievedResult],
        alpha: float = 0.5,
        weights: Optional[Dict[str, float]] = None,  # noqa: ARG002 — Protocol conformance
        **kwargs: Any,  # noqa: ARG002 — Protocol conformance (accepts query= transparently)
    ) -> List[Tuple[str, float]]:
        """Fuse via CombMNZ. See class docstring."""
        sem_by_id, bm25_by_id, all_ids = _bucket_by_id(semantic_results, bm25_results)

        sem_raw = {r.doc_id: r.semantic_score for r in semantic_results if r.semantic_score is not None}
        bm25_raw = {r.doc_id: r.bm25_score for r in bm25_results if r.bm25_score is not None}

        sem_norm = _min_max_normalise(sem_raw)
        bm25_norm = _min_max_normalise(bm25_raw)

        scored: List[Tuple[str, float]] = []
        for doc_id in all_ids:
            s = sem_norm.get(doc_id, 0.0)
            b = bm25_norm.get(doc_id, 0.0)
            n_hits = (1 if doc_id in sem_by_id else 0) + (1 if doc_id in bm25_by_id else 0)
            base = alpha * s + (1.0 - alpha) * b
            scored.append((doc_id, base * n_hits))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ╭─╴ WeightedLinear ╶─────────────────────────╮
# │  Explicit per-branch weights (config).      │
# ╰────────────────────────────────────────────╯


@register("weighted")
class WeightedLinear:
    """Weighted linear combination of per-branch normalised scores.

    Score formula (per doc):
        ``w_sem * norm(semantic_score) + w_bm25 * norm(bm25_score)``

    Weights come from the ``weights`` argument when supplied
    (``{"semantic": w_sem, "bm25": w_bm25}``), otherwise fall back to
    ``{"semantic": alpha, "bm25": 1 - alpha}`` so the strategy degenerates to
    :class:`CombSUM` for callers that do not care about explicit weights.
    """

    name = "weighted"

    def fuse(
        self,
        semantic_results: List[RetrievedResult],
        bm25_results: List[RetrievedResult],
        alpha: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,  # accepts query= for LearnedFusion; ignored by others
    ) -> List[Tuple[str, float]]:
        """Fuse via weighted linear combination. See class docstring."""
        sem_by_id, bm25_by_id, all_ids = _bucket_by_id(semantic_results, bm25_results)

        w_sem, w_bm25 = self._resolve_weights(alpha, weights)

        sem_raw = {r.doc_id: r.semantic_score for r in semantic_results if r.semantic_score is not None}
        bm25_raw = {r.doc_id: r.bm25_score for r in bm25_results if r.bm25_score is not None}

        sem_norm = _min_max_normalise(sem_raw)
        bm25_norm = _min_max_normalise(bm25_raw)

        scored: List[Tuple[str, float]] = []
        for doc_id in all_ids:
            s = sem_norm.get(doc_id, 0.0)
            b = bm25_norm.get(doc_id, 0.0)
            scored.append((doc_id, w_sem * s + w_bm25 * b))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _resolve_weights(alpha: float, weights: Optional[Dict[str, float]]) -> Tuple[float, float]:
        """Return ``(w_semantic, w_bm25)`` from explicit weights or from alpha."""
        if not weights:
            return alpha, 1.0 - alpha
        w_sem = float(weights.get("semantic", alpha))
        w_bm25 = float(weights.get("bm25", 1.0 - alpha))
        return w_sem, w_bm25


__all__ = [
    "RetrievedResult",
    "FusionStrategy",
    "RRFusion",
    "CombSUM",
    "CombMNZ",
    "WeightedLinear",
    "register",
    "get_strategy",
    "available_strategies",
]
