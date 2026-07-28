"""
╭─╴ TRUNCATION HELPERS ╶─────────────────────────────────────────╮
│                                                                │
│   Truncation-aware output guards for MCP tool payloads.        │
│   Extracted verbatim from server.py in the A2.1 refactor.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``

MCP transports and LLM clients both cap how much text a single tool response
may carry. When the cap is hit somewhere downstream the payload is cut without
a word, and the calling model reads that silence as "the knowledge base has
nothing else" — the classic RAG failure where a correct index looks empty.

The helpers below cut deliberately instead: whole results only (never a result
sliced in half), the top-ranked result is never dropped, and every cut is
reported back in the payload so the model can react (narrow the query, raise
max_results, fetch the document directly).
"""

import json
from typing import Any, Dict, List, Tuple

RESULT_CHAR_BUDGET = 100_000
DOCUMENT_CHAR_BUDGET = 100_000


def _make_snippet(content: str, max_chars: int = 500) -> str:
    """Truncate content at a natural break point."""
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    min_pos = int(max_chars * 0.6)
    last_nl = truncated.rfind("\n", min_pos)
    if last_nl > min_pos:
        return truncated[:last_nl].rstrip() + "\n..."
    for sep in (". ", "? ", "! ", "; "):
        last_sep = truncated.rfind(sep, min_pos)
        if last_sep > min_pos:
            return truncated[: last_sep + len(sep) - 1] + " ..."
    last_space = truncated.rfind(" ", min_pos)
    if last_space > min_pos:
        return truncated[:last_space] + " ..."
    return truncated + "..."


def _result_char_cost(result: Dict[str, Any]) -> int:
    """
    Approximate the serialized size of a single result, in characters.

    Args:
        result: One formatted search result.

    Returns:
        int: Length of the JSON encoding, or of ``str(result)`` when the result
        holds something JSON cannot encode.
    """
    try:
        return len(json.dumps(result, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(result))


def _truncate_results_safe(
    results: List[Dict[str, Any]], char_budget: int = RESULT_CHAR_BUDGET
) -> Tuple[List[Dict[str, Any]], bool, int]:
    """
    Trim a result list to a character budget without ever losing the top hit.

    Cuts only at result boundaries, so every returned result is complete. The
    rank-1 result is always kept ("seed protection") even when it alone exceeds
    the budget: an oversized answer is recoverable, a missing best answer is not.

    Args:
        results: Formatted results, already sorted best-first.
        char_budget: Maximum combined serialized size (default: 100 000 chars,
            roughly 25K tokens). Non-positive values disable the budget.

    Returns:
        tuple: ``(kept_results, was_truncated, total_original_count)``.

    Example:
        >>> kept, cut, total = _truncate_results_safe([{"a": "x" * 10}], 4)
        >>> cut, total, len(kept)  # seed protected despite blowing the budget
        (False, 1, 1)
    """
    total = len(results)
    if total == 0 or char_budget <= 0:
        return results, False, total

    kept: List[Dict[str, Any]] = [results[0]]  # seed protection — never dropped
    used = _result_char_cost(results[0])

    for result in results[1:]:
        cost = _result_char_cost(result)
        if used + cost > char_budget:
            break
        kept.append(result)
        used += cost

    return kept, len(kept) < total, total


def _truncation_note(shown: int, total: int) -> str:
    """
    Build the operator-facing warning attached to a truncated payload.

    Args:
        shown: Number of results actually returned.
        total: Number of results that matched before truncation.

    Returns:
        str: Human-readable warning naming both counts and the way out.
    """
    return (
        f"[!] TRUNCATED: showing {shown} of {total} results — "
        f"lower max_results, refine the query, or call get_document() for full content"
    )


def _truncate_document_safe(
    doc: Dict[str, Any], char_budget: int = DOCUMENT_CHAR_BUDGET
) -> Tuple[Dict[str, Any], bool]:
    """
    Trim an oversized document payload at a line boundary, reporting the cut.

    Args:
        doc: Document payload from ``KnowledgeOrchestrator.get_document``.
        char_budget: Maximum content length in characters. Non-positive values
            disable the budget.

    Returns:
        tuple: ``(payload, was_truncated)``. When truncated, the payload is a
        copy carrying ``content_length`` (original size) and ``shown_chars``;
        otherwise the input dict is returned untouched.
    """
    content = doc.get("content", "")
    if char_budget <= 0 or not isinstance(content, str) or len(content) <= char_budget:
        return doc, False

    cut = content[:char_budget]
    last_nl = cut.rfind("\n")
    if last_nl > char_budget // 2:
        cut = cut[:last_nl]

    trimmed = dict(doc)
    trimmed["content"] = cut
    trimmed["content_length"] = len(content)
    trimmed["shown_chars"] = len(cut)
    return trimmed, True
