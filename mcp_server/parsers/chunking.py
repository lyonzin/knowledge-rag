"""
╭─╴ KNOWLEDGE-RAG CHUNKING ╶─────────────────────────────────────╮
│                                                                │
│   Shared chunking helpers used by every parser through the     │
│   registry pipeline.                                           │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Date    ·  2026-07-27

These helpers used to live as private methods on ``DocumentParser``. They
moved here so that:

* Third-party parsers can reuse the exact same chunking heuristics
  without importing the ingestion facade (which pulls optional deps).
* The behaviour is one function per strategy, easy to unit-test in
  isolation and to reason about against fuzz corpora.

The heuristics themselves are byte-identical to the pre-refactor
implementation — this file is a move, not a rewrite. A single behavioural
regression here would silently reshape every chunk in the 70+ user
corpora.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import Chunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..retrieval.semantic_cache import SemanticCache

__all__ = [
    "chunk_hierarchical",
    "chunk_markdown",
    "chunk_text",
    "chunk_text_contextual",
]


def chunk_text(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Split arbitrary text into overlapping chunks for embedding.

    The splitter prefers natural break points (paragraph → line → sentence
    → word) within the last 20 percent of each ``chunk_size`` window, and
    steps forward by ``chunk_size - chunk_overlap`` to keep continuity
    across chunk boundaries. Two safety nets guard against pathological
    inputs:

    * An infinite-loop detector aborts if ``start`` fails to advance.
    * When the overlap would erase forward progress the loop steps to
      ``end`` instead of subtracting.

    Args:
        text: Content to chunk.
        metadata: Parent-document metadata; ``title`` and ``type`` are
            copied onto each chunk.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Number of trailing characters carried into the
            next chunk.

    Returns:
        list[Chunk]: Ordered chunks. Empty when ``text`` is empty.
    """
    if not text:
        return []

    chunks: List[Chunk] = []
    text_len = len(text)
    start = 0
    index = 0
    previous_start = -1  # Track previous start to detect infinite loops

    while start < text_len:
        # Safety: detect infinite loop (start not progressing)
        if start <= previous_start:
            break
        previous_start = start

        # Calculate end position
        end = min(start + chunk_size, text_len)

        # Try to break at sentence/paragraph boundary
        if end < text_len:
            # Look for natural break points within last 20% of chunk
            break_zone_start = start + int(chunk_size * 0.8)
            break_zone = text[break_zone_start:end]

            # Priority: paragraph > sentence > word
            for pattern in ["\n\n", "\n", ". ", " "]:
                last_break = break_zone.rfind(pattern)
                if last_break != -1:
                    end = break_zone_start + last_break + len(pattern)
                    break

        chunk_content = text[start:end].strip()

        if chunk_content:
            chunk = Chunk(
                content=chunk_content,
                index=index,
                start_char=start,
                end_char=end,
                metadata={
                    "title": metadata.get("title", ""),
                    "type": metadata.get("type", ""),
                },
            )
            chunks.append(chunk)
            index += 1

        # Move start position with overlap.
        # Ensure we always make forward progress.
        new_start = end - chunk_overlap

        # If overlap would cause no progress, just move to end
        if new_start <= start:
            start = end
        else:
            start = new_start

    return chunks


def chunk_markdown(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Markdown-aware chunker with code-block protection and min-size merging.

    The heuristic:

    1. Masks fenced code blocks so that ``#`` comments inside them cannot
       be mistaken for headers by the splitter.
    2. Splits on ``##`` and ``###`` headers only. ``#`` alone was catching
       code comments in the wild — an explicit past regression.
    3. Merges sections shorter than ``min_chunk_size`` into their neighbour
       so we do not emit tiny, useless chunks.
    4. Falls back to :func:`chunk_text` for section-less documents or
       sections that exceed ``chunk_size`` (each oversized section is
       re-chunked and the header context is stamped on every sub-chunk).

    Args:
        text: Markdown content.
        metadata: Parent-document metadata; ``title`` and ``type`` are
            copied onto each chunk.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Number of trailing characters carried into the
            next chunk when falling back to :func:`chunk_text`.

    Returns:
        list[Chunk]: Ordered chunks aligned to markdown sections.
    """
    if not text:
        return []

    min_chunk_size = 100  # Minimum chars for a standalone chunk

    # Step 1: Mask code blocks to prevent splitting on # inside them
    code_blocks: List[str] = []

    def mask_code(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    masked_text = re.sub(r"```.*?```", mask_code, text, flags=re.DOTALL)

    # Step 2: Split by ## and ### headers only (not # which catches code comments)
    sections = re.split(r"(?=^#{2,3}\s+)", masked_text, flags=re.MULTILINE)

    # Filter empty sections
    sections = [s for s in sections if s.strip()]

    if len(sections) <= 1:
        return chunk_text(text, metadata, chunk_size, chunk_overlap)

    # Step 3: Restore code blocks in each section
    def restore_code(section_text: str) -> str:
        for i, block in enumerate(code_blocks):
            section_text = section_text.replace(f"__CODE_BLOCK_{i}__", block)
        return section_text

    sections = [restore_code(s) for s in sections]

    # Step 4: Merge small sections with the next one
    merged_sections: List[str] = []
    buffer = ""
    for section in sections:
        if buffer:
            buffer += "\n\n" + section
            if len(buffer.strip()) >= min_chunk_size:
                merged_sections.append(buffer)
                buffer = ""
        elif len(section.strip()) < min_chunk_size:
            buffer = section
        else:
            merged_sections.append(section)

    if buffer:
        if merged_sections:
            merged_sections[-1] += "\n\n" + buffer
        else:
            merged_sections.append(buffer)

    if not merged_sections:
        return chunk_text(text, metadata, chunk_size, chunk_overlap)

    # Step 5: Create chunks from merged sections
    chunks: List[Chunk] = []
    global_index = 0
    char_offset = 0

    for section in merged_sections:
        section_stripped = section.strip()
        if not section_stripped:
            char_offset += len(section)
            continue

        header_match = re.match(r"^(#{2,3}\s+.+)$", section_stripped, re.MULTILINE)
        header_context = header_match.group(1) if header_match else ""

        if len(section_stripped) <= chunk_size:
            chunk = Chunk(
                content=section_stripped,
                index=global_index,
                start_char=char_offset,
                end_char=char_offset + len(section),
                metadata={
                    "title": metadata.get("title", ""),
                    "type": metadata.get("type", ""),
                    "section_header": header_context,
                },
            )
            chunks.append(chunk)
            global_index += 1
        else:
            sub_chunks = chunk_text(section_stripped, metadata, chunk_size, chunk_overlap)
            for i, sub_chunk in enumerate(sub_chunks):
                if i > 0 and header_context:
                    sub_chunk.content = f"{header_context}\n\n{sub_chunk.content}"
                sub_chunk.index = global_index
                sub_chunk.start_char += char_offset
                sub_chunk.end_char += char_offset
                sub_chunk.metadata["section_header"] = header_context
                chunks.append(sub_chunk)
                global_index += 1

        char_offset += len(section)

    return chunks


# ╭─╴ Parent Document Retrieval (A3.7) ╶───────────────────────────╮
# │                                                                │
# │   Small-to-Big / Parent Document Retrieval.                    │
# │                                                                │
# │   Indexes fine-grained small chunks for precise retrieval,     │
# │   but stamps each with its parent chunk's content so the       │
# │   orchestrator can expand hits to the wider context window     │
# │   without a second round-trip to the vector store.             │
# │                                                                │
# ╰────────────────────────────────────────────────────────────────╯


def _parent_id(content: str, start_char: int, end_char: int) -> str:
    """Deterministic short ID for a parent chunk.

    Combines the parent's absolute offsets with a hash of its content so
    two different parents that happen to share text (repeated boilerplate,
    heavily templated docs) still resolve to distinct ids. SHA-1 is used
    purely as a non-cryptographic content hash — the ``usedforsecurity``
    flag makes that intent explicit for auditors.

    Args:
        content: Parent chunk text.
        start_char: Inclusive absolute offset in the source document.
        end_char: Exclusive absolute offset in the source document.

    Returns:
        str: ``parent_<16-hex>`` where the digits are the leading nibbles of a
        SHA-1 of ``start_char:end_char:content``.
    """
    payload = f"{start_char}:{end_char}:{content}".encode("utf-8")
    digest = hashlib.sha1(payload, usedforsecurity=False).hexdigest()
    return f"parent_{digest[:16]}"


def chunk_hierarchical(
    text: str,
    metadata: Dict[str, Any],
    large_size: int = 1500,
    small_size: int = 250,
    small_overlap: int = 50,
) -> List[Chunk]:
    """Two-level chunker for Parent Document / Small-to-Big retrieval.

    Splits ``text`` into two layers:

    1. **Parents** — contiguous, non-overlapping windows of roughly
       ``large_size`` characters. Provide the broad context an LLM (or
       reranker) can reason over.
    2. **Children** — overlapping windows of roughly ``small_size``
       characters carved out of each parent. Only children are returned,
       because only children get indexed and searched.

    Every child chunk carries its parent inline in metadata so the
    orchestrator can swap the returned content for the parent without a
    second vector-store lookup. That is the LangChain/LlamaIndex
    "Small-to-Big" pattern: precise retrieval, broad context.

    Metadata added to each child chunk:

    * ``parent_id`` — deterministic short ID (`parent_<16-hex>`).
    * ``parent_content`` — full parent text, ready to swap into the result.
    * ``parent_start_char`` / ``parent_end_char`` — parent offsets in the
      original ``text``.
    * ``is_small_chunk`` — always ``True``; marks that this chunk expects
      parent expansion at query time.

    The function is a pure composition on top of :func:`chunk_text`, so it
    inherits the same safety nets (infinite-loop detector, boundary
    fallbacks). Zero-overlap parent windows mean the whole text is covered
    at least once by the union of parents (modulo trailing whitespace that
    :func:`chunk_text` strips), so no content is lost.

    Args:
        text: Content to chunk.
        metadata: Parent-document metadata; ``title`` and ``type`` are
            copied onto each child chunk (mirroring :func:`chunk_text`).
        large_size: Target parent-chunk length in characters.
        small_size: Target child-chunk length in characters.
        small_overlap: Trailing characters carried between adjacent
            children of the same parent.

    Returns:
        list[Chunk]: Ordered small chunks with parent metadata attached.
        Empty when ``text`` is empty.

    Raises:
        ValueError: When ``large_size`` is not strictly larger than
            ``small_size``, when either is non-positive, or when
            ``small_overlap`` is negative or would erase forward progress.
    """
    if not text:
        return []

    if large_size <= 0 or small_size <= 0:
        raise ValueError(f"chunk sizes must be positive (large={large_size}, small={small_size})")
    if large_size <= small_size:
        raise ValueError(f"large_size ({large_size}) must be greater than small_size ({small_size})")
    if small_overlap < 0:
        raise ValueError(f"small_overlap must be >= 0 (got {small_overlap})")
    if small_overlap >= small_size:
        raise ValueError(f"small_overlap ({small_overlap}) must be less than small_size ({small_size})")

    # Step 1: Cut parents. Zero overlap keeps the parent set a partition of
    # the text (up to whitespace boundaries) so every child maps to exactly
    # one parent and coverage is complete.
    parents: List[Chunk] = chunk_text(text, metadata, large_size, 0)
    if not parents:
        return []

    # Step 2: Carve children out of each parent. Children inherit the
    # parent's absolute char offsets so downstream consumers still get
    # positions in the original text, not in the parent substring.
    small_chunks: List[Chunk] = []
    global_index = 0

    for parent in parents:
        parent_content = parent.content
        pid = _parent_id(parent_content, parent.start_char, parent.end_char)

        # ``chunk_text`` handles the small-window loop with the same
        # boundary heuristics used everywhere else in the codebase.
        children = chunk_text(parent_content, metadata, small_size, small_overlap)

        # Degenerate case: a parent shorter than one full small window
        # would come back empty from chunk_text (never happens today —
        # parents are >= small_size by construction — but keep the guard
        # so future config tweaks cannot silently drop content).
        if not children:
            children = [
                Chunk(
                    content=parent_content,
                    index=0,
                    start_char=0,
                    end_char=len(parent_content),
                    metadata={
                        "title": metadata.get("title", ""),
                        "type": metadata.get("type", ""),
                    },
                )
            ]

        for child in children:
            child.index = global_index
            child.start_char = parent.start_char + child.start_char
            child.end_char = parent.start_char + child.end_char
            # Preserve any metadata chunk_text set (title/type) while
            # stamping the parent linkage on top.
            child.metadata.update(
                {
                    "parent_id": pid,
                    "parent_content": parent_content,
                    "parent_start_char": parent.start_char,
                    "parent_end_char": parent.end_char,
                    "is_small_chunk": True,
                }
            )
            small_chunks.append(child)
            global_index += 1

    return small_chunks


# ╭─╴ Contextual Chunking (A3.6) ╶─────────────────────────────────╮
# │                                                                │
# │   Anthropic 2024 pattern: fixed-size chunk, then ask an LLM    │
# │   to produce a 1-2 sentence context that situates each chunk   │
# │   in the whole document, and PREPEND that context to the       │
# │   chunk before embedding + indexing. Anthropic reports up to   │
# │   +49% retrieval recall improvement.                           │
# │                                                                │
# │   The heavy lifting (LLM call, cache, fail-open) lives in      │
# │   :mod:`mcp_server.retrieval.llm_features.contextual_chunking` │
# │   — this wrapper stays in the parsers namespace to keep the    │
# │   chunking dispatch table symmetrical with ``chunk_text`` and  │
# │   friends. The LLM import is deferred so a default install     │
# │   (feature off) never touches the ``providers.llm`` subpackage.│
# │                                                                │
# ╰────────────────────────────────────────────────────────────────╯


def chunk_text_contextual(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    provider_name: Optional[str] = None,
    cache: Optional["SemanticCache"] = None,
) -> List[Chunk]:
    """Fixed-size chunk + LLM-generated per-chunk context prepended in-place.

    WARNING — this function issues ONE LLM CALL PER CHUNK at ingestion
    time. On a 100kB document with the default ``chunk_size=1000`` that
    is ~100 provider calls. The upstream ingestion path only invokes
    this function when ``config.contextual_chunking_enabled`` is True,
    which is DEFAULT OFF for exactly this reason.

    Passing a :class:`SemanticCache` is strongly recommended: re-indexing
    an unchanged document then costs zero LLM calls. Without a cache,
    every ingestion pass pays the full LLM bill.

    The returned chunks retain every field the flat chunker produces
    (``index``, ``start_char``, ``end_char``, ``metadata["title"]``,
    ``metadata["type"]``) — only ``content`` is rewritten to
    ``"{context}\\n\\n{original_chunk}"`` and a ``metadata["contextual"] = True``
    marker is added so the orchestrator and the MCP tool layer can
    surface the fact to callers.

    Fail-open contract (inherited from
    :func:`~mcp_server.retrieval.llm_features.contextual_chunking.contextualize_chunk`):

        * A missing/unreachable LLM provider yields the raw chunk with
          NO ``contextual`` marker for that chunk.
        * A total LLM outage means every chunk comes back unmarked and
          identical to what :func:`chunk_text` would produce — no crash,
          no data loss.

    Args:
        text: Content to chunk. Empty input returns an empty list.
        metadata: Parent-document metadata; ``title`` and ``type`` are
            copied onto each chunk exactly like :func:`chunk_text`.
        chunk_size: Target chunk length in characters. Passed straight
            to :func:`chunk_text`.
        chunk_overlap: Number of trailing characters carried into the
            next chunk. Passed straight to :func:`chunk_text`.
        provider_name: Optional LLM provider override
            (e.g. ``"anthropic"``). When ``None`` the LLM feature module
            resolves via ``config.llm_provider`` → ``auto_detect_llm``.
        cache: Optional :class:`SemanticCache`. When supplied, per-chunk
            calls hit the cache on repeat runs. When ``None`` every call
            goes straight to the LLM.

    Returns:
        list[Chunk]: The same shape :func:`chunk_text` would return, with
        per-chunk ``content`` rewritten to include the LLM-generated
        context prefix. Chunks whose contextualisation failed silently
        (LLM error) are still present with their raw content — the
        ``metadata["contextual"]`` marker is the authoritative signal
        for whether a chunk actually got the LLM treatment.
    """
    if not text:
        return []

    # Deferred import (breaks a cycle: llm_features.contextual_chunking
    # imports from providers.llm which imports from ... — keeping the
    # import inside the function guarantees a default install with the
    # feature disabled never even touches the LLM subpackage).
    from ..retrieval.llm_features.contextual_chunking import contextualize_chunk

    raw_chunks = chunk_text(text, metadata, chunk_size, chunk_overlap)

    for chunk in raw_chunks:
        original = chunk.content
        contextualized = contextualize_chunk(
            chunk_content=original,
            doc_content=text,
            provider_name=provider_name,
            cache=cache,
        )
        # Only mark the chunk as ``contextual`` when the LLM actually
        # rewrote it. ``contextualize_chunk`` returns the raw chunk on
        # any failure — using an identity check keeps the marker
        # truthful even when the LLM path degrades silently.
        if contextualized is not original and contextualized != original:
            chunk.content = contextualized
            chunk.metadata["contextual"] = True

    return raw_chunks
