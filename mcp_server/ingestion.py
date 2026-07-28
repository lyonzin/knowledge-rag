"""Document Ingestion System for Knowledge RAG.

Facade over :mod:`mcp_server.parsers`: preserves the historical
``DocumentParser`` API while delegating every per-format extraction
step to the plugin-friendly registry.

Public exports — kept byte-compatible with pre-refactor releases:

* :class:`Document`, :class:`Chunk` — value objects (re-exported from
  :mod:`mcp_server.parsers.base`).
* :class:`DocumentParser` — historical facade. New code should call
  :func:`mcp_server.parsers.registry.parse_content` and build the
  pipeline in place, but ``DocumentParser`` remains the supported entry
  point for anything reading a full :class:`Document`.
* :data:`LANGUAGE_PROFILES` — code-parser dispatch table, re-exported
  from :mod:`mcp_server.parsers.code_parser` for downstream callers.
* :func:`parse_documents` — convenience wrapper over
  ``DocumentParser().parse_directory``.

Supports MD, PDF, TXT, PY, C, H, CPP, JS, JSX, TS, TSX, JSON, XML,
DOCX, XLSX, PPTX, CSV, IPYNB, MQH, MQ4 out of the box; third-party
parsers register via the ``knowledge_rag.parsers`` entry-point group.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import config
from .parsers import registry
from .parsers.base import Chunk, Document, ParserResult
from .parsers.chunking import chunk_hierarchical, chunk_markdown, chunk_text, chunk_text_contextual
from .security import detect_external_marker, is_path_within, neutralize_injection_sentinels

# The following chunkers/classifiers ship in a follow-up PR (advanced chunking +
# confidence labels). They are always resolved via deferred imports inside the
# feature-flag guarded branches below, so a default install (all flags False)
# never touches them at import time. If the follow-up module is not installed
# and the flag is toggled on, the caller sees a clear ImportError at chunk time.
#   - mcp_server.parsers.code_parser (LANGUAGE_PROFILES)
#   - mcp_server.parsers.confidence (classify_chunk_confidence)
#   - mcp_server.parsers.late_chunker (chunk_text_late)
#   - mcp_server.parsers.tree_sitter_chunker (chunk_code_by_ast + SUPPORTED_EXTENSIONS)

log = logging.getLogger(__name__)

__all__ = [
    "Chunk",
    "Document",
    "DocumentParser",
    "parse_documents",
]


_CODE_AWARE_EXTENSIONS_CACHE: Optional[frozenset] = None


def _code_aware_supports(suffix: str) -> bool:
    """Lazy check for tree-sitter code-aware chunking support.

    Returns False when the follow-up advanced-chunking module is not
    installed (default in the base install), so the opt-in
    ``config.code_aware_chunking`` flag stays inert without raising.
    """
    global _CODE_AWARE_EXTENSIONS_CACHE
    if _CODE_AWARE_EXTENSIONS_CACHE is None:
        try:
            from .parsers.tree_sitter_chunker import SUPPORTED_EXTENSIONS
            _CODE_AWARE_EXTENSIONS_CACHE = frozenset(SUPPORTED_EXTENSIONS)
        except ImportError:
            _CODE_AWARE_EXTENSIONS_CACHE = frozenset()
    return suffix in _CODE_AWARE_EXTENSIONS_CACHE


def _resolve_token_aware_provider():
    """Return a token-aware embedding provider instance, or ``None`` on failure.

    Reads ``config.embedding_provider`` and asks the registry for the
    matching class. Returns an instance ONLY when the class exposes an
    ``embed_tokens`` attribute (structural check — cheap enough to run on
    every ingested doc, and safer than trying an ``isinstance`` walk
    against a Protocol that may not import cleanly under every Python
    version).

    Any failure — unknown provider name, constructor exception, provider
    lacks token-aware surface — resolves to ``None`` so
    :func:`~mcp_server.parsers.late_chunker.chunk_text_late` can take its
    documented fallback path (fixed-size chunking + standard embedding).

    Returns:
        TokenAwareEmbeddingProvider | None: A provider instance ready
        for :meth:`embed_tokens` calls, or ``None`` when the configured
        provider is not token-aware.
    """
    try:
        from .providers.registry import get_embedding_class
    except Exception:  # pragma: no cover — defensive
        return None

    provider_name = getattr(config, "embedding_provider", "fastembed") or "fastembed"
    try:
        provider_cls = get_embedding_class(provider_name)
    except KeyError:
        log.warning(
            "Late chunking: embedding provider %r is not registered; falling back "
            "to fixed-size chunking + standard embedding for this document.",
            provider_name,
        )
        return None
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "Late chunking: unexpected error resolving provider %r: %s",
            provider_name,
            exc,
        )
        return None

    # Structural check up-front so we don't waste a constructor call on
    # providers that clearly don't implement the token-aware surface
    # (bundled ``FastEmbedEmbeddings`` is the common example).
    if not hasattr(provider_cls, "embed_tokens"):
        return None

    try:
        return provider_cls()
    except Exception as exc:  # pragma: no cover — defensive; late chunker falls back
        log.warning(
            "Late chunking: provider %r failed to instantiate: %s",
            provider_name,
            exc,
        )
        return None


class DocumentParser:
    """Facade over the parser registry with the historical public API.

    Rewritten as a thin coordinator: per-format extraction lives in
    :mod:`mcp_server.parsers`, and this class owns the cross-cutting
    pipeline — prompt-injection sanitization, provenance-marker
    propagation, category detection, keyword extraction, chunk
    generation, safe directory traversal.

    The API is intentionally identical to the pre-refactor class:

    * ``parser.parse_file(path)`` returns a :class:`Document`.
    * ``parser.parse_directory(dir)`` walks the corpus with symlink and
      exclude-pattern protection.
    * ``parser._parsers`` returns a dict of ``suffix -> parse_callable``
      built live from the registry, so backward-compatible tests
      inspecting the dispatch table keep working.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> None:
        """Initialise with optional per-instance chunking overrides.

        Args:
            chunk_size: Target chunk length. Falls back to
                ``config.chunk_size`` when ``None``.
            chunk_overlap: Trailing chars carried into the next chunk.
                Falls back to ``config.chunk_overlap`` when ``None``.
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

    @property
    def _parsers(self) -> Dict[str, Callable[[Path], ParserResult]]:
        """Return a live view of the dispatch table for backward compatibility.

        Tests historically wrote ``assert ".md" in parser._parsers`` and
        called ``parser._parsers[".md"](path)`` — that shape is preserved
        by snapshotting :func:`registry.get_parsers` on every access, so
        newly registered plugins show up without stale references.

        Returns:
            dict[str, callable]: Suffix → parser ``parse`` callable. Later
            registrations override earlier ones on suffix collision.
        """
        return registry.build_dispatch_view()

    def parse_file(self, filepath: Path) -> Optional[Document]:
        """Parse a single file into a fully hydrated :class:`Document`.

        Runs the full pipeline: dispatch, sanitization (only for content
        that already carries a provenance marker written by
        :func:`mcp_server.security.wrap_external_content`), category
        detection, keyword extraction, chunking, and per-chunk evidence
        stamping for external content.

        Args:
            filepath: Path to parse. Absolute or relative to CWD.

        Returns:
            Document | None: Parsed document, or ``None`` when the file
            exists but contains no non-whitespace content.

        Raises:
            FileNotFoundError: When ``filepath`` does not exist.
            ValueError: When no registered parser claims the suffix.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()
        parser = registry.get_parser_for(filepath)
        if parser is None:
            raise ValueError(f"Unsupported format: {suffix}")

        # Generate unique ID
        doc_id = self._generate_id(filepath)

        # Parse content and metadata
        content, metadata = parser.parse(filepath)

        if not content or not content.strip():
            log.warning("[WARN] Skipping empty file: %s", filepath)
            return None

        # Prompt-injection defense (OWASP LLM01:2025) — layer 2 + layer 3.
        #
        # Only content that carries the provenance marker written by
        # add_from_url() is treated as untrusted. Operator-authored documents
        # are left byte-identical: a hand-written "### System:" header is
        # legitimate prose, and rewriting it would be a false positive.
        #
        # Neutralization is idempotent, so re-indexing an already-sanitized
        # file never stacks separators.
        external_marker = detect_external_marker(content)
        if external_marker is not None:
            external_uri, external_hash = external_marker
            content = neutralize_injection_sentinels(content)
            metadata["external_source"] = True
            metadata["external_source_uri"] = external_uri
            metadata["external_content_sha256"] = external_hash

        # Detect category from path
        category = self._detect_category(filepath)

        # Extract keywords
        keywords = self._extract_keywords(content, category)

        # Create document
        doc = Document(
            id=doc_id,
            content=content,
            source=filepath,
            format=suffix,
            category=category,
            metadata=metadata,
            keywords=keywords,
        )

        # Chunk the content.
        #
        # Precedence — integrated across every opt-in Fase 3+5 feature so
        # only one strategy ever runs per document:
        #
        # 1. Parent Document Retrieval (A3.7) — when enabled, every format
        #    uses the hierarchical small-to-big chunker. Wins over
        #    everything else because it is a whole-corpus retrieval mode.
        # 2. Code-aware chunking (A3.8) — opt-in via
        #    ``documents.code_aware_chunking``. When on and the suffix is
        #    a source-code language tree-sitter understands, chunks split
        #    at function/class boundaries. Falls back to fixed-size when
        #    tree-sitter is not installed. Wins over contextual chunking
        #    because AST-aligned chunk boundaries already carry structural
        #    context — a per-chunk LLM sentence would be redundant.
        # 3. Contextual chunking (A3.6, Anthropic 2024) — opt-in via
        #    ``documents.contextual_chunking``. When on (and the two
        #    higher-priority flags are off), every fixed-size chunk gets a
        #    1-2 sentence LLM-generated context prepended. EXPENSIVE — one
        #    LLM call per chunk at ingestion time. Semantic-cache backed
        #    so re-indexing unchanged docs pays zero LLM cost.
        # 4. Late chunking (R5.7, Jina 2024) — opt-in via
        #    ``documents.late_chunking``. When on (and the three
        #    higher-priority flags are off) AND the configured embedding
        #    provider implements ``embed_tokens``, the whole document is
        #    embedded once in a long-context model and per-chunk vectors
        #    are produced by mean-pooling token embeddings across each
        #    chunk span. Ranks below contextual chunking because
        #    contextual rewrites the CHUNK CONTENT (helps BM25 + reranker
        #    too), while late chunking only reshapes embeddings. Falls
        #    back to fixed-size + standard embedding when the provider
        #    lacks ``embed_tokens`` — never crashes.
        # 5. Markdown — section-aware chunker with header propagation for
        #    ``.md`` files when no higher-priority feature is enabled.
        # 6. Default — the character-window chunker.
        #
        # ``is True`` on the flags (not truthy) so tests that patch
        # ``config`` with a bare ``MagicMock`` — whose attribute lookups
        # return truthy child mocks — still fall through to the flat
        # chunker instead of activating the opt-in path with mock sizes.
        if getattr(config, "parent_document_enabled", False) is True:
            doc.chunks = chunk_hierarchical(
                content,
                metadata,
                large_size=config.parent_document_large_size,
                small_size=config.parent_document_small_size,
                small_overlap=config.parent_document_small_overlap,
            )
            chunker_name = "hierarchical"
        elif getattr(config, "code_aware_chunking", False) and _code_aware_supports(suffix):
            from .parsers.tree_sitter_chunker import chunk_code_by_ast
            doc.chunks = chunk_code_by_ast(
                content,
                metadata,
                ext=suffix,
                max_chunk_size=config.code_aware_max_chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunker_name = "code_aware"
        elif getattr(config, "contextual_chunking_enabled", False) is True:
            # Deferred import: the contextual chunking module pulls in the
            # LLM provider registry. Keeping it out of the module top-level
            # means a default install (feature off) never even touches
            # ``mcp_server.providers.llm`` at ingestion time.
            from .retrieval.llm_features.contextual_chunking import get_ingestion_semantic_cache

            doc.chunks = chunk_text_contextual(
                content,
                metadata,
                self.chunk_size,
                self.chunk_overlap,
                provider_name=(getattr(config, "llm_provider", "") or None),
                cache=get_ingestion_semantic_cache(),
            )
            chunker_name = "contextual"
        elif getattr(config, "late_chunking_enabled", False) is True:
            # Deferred import: only touches the embedding provider registry
            # when late chunking is actually enabled. Resolving the
            # provider here (not in the module body) keeps a default
            # install byte-identical to pre-R5.7 behaviour — no extra
            # imports, no extra registry lookups on the flat-chunker path.
            #
            # Provider resolution failure is treated the same as "no
            # token-aware provider": the chunker's fallback path returns
            # fixed-size chunks + None embeddings so ingestion never
            # crashes because of a misconfigured provider name.
            from .parsers.late_chunker import chunk_text_late
            embed_provider = _resolve_token_aware_provider()
            late_chunks, late_embeddings = chunk_text_late(
                content,
                metadata,
                self.chunk_size,
                self.chunk_overlap,
                embed_provider=embed_provider,
            )
            doc.chunks = late_chunks
            # The chunker already stamps ``chunk.embedding`` on every
            # chunk when embeddings are non-None (returns matched pair
            # by construction). The parallel list is kept in the return
            # tuple for callers that prefer indexed access; the indexing
            # layer downstream reads ``chunk.embedding`` directly.
            chunker_name = "late_chunking" if late_embeddings is not None else "late_chunking_fallback"
        elif suffix == ".md":
            doc.chunks = chunk_markdown(content, metadata, self.chunk_size, self.chunk_overlap)
            chunker_name = "markdown"
        else:
            doc.chunks = chunk_text(content, metadata, self.chunk_size, self.chunk_overlap)
            chunker_name = "flat"

        # M4.7 — Confidence labels in chunks.
        #
        # Stamp the coarse ``source_confidence`` on every chunk based on
        # (1) which chunker produced it and (2) doc-level extraction
        # signals (scanned PDF, encoding fallback, U+FFFD in payload).
        # This is a single-pass classification — chunkers stay pure and
        # every rule change is confined to
        # :func:`mcp_server.parsers.confidence.classify_chunk_confidence`.
        #
        # Backward compat: docs indexed before M4.7 will never have this
        # key on their chunk metadata, so search results for legacy
        # corpora return ``source_confidence=None`` at the orchestrator
        # layer without any storage migration.
        # Deferred import: confidence classifier ships in the advanced-chunking
        # follow-up PR. When the module is absent (default install), skip the
        # per-chunk confidence stamp — legacy behaviour, chunks stay unlabeled.
        try:
            from .parsers.confidence import classify_chunk_confidence
        except ImportError:
            classify_chunk_confidence = None

        source_path_str = str(filepath)
        for chunk in doc.chunks:
            if classify_chunk_confidence is None:
                continue
            confidence = classify_chunk_confidence(
                chunk_text=chunk.content,
                source_path=source_path_str,
                chunker_name=chunker_name,
                metadata=metadata,
            )
            chunk.metadata["source_confidence"] = confidence.value

        # Layer 3 — evidence marker. The <external_content> fence only lands in
        # the first and last chunk, so provenance is stamped on every chunk's
        # metadata instead. search_knowledge() surfaces it as `external_source`,
        # letting the consuming LLM weigh untrusted context accordingly.
        if external_marker is not None:
            for chunk in doc.chunks:
                chunk.metadata["external_source"] = True
                chunk.metadata["external_source_uri"] = external_marker[0]

        return doc

    @staticmethod
    def _should_exclude(path: Path, base_dir: Path, patterns: List[str]) -> bool:
        """Check if a path matches any exclude pattern.

        Uses fnmatch on the relative path (forward-slash normalized) and
        also checks each path component individually for simple name patterns.
        """
        if not patterns:
            return False

        try:
            rel = path.relative_to(base_dir)
        except ValueError:
            rel = path

        rel_str = str(rel).replace("\\", "/")

        for pattern in patterns:
            # Full relative path match (e.g., "docs/drafts/*.tmp")
            if fnmatch.fnmatch(rel_str, pattern):
                return True
            # Check each component (e.g., "node_modules" matches any/node_modules/deep)
            for part in rel.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def parse_directory(self, directory: Path = None) -> List[Document]:
        """Parse all supported files in a directory recursively.

        Symlinks are still followed (``followlinks=True``) so the documented
        "link my notes tree into documents/" workflow keeps working, but every
        directory *and* every file is containment-checked against ``directory``
        after full resolution. A link pointing outside the corpus — say an
        untrusted repo cloned into ``documents/`` that ships ``notes -> /etc``
        — is skipped with a warning instead of leaking host files into the
        index (CWE-59).

        Args:
            directory: Root to walk. Defaults to ``config.documents_dir``.

        Returns:
            List[Document]: Parsed documents that resolved inside ``directory``.
        """
        directory = Path(directory) if directory else config.documents_dir
        documents: List[Document] = []
        seen_dirs = set()
        supported = set(config.supported_formats)
        exclude = config.exclude_patterns

        for root, dirs, files in os.walk(directory, followlinks=True):
            real_root = os.path.realpath(root)
            if real_root in seen_dirs:
                dirs.clear()
                continue
            seen_dirs.add(real_root)

            # CWE-59 — a symlinked directory whose target escapes the corpus is
            # pruned here, before any file inside it is opened.
            if not is_path_within(directory, root):
                log.warning("[WARN] Skipping symlinked directory outside documents_dir: %s", root)
                dirs.clear()
                continue

            # Filter out excluded directories in-place (prevents os.walk from descending)
            if exclude:
                dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d, directory, exclude)]

            for fname in files:
                filepath = Path(root) / fname
                if filepath.suffix.lower() not in supported:
                    continue
                if exclude and self._should_exclude(filepath, directory, exclude):
                    continue
                # CWE-59 — os.walk lists symlinked *files* regardless of
                # followlinks, and open() would follow them. `root` is already
                # known contained, so a non-link file under it is contained by
                # construction; only links need the full resolve. That keeps
                # the common path at one cheap lstat instead of a realpath.
                if filepath.is_symlink() and not is_path_within(directory, filepath):
                    log.warning("[WARN] Skipping symlinked file outside documents_dir: %s", filepath)
                    continue
                try:
                    doc = self.parse_file(filepath)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    log.warning("[WARN] Failed to parse %s: %s", filepath, e)

        return documents

    # =========================================================================
    # Category detection
    # =========================================================================

    def _detect_category(self, filepath: Path) -> str:
        """Detect document category based on file path.

        Args:
            filepath: File whose category is being resolved.

        Returns:
            str: Matching category from ``config.category_mappings`` or
            the fallback ``"general"``.
        """
        try:
            rel_path = filepath.relative_to(config.documents_dir)
            path_str = str(rel_path).replace("\\", "/").lower()
        except ValueError:
            path_str = str(filepath).replace("\\", "/").lower()

        # Check category mappings in order (more specific first)
        for path_pattern, category in sorted(config.category_mappings.items(), key=lambda x: len(x[0]), reverse=True):
            if path_pattern in path_str:
                return category

        return "general"

    # =========================================================================
    # Keyword extraction
    # =========================================================================

    def _extract_keywords(self, content: str, category: str) -> List[str]:
        """Extract technical keywords from content.

        Args:
            content: Document text to scan.
            category: Resolved document category (unused today, kept for
                signature stability with downstream callers).

        Returns:
            list[str]: Sorted unique keywords covering security tools,
            CVE IDs, MITRE ATT&CK technique IDs, and (when few enough
            to be signal) IP addresses.
        """
        keywords = set()
        content_lower = content.lower()

        # Check against all keyword routes
        for route_category, route_keywords in config.keyword_routes.items():
            for keyword in route_keywords:
                if keyword.lower() in content_lower:
                    keywords.add(keyword.lower())

        # Extract additional technical terms
        # CVE patterns
        cve_pattern = r"CVE-\d{4}-\d{4,}"
        keywords.update(re.findall(cve_pattern, content, re.IGNORECASE))

        # MITRE ATT&CK patterns
        mitre_pattern = r"T\d{4}(?:\.\d{3})?"
        keywords.update(re.findall(mitre_pattern, content))

        # IP addresses
        ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        ips = re.findall(ip_pattern, content)
        if len(ips) <= 5:  # Only add if not too many (likely real targets)
            keywords.update(ips)

        # Common security tools mentioned
        security_tools = [
            "nmap",
            "burp",
            "metasploit",
            "wireshark",
            "hydra",
            "john",
            "hashcat",
            "gobuster",
            "nikto",
            "sqlmap",
            "nuclei",
            "ffuf",
            "bloodhound",
            "mimikatz",
            "responder",
            "crackmapexec",
            "impacket",
        ]
        for tool in security_tools:
            if tool in content_lower:
                keywords.add(tool)

        return sorted(list(keywords))

    # =========================================================================
    # Utilities
    # =========================================================================

    def _generate_id(self, filepath: Path) -> str:
        """Generate a stable document ID from path, mtime, and size.

        Args:
            filepath: File whose ID is being generated.

        Returns:
            str: 16-character lowercase hex slice of the SHA-256.
        """
        stat = filepath.stat()
        unique_str = f"{filepath}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]


# Convenience function
def parse_documents(directory: Path = None) -> List[Document]:
    """Parse all documents in ``directory`` using the default parser.

    Args:
        directory: Root to walk. Defaults to ``config.documents_dir``.

    Returns:
        list[Document]: Parsed documents contained inside ``directory``.
    """
    parser = DocumentParser()
    return parser.parse_directory(directory)


# ============================================================================
# Backward-compatibility shims for callers that historically imported the
# now-relocated private helpers directly (rare; kept out of ``__all__``).
# ============================================================================

_chunk_text = chunk_text
_chunk_markdown = chunk_markdown
_chunk_hierarchical = chunk_hierarchical
