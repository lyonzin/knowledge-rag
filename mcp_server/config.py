"""Configuration for Knowledge RAG System v4.0.0 — YAML-configurable"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

# ============================================================================
# BASE DIRECTORY RESOLUTION
# ============================================================================
# Priority: 1. KNOWLEDGE_RAG_DIR env var  2. Source checkout  3. Venv parent  4. CWD

_source_dir = Path(__file__).parent.parent


_SUPPORTED_SUFFIXES = frozenset(
    [
        ".md",
        ".txt",
        ".pdf",
        ".py",
        ".c",
        ".h",
        ".cpp",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".xml",
        ".docx",
        ".xlsx",
        ".pptx",
        ".csv",
        ".ipynb",
    ]
)


def _has_documents(path: Path) -> bool:
    """
    Check if path has a documents/ dir with actual supported files.

    Follows symlinks but enforces containment: symlinks pointing OUTSIDE
    documents_dir are skipped to prevent CWE-59 (symlink escape). Fixes the
    same class of bug that GUARD patched in ``ingestion.py`` during Fase 1.
    """
    docs_dir = path / "documents"
    if not docs_dir.exists():
        return False
    try:
        docs_root = docs_dir.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    for root, dirs, files in os.walk(docs_dir, followlinks=True):
        # Prune subdirectories that escape docs_root via symlink.
        try:
            resolved_root = Path(root).resolve(strict=False)
        except (OSError, RuntimeError):
            dirs[:] = []
            continue
        if not (resolved_root == docs_root or docs_root in resolved_root.parents):
            dirs[:] = []
            continue
        for f in files:
            candidate = Path(root) / f
            try:
                resolved_candidate = candidate.resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            if not (resolved_candidate == docs_root or docs_root in resolved_candidate.parents):
                continue
            if Path(f).suffix.lower() in _SUPPORTED_SUFFIXES:
                return True
    return False


def _venv_project_dir():
    """Detect project root from venv location (pip install from PyPI)."""
    candidates = [Path(sys.prefix), Path(sys.executable), Path(sys.executable).resolve()]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if parent.name in ("venv", ".venv", "env", ".env"):
                return parent.parent
    return None


def _is_project_root(path):
    """Check if path looks like a knowledge-rag project (has config or documents)."""
    if path is None:
        return False
    return (path / "config.yaml").exists() or (path / "config.example.yaml").exists() or _has_documents(path)


_venv_dir = _venv_project_dir()

if os.environ.get("KNOWLEDGE_RAG_DIR"):
    BASE_DIR = Path(os.environ["KNOWLEDGE_RAG_DIR"])
elif _venv_dir is not None and (_venv_dir / "config.yaml").exists():
    # Prefer venv parent if it has an actual config.yaml (editable installs, PyPI installs)
    BASE_DIR = _venv_dir
elif _is_project_root(_source_dir) and (_source_dir / "config.yaml").exists():
    BASE_DIR = _source_dir
elif _is_project_root(Path.cwd()):
    BASE_DIR = Path.cwd()
elif _is_project_root(_source_dir):
    BASE_DIR = _source_dir
elif _is_project_root(_venv_dir):
    BASE_DIR = _venv_dir
else:
    BASE_DIR = _venv_dir if _venv_dir is not None else Path.cwd()


# ============================================================================
# YAML CONFIG LOADER
# ============================================================================


def _load_yaml_config() -> dict:
    """Load config.yaml from BASE_DIR if it exists, otherwise return empty dict."""
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            print("[WARN] config.yaml is not a valid mapping, ignoring")
            return {}
        print(f"[INFO] Loaded config from {config_path}")
        return data
    except yaml.YAMLError as e:
        print(f"[WARN] Failed to parse config.yaml: {e} — using defaults")
        return {}


_yaml = _load_yaml_config()


def _get(section: str, key: str, default):
    """Get a value from the YAML config with section.key path, falling back to default."""
    s = _yaml.get(section, {})
    if not isinstance(s, dict):
        return default
    val = s.get(key)
    if val is None:
        return default
    # Skip type check when default is None (caller handles validation)
    if default is None:
        return val
    # YAML parses "yes"/"no" as bool, but explicit string "yes" stays str
    if not isinstance(val, type(default)):
        print(
            f"[WARN] config.yaml: {section}.{key} has wrong type "
            f"(expected {type(default).__name__}, got {type(val).__name__}), using default"
        )
        return default
    return val


def _get_top(key: str, default):
    """Get a top-level value from YAML, falling back to default if missing or None."""
    val = _yaml.get(key)
    if val is None:
        return default
    if not isinstance(val, type(default)):
        print(f"[WARN] config.yaml: {key} has wrong type, using default")
        return default
    return val


def _normalize_query_term(term: str) -> str:
    """Normalize a query-expansion term for consistent matching."""
    return term.strip().lower()


def _append_unique(values: List[str], value: str) -> None:
    """Append a normalized value while preserving order and removing duplicates."""
    if value and value not in values:
        values.append(value)


def _merge_query_expansion_sources(
    expansions: Dict[str, List[str]], expansion_groups: List[List[str]]
) -> Dict[str, List[str]]:
    """
    Merge legacy directional expansions and symmetric expansion groups.

    Legacy `query_expansions` entries are copied first. Then every
    `query_expansion_groups` entry contributes pairwise synonym links for all
    normalized terms in the group. Overlaps are merged by union while keeping
    insertion order stable.
    """
    merged: Dict[str, List[str]] = {}

    for raw_term, raw_synonyms in expansions.items():
        if not isinstance(raw_term, str):
            continue
        term = _normalize_query_term(raw_term)
        if not term:
            continue

        bucket = merged.setdefault(term, [])
        if isinstance(raw_synonyms, list):
            for synonym in raw_synonyms:
                if not isinstance(synonym, str):
                    continue
                _append_unique(bucket, _normalize_query_term(synonym))

    for raw_group in expansion_groups:
        if not isinstance(raw_group, list):
            continue

        group_terms: List[str] = []
        for raw_term in raw_group:
            if not isinstance(raw_term, str):
                continue
            _append_unique(group_terms, _normalize_query_term(raw_term))

        if len(group_terms) < 2:
            continue

        for term in group_terms:
            bucket = merged.setdefault(term, [])
            for related_term in group_terms:
                if related_term != term:
                    _append_unique(bucket, related_term)

    return merged


# ============================================================================
# DEFAULTS (used when no config.yaml or field is omitted)
# ============================================================================

_DEFAULT_CATEGORY_MAPPINGS = {
    "security/redteam": "redteam",
    "security/blueteam": "blueteam",
    "security/ctf": "ctf",
    "security": "security",
    "aar": "aar",
    "logscale": "logscale",
    "development": "development",
    "general": "general",
}

_DEFAULT_KEYWORD_ROUTES = {
    "logscale": [
        "logscale",
        "lql",
        "cql",
        "humio",
        "crowdstrike query",
        "formattime",
        "groupby",
        "base64decode",
        "case{}",
        "regex",
    ],
    "redteam": [
        "pentest",
        "exploit",
        "payload",
        "reverse shell",
        "privilege escalation",
        "lateral movement",
        "c2",
        "beacon",
        "cobalt strike",
        "metasploit",
        "gtfobins",
        "lolbas",
        "lolbin",
        "suid",
        "sudo",
        "byovd",
        "lol driver",
        "lolad",
        "lolapps",
        "hacktricks",
        "privesc",
        "kerberoast",
        "dcsync",
        "golden ticket",
        "pass-the-hash",
        "bloodhound",
        "mimikatz",
        "rubeus",
        "certipy",
        "adcs",
        "sqli",
        "xss",
        "ssti",
        "ssrf",
        "lfi",
        "rfi",
        "xxe",
        "deserialization",
        "ysoserial",
        "upload bypass",
        "web shell",
        "hash cracking",
        "hashcat",
        "waf bypass",
        "amsi bypass",
        "uac bypass",
        "potato",
        "searchsploit",
        "exploit-db",
        "cve",
    ],
    "blueteam": [
        "detection",
        "sigma",
        "yara",
        "ioc",
        "threat hunting",
        "incident response",
        "forensics",
        "malware analysis",
    ],
    "ctf": [
        "ctf",
        "flag",
        "hackthebox",
        "htb",
        "tryhackme",
        "picoctf",
        "writeup",
        "challenge",
    ],
    "development": [
        "python",
        "typescript",
        "javascript",
        "api",
        "fastapi",
        "django",
        "react",
        "nodejs",
    ],
    "security": [
        "anti-bot",
        "antibot",
        "js challenge",
        "javascript challenge",
        "cdp detection",
        "runtime.enable",
        "puppeteer",
        "playwright",
        "selenium",
        "nodriver",
        "stealth",
        "undetected",
        "ja3",
        "ja4",
        "tls fingerprint",
        "fingerprinting",
        "curl_cffi",
        "got-scraping",
        "impersonate",
        "http/2 settings",
        "browser fingerprint",
        "canvas fingerprint",
        "webgl fingerprint",
        "navigator.webdriver",
        "audio context",
        "hardware concurrency",
        "waf bypass",
        "aws waf",
        "cloudflare bypass",
        "akamai bypass",
        "datadome",
        "perimeterx",
        "imperva bypass",
        "8kb bypass",
        "body size limit",
        "json sqli",
        "behavioral",
        "mouse movement",
        "ghost-cursor",
        "humanized",
        "flaresolverr",
        "turnstile",
        "rebrowser",
        "botbrowser",
    ],
}

_DEFAULT_QUERY_EXPANSIONS = {
    "sqli": ["sql injection", "sqli"],
    "sql injection": ["sql injection", "sqli"],
    "xss": ["cross-site scripting", "xss"],
    "cross-site scripting": ["cross-site scripting", "xss"],
    "ssrf": ["server-side request forgery", "ssrf"],
    "lfi": ["local file inclusion", "lfi"],
    "rfi": ["remote file inclusion", "rfi"],
    "rce": ["remote code execution", "rce"],
    "xxe": ["xml external entity", "xxe"],
    "ssti": ["server-side template injection", "ssti"],
    "idor": ["insecure direct object reference", "idor"],
    "csrf": ["cross-site request forgery", "csrf"],
    "privesc": ["privilege escalation", "privesc"],
    "priv esc": ["privilege escalation", "privesc"],
    "privilege escalation": ["privilege escalation", "privesc"],
    "deserialization": ["deserialization", "deserialisation", "insecure deserialization"],
    "pth": ["pass-the-hash", "pth"],
    "pass-the-hash": ["pass-the-hash", "pth"],
    "dcsync": ["dcsync", "dc sync", "domain controller sync"],
    "kerberoast": ["kerberoasting", "kerberoast"],
    "kerberoasting": ["kerberoasting", "kerberoast"],
    "asrep": ["as-rep roasting", "asrep", "asreproast"],
    "bloodhound": ["bloodhound", "sharphound"],
    "mimikatz": ["mimikatz", "sekurlsa", "logonpasswords"],
    "hashcat": ["hashcat", "hash cracking", "hash crack"],
    "john": ["john the ripper", "john", "jtr"],
    "revshell": ["reverse shell", "revshell", "rev shell"],
    "reverse shell": ["reverse shell", "revshell"],
    "webshell": ["web shell", "webshell"],
    "web shell": ["web shell", "webshell"],
    "waf": ["web application firewall", "waf"],
    "amsi": ["antimalware scan interface", "amsi", "amsi bypass"],
    "uac": ["user account control", "uac", "uac bypass"],
    "potato": ["potato", "juicypotato", "sweetpotato", "godpotato", "efspotato", "printspoofer"],
    "ntlm": ["ntlm", "net-ntlmv2", "ntlmv2"],
    "smb": ["smb", "server message block", "samba"],
    "ldap": ["ldap", "lightweight directory access protocol"],
    "ad": ["active directory", "ad"],
    "active directory": ["active directory", "ad"],
    "defender": ["windows defender", "defender", "wdfilter"],
    "responder": ["responder", "llmnr", "nbt-ns", "netbios"],
    "suid": ["suid", "setuid", "set-uid"],
    "cron": ["cron", "crontab", "cronjob", "scheduled task"],
    "lolbin": ["lolbin", "lolbas", "living off the land"],
    "c2": ["c2", "command and control", "command-and-control", "beacon"],
    "sliver": ["sliver", "sliver c2"],
    "cobalt": ["cobalt strike", "cobalt", "cs beacon"],
    "phishing": ["phishing", "spearphishing", "social engineering"],
    "forensics": ["forensics", "forensic", "dfir"],
    "volatility": ["volatility", "memory forensics", "memory analysis"],
    "steganography": ["steganography", "stego", "steghide"],
    "stego": ["steganography", "stego", "steghide"],
    "rbcd": ["resource-based constrained delegation", "rbcd"],
    "dpapi": ["dpapi", "data protection api", "credential manager"],
    "printnightmare": ["printnightmare", "cve-2021-34527", "spoolsv", "printspooler"],
    "cve-2021-34527": ["printnightmare", "cve-2021-34527", "spoolsv"],
    "eternalblue": ["eternalblue", "ms17-010", "smbv1"],
    "ms17-010": ["eternalblue", "ms17-010", "smbv1"],
    "pwnkit": ["pwnkit", "cve-2021-4034", "pkexec"],
    "cve-2021-4034": ["pwnkit", "cve-2021-4034", "pkexec"],
    "log4shell": ["log4shell", "cve-2021-44228", "log4j"],
    "cve-2021-44228": ["log4shell", "cve-2021-44228", "log4j"],
    "zerologon": ["zerologon", "cve-2020-1472", "netlogon"],
    "cve-2020-1472": ["zerologon", "cve-2020-1472", "netlogon"],
    "petitpotam": ["petitpotam", "cve-2021-36942", "efs", "ntlm relay"],
    "certifried": ["certifried", "cve-2022-26923", "adcs"],
    "nopac": ["nopac", "samaccountname", "cve-2021-42278", "cve-2021-42287"],
    "proxylogon": ["proxylogon", "cve-2021-26855", "exchange"],
    "proxyshell": ["proxyshell", "cve-2021-34473", "exchange"],
}

_DEFAULT_QUERY_EXPANSION_GROUPS: List[List[str]] = []


# ============================================================================
# CONFIG DATACLASS
# ============================================================================


def _resolve_path(raw, default: Path) -> Path:
    """Resolve a path from YAML (string) or use default (Path).

    Expands ``~`` to the user home directory on all platforms
    (Linux/macOS: $HOME, Windows: %USERPROFILE%).
    """
    if raw is None:
        return default
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


@dataclass
class Config:
    """Central configuration for the RAG system — loads from config.yaml when available."""

    # Paths
    data_dir: Path = field(default_factory=lambda: _resolve_path(_get("paths", "data_dir", None), BASE_DIR / "data"))
    chroma_dir: Path = field(
        default_factory=lambda: _resolve_path(_get("paths", "data_dir", None), BASE_DIR / "data") / "chroma_db"
    )
    documents_dir: Path = field(
        default_factory=lambda: _resolve_path(_get("paths", "documents_dir", None), BASE_DIR / "documents")
    )
    models_cache_dir: Path = field(
        default_factory=lambda: _resolve_path(_get("paths", "models_cache_dir", None), BASE_DIR / "models_cache")
    )

    # Chunking
    chunk_size: int = field(
        default_factory=lambda: (
            _get("documents", "chunking", {}).get("chunk_size", 1000)
            if isinstance(_get("documents", "chunking", {}), dict)
            else 1000
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: (
            _get("documents", "chunking", {}).get("chunk_overlap", 200)
            if isinstance(_get("documents", "chunking", {}), dict)
            else 200
        )
    )

    # Code-aware chunking (A3.8) — DEFAULT OFF.
    #
    # When True, source files (.py, .js, .ts, .c, .cpp, .rs, .go, .java, .rb,
    # .php, ...) are chunked at function/class boundaries via tree-sitter
    # instead of the default fixed-size window. This requires the optional
    # `[code]` extra (``pip install knowledge-rag[code]``); without it the
    # chunker logs a single warning and falls back to fixed-size — no crash.
    #
    # Turning this on requires a reindex to take effect (chunks are stored
    # on disk in ChromaDB). Trade-offs are documented in docs/configuration.md.
    code_aware_chunking: bool = field(default_factory=lambda: _get("documents", "code_aware_chunking", False))
    #: Per-chunk character ceiling used by the AST-boundary chunker. Chunks
    #: larger than this (a very long function) are further split via the
    #: fixed-size fallback so no chunk exceeds the ingestion budget.
    code_aware_max_chunk_size: int = field(default_factory=lambda: _get("documents", "code_aware_max_chunk_size", 1500))

    # Contextual chunking (A3.6 — Anthropic 2024) — DEFAULT OFF.
    #
    # When True, every chunk gets a 1-2 sentence LLM-generated context
    # PREPENDED to its content before embedding + indexing. Anthropic
    # reports up to +49% retrieval recall improvement. This feature is
    # EXPENSIVE (one LLM call per chunk at ingestion time) and requires a
    # configured/auto-detectable LLM provider — turning the flag on with
    # no provider silently degrades to raw chunks (one WARN log per
    # ingested document) so retrieval never breaks.
    #
    # The semantic cache (``[semantic_cache]``) is the single control that
    # keeps re-indexing an unchanged corpus free of extra LLM cost. Do
    # NOT disable the semantic cache while this feature is on unless you
    # are certain the corpus never re-indexes.
    #
    # Precedence at ingestion time:
    #   parent_document_enabled > code_aware_chunking > contextual_chunking > late_chunking > markdown > flat
    #
    # Enabling requires a full reindex to take effect (chunks are stored
    # on disk in ChromaDB). Trade-offs documented in docs/configuration.md.
    contextual_chunking_enabled: bool = field(default_factory=lambda: _get("documents", "contextual_chunking", False))

    # Late chunking (R5.7 — Jina AI 2024) — DEFAULT OFF.
    #
    # When True, each document is first embedded IN FULL with a
    # long-context embedding model (jina-embeddings-v3 8K), then chunk
    # embeddings are produced by mean-pooling the token vectors within
    # each chunk's character span. Result: every chunk embedding carries
    # global document context (surrounding paragraphs bias the vector)
    # instead of being embedded in isolation. Jina reports 5-15% recall
    # improvement on BEIR depending on doc length.
    #
    # Requires a token-aware embedding provider — currently only
    # ``jina-v3`` (behind the ``[embed-jina]`` extra). When the flag is
    # on but the configured provider does not implement ``embed_tokens``,
    # the chunker logs a single WARN and falls back to the byte-identical
    # fixed-size chunker + standard embedding path — ingestion never
    # breaks. See :mod:`mcp_server.parsers.late_chunker` for the full
    # fail-open contract.
    #
    # Precedence at ingestion time:
    #   parent_document_enabled > code_aware_chunking > contextual_chunking > late_chunking > markdown > flat
    #
    # Enabling requires a full reindex to take effect (chunks are stored
    # on disk in ChromaDB with their long-context vectors). Trade-offs
    # documented in docs/configuration.md.
    late_chunking_enabled: bool = field(default_factory=lambda: _get("documents", "late_chunking", False))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("model", "BAAI/bge-small-en-v1.5")
            if isinstance(_get("models", "embedding", {}), dict)
            else "BAAI/bge-small-en-v1.5"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("dimensions", 384)
            if isinstance(_get("models", "embedding", {}), dict)
            else 384
        )
    )
    gpu_acceleration: bool = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("gpu", False)
            if isinstance(_get("models", "embedding", {}), dict)
            else False
        )
    )

    # Reranker
    reranker_model: str = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("model", "Xenova/ms-marco-MiniLM-L-6-v2")
            if isinstance(_get("models", "reranker", {}), dict)
            else "Xenova/ms-marco-MiniLM-L-6-v2"
        )
    )
    reranker_enabled: bool = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("enabled", True)
            if isinstance(_get("models", "reranker", {}), dict)
            else True
        )
    )
    reranker_top_k_multiplier: int = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("top_k_multiplier", 3)
            if isinstance(_get("models", "reranker", {}), dict)
            else 3
        )
    )

    # ChromaDB
    collection_name: str = field(default_factory=lambda: _get("search", "collection_name", "knowledge_base"))

    # Provider registry (new in A2.2 — Fase 2 refactor)
    # These select bundled or third-party providers registered under the
    # ``knowledge_rag.embeddings`` / ``knowledge_rag.vector_stores`` /
    # ``knowledge_rag.rerankers`` entry-point groups. Defaults keep the
    # historical stack (FastEmbed + ChromaDB + ms-marco cross-encoder) so
    # users without custom config observe no behaviour change.
    embedding_provider: str = field(
        default_factory=lambda: (
            _get("models", "embedding", {}).get("provider", "fastembed")
            if isinstance(_get("models", "embedding", {}), dict)
            else "fastembed"
        )
    )

    # ── Matryoshka embedding knobs (R5.4 — Fase 5 opt-in) ────────────────
    #
    # DEFAULT: nomic-ai/nomic-embed-text-v1.5 @ 512D. These two fields are
    # ONLY consumed when ``embedding_provider == "matryoshka"`` — the
    # standard FastEmbed path never touches them. Validation is deferred
    # to :class:`mcp_server.providers.embeddings.matryoshka.MatryoshkaEmbedding.
    # _validate_dimension` so an unsupported slice raises at provider
    # construction (with a clear list of valid dimensions for the chosen
    # model) rather than corrupting the vector store on first insert.
    #
    # Switching model OR dimension mid-corpus is destructive: existing
    # ChromaDB embeddings were built against the previous shape and will
    # crash on shape mismatch. A full ``reindex_documents(full_rebuild=
    # True)`` is required — the trade-off is documented under
    # "Matryoshka embeddings" in docs/configuration.md.
    matryoshka_model: str = field(
        default_factory=lambda: _get("models", "matryoshka_model", "nomic-ai/nomic-embed-text-v1.5")
    )
    matryoshka_dimension: int = field(default_factory=lambda: _get("models", "matryoshka_dimension", 512))
    vector_store: str = field(
        default_factory=lambda: (
            _get("models", "vector_store", {}).get("provider", "chromadb")
            if isinstance(_get("models", "vector_store", {}), dict)
            else "chromadb"
        )
    )
    reranker_provider: str = field(
        default_factory=lambda: (
            _get("models", "reranker", {}).get("provider", "cross_encoder")
            if isinstance(_get("models", "reranker", {}), dict)
            else "cross_encoder"
        )
    )

    # ── ColBERT / multi-vector retrieval (R5.3 — Fase 5 opt-in) ─────────────
    #
    # DEFAULT OFF. Only consulted when ``vector_store == "colbert"``. The
    # ColBERT provider lazily reads this field the first time it opens
    # its PLAID index so a stale value on a ChromaDB-only install has
    # zero effect. When left blank the provider falls back to the
    # ``colbert-ir/colbertv2.0`` default (~360 MB checkpoint) — the same
    # baseline documented in :mod:`mcp_server.providers.vector_stores.
    # colbert`. Overriding here (``models.colbert_model: <hf-id-or-path>``)
    # lets a user pin a domain-tuned checkpoint without editing code.
    colbert_model: str = field(default_factory=lambda: _get("models", "colbert_model", "colbert-ir/colbertv2.0"))

    # Supported formats
    supported_formats: List[str] = field(
        default_factory=lambda: _get(
            "documents",
            "supported_formats",
            [
                ".md",
                ".txt",
                ".pdf",
                ".py",
                ".c",
                ".h",
                ".cpp",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".json",
                ".xml",
                ".docx",
                ".xlsx",
                ".pptx",
                ".csv",
                ".ipynb",
            ],
        )
    )

    # Exclude patterns for directory traversal
    exclude_patterns: List[str] = field(default_factory=lambda: _get("documents", "exclude_patterns", []))

    # Category mappings
    category_mappings: Dict[str, str] = field(
        default_factory=lambda: _get_top("category_mappings", _DEFAULT_CATEGORY_MAPPINGS)
    )

    # Keyword routes
    keyword_routes: Dict[str, List[str]] = field(
        default_factory=lambda: _get_top("keyword_routes", _DEFAULT_KEYWORD_ROUTES)
    )

    # Query expansions
    query_expansions: Dict[str, List[str]] = field(
        default_factory=lambda: _get_top("query_expansions", _DEFAULT_QUERY_EXPANSIONS)
    )

    # Symmetric query expansion groups
    query_expansion_groups: List[List[str]] = field(
        default_factory=lambda: _get_top("query_expansion_groups", _DEFAULT_QUERY_EXPANSION_GROUPS)
    )

    # Search settings
    default_results: int = field(default_factory=lambda: _get("search", "default_results", 5))
    max_results: int = field(default_factory=lambda: _get("search", "max_results", 20))

    # Fusion strategy (new in A2.4)
    # ``rrf`` (default) preserves the historical hardcoded K=60 fusion.
    # Alternatives: ``combsum``, ``combmnz``, ``weighted``. Configured under
    # ``search.fusion`` in config.yaml, e.g.:
    #   search:
    #     fusion:
    #       strategy: weighted
    #       weights: {semantic: 0.7, bm25: 0.3}
    # ``weights`` is only consumed by the ``weighted`` strategy.
    fusion_strategy: str = field(
        default_factory=lambda: (
            _get("search", "fusion", {}).get("strategy", "rrf")
            if isinstance(_get("search", "fusion", {}), dict)
            else "rrf"
        )
    )
    fusion_weights: Dict[str, float] = field(
        default_factory=lambda: (
            _get("search", "fusion", {}).get("weights", {}) if isinstance(_get("search", "fusion", {}), dict) else {}
        )
    )

    # MMR (Maximal Marginal Relevance) diversification
    # When ``mmr_enabled`` is True and the candidate pool is larger than the
    # requested top-k, the orchestrator reorders the pool with embedding-based
    # cosine MMR to reduce near-duplicate hits. ``mmr_lambda`` blends relevance
    # (1.0) and diversity (0.0); 0.7 is relevance-biased with mild diversity.
    mmr_enabled: bool = field(default_factory=lambda: _get("search", "mmr_enabled", True))
    mmr_lambda: float = field(default_factory=lambda: _get("search", "mmr_lambda", 0.7))

    # Server (new in v4.0.0)
    transport: str = field(default_factory=lambda: _get("server", "transport", "stdio"))
    server_host: str = field(default_factory=lambda: _get("server", "host", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _get("server", "port", 8179))
    auth_bearer_token: str = field(
        default_factory=lambda: (
            _get("server", "auth", {}).get("bearer_token", "") if isinstance(_get("server", "auth", {}), dict) else ""
        )
    )
    rate_limit_enabled: bool = field(
        default_factory=lambda: (
            _get("server", "rate_limit", {}).get("enabled", False)
            if isinstance(_get("server", "rate_limit", {}), dict)
            else False
        )
    )
    rate_limit_rpm: int = field(
        default_factory=lambda: (
            _get("server", "rate_limit", {}).get("requests_per_minute", 60)
            if isinstance(_get("server", "rate_limit", {}), dict)
            else 60
        )
    )
    rate_limit_burst: int = field(
        default_factory=lambda: (
            _get("server", "rate_limit", {}).get("burst", 10)
            if isinstance(_get("server", "rate_limit", {}), dict)
            else 10
        )
    )
    metrics_enabled: bool = field(
        default_factory=lambda: (
            _get("server", "metrics", {}).get("enabled", False)
            if isinstance(_get("server", "metrics", {}), dict)
            else False
        )
    )
    metrics_port: int = field(
        default_factory=lambda: (
            _get("server", "metrics", {}).get("port", 9179) if isinstance(_get("server", "metrics", {}), dict) else 9179
        )
    )

    # ── Telemetry (A2.6) ─────────────────────────────────────────────────────
    # OpenTelemetry tracing + structured JSON logs. Both fields default off so
    # the pre-A2.6 output shape is preserved for anyone who never edits
    # ``config.yaml``. Tracing requires the ``[otel]`` extra; without it the
    # server falls back to a no-op tracer with a warning.
    telemetry_enabled: bool = field(default_factory=lambda: _get("telemetry", "enabled", False))
    telemetry_exporter: str = field(default_factory=lambda: _get("telemetry", "exporter", "otlp"))
    telemetry_service_name: str = field(default_factory=lambda: _get("telemetry", "service_name", "knowledge-rag"))
    telemetry_otlp_endpoint: str = field(default_factory=lambda: _get("telemetry", "otlp_endpoint", ""))
    telemetry_json_logs: bool = field(default_factory=lambda: _get("telemetry", "json_logs", False))
    telemetry_log_level: str = field(default_factory=lambda: _get("telemetry", "log_level", "INFO"))

    # Multilingual query stopwords (Q1.3). Filters question words + connectives
    # from BM25 tokenization while preserving security-domain PROTECTED_TERMS
    # (os command injection, com hijacking, des cipher, union select from, ...).
    # Empty list [] disables filtering entirely (explicit opt-out). A list of
    # only invalid codes falls back to ["en", "pt"] to avoid silent disabling.
    # Semantic search always sees the raw query — only BM25 and query
    # expansion consume the filtered form.
    stopword_languages: List[str] = field(default_factory=lambda: _get("search", "stopword_languages", ["en", "pt"]))

    # ── Parent Document Retrieval / Small-to-Big (A3.7) ─────────────────────
    # Opt-in hierarchical chunking: index fine-grained "small" chunks for
    # precise retrieval, but stamp each with the "large" parent chunk it
    # belongs to. At query time the orchestrator swaps the retrieved small
    # chunk's content for the parent's, giving the caller broader context
    # without hurting recall. LangChain/LlamaIndex "Small-to-Big" pattern.
    #
    # DEFAULT OFF — behaviour is byte-identical to the pre-A3.7 flat
    # chunker unless the user opts in via config.yaml AND reindexes. The
    # feature adds no LLM call, no new dependency, and no cost when
    # disabled. Enabling requires a manual reindex so pre-existing corpora
    # are never rewritten by surprise.
    parent_document_enabled: bool = field(default_factory=lambda: _get("search", "parent_document_enabled", False))
    parent_document_large_size: int = field(default_factory=lambda: _get("search", "parent_document_large_size", 1500))
    parent_document_small_size: int = field(default_factory=lambda: _get("search", "parent_document_small_size", 250))
    parent_document_small_overlap: int = field(
        default_factory=lambda: _get("search", "parent_document_small_overlap", 50)
    )

    # Semantic cache for LLM-in-loop features (A2.7 — infra prep for Fase 3).
    # Not used by any current retrieval path — LLM features (HyDE / Multi-Query
    # / Query Rewriting / Contextual Chunking) will register prompt fingerprints
    # here when they land. Cache lives at ``data/cache/semantic/`` by default.
    semantic_cache_enabled: bool = field(default_factory=lambda: _get("semantic_cache", "enabled", True))
    semantic_cache_dir: str = field(default_factory=lambda: _get("semantic_cache", "cache_dir", "data/cache/semantic"))
    semantic_cache_ttl_days: int = field(default_factory=lambda: _get("semantic_cache", "ttl_days", 30))
    semantic_cache_max_entries: int = field(default_factory=lambda: _get("semantic_cache", "max_entries", 10000))

    # ── LLM provider (A3.1 — Fase 3 infra) ──────────────────────────────────
    #
    # DEFAULT: none — no LLM calls are made by knowledge-rag on any code
    # path. The four fields below only take effect once an opt-in Fase 3
    # feature (HyDE, Multi-Query, Query Rewriting, Self-Query, Contextual
    # Chunking) is enabled elsewhere in the config AND actually invoked.
    #
    # ``llm_provider`` accepts:
    #   * ``""`` (empty)  → auto-detect on first feature use. The registry
    #                       walks its canonical order (anthropic, openai,
    #                       gemini, deepseek, ollama, openai_compat) and
    #                       picks the first provider whose credentials +
    #                       SDK are present. Returns ``None`` when none
    #                       are — the calling feature stays disabled.
    #   * a registered provider name (see
    #     ``mcp_server.providers.llm.available_llms()`` and
    #     ``pip install knowledge-rag[llm-<name>]`` for extras). An
    #     unknown name is accepted here and only fails when a feature
    #     actually calls ``get_llm(config.llm_provider)`` — the check
    #     is deferred so a typo does not brick server startup.
    #
    # ``llm_model``      : vendor model id (empty ⇒ provider's default)
    # ``llm_max_tokens`` : upper bound on generated tokens per call
    # ``llm_temperature``: sampling temperature (0.0 = deterministic)
    # ``llm_timeout_seconds``: network timeout (Ollama on CPU is slow)
    llm_provider: str = field(default_factory=lambda: _get("llm", "provider", ""))
    llm_model: str = field(default_factory=lambda: _get("llm", "model", ""))
    llm_max_tokens: int = field(default_factory=lambda: _get("llm", "max_tokens", 1024))
    llm_temperature: float = field(default_factory=lambda: _get("llm", "temperature", 0.0))
    llm_timeout_seconds: int = field(default_factory=lambda: _get("llm", "timeout_seconds", 30))

    # ── Query Rewriting via LLM (A3.2 — Fase 3 feature) ─────────────────────
    #
    # DEFAULT OFF. When True, ``retrieval.orchestrator.query()`` routes the
    # raw user query through ``retrieval.llm_features.query_rewrite.rewrite_
    # query`` before tokenisation. The rewritten form (or the raw query, on
    # any LLM failure) is what BM25 and the semantic embedder actually see.
    #
    # Toggle is intentionally isolated in its own ``llm_features`` YAML
    # section so future Fase 3 opt-ins (HyDE, Multi-Query, Self-Query,
    # Contextual Chunking) land as siblings without polluting the ``llm``
    # section — that section stays scoped to provider credentials + knobs.
    #
    # Enabling this without a resolvable LLM provider is safe: the feature
    # fails open at runtime (WARN log, original query returned), so a stale
    # ``True`` after credential rotation never bricks retrieval. The startup
    # path deliberately does NOT probe the provider registry from
    # __post_init__ because config.py is imported before the LLM subpackage
    # is initialised — the provider-availability check belongs at feature
    # invocation time, not at server boot.
    query_rewrite_enabled: bool = field(default_factory=lambda: _get("llm_features", "query_rewrite", False))

    # ── HyDE — Hypothetical Document Embeddings (A3.3 — Fase 3 feature) ─────
    #
    # DEFAULT OFF. When True, ``retrieval.orchestrator.query()`` calls
    # ``retrieval.llm_features.hyde.apply_hyde`` BEFORE semantic search: the
    # LLM writes N short passages that would answer the query, each is
    # embedded, and the mean of those embeddings PLUS the raw query
    # embedding is what ChromaDB actually searches against. BM25 keeps using
    # the raw query text — only the dense branch benefits from HyDE (and
    # this is on purpose; sparse retrieval on hypothetical text tends to
    # hurt exact-term matching).
    #
    # Reference: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval
    # without Relevance Labels" (https://arxiv.org/abs/2212.10496).
    #
    # ``hyde_num_hypotheses`` controls the size of the passage bag. ``1``
    # is deterministic (temperature 0.0, one cache hit per query). ``2+``
    # uses higher sampling temperature so the bag is genuinely diverse —
    # a bag of identical copies collapses back to the raw query embedding.
    # Cost scales linearly with N (N LLM calls + N embed calls per query).
    #
    # Same fail-open contract as ``query_rewrite_enabled``: any failure
    # (missing provider, LLM error, embedding failure) collapses to the
    # default ``ChromaDB.query(query_texts=[query])`` path — retrieval
    # never breaks. Enabling this without a resolvable LLM provider is safe.
    hyde_enabled: bool = field(default_factory=lambda: _get("llm_features", "hyde", False))
    hyde_num_hypotheses: int = field(default_factory=lambda: _get("llm_features", "hyde_num_hypotheses", 1))

    # ── Self-Query via LLM (A3.5 — Fase 3 feature) ──────────────────────────
    #
    # DEFAULT OFF. When True, ``retrieval.orchestrator.query()`` routes the
    # raw user query through ``retrieval.llm_features.self_query.extract_
    # filters`` before running retrieval. The LLM extracts structured
    # filters (category, source substring, date range) from natural language
    # so the ChromaDB metadata where-clause can narrow the candidate set
    # *before* semantic + BM25 do their work. Only fires when the caller
    # did NOT pass ``category_filter`` explicitly — explicit user intent
    # always wins over inferred intent, no exceptions.
    #
    # Same fail-open contract as ``query_rewrite_enabled``: any failure
    # (missing provider, malformed JSON, unknown category, malformed
    # date) collapses to the fallback (raw query + all filters None) and
    # retrieval runs as if the flag were off. Enabling this without a
    # resolvable LLM provider is safe.
    self_query_enabled: bool = field(default_factory=lambda: _get("llm_features", "self_query", False))

    # ── Multi-Query fan-out via LLM (A3.4 — Fase 3 feature) ─────────────────
    #
    # DEFAULT OFF. When True and ``multi_query_n > 1``,
    # ``retrieval.orchestrator.query()`` asks the configured LLM for N-1
    # alternative formulations of the raw query, runs an independent
    # semantic + BM25 pass per variation (including the original), and
    # fuses the N per-variation candidate sets via a top-level RRF (K=60)
    # BEFORE reranking / MMR / adjacent-chunk expansion. This trades N x
    # retrieval cost for improved recall on ambiguous or terminology-rich
    # queries — the classic use case is a user query that misses the
    # exact vocabulary the corpus was written in.
    #
    # ``multi_query_n`` is the TOTAL query count including the original,
    # not the number of extra variations. ``n=1`` is a no-op (single-query
    # semantics, no LLM call). Values above the internal cap in
    # ``retrieval.llm_features.multi_query`` are silently clipped so a
    # runaway config never explodes the LLM budget or the retrieval
    # latency budget.
    #
    # Same fail-open contract as ``query_rewrite_enabled`` and
    # ``hyde_enabled``: any failure (no provider, LLM error, empty
    # response) collapses to the pre-A3.4 single-query path — retrieval
    # never breaks. Enabling this without a resolvable LLM provider is
    # safe; the feature simply short-circuits on every call.
    multi_query_enabled: bool = field(default_factory=lambda: _get("llm_features", "multi_query", False))
    multi_query_n: int = field(default_factory=lambda: _get("llm_features", "multi_query_n", 3))

    # ── Global cross-corpus index (M4.3 — Fase 4 feature) ───────────────────
    #
    # DEFAULT ON — but the feature only fires when at least one corpus has
    # been registered via ``knowledge-rag global add <path> --as <tag>``, so
    # a user with a single corpus continues seeing byte-identical behaviour.
    # The toggle exists so operators can hard-disable the ``search_global``
    # MCP tool + the ``global`` CLI subcommand without deleting the
    # registry (useful in kiosk deployments where cross-corpus queries are
    # policy-blocked).
    #
    # The registry itself lives at ``~/.knowledge-rag/global/registry.json``
    # and is NOT part of ``config.yaml`` — it is per-user global state, not
    # per-project. When ``global_index_enabled`` is False, ``search_global``
    # returns a clear ``status="error"`` payload and the ``global``
    # CLI subcommands exit non-zero with an actionable message.
    global_index_enabled: bool = field(default_factory=lambda: _get("global_index", "enabled", True))

    # ── Adaptive Retrieval router via LLM (A3.9 — Fase 3 feature) ───────────
    #
    # DEFAULT OFF. When True, ``retrieval.orchestrator.query()`` asks the
    # configured LLM to pick a retrieval strategy per query (simple /
    # hybrid / multi_hop / code / filter) and applies the per-strategy
    # overrides in ``retrieval.llm_features.adaptive.ROUTE_TO_FLAGS``
    # BEFORE running the pipeline. The router's effect is compositional:
    # it can flip on HyDE (A3.3), Multi-Query (A3.4), Self-Query (A3.5)
    # and Code-Aware chunking (A3.8) for a single query without editing
    # the operator's static config.
    #
    # Feature is architecturally last-of-value: the flag dicts it emits
    # only bite when the sibling Fase 3 features are enabled elsewhere.
    # Enabling ``adaptive_retrieval`` in isolation still exercises the
    # ``hybrid_alpha`` override — the other hints become no-ops the
    # orchestrator can log but not act on.
    #
    # Same fail-open contract as the other Fase 3 features: any failure
    # (no provider, malformed strategy, LLM error) collapses the router
    # output to ``"hybrid"`` — the standard dense+sparse path — and
    # retrieval never breaks. Enabling this without a resolvable LLM
    # provider is safe; every query short-circuits to ``"hybrid"`` and
    # the semantic cache never grows.
    #
    # Per-call override lives on ``orchestrator.query`` and the exposed
    # ``adaptive`` param on the ``search_knowledge`` MCP tool + the
    # ``--adaptive`` / ``--no-adaptive`` CLI flag. Explicit user intent
    # wins over the config toggle in both directions.
    adaptive_retrieval_enabled: bool = field(default_factory=lambda: _get("llm_features", "adaptive", False))

    # ── Work Memory / Lessons Learned Loop (M4.2 — Fase 4 feature) ──────────
    #
    # DEFAULT OFF. When enabled, the two new MCP tools ``save_result`` and
    # ``reflect`` become active. ``save_result`` persists retrieval outcomes
    # (useful / dead_end / corrected) as YAML-frontmatter Markdown files
    # under ``work_memory_dir``; ``reflect`` aggregates every entry into a
    # time-decayed corroboration score, classifies each doc as
    # preferred/tentative/contested/dead_end/corrected and emits a sidecar
    # ``.knowledge_rag_learning.json`` overlay. The orchestrator's
    # ``query()`` reads the overlay and stamps each result with a
    # ``learning`` field so the LLM caller can weight docs across sessions.
    #
    # Same fail-safe posture as every other opt-in feature: with the flag
    # off, both new MCP tools short-circuit with a "disabled" status, no
    # files are read or written, and ``search_knowledge`` output is
    # byte-identical to pre-M4.2. Turning the flag on without any saved
    # entries is safe — the overlay is empty, so the ``learning`` field
    # simply never appears on results.
    #
    # ``work_memory_dir`` is resolved relative to ``BASE_DIR`` when
    # relative, absolute when starts with a drive/root. The overlay JSON
    # is written to ``BASE_DIR / .knowledge_rag_learning.json``, sibling
    # to the data/ tree — this keeps the learned preferences out of the
    # index (so wiping ChromaDB does NOT erase them) while still being
    # trivially discoverable next to the config.
    work_memory_enabled: bool = field(default_factory=lambda: _get("work_memory", "enabled", False))
    work_memory_dir: str = field(default_factory=lambda: _get("work_memory", "dir", "data/memory"))
    work_memory_half_life_days: int = field(default_factory=lambda: _get("work_memory", "half_life_days", 30))
    work_memory_min_corroboration: int = field(default_factory=lambda: _get("work_memory", "min_corroboration", 2))

    # ── Dashboard analytics + query log (M4.4 — Fase 4 feature) ─────────────
    #
    # DEFAULT OFF. Both fields power the opt-in ``list_dashboard`` MCP tool
    # and the sibling ``knowledge-rag dashboard`` CLI subcommand. Nothing in
    # the retrieval path reads or writes these fields until an operator
    # explicitly opts in via ``dashboard.query_log: true`` in config.yaml.
    #
    # ``query_log_enabled`` — when True, ``KnowledgeOrchestrator.query``
    # appends one JSONL line per call (raw query + returned doc sources)
    # into ``query_log_dir`` so downstream analytics (``docs_never_queried``,
    # ``docs_high_volume``) can answer "which docs is anyone actually
    # reading?" Log write is best-effort — a full disk, a permission error
    # or a torn append never breaks retrieval; the exception is swallowed
    # and the search returns as usual. Byte-identical to the pre-M4.4
    # pipeline when this flag stays False (zero I/O, zero code path taken).
    #
    # ``query_log_dir`` — relative paths anchor at ``data_dir.parent``, same
    # semantics as ``semantic_cache_dir``. ``~`` is expanded. The directory
    # is created lazily on first write; nothing here materialises it on
    # startup so an unused feature leaves the filesystem untouched.
    query_log_enabled: bool = field(default_factory=lambda: _get("dashboard", "query_log", False))
    query_log_dir: str = field(default_factory=lambda: _get("dashboard", "query_log_dir", "data/query_log"))

    def __post_init__(self):
        """Validate config values and ensure directories exist."""
        # Bounds validation
        if not isinstance(self.chunk_size, int) or self.chunk_size < 100:
            print(f"[WARN] chunk_size={self.chunk_size} invalid, using 1000")
            self.chunk_size = 1000
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            print(f"[WARN] chunk_overlap={self.chunk_overlap} invalid, using 200")
            self.chunk_overlap = 200
        if self.chunk_overlap >= self.chunk_size:
            print(
                f"[WARN] chunk_overlap ({self.chunk_overlap}) >= chunk_size ({self.chunk_size}), using {self.chunk_size // 5}"
            )
            self.chunk_overlap = self.chunk_size // 5

        # Code-aware chunking validation (A3.8). Both fields default to safe
        # values; a bogus config never bricks ingestion — the flag just
        # silently reverts to False and chunking behaves as before.
        if not isinstance(self.code_aware_chunking, bool):
            print(f"[WARN] documents.code_aware_chunking={self.code_aware_chunking!r} invalid, using False")
            self.code_aware_chunking = False
        if not isinstance(self.code_aware_max_chunk_size, int) or self.code_aware_max_chunk_size < 100:
            print(f"[WARN] documents.code_aware_max_chunk_size={self.code_aware_max_chunk_size!r} invalid, using 1500")
            self.code_aware_max_chunk_size = 1500

        # Contextual chunking validation (A3.6). Same fail-safe posture as
        # code_aware_chunking: an invalid value silently reverts to False so
        # a typo in config.yaml never turns an ingestion pass into a runaway
        # LLM bill. The presence of a working LLM provider is deliberately
        # NOT checked here — see the docstring on ``llm_provider`` for the
        # rationale (config.py runs before the LLM subpackage is importable).
        if not isinstance(self.contextual_chunking_enabled, bool):
            print(f"[WARN] documents.contextual_chunking={self.contextual_chunking_enabled!r} invalid, using False")
            self.contextual_chunking_enabled = False

        # Late chunking validation (R5.7 — Jina AI 2024). Same fail-safe
        # posture as the sibling opt-in chunking flags: an invalid value
        # silently reverts to False so a typo in config.yaml never crashes
        # ingestion. The presence of a token-aware embedding provider is
        # NOT checked here — the chunker itself logs a single WARN and
        # falls back to the fixed-size chunker at first use if the
        # provider does not implement ``embed_tokens``.
        if not isinstance(self.late_chunking_enabled, bool):
            print(f"[WARN] documents.late_chunking={self.late_chunking_enabled!r} invalid, using False")
            self.late_chunking_enabled = False

        if not isinstance(self.default_results, int) or self.default_results < 1:
            self.default_results = 5
        if not isinstance(self.max_results, int) or self.max_results < 1:
            self.max_results = 20
        if not isinstance(self.embedding_dim, int) or self.embedding_dim < 1:
            self.embedding_dim = 384
        if not isinstance(self.reranker_enabled, bool):
            print(f"[WARN] reranker_enabled={self.reranker_enabled!r} invalid, using True")
            self.reranker_enabled = True
        if not isinstance(self.reranker_top_k_multiplier, int) or self.reranker_top_k_multiplier < 1:
            self.reranker_top_k_multiplier = 3

        # A2.4 fusion strategy validation. Import is deferred so config.py has
        # no import-time dependency on the retrieval subpackage.
        _VALID_FUSION = ("rrf", "combsum", "combmnz", "weighted")
        if not isinstance(self.fusion_strategy, str) or self.fusion_strategy not in _VALID_FUSION:
            print(
                f"[WARN] search.fusion.strategy={self.fusion_strategy!r} invalid, using 'rrf' "
                f"(valid: {', '.join(_VALID_FUSION)})"
            )
            self.fusion_strategy = "rrf"
        if not isinstance(self.fusion_weights, dict):
            print(f"[WARN] search.fusion.weights={self.fusion_weights!r} invalid, using {{}}")
            self.fusion_weights = {}
        else:
            # Coerce numeric values, drop malformed entries. Keep only the
            # branch names the ``weighted`` strategy actually reads today.
            _clean_weights: Dict[str, float] = {}
            for _fw_key, _fw_val in self.fusion_weights.items():
                if _fw_key in ("semantic", "bm25") and isinstance(_fw_val, (int, float)):
                    _clean_weights[str(_fw_key)] = float(_fw_val)
                else:
                    print(f"[WARN] search.fusion.weights.{_fw_key}={_fw_val!r} invalid, dropping")
            self.fusion_weights = _clean_weights

        # Provider registry validation (A2.2). Defaults keep the legacy stack
        # (FastEmbed + ChromaDB + ms-marco cross-encoder). Empty strings or
        # wrong types silently fall back to the historical default so a typo
        # in config.yaml never bricks startup.
        if not isinstance(self.embedding_provider, str) or not self.embedding_provider.strip():
            print(f"[WARN] models.embedding.provider={self.embedding_provider!r} invalid, using 'fastembed'")
            self.embedding_provider = "fastembed"
        else:
            self.embedding_provider = self.embedding_provider.strip()
        if not isinstance(self.vector_store, str) or not self.vector_store.strip():
            print(f"[WARN] models.vector_store.provider={self.vector_store!r} invalid, using 'chromadb'")
            self.vector_store = "chromadb"
        else:
            self.vector_store = self.vector_store.strip()
        if not isinstance(self.reranker_provider, str) or not self.reranker_provider.strip():
            print(f"[WARN] models.reranker.provider={self.reranker_provider!r} invalid, using 'cross_encoder'")
            self.reranker_provider = "cross_encoder"
        else:
            self.reranker_provider = self.reranker_provider.strip()

        # R5.3 — ColBERT checkpoint id/path. Only consulted when
        # ``vector_store == "colbert"``; on ChromaDB installs the value is
        # inert. Same permissive contract as every other opt-in field:
        # a non-string silently reverts to the documented default so a
        # typo in config.yaml never bricks startup.
        if not isinstance(self.colbert_model, str) or not self.colbert_model.strip():
            print(f"[WARN] models.colbert_model={self.colbert_model!r} invalid, using 'colbert-ir/colbertv2.0'")
            self.colbert_model = "colbert-ir/colbertv2.0"
        else:
            self.colbert_model = self.colbert_model.strip()

        # R5.4 — Matryoshka checkpoint id + slice width. Only consulted
        # when ``embedding_provider == "matryoshka"``; the default
        # ``fastembed`` path never reads either field. Same permissive
        # contract as every other opt-in field: wrong types silently
        # revert to the documented default so a typo in config.yaml
        # never bricks startup. The dimension-vs-model semantic check
        # (was 200 requested but the model only supports 64/128/256/…?)
        # lives in :meth:`MatryoshkaEmbedding._validate_dimension` so the
        # error surfaces at provider construction with the exact list of
        # valid slices — much better UX than a numeric range check here.
        if not isinstance(self.matryoshka_model, str) or not self.matryoshka_model.strip():
            print(
                f"[WARN] models.matryoshka_model={self.matryoshka_model!r} invalid, "
                f"using 'nomic-ai/nomic-embed-text-v1.5'"
            )
            self.matryoshka_model = "nomic-ai/nomic-embed-text-v1.5"
        else:
            self.matryoshka_model = self.matryoshka_model.strip()
        if not isinstance(self.matryoshka_dimension, int) or self.matryoshka_dimension < 1:
            print(f"[WARN] models.matryoshka_dimension={self.matryoshka_dimension!r} invalid, using 512")
            self.matryoshka_dimension = 512

        # Server transport validation
        if self.transport not in ("stdio", "sse", "streamable-http"):
            print(f"[WARN] server.transport={self.transport!r} invalid, using 'stdio'")
            self.transport = "stdio"
        if not isinstance(self.server_port, int) or not (1 <= self.server_port <= 65535):
            self.server_port = 8179
        if not isinstance(self.metrics_port, int) or not (1 <= self.metrics_port <= 65535):
            self.metrics_port = 9179
        if not isinstance(self.rate_limit_rpm, int) or self.rate_limit_rpm < 1:
            self.rate_limit_rpm = 60
        if not isinstance(self.rate_limit_burst, int) or self.rate_limit_burst < 0:
            self.rate_limit_burst = 10

        # MMR validation — bounds keep the diversification well-defined.
        if not isinstance(self.mmr_enabled, bool):
            print(f"[WARN] search.mmr_enabled={self.mmr_enabled!r} invalid, using True")
            self.mmr_enabled = True
        try:
            lam = float(self.mmr_lambda)
        except (TypeError, ValueError):
            print(f"[WARN] search.mmr_lambda={self.mmr_lambda!r} invalid, using 0.7")
            lam = 0.7
        if not 0.0 <= lam <= 1.0:
            print(f"[WARN] search.mmr_lambda={self.mmr_lambda!r} out of [0.0, 1.0], using 0.7")
            lam = 0.7
        self.mmr_lambda = lam

        # Telemetry validation (A2.6)
        if not isinstance(self.telemetry_enabled, bool):
            print(f"[WARN] telemetry.enabled={self.telemetry_enabled!r} invalid, using False")
            self.telemetry_enabled = False
        if not isinstance(self.telemetry_json_logs, bool):
            print(f"[WARN] telemetry.json_logs={self.telemetry_json_logs!r} invalid, using False")
            self.telemetry_json_logs = False
        _valid_exporters = ("otlp", "console", "none")
        if not isinstance(self.telemetry_exporter, str) or self.telemetry_exporter not in _valid_exporters:
            print(
                f"[WARN] telemetry.exporter={self.telemetry_exporter!r} invalid "
                f"(valid: {', '.join(_valid_exporters)}), using 'otlp'"
            )
            self.telemetry_exporter = "otlp"
        if not isinstance(self.telemetry_service_name, str) or not self.telemetry_service_name.strip():
            self.telemetry_service_name = "knowledge-rag"
        if not isinstance(self.telemetry_otlp_endpoint, str):
            self.telemetry_otlp_endpoint = ""
        _valid_log_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if not isinstance(self.telemetry_log_level, str) or self.telemetry_log_level.upper() not in _valid_log_levels:
            print(
                f"[WARN] telemetry.log_level={self.telemetry_log_level!r} invalid "
                f"(valid: {', '.join(_valid_log_levels)}), using 'INFO'"
            )
            self.telemetry_log_level = "INFO"
        else:
            self.telemetry_log_level = self.telemetry_log_level.upper()

        # Stopword languages validation (Q1.3). Empty list = opt-out.
        # A list with NO recognized codes falls back to defaults rather than
        # silently disabling — a typo shouldn't kill the filter.
        _KNOWN_STOPWORD_LANGS = ("en", "pt", "es", "de", "fr", "it")
        if not isinstance(self.stopword_languages, list):
            print(f"[WARN] search.stopword_languages={self.stopword_languages!r} invalid, using ['en', 'pt']")
            self.stopword_languages = ["en", "pt"]
        else:
            _valid_langs = [str(lang).lower().strip() for lang in self.stopword_languages if isinstance(lang, str)]
            _known = [lang for lang in _valid_langs if lang in _KNOWN_STOPWORD_LANGS]
            if _valid_langs and not _known:
                print(
                    f"[WARN] search.stopword_languages={self.stopword_languages!r} contains no "
                    f"known language codes (valid: {', '.join(_KNOWN_STOPWORD_LANGS)}), "
                    f"using ['en', 'pt']"
                )
                self.stopword_languages = ["en", "pt"]
            else:
                # Preserve original ordering; empty list [] stays empty (opt-out)
                self.stopword_languages = _known

        # Parent Document Retrieval validation (A3.7). Fallback to safe
        # defaults on any misconfig — a typo in config.yaml must never brick
        # startup. When invalid sizes force the feature into an unusable
        # state, silently disable it and warn.
        if not isinstance(self.parent_document_enabled, bool):
            print(f"[WARN] search.parent_document_enabled={self.parent_document_enabled!r} invalid, using False")
            self.parent_document_enabled = False
        if not isinstance(self.parent_document_large_size, int) or self.parent_document_large_size < 100:
            print(f"[WARN] search.parent_document_large_size={self.parent_document_large_size!r} invalid, using 1500")
            self.parent_document_large_size = 1500
        if not isinstance(self.parent_document_small_size, int) or self.parent_document_small_size < 50:
            print(f"[WARN] search.parent_document_small_size={self.parent_document_small_size!r} invalid, using 250")
            self.parent_document_small_size = 250
        if not isinstance(self.parent_document_small_overlap, int) or self.parent_document_small_overlap < 0:
            print(
                f"[WARN] search.parent_document_small_overlap={self.parent_document_small_overlap!r} invalid, using 50"
            )
            self.parent_document_small_overlap = 50
        if self.parent_document_large_size <= self.parent_document_small_size:
            print(
                f"[WARN] search.parent_document_large_size "
                f"({self.parent_document_large_size}) must be > "
                f"search.parent_document_small_size ({self.parent_document_small_size}), "
                f"disabling Parent Document Retrieval"
            )
            self.parent_document_enabled = False
            self.parent_document_large_size = 1500
            self.parent_document_small_size = 250
        if self.parent_document_small_overlap >= self.parent_document_small_size:
            print(
                f"[WARN] search.parent_document_small_overlap "
                f"({self.parent_document_small_overlap}) must be < "
                f"search.parent_document_small_size ({self.parent_document_small_size}), "
                f"using {self.parent_document_small_size // 5}"
            )
            self.parent_document_small_overlap = self.parent_document_small_size // 5

        # Semantic cache validation (A2.7)
        if not isinstance(self.semantic_cache_enabled, bool):
            self.semantic_cache_enabled = True
        if not isinstance(self.semantic_cache_ttl_days, int) or self.semantic_cache_ttl_days < 0:
            self.semantic_cache_ttl_days = 30
        if not isinstance(self.semantic_cache_max_entries, int) or self.semantic_cache_max_entries < 1:
            self.semantic_cache_max_entries = 10000
        if not isinstance(self.semantic_cache_dir, str) or not self.semantic_cache_dir.strip():
            self.semantic_cache_dir = "data/cache/semantic"

        # ── LLM provider validation (A3.1) ─────────────────────────────
        # Structural checks only — we deliberately do NOT import
        # ``mcp_server.providers.llm`` to validate ``llm_provider`` name
        # here because config.py loads at process start, before the LLM
        # subpackage is imported. An unknown provider name surfaces the
        # first time a Fase 3 feature calls ``get_llm(config.llm_provider)``
        # with a clear ``ValueError`` — that is the right failure boundary.
        # A typo in ``llm_provider`` MUST NOT brick server startup because
        # knowledge-rag runs perfectly without any LLM by default.
        if not isinstance(self.llm_provider, str):
            print(f"[WARN] llm.provider={self.llm_provider!r} invalid, using '' (auto-detect)")
            self.llm_provider = ""
        else:
            self.llm_provider = self.llm_provider.strip()
        if not isinstance(self.llm_model, str):
            print(f"[WARN] llm.model={self.llm_model!r} invalid, using '' (provider default)")
            self.llm_model = ""
        else:
            self.llm_model = self.llm_model.strip()
        if not isinstance(self.llm_max_tokens, int) or self.llm_max_tokens < 1:
            print(f"[WARN] llm.max_tokens={self.llm_max_tokens!r} invalid, using 1024")
            self.llm_max_tokens = 1024
        try:
            _llm_temp = float(self.llm_temperature)
        except (TypeError, ValueError):
            print(f"[WARN] llm.temperature={self.llm_temperature!r} invalid, using 0.0")
            _llm_temp = 0.0
        if not 0.0 <= _llm_temp <= 2.0:
            print(f"[WARN] llm.temperature={self.llm_temperature!r} out of [0.0, 2.0], using 0.0")
            _llm_temp = 0.0
        self.llm_temperature = _llm_temp
        if not isinstance(self.llm_timeout_seconds, int) or self.llm_timeout_seconds < 1:
            print(f"[WARN] llm.timeout_seconds={self.llm_timeout_seconds!r} invalid, using 30")
            self.llm_timeout_seconds = 30

        # ── Query rewriting toggle (A3.2) ──────────────────────────────
        # Same permissive contract as the ``llm.*`` fields above: a wrong
        # type silently falls back to the default (disabled). We do NOT
        # probe ``auto_detect_llm`` here — the LLM subpackage may not be
        # importable this early, and enabling the flag without a resolvable
        # provider is documented as safe (runtime fail-open).
        if not isinstance(self.query_rewrite_enabled, bool):
            print(f"[WARN] llm_features.query_rewrite={self.query_rewrite_enabled!r} invalid, using False")
            self.query_rewrite_enabled = False

        # A3.3 — HyDE toggles. Same coercion contract as
        # ``query_rewrite_enabled``: bad type ⇒ safe default, no LLM
        # probe here. ``hyde_num_hypotheses`` is clamped to a modest
        # ceiling because cost scales linearly with N; a stray ``100``
        # in config.yaml would silently rack up LLM bills per query.
        if not isinstance(self.hyde_enabled, bool):
            print(f"[WARN] llm_features.hyde={self.hyde_enabled!r} invalid, using False")
            self.hyde_enabled = False
        if not isinstance(self.hyde_num_hypotheses, int) or self.hyde_num_hypotheses < 1:
            print(f"[WARN] llm_features.hyde_num_hypotheses={self.hyde_num_hypotheses!r} invalid, using 1")
            self.hyde_num_hypotheses = 1
        elif self.hyde_num_hypotheses > 10:
            print(
                f"[WARN] llm_features.hyde_num_hypotheses={self.hyde_num_hypotheses!r} > 10 "
                f"(would trigger {self.hyde_num_hypotheses} LLM calls per query), clamping to 10"
            )
            self.hyde_num_hypotheses = 10

        # A3.5 — self-query flag. Same coercion contract as
        # ``query_rewrite_enabled``: bad type -> False, no LLM probe here.
        if not isinstance(self.self_query_enabled, bool):
            print(f"[WARN] llm_features.self_query={self.self_query_enabled!r} invalid, using False")
            self.self_query_enabled = False

        # A3.4 — multi-query fan-out flag + N. Same permissive contract as
        # the other Fase 3 knobs: a wrong type silently falls back to the
        # safe default. Non-positive N or N==1 degrades to single-query
        # semantics (no LLM call). Runaway N is clipped to 10 — the same
        # bound the feature module itself enforces internally — so a
        # config typo can never explode retrieval latency or LLM cost.
        if not isinstance(self.multi_query_enabled, bool):
            print(f"[WARN] llm_features.multi_query={self.multi_query_enabled!r} invalid, using False")
            self.multi_query_enabled = False
        if not isinstance(self.multi_query_n, int) or self.multi_query_n < 1:
            print(f"[WARN] llm_features.multi_query_n={self.multi_query_n!r} invalid, using 3")
            self.multi_query_n = 3
        elif self.multi_query_n > 10:
            print(
                f"[WARN] llm_features.multi_query_n={self.multi_query_n!r} > 10 "
                f"(would trigger {self.multi_query_n}x retrieval + LLM cost per call), clamping to 10"
            )
            self.multi_query_n = 10

        # M4.3 — global cross-corpus index toggle. Same permissive contract
        # as every other Fase 3/4 knob: a wrong type silently falls back to
        # the default. We do NOT touch ``~/.knowledge-rag/global/registry.
        # json`` from here — the registry is per-user state validated by
        # ``mcp_server.global_index.registry`` at call time.
        if not isinstance(self.global_index_enabled, bool):
            print(f"[WARN] global_index.enabled={self.global_index_enabled!r} invalid, using True")
            self.global_index_enabled = True

        # A3.9 — adaptive retrieval router flag. Same permissive contract
        # as the other Fase 3 knobs: a wrong type silently falls back to
        # the default (disabled). We do NOT probe ``auto_detect_llm`` here
        # because config.py is loaded before the LLM subpackage; a stale
        # ``True`` after credential rotation is safe — the router itself
        # fails open to ``"hybrid"`` at runtime.
        if not isinstance(self.adaptive_retrieval_enabled, bool):
            print(f"[WARN] llm_features.adaptive={self.adaptive_retrieval_enabled!r} invalid, using False")
            self.adaptive_retrieval_enabled = False

        if not isinstance(self.supported_formats, list) or not self.supported_formats:
            print("[WARN] supported_formats is empty or invalid, using defaults")
            self.supported_formats = [
                ".md",
                ".txt",
                ".pdf",
                ".py",
                ".c",
                ".h",
                ".cpp",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".json",
                ".xml",
                ".docx",
                ".xlsx",
                ".pptx",
                ".csv",
                ".ipynb",
            ]

        # Validate exclude_patterns is a list of strings
        if not isinstance(self.exclude_patterns, list):
            print(f"[WARN] exclude_patterns={self.exclude_patterns!r} invalid, using []")
            self.exclude_patterns = []
        else:
            self.exclude_patterns = [p for p in self.exclude_patterns if isinstance(p, str)]

        # Validate keyword_routes values are lists (not strings)
        for cat, keywords in list(self.keyword_routes.items()):
            if not isinstance(keywords, list):
                print(f"[WARN] keyword_routes.{cat} is not a list, removing")
                del self.keyword_routes[cat]

        if not isinstance(self.query_expansions, dict):
            print("[WARN] query_expansions is invalid, using defaults")
            self.query_expansions = dict(_DEFAULT_QUERY_EXPANSIONS)

        for term, synonyms in list(self.query_expansions.items()):
            if not isinstance(term, str) or not isinstance(synonyms, list):
                print(f"[WARN] query_expansions.{term} is invalid, removing")
                del self.query_expansions[term]

        if not isinstance(self.query_expansion_groups, list):
            print("[WARN] query_expansion_groups is invalid, ignoring")
            self.query_expansion_groups = []

        self.query_expansions = _merge_query_expansion_sources(self.query_expansions, self.query_expansion_groups)

        # ── Work Memory validation (M4.2) ──────────────────────────────
        # Same permissive contract as every other opt-in Fase 4 feature:
        # a wrong type silently falls back to the safe default. The
        # directory is NOT created here — creation is deferred to the
        # first ``save_result`` call so that flipping the flag on without
        # writing anything leaves the filesystem untouched.
        if not isinstance(self.work_memory_enabled, bool):
            print(f"[WARN] work_memory.enabled={self.work_memory_enabled!r} invalid, using False")
            self.work_memory_enabled = False
        if not isinstance(self.work_memory_dir, str) or not self.work_memory_dir.strip():
            print(f"[WARN] work_memory.dir={self.work_memory_dir!r} invalid, using 'data/memory'")
            self.work_memory_dir = "data/memory"
        else:
            self.work_memory_dir = self.work_memory_dir.strip()
        if not isinstance(self.work_memory_half_life_days, int) or self.work_memory_half_life_days < 0:
            print(f"[WARN] work_memory.half_life_days={self.work_memory_half_life_days!r} invalid, using 30")
            self.work_memory_half_life_days = 30
        if not isinstance(self.work_memory_min_corroboration, int) or self.work_memory_min_corroboration < 1:
            print(f"[WARN] work_memory.min_corroboration={self.work_memory_min_corroboration!r} invalid, using 2")
            self.work_memory_min_corroboration = 2

        # ── Dashboard / query log validation (M4.4) ────────────────────
        # Same permissive contract as every other opt-in Fase 4 feature:
        # a wrong type silently falls back to the safe default. The log
        # directory is NOT created here — creation happens lazily inside
        # the first ``append_query_log`` call so flipping the flag on
        # without ever running a query leaves the filesystem untouched.
        if not isinstance(self.query_log_enabled, bool):
            print(f"[WARN] dashboard.query_log={self.query_log_enabled!r} invalid, using False")
            self.query_log_enabled = False
        if not isinstance(self.query_log_dir, str) or not self.query_log_dir.strip():
            print(f"[WARN] dashboard.query_log_dir={self.query_log_dir!r} invalid, using 'data/query_log'")
            self.query_log_dir = "data/query_log"
        else:
            self.query_log_dir = self.query_log_dir.strip()

        # Warn when documents_dir was explicitly set but does not exist
        raw_docs = _get("paths", "documents_dir", None)
        if raw_docs is not None and not self.documents_dir.exists():
            print(
                f"[WARN] documents_dir '{raw_docs}' resolved to "
                f"'{self.documents_dir}' which does not exist — creating it. "
                f"Verify the path in config.yaml if reindex returns 0 files."
            )

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
