"""Configuration for Knowledge RAG System v3.4.1 — YAML-configurable"""

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
    """Check if path has a documents/ dir with actual supported files (follows symlinks)."""
    docs_dir = path / "documents"
    if not docs_dir.exists():
        return False
    for root, _, files in os.walk(docs_dir, followlinks=True):
        for f in files:
            if Path(f).suffix.lower() in _SUPPORTED_SUFFIXES:
                return True
    return False


def _venv_project_dir():
    """Detect project root from venv location (pip install from PyPI)."""
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
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
    if not isinstance(val, dict):
        print(f"[WARN] config.yaml: {key} has wrong type (expected dict, got {type(val).__name__}), using default")
        return default
    return val


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


# ============================================================================
# CONFIG DATACLASS
# ============================================================================


def _resolve_path(raw, default: Path) -> Path:
    """Resolve a path from YAML (string) or use default (Path)."""
    if raw is None:
        return default
    p = Path(raw)
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

    # Search settings
    default_results: int = field(default_factory=lambda: _get("search", "default_results", 5))
    max_results: int = field(default_factory=lambda: _get("search", "max_results", 20))

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

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
