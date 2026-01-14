"""Configuration for Knowledge RAG System"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    """Central configuration for the RAG system"""

    # Paths
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    chroma_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "chroma_db")
    documents_dir: Path = field(default_factory=lambda: BASE_DIR / "documents")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embeddings (Ollama)
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # ChromaDB
    collection_name: str = "knowledge_base"

    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        ".md", ".txt", ".pdf", ".py", ".json"
    ])

    # Category mappings based on path
    category_mappings: Dict[str, str] = field(default_factory=lambda: {
        "security/redteam": "redteam",
        "security/blueteam": "blueteam",
        "security/ctf": "ctf",
        "security": "security",
        "logscale": "logscale",
        "development": "development",
        "general": "general",
    })

    # Keyword routing rules (deterministic routing before semantic search)
    keyword_routes: Dict[str, List[str]] = field(default_factory=lambda: {
        "logscale": [
            "logscale", "lql", "cql", "humio", "crowdstrike query",
            "formattime", "groupby", "base64decode", "case{}", "regex"
        ],
        "redteam": [
            "pentest", "exploit", "payload", "reverse shell", "privilege escalation",
            "lateral movement", "c2", "beacon", "cobalt strike", "metasploit"
        ],
        "blueteam": [
            "detection", "sigma", "yara", "ioc", "threat hunting",
            "incident response", "forensics", "malware analysis"
        ],
        "ctf": [
            "ctf", "flag", "hackthebox", "htb", "tryhackme", "picoctf",
            "writeup", "challenge"
        ],
        "development": [
            "python", "typescript", "javascript", "api", "fastapi",
            "django", "react", "nodejs"
        ],
    })

    # Search settings
    default_results: int = 5
    max_results: int = 20

    def __post_init__(self):
        """Ensure directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
