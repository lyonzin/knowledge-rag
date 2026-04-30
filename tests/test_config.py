"""Tests for configuration integrity."""

from mcp_server.config import config


def test_no_ollama_references():
    """Config must not reference Ollama (removed in v3.0)."""
    assert not hasattr(config, "ollama_model")
    assert not hasattr(config, "ollama_base_url")


def test_embedding_model():
    """FastEmbed model must be configured."""
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.embedding_dim == 384


def test_reranker_config():
    """Reranker must be configured and enabled."""
    assert "ms-marco" in config.reranker_model
    assert config.reranker_enabled is True
    assert config.reranker_top_k_multiplier >= 2


def test_supported_formats():
    """Core formats must be present in supported_formats."""
    core = {".md", ".txt", ".pdf"}
    assert core.issubset(set(config.supported_formats))


def test_query_expansions_count():
    """Must have 50+ query expansion terms."""
    assert len(config.query_expansions) >= 50


def test_query_expansion_security_terms():
    """Key security terms must have expansions."""
    must_have = ["sqli", "xss", "privesc", "amsi", "suid", "kerberoast"]
    for term in must_have:
        assert term in config.query_expansions, f"Missing expansion for: {term}"


def test_cve_aliases():
    """CVE aliases must be present."""
    must_have = ["printnightmare", "eternalblue", "pwnkit", "log4shell", "zerologon"]
    for term in must_have:
        assert term in config.query_expansions, f"Missing CVE alias for: {term}"


def test_category_mappings():
    """Essential categories must exist."""
    must_have = ["security", "ctf", "logscale", "development", "general", "aar"]
    for cat in must_have:
        found = any(cat in v for v in config.category_mappings.values())
        assert found, f"Missing category mapping for: {cat}"


def test_chunk_settings():
    """Chunk settings must be reasonable."""
    assert 500 <= config.chunk_size <= 2000
    assert 100 <= config.chunk_overlap <= 500
    assert config.chunk_overlap < config.chunk_size


# ── v3.4.0 Features ──


def test_models_cache_dir_exists():
    """models_cache_dir must be a Path and directory must be created."""
    from pathlib import Path

    assert hasattr(config, "models_cache_dir")
    assert isinstance(config.models_cache_dir, Path)
    assert config.models_cache_dir.exists()


def test_venv_project_dir_uses_unresolved_executable(monkeypatch):
    """Venv detection must survive python symlinks that resolve to system Python."""
    from pathlib import Path

    import mcp_server.config as config_module

    monkeypatch.setattr(config_module.sys, "prefix", "/usr")
    monkeypatch.setattr(config_module.sys, "executable", "/opt/knowledge-rag/venv/bin/python")

    assert config_module._venv_project_dir() == Path("/opt/knowledge-rag")


def test_exclude_patterns_default_empty():
    """Default exclude_patterns must be an empty list."""
    assert hasattr(config, "exclude_patterns")
    assert isinstance(config.exclude_patterns, list)


def test_ipynb_in_supported_suffixes():
    """.ipynb must be in the internal supported suffixes set."""
    from mcp_server.config import _SUPPORTED_SUFFIXES

    assert ".ipynb" in _SUPPORTED_SUFFIXES


def test_new_code_formats_in_supported_suffixes():
    """New code formats must be in _SUPPORTED_SUFFIXES for directory detection."""
    from mcp_server.config import _SUPPORTED_SUFFIXES

    for ext in [".c", ".h", ".cpp", ".js", ".jsx", ".ts", ".tsx", ".xml"]:
        assert ext in _SUPPORTED_SUFFIXES, f"{ext} missing from _SUPPORTED_SUFFIXES"


def test_new_code_formats_default_enabled():
    """New code formats must be in default supported_formats (not opt-in)."""
    for ext in [".c", ".h", ".cpp", ".js", ".jsx", ".ts", ".tsx", ".xml"]:
        assert ext in config.supported_formats, f"{ext} missing from supported_formats defaults"
