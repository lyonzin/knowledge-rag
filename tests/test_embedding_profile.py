"""Coverage for embedding profile + prefix support (v4.8.0 Fase 1).

Two concerns exercised here:

* **TestProfileResolution** — the profile resolver in
  ``Config.__post_init__`` correctly maps ``profile: "<name>"`` onto the
  matching ``_EMBEDDING_PROFILES`` entry, respects user overrides for
  prefix, warns and yields to profile when ``models.embedding.model`` is
  redeclared, and falls back to ``custom`` for unknown / non-string
  profile names.
* **TestPrefixApplication** — the ``FastEmbedEmbeddings`` pipeline
  prepends the right prefix per scope: ``passage_prefix`` for
  ``__call__`` and ``embed_documents``, ``query_prefix`` for
  ``embed_query``. Empty prefixes are pass-through (the input list is
  returned unchanged) so the default profile allocates zero new strings.

The Config tests rebuild the dataclass under a monkey-patched ``_yaml``
dict rather than mutating the process-global ``config`` instance so we
never leak state into other test modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mcp_server import config as cfg_module
from mcp_server.server import FastEmbedEmbeddings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rebuild_config(monkeypatch: pytest.MonkeyPatch, yaml_dict: dict) -> cfg_module.Config:
    """Return a fresh Config resolved against ``yaml_dict``.

    Monkey-patches ``mcp_server.config._yaml`` for the duration of the
    test — factory lambdas read the module-level name dynamically, so
    swapping the dict here reroutes all defaults.
    """
    monkeypatch.setattr(cfg_module, "_yaml", yaml_dict)
    return cfg_module.Config()


# ---------------------------------------------------------------------------
# TestProfileResolution
# ---------------------------------------------------------------------------


class TestProfileResolution:
    """Config.__post_init__ profile resolver — 5 cases."""

    def test_profile_custom_preserves_explicit_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: user opts out of profile shorthand and declares model + dim.
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "custom",
                    "model": "sentence-transformers/all-MiniLM-L12-v2",
                    "dimensions": 384,
                    "query_prefix": "",
                    "passage_prefix": "",
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)

        # Assert
        assert cfg.embedding_profile == "custom"
        assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L12-v2"
        assert cfg.embedding_dim == 384
        assert cfg.query_prefix == ""
        assert cfg.passage_prefix == ""

    def test_profile_multilingual_populates_e5_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: YAML only declares the profile — resolver fills the rest.
        yaml_dict = {"models": {"embedding": {"profile": "multilingual"}}}

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)

        # Assert
        assert cfg.embedding_profile == "multilingual"
        assert cfg.embedding_model == "intfloat/multilingual-e5-large"
        assert cfg.embedding_dim == 1024
        assert cfg.query_prefix == "query: "
        assert cfg.passage_prefix == "passage: "

    def test_profile_overrides_model_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange: user set BOTH a named profile AND an explicit model.
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "quality",
                    "model": "BAAI/bge-small-en-v1.5",  # would-be override, loses
                    "dimensions": 384,  # ditto
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)
        captured = capsys.readouterr()

        # Assert: profile wins, WARN was emitted, dim promoted to 1024
        assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
        assert cfg.embedding_dim == 1024
        assert "profile takes precedence" in captured.out

    def test_invalid_profile_falls_back_to_custom(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange: bogus profile name + explicit model that must be honored
        # after the fallback (custom leaves models.embedding.* alone).
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "does-not-exist",
                    "model": "BAAI/bge-small-en-v1.5",
                    "dimensions": 384,
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)
        captured = capsys.readouterr()

        # Assert: profile normalized to custom, WARN emitted, explicit model kept
        assert cfg.embedding_profile == "custom"
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.embedding_dim == 384
        assert "Invalid embedding profile" in captured.out
        assert "does-not-exist" in captured.out

    def test_user_prefix_overrides_profile_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: profile=multilingual ships "query: "/"passage: " but the
        # user prefers alternative sentinels for a fine-tuned e5 checkpoint.
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "multilingual",
                    "query_prefix": "search_query: ",
                    "passage_prefix": "search_document: ",
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)

        # Assert: model + dim still come from profile; prefixes stay user's
        assert cfg.embedding_model == "intfloat/multilingual-e5-large"
        assert cfg.embedding_dim == 1024
        assert cfg.query_prefix == "search_query: "
        assert cfg.passage_prefix == "search_document: "


# ---------------------------------------------------------------------------
# TestPrefixApplication
# ---------------------------------------------------------------------------


def _fake_embed_output(texts):
    """Yield deterministic 4D vectors so length + dim invariants hold."""
    for i, _ in enumerate(texts):
        yield np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32)


def _prepared_embedder(monkeypatch: pytest.MonkeyPatch) -> tuple[FastEmbedEmbeddings, MagicMock]:
    """Return a FastEmbedEmbeddings with load short-circuited and a spy model.

    The spy captures the ``texts`` argument passed to ``model.embed`` so
    the test can assert what actually reached the ONNX layer after any
    prefix massaging.
    """
    embedder = FastEmbedEmbeddings()
    embedder._dim = 4  # match the fake output
    spy = MagicMock()
    spy.embed.side_effect = lambda texts: _fake_embed_output(texts)
    embedder._model = spy
    # Bypass the lazy loader (model is already "loaded")
    monkeypatch.setattr(embedder, "_load_model", lambda: None)
    return embedder, spy


class TestPrefixApplication:
    """FastEmbedEmbeddings prefix wiring — 3 cases."""

    def test_passage_prefix_prepended_in_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(cfg_module.config, "passage_prefix", "passage: ")
        monkeypatch.setattr(cfg_module.config, "query_prefix", "query: ")
        embedder, spy = _prepared_embedder(monkeypatch)

        # Act — __call__ is the ChromaDB embedding_function entrypoint (passage)
        embedder(["mitre att&ck", "cve-2024-1234"])

        # Assert: model saw the passage-prefixed strings, not the raw ones
        spy.embed.assert_called_once()
        (called_texts,) = spy.embed.call_args.args
        assert called_texts == ["passage: mitre att&ck", "passage: cve-2024-1234"]

    def test_query_prefix_prepended_in_embed_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(cfg_module.config, "passage_prefix", "passage: ")
        monkeypatch.setattr(cfg_module.config, "query_prefix", "query: ")
        embedder, spy = _prepared_embedder(monkeypatch)

        # Act
        embedder.embed_query("what is kerberoasting")

        # Assert: query prefix wins on this path, passage prefix never leaks
        spy.embed.assert_called_once()
        (called_texts,) = spy.embed.call_args.args
        assert called_texts == ["query: what is kerberoasting"]

    def test_empty_prefix_does_not_reallocate_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: empty prefix must be a no-op — inputs pass through as-is.
        # This matters on the hot ingest path where the default "compact"
        # profile allocates zero new strings.
        monkeypatch.setattr(cfg_module.config, "passage_prefix", "")
        monkeypatch.setattr(cfg_module.config, "query_prefix", "")
        embedder, spy = _prepared_embedder(monkeypatch)
        raw = ["alpha", "beta", "gamma"]

        # Act
        embedder(raw)

        # Assert: model got the same list object (identity) because
        # ``_apply_prefix`` returns the input unchanged on empty prefix.
        spy.embed.assert_called_once()
        (called_texts,) = spy.embed.call_args.args
        assert called_texts is raw, "empty prefix must be identity, not a copy"
