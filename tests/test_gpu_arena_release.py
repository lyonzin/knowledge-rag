"""GH #257: CUDA sessions must return their arena instead of pinning peak VRAM.

ONNX Runtime's default CUDA arena grows by powers of two and keeps every
region until the session is destroyed, so one indexing batch or one wide
rerank held gigabytes of VRAM for the life of the server. These tests mock
ORT and FastEmbed; the GPU measurement is the manual check in docs/gpu-setup.md.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import onnxruntime as ort
import pytest

from mcp_server.server import CrossEncoderReranker, FastEmbedEmbeddings, GPUStatus, config

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
DIRECTML = "DmlExecutionProvider"
SHRINK_KEY = "memory.enable_memory_arena_shrinkage"
SAME_AS_REQUESTED = (CUDA, {"arena_extend_strategy": "kSameAsRequested"})


class _Session:
    """ORT InferenceSession stand-in recording the RunOptions of each run."""

    def __init__(self, providers):
        self.providers = list(providers)
        self.run_options = []

    def get_providers(self):
        return self.providers

    def get_provider_options(self):
        return {name: {"device_id": "0"} if name == CUDA else {} for name in self.providers}

    def run(self, output_names, input_feed, run_options=None):
        self.run_options.append(run_options)
        return [[0.0]]


class _FakeCrossEncoder:
    """TextCrossEncoder stand-in; FastEmbed keeps the session at ``model.model``."""

    def __init__(self, providers):
        self.model = SimpleNamespace(model=_Session(providers))
        self.rerank_kwargs = []

    def rerank(self, query, documents, **kwargs):
        self.rerank_kwargs.append(kwargs)
        return [0.0 for _ in documents]


def _embedding_model(providers):
    """TextEmbedding stand-in with FastEmbed's ``model.model.model`` session nesting."""
    return SimpleNamespace(model=SimpleNamespace(model=_Session(providers)))


@pytest.mark.parametrize("mode", ["auto", "true"])
def test_cuda_embedding_load_extends_arena_by_request_size(mode):
    """Given a ready GPU, when the embedder loads, then CUDA is asked for kSameAsRequested growth."""
    embedder = FastEmbedEmbeddings()
    embedder._gpu_mode = mode
    with (
        patch.object(FastEmbedEmbeddings, "verify_gpu_readiness", return_value=GPUStatus(available=True)),
        patch.object(FastEmbedEmbeddings, "_setup_cuda_dll_paths"),
        patch("mcp_server.server.TextEmbedding", return_value=_embedding_model([CUDA, CPU])) as factory,
    ):
        embedder._load_model()

    assert factory.call_args.kwargs["providers"] == [SAME_AS_REQUESTED, CPU]


def test_cuda_embedding_run_shrinks_arena_afterwards():
    """Given a CUDA embedding session, when FastEmbed runs it, then ORT is asked to shrink the arena."""
    model = _embedding_model([CUDA, CPU])
    with patch("mcp_server.server.TextEmbedding", return_value=model):
        FastEmbedEmbeddings()._load_with_providers([SAME_AS_REQUESTED, CPU], "test")
    session = model.model.model

    session.run(["embeddings"], {"input_ids": []})

    assert session.run_options[0].get_run_config_entry(SHRINK_KEY) == "gpu:0"


def test_cuda_run_keeps_run_options_passed_by_caller():
    """Given a wrapped CUDA session, when a caller passes RunOptions, then those reach ORT unchanged."""
    model = _embedding_model([CUDA, CPU])
    with patch("mcp_server.server.TextEmbedding", return_value=model):
        FastEmbedEmbeddings()._load_with_providers([SAME_AS_REQUESTED, CPU], "test")
    session = model.model.model
    caller_options = ort.RunOptions()

    session.run(["embeddings"], {}, caller_options)

    assert session.run_options == [caller_options]


@pytest.mark.parametrize("providers", [[CPU], [DIRECTML, CPU]])
def test_cpu_and_directml_embedding_runs_are_untouched(providers):
    """Given a CPU or DirectML session, when FastEmbed runs it, then no RunOptions are injected."""
    model = _embedding_model(providers)
    with (
        patch("mcp_server.server.TextEmbedding", return_value=model),
        patch.object(FastEmbedEmbeddings, "_validate_directml_session"),
    ):
        FastEmbedEmbeddings()._load_with_providers(providers, "test")
    session = model.model.model

    session.run(["embeddings"], {})

    assert session.run_options == [None]


@pytest.mark.parametrize(
    ("mode", "available", "expected"),
    [
        ("false", [CUDA, CPU], [CPU]),
        ("directml", [DIRECTML, CPU], [CPU]),
        ("auto", [CPU], [CPU]),
        ("auto", [CUDA, CPU], [SAME_AS_REQUESTED, CPU]),
        ("true", [CUDA, CPU], [SAME_AS_REQUESTED, CPU]),
    ],
)
def test_reranker_providers_follow_gpu_setting(monkeypatch, mode, available, expected):
    """Given models.embedding.gpu, when the reranker loads, then FastEmbed gets matching providers.

    Without explicit providers FastEmbed picks CUDA whenever ORT offers it, even with gpu: false.
    """
    monkeypatch.setattr(config, "gpu_mode", mode)
    monkeypatch.setattr(ort, "get_available_providers", lambda: available)
    with patch("mcp_server.server.TextCrossEncoder", return_value=_FakeCrossEncoder([CPU])) as factory:
        assert CrossEncoderReranker()._ensure_model()

    assert factory.call_args.kwargs["providers"] == expected


def test_cuda_reranker_run_shrinks_arena_afterwards(monkeypatch):
    """Given a CUDA reranker session, when FastEmbed runs it, then ORT is asked to shrink the arena."""
    monkeypatch.setattr(config, "gpu_mode", "auto")
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])
    encoder = _FakeCrossEncoder([CUDA, CPU])
    with patch("mcp_server.server.TextCrossEncoder", return_value=encoder):
        CrossEncoderReranker()._ensure_model()
    session = encoder.model.model

    session.run(["logits"], {"input_ids": []})

    assert session.run_options[0].get_run_config_entry(SHRINK_KEY) == "gpu:0"


def test_reranker_splits_candidates_into_bounded_batches(monkeypatch):
    """Given 60 candidates, when reranking, then FastEmbed runs batches of 16 instead of one padded run of 60.

    Sixteen still scores the default 15 candidates (5 results x 3) in a single run.
    """
    monkeypatch.setattr(config, "reranker_enabled", True)
    monkeypatch.setattr(config, "gpu_mode", "false")
    encoder = _FakeCrossEncoder([CPU])
    candidates = [{"document": f"candidate {index}"} for index in range(60)]
    with patch("mcp_server.server.TextCrossEncoder", return_value=encoder):
        CrossEncoderReranker().rerank("query", candidates, top_k=5)

    assert encoder.rerank_kwargs == [{"batch_size": 16}]
