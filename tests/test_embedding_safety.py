"""Regression coverage for provider detection and invalid embedding output."""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcp_server.server import EmbeddingError, FastEmbedEmbeddings, GPUStatus, config


def _model(providers=("CPUExecutionProvider",)):
    """Expose the same session nesting as FastEmbed's ONNX text models."""
    model = MagicMock()
    model.model.model.get_providers.return_value = list(providers)
    model.embed.side_effect = lambda texts, batch_size: (np.ones(384, dtype=np.float32) for _ in texts)
    return model


@pytest.fixture(autouse=True)
def _isolate_embedding_config(monkeypatch):
    """Keep provider and batching tests independent of user config/environment."""
    monkeypatch.delenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", raising=False)
    monkeypatch.setattr(config, "gpu_mode", "false")
    monkeypatch.setattr(config, "embed_batch_size", None)
    monkeypatch.setattr(config, "embedding_dim", 384)


@pytest.mark.parametrize("mode", ["auto", "true"])
def test_cuda_silent_fallback_uses_cpu_batch_and_banner(mode, capsys):
    """ORT accepting CUDA must not be mistaken for the model using CUDA."""
    model = _model()
    embedder = FastEmbedEmbeddings()
    embedder._gpu_mode = mode
    with (
        patch("mcp_server.server.TextEmbedding", return_value=model) as factory,
        patch.object(embedder, "verify_gpu_readiness", return_value=GPUStatus(available=True)),
        patch.object(embedder, "_setup_cuda_dll_paths"),
    ):
        embedder(["test"])
    assert embedder._active_provider == "CPU"
    assert model.embed.call_args.kwargs["batch_size"] == 32
    assert factory.call_count == 1  # Reuse the already initialized CPU fallback session.
    assert "GPU STATUS: ACTIVE" not in capsys.readouterr().err


@pytest.mark.parametrize("raw", ["invalid", "", "0", "-2"])
def test_invalid_batch_environment_uses_valid_yaml(raw, monkeypatch):
    """Invalid environment input must not crash or disable bounded inference."""
    monkeypatch.setenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", raw)
    monkeypatch.setattr(config, "embed_batch_size", 7)
    model = _model()
    embedder = FastEmbedEmbeddings()
    embedder._model = model
    embedder._active_provider = "CPU"
    assert len(embedder(["test"])) == 1
    assert model.embed.call_args.kwargs["batch_size"] == 7


def test_batch_environment_obeys_maximum(monkeypatch):
    """The environment override shares the YAML upper bound."""
    monkeypatch.setenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", "999999")
    model = _model()
    embedder = FastEmbedEmbeddings()
    embedder._model = model
    embedder._active_provider = "CPU"
    embedder(["test"])
    assert model.embed.call_args.kwargs["batch_size"] == 512


def test_unknown_provider_uses_conservative_batch():
    """An unclassified session must never default to CUDA's batch size."""
    model = _model()
    embedder = FastEmbedEmbeddings()
    embedder._model = model
    embedder(["test"])
    assert model.embed.call_args.kwargs["batch_size"] == 32


def test_actual_cuda_session_uses_cuda_batch():
    """The larger batch is allowed only after inspecting the real session."""
    model = _model(("CUDAExecutionProvider", "CPUExecutionProvider"))
    embedder = FastEmbedEmbeddings()
    with patch("mcp_server.server.TextEmbedding", return_value=model):
        embedder._load_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], "test")
        embedder(["test"])
    assert embedder._active_provider == "CUDA"
    assert model.embed.call_args.kwargs["batch_size"] == 256


def test_concurrent_load_publishes_model_after_provider_is_known():
    """A second caller cannot observe a half-initialized model or load twice."""
    entered, release = threading.Event(), threading.Event()
    model = _model()

    def providers():
        entered.set()
        assert release.wait(10), "test did not release provider inspection"
        return ["CPUExecutionProvider"]

    model.model.model.get_providers.side_effect = providers
    embedder = FastEmbedEmbeddings()
    with patch("mcp_server.server.TextEmbedding", return_value=model) as factory, ThreadPoolExecutor(2) as pool:
        first = pool.submit(embedder._load_model)
        try:
            assert entered.wait(10)
            second = pool.submit(embedder._load_model)
            assert embedder._model is None
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert factory.call_count == 1
    assert embedder._active_provider == "CPU"


def test_embedding_arrays_are_released_as_the_generator_advances():
    """Streaming conversion avoids holding every ONNX batch until completion."""
    references = []

    def vectors(texts, batch_size):
        for _ in texts:
            if len(references) > 1:
                assert references[-2]() is None, "old ONNX vector retained"
            vector = np.ones(384, dtype=np.float32)
            references.append(weakref.ref(vector))
            yield vector

    embedder = FastEmbedEmbeddings()
    embedder._model = _model()
    embedder._model.embed.side_effect = vectors
    assert len(embedder(["text"] * 10)) == 10


@pytest.mark.parametrize("api", ["__call__", "embed_documents", "embed_query"])
@pytest.mark.parametrize(
    "bad_vector",
    [np.ones(3), np.ones((384, 1)), np.full(384, np.nan), np.full(384, np.inf), np.full(384, "bad")],
    ids=["wrong-dimension", "wrong-rank", "nan", "infinity", "nonnumeric"],
)
def test_all_embedding_vectors_are_validated(api, bad_vector):
    """Invalid vectors after a valid first vector must never reach the index."""
    embedder = FastEmbedEmbeddings()
    model = _model()
    model.embed.side_effect = lambda texts, batch_size: iter([np.ones(384), bad_vector])
    embedder._model = model
    embedder._active_provider = "CPU"
    with pytest.raises(EmbeddingError):
        getattr(embedder, api)(["valid", "invalid"])


def test_gpu_session_probe_failure_is_not_success():
    """A failed ONNX probe must fail closed even if provider and DLLs exist."""
    with (
        patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]),
        patch("mcp_server.server.os.path.isfile", return_value=True),
        patch("mcp_server.server.os.path.isdir", return_value=False),
        patch("mcp_server.server.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="GPU, 12288")),
        patch("onnxruntime.InferenceSession", side_effect=RuntimeError("driver initialization failed")),
    ):
        status = FastEmbedEmbeddings.verify_gpu_readiness()
    assert status.available is False
    assert "driver initialization failed" in status.fallback_reason


def test_gpu_probe_graph_can_really_execute_on_ort_cpu():
    """Exercise the serialized graph with real ORT, without requiring a GPU."""
    import onnxruntime as ort

    factory = ort.InferenceSession

    def cpu_session(model, **kwargs):
        session = factory(model, providers=["CPUExecutionProvider"])
        return SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"], run=session.run)

    with (
        patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]),
        patch("mcp_server.server.os.path.isfile", return_value=True),
        patch("mcp_server.server.os.path.isdir", return_value=False),
        patch("mcp_server.server.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="GPU, 12288")),
        patch("onnxruntime.InferenceSession", side_effect=cpu_session),
    ):
        status = FastEmbedEmbeddings.verify_gpu_readiness()
    assert status.available is True, status.fallback_reason
