"""GH #257: CUDA sessions must return their arena instead of pinning peak VRAM.

ONNX Runtime's default CUDA arena grows by powers of two and keeps every
region until the session is destroyed, so one indexing batch or one wide
rerank held gigabytes of VRAM for the life of the server. These tests mock
ORT and FastEmbed; the GPU measurement is the manual check in docs/gpu-setup.md.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest

from mcp_server.server import CrossEncoderReranker, FastEmbedEmbeddings, GPUStatus, config

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
DIRECTML = "DmlExecutionProvider"
SHRINK_KEY = "memory.enable_memory_arena_shrinkage"
SAME_AS_REQUESTED = (CUDA, {"arena_extend_strategy": "kSameAsRequested"})
NO_SHRINK_WARNING = "[WARN] CUDA arena shrink not installed"
# Serialized ModelProto: float[1] x -> Add(x, x) -> float[1] y (IR 7, opset 13), the
# same graph as the server's GPU probe, so the test needs no `onnx` dependency.
ADD_MODEL = (
    b"\x08\x07\x12\x0dknowledge-rag\x3a\x3d"
    b"\x0a\x0e\x0a\x01x\x0a\x01x\x12\x01y\x22\x03Add"
    b"\x12\x09gpu_probe"
    b"\x5a\x0f\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01"
    b"\x62\x0f\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01"
    b"\x42\x02\x10\x0d"
)


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


class _RenamedApiSession(_Session):
    """A CUDA session from an ONNX Runtime whose provider-options API changed."""

    def get_provider_options(self):
        raise AttributeError("get_provider_options was renamed")


def _embedding_model(providers):
    """TextEmbedding stand-in with FastEmbed's ``model.model.model`` session nesting."""
    return SimpleNamespace(model=SimpleNamespace(model=_Session(providers)))


def _load_as_embedder(model, monkeypatch):
    """Load ``model`` through the embedder's CUDA path; return what it published."""
    embedder = FastEmbedEmbeddings()
    with patch("mcp_server.server.TextEmbedding", return_value=model):
        embedder._load_with_providers([SAME_AS_REQUESTED, CPU], "test")
    return embedder._model


def _load_as_reranker(model, monkeypatch):
    """Load ``model`` as a CUDA reranker; ``None`` means reranking was turned off."""
    monkeypatch.setattr(config, "gpu_mode", "auto")
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])
    reranker = CrossEncoderReranker()
    with patch("mcp_server.server.TextCrossEncoder", return_value=model):
        reranker._ensure_model()
    return reranker._model


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


@pytest.mark.parametrize("load", [_load_as_embedder, _load_as_reranker], ids=["embedder", "reranker"])
def test_missing_session_warns_and_model_still_loads(load, monkeypatch, capsys):
    """Given FastEmbed moved its session, when a CUDA model loads, then it warns instead of failing or going quiet."""
    model = SimpleNamespace()

    loaded = load(model, monkeypatch)

    assert loaded is model
    assert NO_SHRINK_WARNING in capsys.readouterr().err


@pytest.mark.parametrize("load", [_load_as_embedder, _load_as_reranker], ids=["embedder", "reranker"])
def test_changed_ort_api_warns_and_leaves_runs_untouched(load, monkeypatch, capsys):
    """Given ORT renamed an API the shrink needs, when a CUDA model loads, then it warns, loads and runs as before."""
    session = _RenamedApiSession([CUDA, CPU])
    model = SimpleNamespace(model=SimpleNamespace(model=session))

    loaded = load(model, monkeypatch)
    session.run(["out"], {})

    assert loaded is model
    assert NO_SHRINK_WARNING in capsys.readouterr().err
    assert session.run_options == [None]


def test_shrink_wraps_run_on_a_real_onnxruntime_session(monkeypatch):
    """Given a real InferenceSession, when the shrink is installed, then ORT runs it with the shrink options.

    The session runs on CPU but reports CUDA so the installer acts; the arena name points at
    the CPU arena, the only one it has. A spy between wrapper and ORT records the options.
    """
    session = ort.InferenceSession(ADD_MODEL, providers=[CPU])
    real_run, seen = session.run, []

    def spy(output_names, input_feed, run_options=None):
        seen.append(run_options.get_run_config_entry(SHRINK_KEY))
        return real_run(output_names, input_feed, run_options)

    monkeypatch.setattr(session, "run", spy)
    monkeypatch.setattr(session, "get_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(session, "get_provider_options", lambda: {CUDA: {"device_id": "0"}, CPU: {}})
    monkeypatch.setattr("mcp_server.server._ARENA_SHRINKAGE_DEVICE", "cpu:{device_id}")
    with patch("mcp_server.server.TextEmbedding", return_value=SimpleNamespace(model=SimpleNamespace(model=session))):
        FastEmbedEmbeddings()._load_with_providers([SAME_AS_REQUESTED, CPU], "test")

    outputs = session.run(None, {"x": np.ones(1, dtype=np.float32)})

    assert (seen, outputs[0].tolist()) == (["cpu:0"], [2.0])


def _real_cpu_session_reporting_cuda(monkeypatch):
    """A real InferenceSession on CPU that reports CUDA, wrapped by the embedder's CUDA load.

    ORT itself then rejects the real "gpu:0" shrink request: this session has no GPU arena.
    """
    session = ort.InferenceSession(ADD_MODEL, providers=[CPU])
    monkeypatch.setattr(session, "get_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(session, "get_provider_options", lambda: {CUDA: {"device_id": "0"}, CPU: {}})
    with patch("mcp_server.server.TextEmbedding", return_value=SimpleNamespace(model=SimpleNamespace(model=session))):
        FastEmbedEmbeddings()._load_with_providers([SAME_AS_REQUESTED, CPU], "test")
    return session


def test_rejected_shrink_warns_once_and_hands_back_the_original_run(monkeypatch, capsys):
    """Given ORT has no arena for gpu:0, when FastEmbed runs twice, then both succeed and only the first warns."""
    session = _real_cpu_session_reporting_cuda(monkeypatch)
    feed = {"x": np.ones(1, dtype=np.float32)}

    first = session.run(None, feed)
    later = session.run(None, feed)

    assert first[0].tolist() == later[0].tolist() == [2.0]
    assert capsys.readouterr().err.count("[WARN] CUDA arena shrink disabled: InvalidArgument") == 1
    assert session.run.__func__ is ort.InferenceSession.run


@pytest.mark.parametrize(
    "feed",
    [{"wrong_name": np.ones(1, dtype=np.float32)}, {"x": np.ones(1, dtype=np.float64)}],
    ids=["wrong-input-name", "wrong-dtype-same-invalid-argument-type"],
)
def test_other_ort_errors_still_raise_and_keep_the_shrink(monkeypatch, capsys, feed):
    """Given the shrink is installed, when a run fails for another reason, then that error raises unchanged."""
    session = _real_cpu_session_reporting_cuda(monkeypatch)
    wrapper = session.run

    with pytest.raises(Exception) as raised:
        session.run(None, feed)

    assert "memory arena shrink list" not in str(raised.value)
    assert session.run is wrapper
    assert "CUDA arena shrink disabled" not in capsys.readouterr().err
