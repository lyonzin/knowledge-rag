"""DirectML is opt-in, fail-closed, and serialized independently of CUDA."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import onnxruntime as ort
import pytest

from mcp_server import server
from mcp_server.config import Config


def _model(providers, *, memory_patterns=False, sequential=True):
    model = MagicMock()
    model.model.model.get_providers.return_value = providers
    model.model.model.get_session_options.return_value = SimpleNamespace(
        enable_mem_pattern=memory_patterns,
        execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL if sequential else ort.ExecutionMode.ORT_PARALLEL,
    )
    model.embed.side_effect = lambda texts, **kw: (np.ones(384, dtype=np.float32) for _ in texts)
    return model


@pytest.fixture
def directml(monkeypatch):
    monkeypatch.setattr(server.config, "gpu_mode", "directml")
    monkeypatch.setattr(server.config, "gpu_device_id", 1, raising=False)
    monkeypatch.setattr(server.config, "embed_batch_size", None)
    monkeypatch.setattr(server.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.delenv("KNOWLEDGE_RAG_EMBED_BATCH_SIZE", raising=False)
    return server.FastEmbedEmbeddings()


def test_directml_configuration_is_explicit():
    cfg = Config(gpu_mode=" DirectML ", gpu_device_id=1)
    assert cfg.gpu_mode == "directml"
    assert cfg.gpu_device_id == 1
    assert cfg.gpu_acceleration
    assert Config().gpu_mode == "auto"


@pytest.mark.parametrize("invalid", [True, False, -1, 2**31, "1", 1.5])
def test_directml_device_id_rejects_bool_string_negative_or_overflow(invalid):
    cfg = Config(gpu_mode="directml", gpu_device_id=invalid)
    assert cfg.gpu_device_id is None


def test_directml_uses_explicit_adapter_actual_provider_and_small_batch(directml, monkeypatch, capsys):
    model = _model(["DmlExecutionProvider", "CPUExecutionProvider"])
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(server, "TextEmbedding", factory)
    result = directml.embed_documents(["document"])
    assert len(result) == 1
    assert directml._active_provider == "DirectML"
    assert factory.call_args.kwargs["providers"] == [
        ("DmlExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]
    assert model.embed.call_args.kwargs["batch_size"] == 8
    assert "GPU STATUS: ACTIVE" in capsys.readouterr().err


@pytest.mark.parametrize("platform", ["Linux", "Darwin"])
def test_directml_non_windows_only_loads_cpu(directml, monkeypatch, platform):
    monkeypatch.setattr(server.platform, "system", lambda: platform)
    factory = MagicMock(return_value=_model(["CPUExecutionProvider"]))
    monkeypatch.setattr(server, "TextEmbedding", factory)
    directml.embed_query("query")
    assert factory.call_count == 1
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert directml._active_provider == "CPU"


@pytest.mark.parametrize("reason", ["missing-device", "bad-device", "missing-provider"])
def test_directml_unavailable_falls_back_without_attempting_dml(directml, monkeypatch, reason):
    if reason == "missing-provider":
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    else:
        directml._gpu_device_id = None if reason == "missing-device" else -1
    factory = MagicMock(return_value=_model(["CPUExecutionProvider"]))
    monkeypatch.setattr(server, "TextEmbedding", factory)
    directml.embed_query("query")
    assert factory.call_count == 1
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize("unsafe", ["memory-patterns", "parallel", "missing-options"])
def test_unsafe_directml_session_never_runs_or_becomes_published(directml, monkeypatch, unsafe):
    model = _model(
        ["DmlExecutionProvider"], memory_patterns=unsafe == "memory-patterns", sequential=unsafe != "parallel"
    )
    if unsafe == "missing-options":
        model.model.model.get_session_options = None
    cpu = _model(["CPUExecutionProvider"])
    factory = MagicMock(side_effect=[model, cpu])
    monkeypatch.setattr(server, "TextEmbedding", factory)
    directml.embed_query("query")
    model.embed.assert_not_called()
    assert directml._model is cpu
    assert directml._active_provider == "CPU"
    assert cpu.embed.call_args.kwargs["batch_size"] == 32


def test_directml_silent_cpu_fallback_reuses_valid_cpu_session(directml, monkeypatch, capsys):
    factory = MagicMock(return_value=_model(["CPUExecutionProvider"]))
    monkeypatch.setattr(server, "TextEmbedding", factory)
    directml.embed_query("query")
    assert factory.call_count == 1
    assert directml._active_provider == "CPU"
    assert "GPU STATUS: ACTIVE" not in capsys.readouterr().err


def test_directml_load_exception_falls_back_to_cpu(directml, monkeypatch):
    factory = MagicMock(side_effect=[RuntimeError("DML device unavailable"), _model(["CPUExecutionProvider"])])
    monkeypatch.setattr(server, "TextEmbedding", factory)
    assert len(directml.embed_query("query")) == 1
    assert directml._active_provider == "CPU"


def test_directml_serializes_generator_consumption_for_concurrent_api_calls(directml, monkeypatch):
    model = _model(["DmlExecutionProvider"])
    entered, release = threading.Event(), threading.Event()
    calls = []

    def embed(texts, **kwargs):
        calls.append(texts)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        yield np.ones(384, dtype=np.float32)

    model.embed.side_effect = embed
    monkeypatch.setattr(server, "TextEmbedding", lambda **kwargs: model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(directml.embed_query, "first")
        assert entered.wait(5)
        assert directml._inference_lock.locked()
        second = pool.submit(directml.embed_documents, ["second"])
        try:
            assert len(calls) == 1
        finally:
            release.set()
        assert len(first.result(timeout=5)) == len(second.result(timeout=5)) == 1
    assert len(calls) == 2
