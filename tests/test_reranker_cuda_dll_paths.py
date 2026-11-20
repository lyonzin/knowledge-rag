"""GH #317: reranker loads cudnn DLLs before building its ORT session.

Regression — when the first search is keyword-only, the reranker is
instantiated before the embedding model and used to rely on PATH entries
only _setup_cuda_dll_paths registers. This test freezes the invariant:
if _providers returns a CUDA provider, _setup_cuda_dll_paths runs first.
"""

from __future__ import annotations

from unittest.mock import patch


def test_reranker_registers_cuda_dll_paths_before_session(monkeypatch):
    from mcp_server.server import _CUDA_PROVIDER, CrossEncoderReranker

    monkeypatch.setattr("mcp_server.server.config.reranker_enabled", True)

    reranker = CrossEncoderReranker()
    call_order: list[str] = []

    def fake_setup():
        call_order.append("setup_cuda_dll_paths")

    def fake_cross_encoder(*args, **kwargs):
        call_order.append("TextCrossEncoder")
        raise RuntimeError("stop after construction attempt")

    with (
        patch.object(
            CrossEncoderReranker,
            "_providers",
            return_value=[_CUDA_PROVIDER, "CPUExecutionProvider"],
        ),
        patch(
            "mcp_server.server.FastEmbedEmbeddings._setup_cuda_dll_paths",
            side_effect=fake_setup,
        ),
        patch("mcp_server.server.TextCrossEncoder", side_effect=fake_cross_encoder),
    ):
        reranker._ensure_model()

    assert call_order == ["setup_cuda_dll_paths", "TextCrossEncoder"], (
        f"Reranker must register CUDA DLL paths before building its session; got {call_order}"
    )


def test_reranker_skips_cuda_dll_setup_on_cpu_provider(monkeypatch):
    from mcp_server.server import CrossEncoderReranker

    monkeypatch.setattr("mcp_server.server.config.reranker_enabled", True)

    reranker = CrossEncoderReranker()
    setup_called = []

    with (
        patch.object(
            CrossEncoderReranker,
            "_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch(
            "mcp_server.server.FastEmbedEmbeddings._setup_cuda_dll_paths",
            side_effect=lambda: setup_called.append(True),
        ),
        patch("mcp_server.server.TextCrossEncoder", side_effect=RuntimeError("stop")),
    ):
        reranker._ensure_model()

    assert setup_called == [], "CPU-only reranker must not touch NVIDIA DLL paths"
