"""Coverage for GPU auto-detect tri-state routing (v4.8.0 Fase 2).

Tests the tri-state gpu_mode ("auto" | "true" | "false") and its effect on
FastEmbedEmbeddings._load_model routing. All tests mock verify_gpu_readiness
and TextEmbedding so they do not require a real GPU or downloadable model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mcp_server.config import Config
from mcp_server.server import FastEmbedEmbeddings, GPUStatus

# ============================================================================
# Config normalization — gpu_mode accepts legacy bool AND new string values
# ============================================================================


class TestGpuModeNormalization:
    """Verify Config.__post_init__ normalizes gpu_mode to the tri-state string."""

    def test_legacy_bool_true_normalizes_to_true_string(self):
        c = Config(gpu_mode=True)
        assert c.gpu_mode == "true"
        assert c.gpu_acceleration is True

    def test_legacy_bool_false_normalizes_to_false_string(self):
        c = Config(gpu_mode=False)
        assert c.gpu_mode == "false"
        assert c.gpu_acceleration is False

    def test_string_auto_stays_auto(self):
        c = Config(gpu_mode="auto")
        assert c.gpu_mode == "auto"
        # "auto" means CUDA MAY be attempted, so the legacy alias is True
        assert c.gpu_acceleration is True

    def test_string_true_stays_true(self):
        c = Config(gpu_mode="true")
        assert c.gpu_mode == "true"
        assert c.gpu_acceleration is True

    def test_string_false_stays_false(self):
        c = Config(gpu_mode="false")
        assert c.gpu_mode == "false"
        assert c.gpu_acceleration is False

    def test_string_is_case_insensitive_and_stripped(self):
        c = Config(gpu_mode="  AUTO  ")
        assert c.gpu_mode == "auto"

    def test_invalid_string_falls_back_to_auto(self, capsys):
        c = Config(gpu_mode="banana")
        assert c.gpu_mode == "auto"
        assert c.gpu_acceleration is True
        captured = capsys.readouterr()
        assert "Invalid gpu value" in captured.out
        assert "banana" in captured.out

    def test_invalid_type_falls_back_to_auto(self, capsys):
        c = Config(gpu_mode=42)  # type: ignore[arg-type]
        assert c.gpu_mode == "auto"
        assert c.gpu_acceleration is True
        captured = capsys.readouterr()
        assert "Invalid gpu value" in captured.out


# ============================================================================
# _load_model routing — dispatches on gpu_mode
# ============================================================================


class TestLoadModelRouting:
    """Verify _load_model routes correctly per gpu_mode without touching a real GPU."""

    @patch("mcp_server.server.FastEmbedEmbeddings.verify_gpu_readiness")
    @patch("mcp_server.server.TextEmbedding")
    def test_gpu_mode_false_skips_probe_and_loads_cpu(self, mock_te, mock_verify):
        # Arrange
        mock_te.return_value = MagicMock()
        e = FastEmbedEmbeddings()
        e._gpu_mode = "false"

        # Act
        e._load_model()

        # Assert — probe was NEVER called (zero startup overhead when forced CPU)
        mock_verify.assert_not_called()
        # And TextEmbedding got CPU-only providers
        _, kwargs = mock_te.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]

    @patch("mcp_server.server.FastEmbedEmbeddings.verify_gpu_readiness")
    @patch("mcp_server.server.TextEmbedding")
    def test_gpu_mode_auto_probes_and_uses_cuda_when_available(self, mock_te, mock_verify):
        # Arrange
        mock_verify.return_value = GPUStatus(
            available=True,
            provider="CUDAExecutionProvider",
            device_name="Fake GPU",
            vram_mb=8192,
        )
        mock_te.return_value = MagicMock()
        e = FastEmbedEmbeddings()
        e._gpu_mode = "auto"

        # Act
        e._load_model()

        # Assert — probe called, CUDA providers used with CPU fallback in the list
        mock_verify.assert_called_once()
        _, kwargs = mock_te.call_args
        assert kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @patch("mcp_server.server.FastEmbedEmbeddings.verify_gpu_readiness")
    @patch("mcp_server.server.TextEmbedding")
    def test_gpu_mode_auto_falls_back_to_cpu_when_probe_fails(self, mock_te, mock_verify, capsys):
        # Arrange — probe reports GPU unavailable
        mock_verify.return_value = GPUStatus(
            available=False,
            fallback_reason="CUDAExecutionProvider not in onnxruntime providers",
        )
        mock_te.return_value = MagicMock()
        e = FastEmbedEmbeddings()
        e._gpu_mode = "auto"

        # Act
        e._load_model()

        # Assert — CPU-only providers passed to TextEmbedding
        _, kwargs = mock_te.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        # And the fallback reason is surfaced in the banner
        captured = capsys.readouterr()
        assert "GPU STATUS: UNAVAILABLE" in captured.out
        assert "auto-cpu-fallback" in captured.out
        assert "CUDAExecutionProvider not in onnxruntime providers" in captured.out

    @patch("mcp_server.server.FastEmbedEmbeddings.verify_gpu_readiness")
    @patch("mcp_server.server.TextEmbedding")
    def test_gpu_mode_true_forces_cuda_and_falls_back_when_probe_fails(self, mock_te, mock_verify, capsys):
        # Arrange — forced CUDA but GPU is not ready
        mock_verify.return_value = GPUStatus(
            available=False,
            fallback_reason="no driver",
        )
        mock_te.return_value = MagicMock()
        e = FastEmbedEmbeddings()
        e._gpu_mode = "true"

        # Act — must not raise (fallback path)
        e._load_model()

        # Assert — CPU providers passed, WARN emitted, banner reports forced-cuda-fallback
        _, kwargs = mock_te.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        captured = capsys.readouterr()
        assert "gpu: true but GPU not ready" in captured.out
        assert "forced-cuda-fallback" in captured.out

    @patch("mcp_server.server.FastEmbedEmbeddings.verify_gpu_readiness")
    @patch("mcp_server.server.TextEmbedding")
    def test_gpu_mode_true_falls_back_when_cuda_load_raises(self, mock_te, mock_verify, capsys):
        # Arrange — probe passes but TextEmbedding raises on CUDA providers,
        # then succeeds on CPU providers (call_args_list keeps both calls)
        mock_verify.return_value = GPUStatus(
            available=True,
            provider="CUDAExecutionProvider",
            device_name="Fake GPU",
        )
        cpu_instance = MagicMock()
        mock_te.side_effect = [RuntimeError("CUDA OOM"), cpu_instance]

        e = FastEmbedEmbeddings()
        e._gpu_mode = "true"

        # Act — must not raise (fallback path)
        e._load_model()

        # Assert — two calls total: first CUDA (raised), then CPU (succeeded)
        assert mock_te.call_count == 2
        first_call_kwargs = mock_te.call_args_list[0].kwargs
        second_call_kwargs = mock_te.call_args_list[1].kwargs
        assert first_call_kwargs["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert second_call_kwargs["providers"] == ["CPUExecutionProvider"]
        captured = capsys.readouterr()
        assert "CUDA load failed" in captured.out
        assert "forced-cuda-fallback" in captured.out
