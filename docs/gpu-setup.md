# GPU Setup for knowledge-rag v4.8.0+

## TL;DR

The default `gpu: auto` probes CUDA at startup and uses it if all deps are installed. If not, it silently falls back to CPU with a diagnostic banner explaining why. You do not need to touch config to enable CUDA — installing the dependency chain is enough.

## Modes

| `models.embedding.gpu` | Behavior | Cold-start overhead when GPU absent |
|---|---|---|
| `"auto"` (default) | Probe. Use CUDA if all four checks pass. CPU otherwise. | ~500ms–2s (probe rejects fast on first failing check) |
| `"true"` | Force CUDA attempt. Fall back to CPU only if the actual `TextEmbedding(providers=["CUDA..."])` load raises. | Same as auto |
| `"false"` | Never probe. Load CPU directly. | 0ms |

Legacy bool `gpu: true` and `gpu: false` continue to work; they are normalized to `"true"` and `"false"` strings internally.

## GPU auto-detect flow

`FastEmbedEmbeddings.verify_gpu_readiness()` runs four independent checks in this order. The first failure short-circuits and returns a `GPUStatus` with a `fallback_reason`.

1. **Provider check** — `CUDAExecutionProvider` must be present in `onnxruntime.get_available_providers()`. If `onnxruntime-gpu` is not installed (only the CPU-only `onnxruntime` package), this check fails immediately.
2. **DLL check** — All three required NVIDIA runtime libraries must be reachable via the process `PATH`:
   - Windows: `cudart64_12.dll`, `cudnn64_9.dll`, `cublasLt64_12.dll`
   - Linux: `libcudart.so.12`, `libcudnn.so.9`, `libcublasLt.so.12`

   Search order: process `PATH`, then every `<site-packages>/nvidia/*/bin/` and `<site-packages>/nvidia/*/lib/` inside the venv. Falls back to `ctypes.WinDLL` / `ctypes.CDLL` for system-wide installs.
3. **Device check** — `nvidia-smi --query-gpu=name,memory.total` must exit 0. Populates `device_name` and `vram_mb` on success.
4. **Minimal ONNX session** — Creates a trivial 1-node ONNX graph and instantiates `InferenceSession(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`. Confirms the CUDA provider is actually the active one (guards against ORT silently falling back to CPU).

If all four pass, `available=True` and CUDA loads. Any one fails → CPU, with `fallback_reason` printed in the startup banner.

## Dependency chain (pip)

CUDA 12 variant only — **CUDA 13 is NOT supported by FastEmbed as of writing**. Installing plain `pip install onnxruntime-gpu` from PyPI will pull the CUDA 13 build on some indexes and silently break the CUDA provider.

Install inside the same venv where `knowledge-rag` runs:

```bash
# 1. onnxruntime-gpu, CUDA 12 variant
pip install onnxruntime-gpu \
  --extra-index-url \
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 2. NVIDIA runtime libraries (pulls the DLLs/.so files the probe looks for)
pip install \
  nvidia-cudnn-cu12 \
  nvidia-cublas-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cufft-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-cuda-nvrtc-cu12
```

Host prerequisites:

- **NVIDIA driver** ≥ 525 (for CUDA 12 runtime compatibility). Check with `nvidia-smi`.
- **cuDNN 9** — comes bundled with `nvidia-cudnn-cu12`. Do not install a separate system cuDNN.

## Verify readiness manually

Run this before loading the embedding model to get the full diagnostic dump:

```bash
python -c "from mcp_server.server import FastEmbedEmbeddings; \
  import json; \
  print(json.dumps(FastEmbedEmbeddings.verify_gpu_readiness().__dict__, \
  default=str, indent=2))"
```

Expected output when everything is ready:

```json
{
  "available": true,
  "provider": "CUDAExecutionProvider",
  "device_name": "NVIDIA GeForce RTX 3080 Ti",
  "vram_mb": 12288,
  "missing_deps": [],
  "fallback_reason": null
}
```

Expected output when the driver is fine but pip deps are missing:

```json
{
  "available": false,
  "provider": "CPUExecutionProvider",
  "device_name": "",
  "vram_mb": 0,
  "missing_deps": ["cudnn64_9.dll (pip install nvidia-cudnn-cu12)"],
  "fallback_reason": "Missing CUDA dependencies: cudnn64_9.dll (pip install nvidia-cudnn-cu12)"
}
```

## Startup banner

Since v4.8.0, the banner prints on **every** startup path, not only when CUDA succeeds. Reading the banner is the fastest way to know what actually happened.

Success (auto mode probed and CUDA loaded):

```
============================================================
  GPU STATUS: ACTIVE
  Provider:   CUDAExecutionProvider
  Device:     NVIDIA GeForce RTX 3080 Ti
  VRAM:       12.0 GB
  Mode:       auto-cuda
============================================================
```

Auto-fallback (probe failed, CPU loaded):

```
============================================================
  GPU STATUS: UNAVAILABLE — running on CPU
  Reason:     CUDAExecutionProvider not in onnxruntime providers (available: CPUExecutionProvider, AzureExecutionProvider). Fix: pip install onnxruntime-gpu
  Mode:       auto-cpu-fallback
  Hint:       pip install onnxruntime-gpu --extra-index-url \
              https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
              plus nvidia-cudnn-cu12, nvidia-cublas-cu12, nvidia-cuda-runtime-cu12
============================================================
```

Forced CPU (gpu: false, no probe, zero overhead):

```
============================================================
  GPU STATUS: UNAVAILABLE — running on CPU
  Mode:       forced-cpu
============================================================
```

## Troubleshooting

### `cublasLt64_12.dll` missing on Windows

Symptom: probe reports `Missing CUDA dependencies: cublasLt64_12.dll (pip install nvidia-cublas-cu12)`. `nvidia-cublas-cu12` is installed but the DLL is not found.

Cause: `pip install onnxruntime-gpu` pulled the wrong variant (CUDA 13 build) which does not co-load with `nvidia-cublas-cu12`'s CUDA 12 DLLs.

Fix: uninstall and reinstall with the `--extra-index-url`:

```bash
pip uninstall onnxruntime-gpu
pip install onnxruntime-gpu \
  --extra-index-url \
  https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### `nvidia-smi not found on PATH`

Symptom: probe reports `nvidia-smi not found on PATH`.

Cause: NVIDIA driver not installed, or `nvidia-smi` is not on the user's `PATH` (common on Windows if the driver was installed with a custom option).

Fix: reinstall the NVIDIA driver from `https://www.nvidia.com/drivers/` and let the installer add `nvidia-smi` to `PATH`. On Linux, `nvidia-smi` ships with the driver package.

### Driver too old for CUDA 12

Symptom: `nvidia-smi` runs, but the probe fails at the ONNX session with `CUDA driver version is insufficient for CUDA runtime version`.

Fix: upgrade the NVIDIA driver to ≥ 525. On Linux (Ubuntu/Debian):

```bash
sudo apt install nvidia-driver-535
sudo reboot
```

### WSL2 sees no GPU

Symptom: `nvidia-smi` fails inside WSL2 even though it works on Windows host.

Cause: NVIDIA driver for WSL2 requires the Windows-side driver to be recent (≥ 470.14) AND the WSL2 kernel to be up to date.

Fix: on Windows, update the NVIDIA driver. In WSL2, run `wsl --update` from PowerShell. Do not install any `nvidia-driver-*` package inside WSL2 — the driver is virtualized through the Windows host.

### Docker/Podman container has no GPU access

Symptom: probe fails inside container even though host has GPU.

Fix: install `nvidia-container-toolkit` and run with `--gpus all`:

```bash
# Docker
docker run --gpus all knowledge-rag:latest

# Podman
podman run --device nvidia.com/gpu=all knowledge-rag:latest
```

### ORT silently fell back to CPU

Symptom: probe passes but banner says `CUDA session created but active provider is CPUExecutionProvider. ORT silently fell back to CPU.`

Cause: onnxruntime accepted the CUDAExecutionProvider registration but failed to initialize the CUDA session, then quietly used the CPU fallback provider from the list. Usually a version mismatch between `onnxruntime-gpu` and the NVIDIA runtime libs.

Fix: reinstall onnxruntime-gpu with the pinned CUDA 12 variant (see dependency chain above), and make sure all `nvidia-*-cu12` packages are the same major version.

## Performance expectations

Reindexing 3865 markdown docs (v4.8.0 reference corpus):

| Setup | Wall time | Speedup |
|---|---|---|
| CPU-only (Ryzen 9 5950X) | ~4.4h | 1.0x |
| GPU (RTX 3080 Ti, CUDA 12) | ~15m | ~17x |

Search latency is essentially identical between CPU and GPU because it only needs one query embedding — GPU wins on indexing, not per-query cost.
