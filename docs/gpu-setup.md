# GPU setup and CPU fallback

Compatibility information checked on **2026-10-03**. CPU is supported on Windows, Linux, and macOS. Automatic selection chooses NVIDIA CUDA or CPU. AMD on Windows is available through an experimental, explicit DirectML setting; Linux AMD and Apple GPU providers are not selected.

## Configuration and actual behavior

```yaml
models:
  embedding:
    gpu: "auto"
documents:
  embed_batch_size: null
```

| `models.embedding.gpu` | Behavior on the first embedding request |
|---|---|
| `"auto"` (default) | Probe CUDA, attempt it if ready, otherwise use CPU and print the reason. |
| `"true"` | Request CUDA with a warning and CPU fallback if the probe or model load fails. |
| `"false"` | Load CPU directly without a CUDA probe. |
| `"directml"` | On Windows, request the explicitly configured DXGI adapter through DirectML. Validate the loaded session's safety options; otherwise fall back to CPU with the reason. |

Legacy YAML booleans `true` and `false` remain accepted. Models load lazily: importing the server or completing the MCP handshake does not itself load an embedding model. Initial indexing can trigger that load in a background thread. Diagnostics go to stderr throughout the process, keeping stdout available for MCP JSON-RPC.

The loaded ONNX session determines the reported provider and default micro-batch: **CPU 32, CUDA 256, DirectML 8**. Requesting a GPU does not imply that it became active. A session that silently falls back to CPU uses the CPU batch and never receives an ACTIVE GPU banner.

`documents.embed_batch_size` sets an explicit positive micro-batch, separate from the ChromaDB write `documents.batch_size`. A positive integer in `KNOWLEDGE_RAG_EMBED_BATCH_SIZE` takes precedence; invalid environment values produce a warning and use the YAML/provider value. Values above 512 are clamped. Lower the batch when RAM or VRAM is constrained; larger batches are not guaranteed to be faster or safe for every model.

The GPU setting applies to embeddings. The cross-encoder reranker has an independent FastEmbed session and uses upstream provider defaults; this option does not configure or certify reranker acceleration. Search latency also includes retrieval and reranking, so embedding throughput alone does not predict end-to-end search latency.

## NVIDIA runtime compatibility

The [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) distinguish the CUDA major version of each wheel. As checked on 2026-10-03:

| ONNX Runtime GPU distribution | CUDA runtime | cuDNN |
|---|---|---|
| PyPI versions 1.21–1.26 | 12.8 or newer within CUDA 12 | 9.x |
| PyPI versions 1.27+ | 13.x by default | 9.x |
| Separate CUDA 12 builds of 1.27+ | 12.8 or newer within CUDA 12 | 9.x |

The project's `[gpu]` extra is constrained to `onnxruntime-gpu>=1.21,<1.27` and CUDA 12/cuDNN 9 libraries on Windows/Linux. This prevents a future CUDA 13 wheel from being combined automatically with the extra's CUDA 12 libraries. Python 3.11–3.14 Windows x64 and Linux x86-64 wheels exist for [ORT 1.26.0](https://pypi.org/project/onnxruntime-gpu/1.26.0/); wheel availability is not a substitute for testing a GPU/driver combination.

The current readiness probe checks CUDA 12 library names. A CUDA 13-only environment is not covered by that probe. An existing mixed CUDA 12/13 environment can pass and execute CUDA, but that is not a reproducible clean-install recipe. FastEmbed itself should not be described as categorically incompatible with CUDA 13.

`--extra-index-url` does **not** select a CUDA variant or prioritize an index. Pip compares candidates across the configured locations and selects a matching version. Use explicit versions and the appropriate wheel source when testing another runtime family. See [pip's package-selection rules](https://pip.pypa.io/en/stable/cli/pip_install/).

## Install in an isolated environment

Choose a GPU-compatible Python interpreter and create a separate environment first:

```bash
python -m venv .venv-gpu
```

Activate it with `.venv-gpu\Scripts\Activate.ps1` in PowerShell or `source .venv-gpu/bin/activate` on Linux. Install the matching NVIDIA host driver through your normal system administration process. Check the [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) against the chosen runtime; a driver version copied from an old guide is not sufficient validation.

```bash
python -m pip install "knowledge-rag[gpu]" "onnxruntime-gpu>=1.21,<1.27" "nvidia-cublas-cu12>=12.8,<13" "nvidia-cuda-runtime-cu12>=12.8,<13" "nvidia-cudnn-cu12>=9,<10"
python -m pip list
```

**Packaging limitation:** knowledge-rag's CPU dependencies include FastEmbed/ChromaDB, whose dependency metadata can also install the `onnxruntime` CPU distribution. CPU, GPU, and DirectML distributions share the `onnxruntime` import namespace and should not coexist. The extra alone cannot remove a transitive CPU wheel. FastEmbed's [GPU installation guidance](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/) describes this namespace conflict.

If both distributions appear, repair **only the isolated GPU environment** by removing both before reinstalling the chosen GPU wheel. Removing only the CPU distribution can delete files belonging to the GPU installation:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install "onnxruntime-gpu==1.26.0"
```

This is a runtime substitution: dependency metadata still names the CPU distribution, so `pip check` can report it missing, and later package upgrades can reinstall it. Preserve the environment's package inventory and repeat provider/inference verification after upgrades. A successful import or installation does not certify a clean dependency graph. Do not apply this sequence to a running production environment without first validating a replacement environment.

## Verify the real embedding path

First inspect what the installed runtime makes available:

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Then run the application's embedding path. This may download the model on first use if it is not already cached:

```python
from mcp_server.server import FastEmbedEmbeddings

embedding = FastEmbedEmbeddings()
vectors = embedding.embed_query("How is CPU fallback configured?")
print("vectors:", len(vectors), "dimensions:", len(vectors[0]))
print("session providers:", embedding._model.model.model.get_providers())
```

The final diagnostic line inspects the FastEmbed 0.8 session. Internal object paths can differ between upstream versions. For NVIDIA, expect `CUDAExecutionProvider` in the **loaded session**, followed by CPU fallback where appropriate. The probe's `available=True` means a small valid ONNX graph both loaded and produced the expected output; it is not proof that every node of the embedding model executes on GPU.

To prove graph placement, use [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) with `SessionOptions.enable_profiling=True` and inspect node events' `args.provider`. Quantized models can have CPU nodes even when the session includes CUDA. Record CPU node work as well as CUDA node work.

## Readiness checks and failures

`FastEmbedEmbeddings.verify_gpu_readiness()` checks, in order:

1. The runtime lists `CUDAExecutionProvider`.
2. CUDA 12 runtime, cuDNN 9, and cuBLAS libraries are discoverable. Windows names include `cudart64_12.dll`, `cudnn64_9.dll`, and `cublasLt64_12.dll`; Linux equivalents end in `.so.12`, `.so.9`, and `.so.12`.
3. `nvidia-smi` can report a device.
4. A tiny ONNX Add graph loads with CUDA present and produces the expected finite result.

Any probe exception is a failure and leads to CPU fallback. The application also checks the actual embedding session after model load. If CPU loading fails too, a model-load error is reported; fallback is not a promise that startup or indexing can never fail.

| Symptom | Investigation / action |
|---|---|
| CUDA missing from available providers | Confirm the interpreter path and installed runtime distributions; repair an isolated environment if CPU/GPU wheels overlap. |
| Missing DLL/shared library | Check the named file, runtime family, package location, and loader search path. A missing CUDA 12 DLL alone does not establish which ORT wheel was installed. |
| `nvidia-smi` absent or failing | Check driver installation and executable search path. Avoid replacing drivers just to fix a Python package conflict. |
| Provider listed but session uses CPU | Inspect ORT's stderr for loader/driver errors; verify CUDA and cuDNN compatibility, then rerun an actual embedding. |
| CUDA-only dependencies on macOS | Use CPU. This extra does not provide Apple GPU acceleration. |
| RAM/VRAM exhaustion | Reduce `documents.embed_batch_size`, measure peak memory, and test the intended model/corpus. Moving work to GPU does not eliminate host RAM use. |

Containers need host GPU access and a compatible runtime image. WSL2 uses the Windows host driver; follow [NVIDIA's WSL guidance](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) rather than installing a Linux display driver inside WSL.

## AMD and Apple GPU status

| Platform | Upstream option | Status in knowledge-rag |
|---|---|---|
| AMD on Windows | `DmlExecutionProvider` / ONNX Runtime DirectML | Experimental opt-in; validated on one integrated Radeon GPU with FastEmbed 0.8.0 and ORT DirectML 1.24.4. |
| AMD on Linux | `MIGraphXExecutionProvider` with a matching ROCm/GPU matrix | Not enabled or hardware validated in this audit. |
| macOS / Apple silicon | `CoreMLExecutionProvider` | Not selected; CPU remains the tested fallback path. |

[DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) requires sequential execution, disabled memory patterns, and serialized calls on a shared session. In the tested ORT DirectML 1.24.4 runtime, session creation automatically disables memory patterns even when the caller requested them. The application checks the **effective** options through `get_session_options()` and rejects a DirectML session unless `enable_mem_pattern=False` and `execution_mode=ORT_SEQUENTIAL`. It also holds a per-instance lock throughout embedding-generator consumption, preventing concurrent `Run` calls from document and query APIs. These checks use public runtime methods; no session replacement or global patch is installed.

Create a dedicated Windows environment with Python 3.12 and install knowledge-rag there first. Then replace its CPU runtime with the tested DirectML distribution, using that environment's interpreter:

```powershell
python -m pip install "knowledge-rag" "fastembed==0.8.0"
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
python -m pip install "onnxruntime-directml==1.24.4"
```

Inspect `python -m pip list` to confirm that DirectML is the only ONNX Runtime distribution. The packaging limitation and upgrade precautions in the NVIDIA section also apply here: FastEmbed/ChromaDB metadata still names the CPU distribution, and future dependency upgrades can reinstall it. Do not mix NVIDIA and DirectML runtime wheels in one environment.

```yaml
models:
  embedding:
    gpu: "directml"
    device_id: 1  # Example only: replace with your verified AMD DXGI adapter index.
documents:
  embed_batch_size: 8
```

`device_id` is the non-negative adapter ordinal returned by [DXGI `EnumAdapters`](https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgifactory-enumadapters), not an NVIDIA CUDA ordinal or PCI device ID. Confirm the adapter description and AMD vendor ID `0x1002` before selecting it. On the audit host, index 0 was NVIDIA and index 1 was AMD; the order can differ on another machine or after hardware changes. There is deliberately no assumed adapter when `device_id` is absent. Invalid/missing indices, a non-Windows host, unavailable DirectML, or unsafe session options cause CPU fallback. Device creation failures also fall back; inference failures after initialization remain visible errors.

DirectML is not enabled by `gpu: "auto"`. Integrated GPUs share memory with the system, and they can be slower than CPU for this quantized model. Measure the actual workload before choosing DirectML for routine indexing. The setting configures embeddings; it does not select the reranker's provider.

For Linux AMD, use the [MIGraphX documentation](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) and [AMD's supported hardware/runtime matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html). The older [ROCm execution provider](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) was removed from ORT 1.23. A provider name in an installed wheel does not establish support for an arbitrary integrated Radeon GPU.

[CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) has its own operator, shape, and device constraints. Enabling it safely requires per-model parity, placement, memory, and concurrency tests on actual Apple hardware.

## Validation and performance claims

The 2026-10-03 audit ran real BGE-small embeddings on Windows 11 with CPU, an NVIDIA RTX 3080 Ti (12 GiB), and AMD Radeon(TM) integrated graphics (driver 32.0.21045.9003). The CUDA 12 recipe above was also exercised in a new Python 3.12.7 environment: FastEmbed 0.8.1, ORT GPU 1.26.0, cuBLAS 12.9.2.10, CUDA runtime 12.9.79, and cuDNN 9.27.0.42. After removing both runtime distributions and reinstalling only ORT GPU, the application produced finite 384-dimensional vectors with CUDA active. `pip check` reported the expected missing CPU-distribution metadata for FastEmbed/ChromaDB; there was no CPU runtime left sharing the namespace.

An additional CUDA run used an existing Python 3.14/ORT 1.28 environment with mixed runtimes; it was not used as evidence for the clean-install recipe. The AMD run used an isolated Python 3.12.7 environment with FastEmbed 0.8.0 and ORT DirectML 1.24.4 as its only runtime distribution. Linux/macOS GPU execution was not verified.

Raw ONNX profiling of the same quantized BGE-small model recorded 95 CUDA and 2 CPU node events on NVIDIA, and 95 DirectML and 2 CPU events on AMD. The FastEmbed AMD/CPU smoke test produced finite 384-dimensional vectors with cosine similarity above 0.999995 for four texts. The application's DirectML path also completed a 16-document embedding workload using its default batch of 8. These measurements establish execution and basic numeric parity on these devices, not a universal acceleration or quality guarantee.

The offline [retrieval audit](../scripts/audit_retrieval.py) additionally passed on DirectML with seven public documents: finite stored vectors, matching Chroma/FTS5 counts, English/Portuguese queries, concurrent reads, incremental edit/delete, forced reindex, and reopening the persistent index. Run it with `--backend directml --device-id <verified-index> --cache <copied-model-cache> --output <report.json>` in the validated environment. It uses a disposable index and keeps corpus/model hashes in the report; the cross-encoder is disabled to isolate indexing and retrieval.

For a comparable benchmark, keep the model hash, corpus hash, Python/runtime versions, thread count, model prefixes, and batch settings in the report. Separate cold model load, warm embedding throughput, query embedding p50/p95, full search latency, and peak process RSS/VRAM. Include vector dimensions/finite checks and retrieval parity. Run CPU and GPU sequentially to avoid resource contention.

The previous approximate 17x reindex claim lacked a reproducible artifact and has been removed. A speedup for one batch/corpus is not a promise for all models or end-to-end indexing, and per-query CPU/GPU latency must be measured rather than assumed equal.
