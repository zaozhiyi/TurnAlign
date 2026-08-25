# Cross-platform inference

The orchestration core has no mandatory ML dependency. Heavy model runtimes are
installed per host, while `turnalign doctor` exposes one common selection result
to ASR, VAD, alignment and diarization plugins.

## Selection contract

| Host | Selected accelerator | PyTorch device | Default dtype | Preferred engines |
|---|---|---|---|---|
| NVIDIA | `cuda` | `cuda:N` | `float16` | faster-whisper, Transformers, FunASR |
| AMD ROCm | `rocm` | `cuda:N` | `float16` | Transformers, FunASR |
| Apple Silicon | `mps` | `mps` | `float16` | Transformers, FunASR |
| Any CPU | `cpu` | `cpu` | `float32` | ONNX Runtime, faster-whisper, Transformers |

## Execution profiles

Use `turnalign profiles` to inspect all policies and `turnalign doctor --device
auto` to see the selected one. Profiles are scheduling defaults, not capability
claims for every optional model on every GPU.

- Apple MPS keeps GLM/Transformers ASR on MPS and current FunASR components on
  CPU, overlapping diarization with ASR.
- A single NVIDIA GPU keeps ASR and post-processing on the same CUDA device and
  schedules stages to avoid VRAM contention.
- Multiple NVIDIA GPUs place ASR on `cuda:0` and post-processing on `cuda:1`, so
  the tracks can overlap without sharing one GPU.
- ROCm Linux keeps PyTorch stages on one AMD GPU. The public profile uses
  `rocm:0`, while PyTorch/FunASR correctly receives `cuda:0` under HIP.
- ROCm Windows uses a conservative GPU-ASR/CPU-post-processing split because the
  available PyTorch and ROCm component surface depends on the official hardware
  support matrix.
- CPU-only hosts avoid model concurrency and use a smaller alignment batch.

Explicit component options always override these defaults. Re-run a representative
recording after changing the profile because optimal batch size is hardware- and
model-specific.

PyTorch uses the `torch.cuda` namespace and `cuda:N` device strings for both
CUDA and ROCm. TurnAlign distinguishes them with `torch.version.hip`; a Radeon
will therefore be reported as `rocm` while model code correctly receives
`cuda:0`.

`faster-whisper` uses CTranslate2. Its GPU path targets NVIDIA CUDA, so TurnAlign
does not recommend that GPU engine on ROCm or MPS. On AMD and Mac, use the
PyTorch Transformers Whisper implementation or another PyTorch-native ASR.

## Common command

From a source checkout:

```bash
export PYTHONPATH="$PWD/src"       # PowerShell: $env:PYTHONPATH = "$PWD\src"
python -m turnalign.cli doctor --device auto
```

The JSON contains `selected` plus a `backend_plan`. Service deployments can set
one explicit value:

```bash
export TURNALIGN_DEVICE=cpu        # auto, cpu, cuda[:N], rocm[:N], mps
```

An explicit unavailable device fails fast. `auto` prefers CUDA, ROCm, MPS and
then CPU, restricted by the selected plugin's declared capabilities.

## macOS / Apple Silicon

Use a native arm64 Python environment, install a current macOS PyTorch build,
then run `doctor`. A successful report selects `mps` and supplies
`PYTORCH_ENABLE_MPS_FALLBACK=1` in the plan for operators without an MPS kernel.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip torch
python -m turnalign.cli doctor --device mps
```

Run the service natively on macOS; Docker Desktop does not pass Metal/MPS
through to Linux containers. Unified memory is shared with the system, so leave
headroom instead of treating all installed RAM as model memory.

## NVIDIA / CUDA

Install the NVIDIA driver and the PyTorch wheel matching the supported CUDA
runtime from the official PyTorch installer. Verify that `doctor` selects
`cuda:0`. Multiple GPUs can be pinned with `cuda:1`, etc.

For containers, expose GPUs through the NVIDIA Container Toolkit and keep model
weights in a mounted cache. CUDA is the only target where the CTranslate2 GPU
backend is recommended by the current runtime plan.

## AMD / ROCm

Install the ROCm-compatible PyTorch build recommended for the OS and GPU. Linux
is the broadly supported deployment target; this repository has additionally
been validated on Windows with AMD's ROCm-enabled PyTorch 2.9.1 build and an RX
7650 GRE.

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m turnalign.cli doctor --device rocm:0
```

The tested Windows host reports `rocm`, `cuda:0`, FP16 and 7.98 GiB VRAM. The
full 129.4-minute FunASR/Paraformer/CAM++ pipeline completed in 369.2 seconds
with 1.61 GB peak allocated VRAM.

For Linux containers, pass the AMD render and KFD devices and use a ROCm base
image compatible with the host driver. Exact flags and group IDs vary by host;
keep those deployment details outside the portable core.

## CPU fallback and ONNX

CPU is always present and is selected only when no supported accelerator is
usable, or when explicitly requested. If ONNX Runtime is installed, `doctor`
also reports available execution providers. Prefer quantized/int8 models only
when the chosen model and backend explicitly support them.

## Plugin integration

A plugin should call `select_device` with its capability list, then consume
`backend_plan` rather than duplicating vendor checks:

```python
from turnalign.devices import backend_plan, select_device
from turnalign.plugins import Accelerator

device = select_device(
    "auto",
    supported=(Accelerator.CPU, Accelerator.CUDA, Accelerator.ROCM, Accelerator.MPS),
)
plan = backend_plan(device)
model.to(plan["pytorch"]["device"])
```

This keeps model plugins replaceable and makes an unsupported explicit target a
startup error instead of a silent CPU fallback.

## Validation matrix

- Real AMD ROCm run: RX 7650 GRE, Windows, full ASR + diarization workload.
- Simulated hardware tests: two NVIDIA GPUs, two AMD GPUs, Apple MPS and CPU.
- Selection tests: automatic priority, plugin capability filtering, explicit
  index, environment override and unavailable-device failure.
- CLI test: `doctor` returns parseable JSON with a backend plan.

Physical Mac and NVIDIA performance runs are still required before publishing
platform-specific throughput claims; their control paths are covered without
pretending a simulated probe is a hardware benchmark.
