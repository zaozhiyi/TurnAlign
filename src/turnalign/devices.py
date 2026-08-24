from __future__ import annotations

import importlib
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Protocol

from .plugins import Accelerator


class TorchProbe(Protocol):
    version: Any
    cuda: Any
    backends: Any


@dataclass(frozen=True, slots=True)
class Device:
    accelerator: Accelerator
    available: bool
    runtime: str
    device: str
    vendor: str
    name: str
    dtype: str
    reason: str = ""
    memory_bytes: int | None = None
    onnx_providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["accelerator"] = self.accelerator.value
        result["memory_gib"] = (
            round(self.memory_bytes / 1024**3, 2) if self.memory_bytes is not None else None
        )
        return result


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        # A partially installed ML runtime often raises a DLL/driver RuntimeError
        # rather than ImportError. Doctor must still be able to report CPU.
        return None


def _onnx_providers(ort: Any | None) -> tuple[str, ...]:
    if ort is None:
        return ()
    try:
        return tuple(ort.get_available_providers())
    except (AttributeError, RuntimeError):
        return ()


def _cuda_memory(torch: TorchProbe, index: int) -> int | None:
    try:
        return int(torch.cuda.get_device_properties(index).total_memory)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def detect_devices(torch: TorchProbe | None = None, ort: Any | None = None) -> list[Device]:
    """Detect usable inference targets without making torch a core dependency.

    PyTorch deliberately exposes ROCm devices through its ``torch.cuda`` API.
    ``torch.version.hip`` is therefore the reliable CUDA-vs-ROCm discriminator.
    Optional probe arguments make every platform branch testable on any host.
    """

    if torch is None:
        torch = _optional_import("torch")
    if ort is None:
        ort = _optional_import("onnxruntime")
    providers = _onnx_providers(ort)
    system = platform.system()
    machine = platform.machine()
    devices: list[Device] = []

    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
        except (AttributeError, RuntimeError):
            cuda_available = False
        if cuda_available:
            hip_version = getattr(getattr(torch, "version", None), "hip", None)
            accelerator = Accelerator.ROCM if hip_version else Accelerator.CUDA
            vendor = "AMD" if hip_version else "NVIDIA"
            runtime = f"ROCm {hip_version}" if hip_version else f"CUDA {getattr(torch.version, 'cuda', 'unknown')}"
            try:
                count = int(torch.cuda.device_count())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                count = 1
            for index in range(max(1, count)):
                try:
                    name = str(torch.cuda.get_device_name(index))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    name = f"{vendor} GPU {index}"
                devices.append(
                    Device(
                        accelerator=accelerator,
                        available=True,
                        runtime=runtime,
                        device=f"cuda:{index}",
                        vendor=vendor,
                        name=name,
                        dtype="float16",
                        memory_bytes=_cuda_memory(torch, index),
                        onnx_providers=tuple(
                            p for p in providers if p in {"CUDAExecutionProvider", "ROCMExecutionProvider"}
                        ),
                    )
                )

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        try:
            mps_available = bool(mps_backend and mps_backend.is_available())
        except (AttributeError, RuntimeError):
            mps_available = False
        if mps_available:
            devices.append(
                Device(
                    accelerator=Accelerator.MPS,
                    available=True,
                    runtime="PyTorch MPS",
                    device="mps",
                    vendor="Apple",
                    name=f"Apple Silicon ({machine})",
                    dtype="float16",
                    onnx_providers=tuple(p for p in providers if p == "CoreMLExecutionProvider"),
                )
            )

    devices.append(
        Device(
            accelerator=Accelerator.CPU,
            available=True,
            runtime="native",
            device="cpu",
            vendor=platform.processor() or machine or "unknown",
            name=f"{system} CPU",
            dtype="float32",
            onnx_providers=tuple(p for p in providers if p == "CPUExecutionProvider"),
        )
    )
    return devices


def select_device(
    requested: str | Accelerator = Accelerator.AUTO,
    *,
    supported: Iterable[Accelerator] | None = None,
    devices: Iterable[Device] | None = None,
) -> Device:
    """Select a device deterministically and fail clearly for explicit requests."""

    value = requested.value if isinstance(requested, Accelerator) else requested
    value = os.environ.get("TURNALIGN_DEVICE", value).strip().lower()
    requested_index: int | None = None
    if ":" in value:
        value, raw_index = value.split(":", 1)
        try:
            requested_index = int(raw_index)
        except ValueError as error:
            raise ValueError(f"invalid device index: {raw_index!r}") from error
    try:
        target = Accelerator(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in Accelerator if item not in {Accelerator.COREML, Accelerator.ONNX})
        raise ValueError(f"unknown device {value!r}; choose one of: {choices}") from error

    supported_set = set(supported or (Accelerator.CPU, Accelerator.CUDA, Accelerator.ROCM, Accelerator.MPS))
    candidates = [item for item in (devices or detect_devices()) if item.available]
    if target is Accelerator.AUTO:
        for accelerator in (Accelerator.CUDA, Accelerator.ROCM, Accelerator.MPS, Accelerator.CPU):
            if accelerator in supported_set:
                match = next((item for item in candidates if item.accelerator is accelerator), None)
                if match:
                    return match
        raise RuntimeError("no device is supported by both the host and the selected backend")

    if target not in supported_set:
        raise RuntimeError(f"backend does not support requested accelerator: {target.value}")
    matches = [item for item in candidates if item.accelerator is target]
    if requested_index is not None:
        matches = [item for item in matches if item.device.endswith(f":{requested_index}")]
    if not matches:
        available = ", ".join(sorted({item.accelerator.value for item in candidates})) or "none"
        raise RuntimeError(f"requested accelerator {target.value!r} is unavailable; available: {available}")
    return matches[0]


def runtime_report(requested: str | Accelerator = Accelerator.AUTO) -> dict[str, Any]:
    devices = detect_devices()
    selected = select_device(requested, devices=devices)
    torch = _optional_import("torch")
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "torch": getattr(torch, "__version__", None),
        "requested": requested.value if isinstance(requested, Accelerator) else requested,
        "selected": selected.to_dict(),
        "backend_plan": backend_plan(selected),
        "devices": [device.to_dict() for device in devices],
    }


def backend_plan(device: Device) -> dict[str, Any]:
    """Translate a selected accelerator into stable backend-facing settings."""

    plan: dict[str, Any] = {
        "pytorch": {"device": device.device, "dtype": device.dtype},
        "funasr": {"device": device.device},
        "transformers": {"device": device.device, "torch_dtype": device.dtype},
        "onnxruntime": {"providers": list(device.onnx_providers)},
        "environment": {},
        "recommended_engines": [],
        "unsupported_engines": [],
    }
    if device.accelerator is Accelerator.ROCM:
        plan["recommended_engines"] = ["transformers", "funasr-pytorch"]
        plan["unsupported_engines"] = ["faster-whisper/CTranslate2 GPU"]
        plan["notes"] = ["PyTorch ROCm intentionally uses cuda:N device strings."]
    elif device.accelerator is Accelerator.CUDA:
        plan["recommended_engines"] = ["faster-whisper", "transformers", "funasr-pytorch"]
    elif device.accelerator is Accelerator.MPS:
        plan["recommended_engines"] = ["transformers", "funasr-pytorch"]
        plan["unsupported_engines"] = ["faster-whisper/CTranslate2 GPU"]
        plan["environment"] = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        plan["notes"] = ["Fallback lets unsupported MPS operators run on CPU instead of aborting."]
    else:
        plan["recommended_engines"] = ["onnxruntime", "faster-whisper", "transformers", "funasr-pytorch"]
        plan["notes"] = ["Use int8 where the chosen CPU backend and model support it."]
    return plan
