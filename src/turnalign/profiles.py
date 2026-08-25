from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from .devices import Device, detect_devices, select_device
from .plugins import Accelerator


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    name: str
    asr_device: str
    vad_device: str
    alignment_device: str
    diarization_device: str
    alignment_batch_size: int
    parallel_diarization: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["notes"] = list(self.notes)
        return result


PROFILE_NAMES = (
    "auto",
    "mac-balanced",
    "cuda-single-gpu",
    "cuda-multi-gpu",
    "rocm-linux",
    "rocm-windows",
    "cpu-low-memory",
)


def _gpu_count(devices: Iterable[Device], accelerator: Accelerator) -> int:
    return sum(item.available and item.accelerator is accelerator for item in devices)


def _profile(name: str) -> ExecutionProfile:
    if name == "mac-balanced":
        return ExecutionProfile(
            name, "mps", "cpu", "cpu", "cpu", 4, True,
            ("Run MPS ASR alongside CPU diarization; unified memory still requires bounded batches.",),
        )
    if name == "cuda-single-gpu":
        return ExecutionProfile(
            name, "cuda:0", "cuda:0", "cuda:0", "cuda:0", 4, False,
            ("Keep single-GPU model stages scheduled and benchmark batches on the target GPU.",),
        )
    if name == "cuda-multi-gpu":
        return ExecutionProfile(
            name, "cuda:0", "cpu", "cuda:1", "cuda:1", 8, True,
            ("Keep ASR on GPU 0 and post-processing on GPU 1.",),
        )
    if name == "rocm-linux":
        return ExecutionProfile(
            name, "rocm:0", "cuda:0", "cuda:0", "cuda:0", 4, False,
            ("PyTorch ROCm components intentionally receive cuda:N device strings.",),
        )
    if name == "rocm-windows":
        return ExecutionProfile(
            name, "rocm:0", "cpu", "cpu", "cpu", 1, True,
            ("Use the supported PyTorch GPU path while keeping optional post-processing conservative.",),
        )
    if name == "cpu-low-memory":
        return ExecutionProfile(
            name, "cpu", "cpu", "cpu", "cpu", 2, False,
            ("Avoid model concurrency and use small alignment batches.",),
        )
    raise ValueError(f"unknown execution profile {name!r}; choose one of: {', '.join(PROFILE_NAMES)}")


def select_execution_profile(
    requested: str = "auto",
    *,
    requested_device: str = "auto",
    devices: Iterable[Device] | None = None,
    system: str | None = None,
) -> ExecutionProfile:
    available = list(devices or detect_devices())
    host_system = system or platform.system()
    selected_for_request = None
    if requested not in PROFILE_NAMES:
        return _profile(requested)

    if requested == "auto":
        selected = select_device(requested_device, devices=available)
        selected_for_request = selected
        if selected.accelerator is Accelerator.MPS:
            requested = "mac-balanced"
        elif selected.accelerator is Accelerator.CUDA:
            requested = (
                "cuda-multi-gpu"
                if requested_device == "auto" and _gpu_count(available, Accelerator.CUDA) >= 2
                else "cuda-single-gpu"
            )
        elif selected.accelerator is Accelerator.ROCM:
            requested = "rocm-windows" if host_system == "Windows" else "rocm-linux"
        else:
            requested = "cpu-low-memory"

    profile = _profile(requested)
    if selected_for_request is not None and requested_device != "auto":
        index = selected_for_request.device.split(":", 1)[-1]
        if profile.name == "cuda-single-gpu":
            target = f"cuda:{index}"
            profile = replace(
                profile,
                asr_device=target,
                vad_device=target,
                alignment_device=target,
                diarization_device=target,
            )
        elif profile.name in {"rocm-linux", "rocm-windows"}:
            profile = replace(profile, asr_device=f"rocm:{index}")
            if profile.name == "rocm-linux":
                target = f"cuda:{index}"
                profile = replace(
                    profile,
                    vad_device=target,
                    alignment_device=target,
                    diarization_device=target,
                )
    required_system = {
        "mac-balanced": "Darwin",
        "rocm-linux": "Linux",
        "rocm-windows": "Windows",
    }.get(profile.name)
    if required_system is not None and host_system != required_system:
        raise RuntimeError(
            f"execution profile {profile.name!r} requires {required_system}, found {host_system}"
        )
    required = {
        "mac-balanced": (Accelerator.MPS, 1),
        "cuda-single-gpu": (Accelerator.CUDA, 1),
        "cuda-multi-gpu": (Accelerator.CUDA, 2),
        "rocm-linux": (Accelerator.ROCM, 1),
        "rocm-windows": (Accelerator.ROCM, 1),
        "cpu-low-memory": (Accelerator.CPU, 1),
    }[profile.name]
    if _gpu_count(available, required[0]) < required[1]:
        raise RuntimeError(
            f"execution profile {profile.name!r} requires {required[1]} "
            f"{required[0].value} device(s)"
        )
    return profile


def profile_catalog() -> list[dict[str, object]]:
    return [_profile(name).to_dict() for name in PROFILE_NAMES if name != "auto"]
