from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable

from .models import AudioChunk, Hypothesis, SpeakerTurn, Word


class Accelerator(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    COREML = "coreml"
    ONNX = "onnx"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    streaming: bool = False
    word_timestamps: bool = False
    hotwords: bool = False
    languages: tuple[str, ...] = ()
    accelerators: tuple[Accelerator, ...] = (Accelerator.CPU,)


@dataclass(frozen=True, slots=True)
class AsrConfig:
    """Common, serializable configuration passed to ASR plugins."""

    model: str | None = None
    device: str = "auto"
    language: str | None = None
    compute_type: str | None = None
    executable: str | None = None
    model_path: str | None = None
    extra: dict[str, Any] | None = None


@runtime_checkable
class AsrBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]: ...

    def close(self) -> None: ...


@runtime_checkable
class VadBackend(Protocol):
    name: str

    def accept(self, chunk: AudioChunk) -> Iterable[AudioChunk]: ...

    def flush(self) -> Iterable[AudioChunk]: ...


@runtime_checkable
class AlignmentBackend(Protocol):
    name: str

    def align(self, audio: AudioChunk, text: str) -> list[Word]: ...


@runtime_checkable
class DiarizationBackend(Protocol):
    name: str

    def diarize(self, chunks: Iterable[AudioChunk]) -> Iterable[SpeakerTurn]: ...
