from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .hints import AsrHints
from .models import AudioChunk, Hypothesis, SpeakerTurn, SpeechSegment, Word


class Accelerator(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    MPS = "mps"
    COREML = "coreml"
    ONNX = "onnx"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    streaming: bool = False
    word_timestamps: bool = False
    hotwords: bool = False
    context_prompt: bool = False
    hotword_boost: bool = False
    min_chunk_ms: int | None = None
    lookahead_ms: int = 0
    external_vad: bool = True
    state_serialization: bool = False
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
    hints: AsrHints = field(default_factory=AsrHints)
    require_local_model: bool = False


@runtime_checkable
class AsrBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]: ...

    def close(self) -> None: ...


@runtime_checkable
class StreamingAsrSession(Protocol):
    """Per-audio-stream state owned separately from a shared model."""

    def accept_audio(self, chunk: AudioChunk) -> Iterable[Hypothesis]: ...

    def finish(self) -> Iterable[Hypothesis]: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class StatefulStreamingAsrBackend(AsrBackend, Protocol):
    def start_session(self) -> StreamingAsrSession: ...


@runtime_checkable
class OnlineDiarizationSession(Protocol):
    def accept_audio(self, chunk: AudioChunk) -> Iterable[SpeakerTurn]: ...

    def finish(self) -> Iterable[SpeakerTurn]: ...

    def close(self) -> None: ...


@runtime_checkable
class OnlineDiarizationBackend(Protocol):
    name: str

    def start_session(self) -> OnlineDiarizationSession: ...

    def close(self) -> None: ...


@runtime_checkable
class VadBackend(Protocol):
    name: str

    def segment(self, chunks: Iterable[AudioChunk]) -> Iterable[SpeechSegment]: ...

    def close(self) -> None: ...


@runtime_checkable
class AlignmentBackend(Protocol):
    name: str

    def align(self, audio: AudioChunk, text: str) -> list[Word]: ...


@runtime_checkable
class DiarizationBackend(Protocol):
    name: str

    def diarize(self, chunks: Iterable[AudioChunk]) -> Iterable[SpeakerTurn]: ...
