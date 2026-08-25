from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal


def _valid_range(start: float, end: float) -> bool:
    return isfinite(start) and isfinite(end) and start >= 0 and end >= start


@dataclass(frozen=True, slots=True)
class AudioChunk:
    pcm_s16le: bytes
    start: float
    sample_rate: int = 16_000
    channels: int = 1
    is_final: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.start) or self.start < 0:
            raise ValueError("audio chunk start must be non-negative")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        frame_bytes = 2 * self.channels
        if len(self.pcm_s16le) % frame_bytes:
            raise ValueError("PCM data must contain complete signed 16-bit frames")

    @property
    def duration(self) -> float:
        return len(self.pcm_s16le) / (2 * self.channels * self.sample_rate)


@dataclass(slots=True)
class SpeechSegment:
    """One original-timeline speech region emitted by a VAD backend."""

    chunks: list[AudioChunk]
    start: float
    end: float
    confidence: float | None = None
    forced_split: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid speech segment time range")
        if not self.chunks:
            raise ValueError("speech segment must contain audio chunks")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("speech confidence must be between zero and one")


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid word time range")


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    overlap: bool = False

    def __post_init__(self) -> None:
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid speaker turn time range")
        if not self.speaker:
            raise ValueError("speaker must not be empty")


@dataclass(slots=True)
class Hypothesis:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    confidence: float | None = None
    final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid hypothesis time range")


EventKind = Literal["partial", "commit", "replace", "speaker_merge", "end"]


@dataclass(slots=True)
class TranscriptEvent:
    kind: EventKind
    segment_id: str
    revision: int
    start: float
    end: float
    text: str = ""
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must not be empty")
        if self.revision < 1:
            raise ValueError("revision must be at least 1")
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid event time range")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "TranscriptEvent":
        data = dict(item)
        data["words"] = [word if isinstance(word, Word) else Word(**word) for word in data.get("words", [])]
        return cls(**data)
