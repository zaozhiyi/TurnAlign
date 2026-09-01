from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal, TypeGuard


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def _valid_range(start: object, end: object) -> bool:
    return (
        _finite_number(start)
        and _finite_number(end)
        and start >= 0
        and end >= start
    )


def _valid_confidence(value: object) -> bool:
    return value is None or (
        _finite_number(value) and 0 <= value <= 1
    )


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


@dataclass(frozen=True, slots=True)
class AudioChunk:
    pcm_s16le: bytes
    start: float
    sample_rate: int = 16_000
    channels: int = 1
    is_final: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pcm_s16le, bytes):
            raise TypeError("PCM data must be bytes")
        if not _valid_range(self.start, self.start):
            raise ValueError("audio chunk start must be non-negative")
        if not _positive_int(self.sample_rate) or not _positive_int(self.channels):
            raise ValueError("sample_rate and channels must be positive integers")
        if not isinstance(self.is_final, bool):
            raise TypeError("is_final must be a boolean")
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
        if (
            not isinstance(self.chunks, list)
            or not self.chunks
            or not all(isinstance(chunk, AudioChunk) for chunk in self.chunks)
        ):
            raise ValueError("speech segment must contain audio chunks")
        if not _valid_confidence(self.confidence):
            raise ValueError("speech confidence must be between zero and one")
        if not isinstance(self.forced_split, bool):
            raise TypeError("forced_split must be a boolean")
        if not isinstance(self.metadata, dict):
            raise TypeError("speech metadata must be a dictionary")


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("word text must be a string")
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid word time range")
        if not _valid_confidence(self.confidence):
            raise ValueError("word confidence must be between zero and one")
        if self.speaker is not None and not isinstance(self.speaker, str):
            raise TypeError("word speaker must be a string")


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
        if not isinstance(self.speaker, str) or not self.speaker:
            raise ValueError("speaker must not be empty")
        if not _valid_confidence(self.confidence):
            raise ValueError("speaker confidence must be between zero and one")
        if not isinstance(self.overlap, bool):
            raise TypeError("overlap must be a boolean")


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
        if not isinstance(self.text, str):
            raise TypeError("hypothesis text must be a string")
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid hypothesis time range")
        if not isinstance(self.words, list) or not all(
            isinstance(word, Word) for word in self.words
        ):
            raise TypeError("hypothesis words must be Word objects")
        if self.language is not None and not isinstance(self.language, str):
            raise TypeError("hypothesis language must be a string")
        if not _valid_confidence(self.confidence):
            raise ValueError("hypothesis confidence must be between zero and one")
        if not isinstance(self.final, bool):
            raise TypeError("hypothesis final must be a boolean")
        if not isinstance(self.metadata, dict):
            raise TypeError("hypothesis metadata must be a dictionary")


EventKind = Literal["partial", "commit", "replace", "speaker_merge", "end"]
_EVENT_KINDS = frozenset({"partial", "commit", "replace", "speaker_merge", "end"})


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
    protocol_version: int = 1
    session_id: str | None = None
    sequence: int | None = None
    source_timestamp: float | None = None
    acknowledged_sequence: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError("unknown event kind")
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError("segment_id must not be empty")
        if not _positive_int(self.revision):
            raise ValueError("revision must be at least 1")
        if not _positive_int(self.protocol_version):
            raise ValueError("protocol_version must be at least 1")
        if self.sequence is not None and not _non_negative_int(self.sequence):
            raise ValueError("sequence must be non-negative")
        if (
            self.acknowledged_sequence is not None
            and not _non_negative_int(self.acknowledged_sequence)
        ):
            raise ValueError("acknowledged_sequence must be non-negative")
        if self.source_timestamp is not None and (
            not _finite_number(self.source_timestamp) or self.source_timestamp < 0
        ):
            raise ValueError("source_timestamp must be non-negative")
        if not _valid_range(self.start, self.end):
            raise ValueError("invalid event time range")
        if not isinstance(self.text, str):
            raise TypeError("event text must be a string")
        if not isinstance(self.words, list) or not all(
            isinstance(word, Word) for word in self.words
        ):
            raise TypeError("event words must be Word objects")
        if self.speaker is not None and not isinstance(self.speaker, str):
            raise TypeError("event speaker must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("event metadata must be a dictionary")
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id
        ):
            raise ValueError("session_id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> TranscriptEvent:
        data = dict(item)
        data["words"] = [word if isinstance(word, Word) else Word(**word) for word in data.get("words", [])]
        return cls(**data)
