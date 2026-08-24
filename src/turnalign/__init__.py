"""Model-agnostic streaming ASR orchestration."""

from .models import AudioChunk, Hypothesis, SpeakerTurn, TranscriptEvent, Word
from .plugins import AsrConfig, BackendCapabilities
from .session import transcribe_events

__all__ = [
    "AsrConfig",
    "AudioChunk",
    "BackendCapabilities",
    "Hypothesis",
    "SpeakerTurn",
    "TranscriptEvent",
    "Word",
    "transcribe_events",
]
__version__ = "0.1.0"
