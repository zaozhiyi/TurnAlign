"""Model-agnostic streaming ASR orchestration."""

from .audio import AudioTimeline
from .models import AudioChunk, Hypothesis, SpeakerTurn, TranscriptEvent, Word
from .offline import OfflineRefinementPipeline
from .pipelines import TwoPassPipeline
from .plugins import AsrConfig, BackendCapabilities
from .realtime import RealtimePipeline
from .session import transcribe_events

__all__ = [
    "AsrConfig",
    "AudioChunk",
    "AudioTimeline",
    "BackendCapabilities",
    "Hypothesis",
    "OfflineRefinementPipeline",
    "RealtimePipeline",
    "SpeakerTurn",
    "TranscriptEvent",
    "TwoPassPipeline",
    "Word",
    "transcribe_events",
]
__version__ = "0.1.0"
