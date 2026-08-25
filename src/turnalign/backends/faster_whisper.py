from __future__ import annotations

from collections.abc import Iterable

from ..models import AudioChunk, Hypothesis, Word
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from .common import collect_pcm, pcm_to_float32


class FasterWhisperBackend:
    name = "faster-whisper"
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=True,
        hotwords=True,
        context_prompt=True,
        accelerators=(Accelerator.CUDA, Accelerator.CPU),
    )

    def __init__(self, config: AsrConfig):
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as error:
            if error.name == "faster_whisper":
                raise RuntimeError("install this backend with: pip install 'turnalign[faster-whisper]'") from error
            raise RuntimeError(f"faster-whisper dependency failed to initialize: {error}") from error
        requested = config.device.lower()
        device = "cuda" if requested.startswith("cuda") else "cpu"
        if requested.startswith("rocm") or requested == "mps":
            device = "cpu"
        compute_type = config.compute_type or ("float16" if device == "cuda" else "int8")
        self.language = config.language
        self.hints = config.hints
        self.model = WhisperModel(config.model or "small", device=device, compute_type=compute_type)

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data:
            return
        audio = pcm_to_float32(data, channels)
        if sample_rate != 16_000:
            raise ValueError("faster-whisper backend expects 16 kHz audio")
        options = {
            "language": self.language,
            "word_timestamps": True,
            "vad_filter": False,
        }
        if self.hints.hotwords:
            options["hotwords"] = " ".join(self.hints.hotwords)
        if self.hints.context:
            options["initial_prompt"] = self.hints.context
        segments, info = self.model.transcribe(audio, **options)
        for segment in segments:
            words = [
                Word(item.word, offset + item.start, offset + item.end, getattr(item, "probability", None))
                for item in (segment.words or [])
            ]
            yield Hypothesis(
                segment.text.strip(), offset + segment.start, offset + segment.end,
                words=words, language=getattr(info, "language", self.language), final=True,
                metadata=(
                    self.hints.private_metadata("faster-whisper-native")
                    if self.hints.active else {}
                ),
            )

    def close(self) -> None:
        self.model = None
