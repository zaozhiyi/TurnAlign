from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ..models import AudioChunk, Hypothesis
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from .common import pcm_to_float32


class FunAsrStreamingSession:
    """Per-stream Paraformer cache and timing state."""

    def __init__(self, backend: FunAsrStreamingBackend) -> None:
        self.backend = backend
        self.cache: dict[str, Any] = {}
        self.start: float | None = None
        self.end = 0.0
        self.closed = False
        self.finished = False
        self.pending = bytearray()
        self.last_text = ""

    def _accumulate(self, text: str) -> str:
        if not self.last_text:
            return text
        if text.startswith(self.last_text):
            return text
        if self.last_text.endswith(text):
            return self.last_text
        separator = (
            " "
            if self.last_text[-1].isascii()
            and self.last_text[-1].isalnum()
            and text[0].isascii()
            and text[0].isalnum()
            else ""
        )
        return f"{self.last_text}{separator}{text}"

    def _generate(self, data: bytes, *, final: bool) -> Iterator[Hypothesis]:
        if self.closed:
            raise RuntimeError("streaming ASR session is closed")
        options: dict[str, Any] = {
            "input": pcm_to_float32(data, 1),
            "cache": self.cache,
            "is_final": final,
            "chunk_size": self.backend.chunk_size,
            "encoder_chunk_look_back": self.backend.encoder_chunk_look_back,
            "decoder_chunk_look_back": self.backend.decoder_chunk_look_back,
        }
        if self.backend.hints.hotwords:
            options["hotword"] = " ".join(self.backend.hints.hotwords)
        result = self.backend.model.generate(**options)
        items = result if isinstance(result, list) else [result]
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                self.last_text = self._accumulate(text)
                yield Hypothesis(
                    text=self.last_text,
                    start=self.start or 0.0,
                    end=self.end,
                    language=self.backend.language,
                    final=final or bool(item.get("is_final", False)),
                    metadata={
                        "native_streaming": True,
                        "funasr_timestamp": item.get("timestamp"),
                        **(
                            self.backend.hints.private_metadata("funasr-streaming-hotword")
                            if self.backend.hints.active else {}
                        ),
                    },
                )

    def accept_audio(self, chunk: AudioChunk) -> Iterable[Hypothesis]:
        if chunk.sample_rate != 16_000 or chunk.channels != 1:
            raise ValueError("FunASR streaming backend expects 16 kHz mono audio")
        if self.start is None:
            self.start = chunk.start
        self.end = max(self.end, chunk.start + chunk.duration)
        self.pending.extend(chunk.pcm_s16le)
        target_bytes = round(16_000 * self.backend.chunk_ms / 1000) * 2
        while len(self.pending) >= target_bytes and not chunk.is_final:
            data = bytes(self.pending[:target_bytes])
            del self.pending[:target_bytes]
            yield from self._generate(data, final=False)
        if chunk.is_final:
            yield from self._generate(bytes(self.pending), final=True)
            self.pending.clear()
            self.finished = True

    def finish(self) -> Iterable[Hypothesis]:
        if self.finished:
            return
        self.finished = True
        if self.pending:
            yield from self._generate(bytes(self.pending), final=True)
            self.pending.clear()
        elif self.last_text and self.start is not None:
            yield Hypothesis(
                text=self.last_text,
                start=self.start,
                end=self.end,
                language=self.backend.language,
                final=True,
                metadata={"native_streaming": True, "synthetic_final_flush": True},
            )

    def cancel(self) -> None:
        self.finished = True
        self.pending.clear()
        self.cache.clear()

    def close(self) -> None:
        if not self.closed:
            self.pending.clear()
            self.cache.clear()
            self.closed = True


class FunAsrStreamingBackend:
    """Native incremental Paraformer backend using one cache per session."""

    name = "funasr-streaming"
    capabilities = BackendCapabilities(
        streaming=True,
        word_timestamps=False,
        hotwords=True,
        min_chunk_ms=600,
        lookahead_ms=300,
        external_vad=False,
        languages=("zh",),
        accelerators=(Accelerator.CUDA, Accelerator.ROCM, Accelerator.CPU),
    )

    def __init__(self, config: AsrConfig):
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as error:
            if error.name == "funasr":
                raise RuntimeError("install this backend with: pip install 'turnalign[funasr]'") from error
            raise RuntimeError(f"FunASR dependency failed to initialize: {error}") from error
        except ImportError as error:
            raise RuntimeError(f"FunASR dependency failed to initialize: {error}") from error
        device = config.device.replace("rocm", "cuda", 1)
        if device == "auto":
            device = "cuda:0" if self._cuda_available() else "cpu"
        options = dict(config.extra or {})
        self.chunk_ms = int(options.pop("chunk_ms", 600))
        if not 20 <= self.chunk_ms <= 2_000:
            raise ValueError("FunASR streaming chunk_ms must be between 20 and 2000")
        self.chunk_size = options.pop("chunk_size", [0, 10, 5])
        self.encoder_chunk_look_back = int(options.pop("encoder_chunk_look_back", 4))
        self.decoder_chunk_look_back = int(options.pop("decoder_chunk_look_back", 1))
        self.model = AutoModel(
            model=config.model or "paraformer-zh-streaming",
            device=device,
            **options,
        )
        self.language = config.language or "zh"
        self.hints = config.hints

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - optional torch runtimes fail in many ways
            return False

    def start_session(self) -> FunAsrStreamingSession:
        if self.model is None:
            raise RuntimeError("FunASR streaming backend is closed")
        return FunAsrStreamingSession(self)

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        session = self.start_session()
        try:
            for chunk in chunks:
                yield from session.accept_audio(chunk)
            yield from session.finish()
        finally:
            session.close()

    def close(self) -> None:
        self.model = None
