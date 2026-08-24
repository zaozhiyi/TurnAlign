from __future__ import annotations

from collections.abc import Iterable

from ..models import AudioChunk, Hypothesis
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from .common import collect_pcm, pcm_to_float32


class FunAsrBackend:
    name = "funasr"
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=False,
        languages=("zh", "en", "yue", "ja", "ko"),
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
        self.model = AutoModel(model=config.model or "paraformer-zh", device=device, **options)
        self.language = config.language

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data:
            return
        if sample_rate != 16_000:
            raise ValueError("FunASR backend expects 16 kHz audio")
        result = self.model.generate(input=pcm_to_float32(data, channels))
        duration = len(data) / (2 * channels * sample_rate)
        for item in result or []:
            text = str(item.get("text", "")).strip()
            if text:
                yield Hypothesis(
                    text, offset, offset + duration, language=self.language, final=True,
                    metadata={"funasr_timestamp": item.get("timestamp")},
                )

    def close(self) -> None:
        self.model = None
