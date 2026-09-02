from __future__ import annotations

from collections.abc import Iterable

from ..models import AudioChunk, Hypothesis
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from .common import (
    collect_pcm,
    local_model_files,
    pcm_to_float32,
    require_local_model_path,
)


class FunAsrBackend:
    name = "funasr"
    session_hints = True
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=False,
        hotwords=True,
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
        self._local_model_path = None
        if config.require_local_model:
            self._local_model_path = require_local_model_path(
                config.model_path,
                directory=True,
            )
            model = str(self._local_model_path)
        else:
            model = config.model_path or config.model or "paraformer-zh"
        options.setdefault("disable_update", True)
        if model == "paraformer-zh" and self._local_model_path is None:
            options.setdefault(
                "model_revision",
                "71684869ca6d8bfa59057d8a367b3fb7345a0c02",
            )
        self.model_revision = options.get("model_revision")
        self.model = AutoModel(model=model, device=device, **options)
        self.language = config.language
        self.hints = config.hints

    def loaded_model_files(self):
        return local_model_files(self._local_model_path)

    def set_hints(self, hints) -> None:
        self.hints = hints

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - optional torch runtimes fail in many ways
            return False

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data:
            return
        if sample_rate != 16_000:
            raise ValueError("FunASR backend expects 16 kHz audio")
        generate_options = {"input": pcm_to_float32(data, channels)}
        if self.hints.hotwords:
            generate_options["hotword"] = " ".join(self.hints.hotwords)
        result = self.model.generate(**generate_options)
        duration = len(data) / (2 * channels * sample_rate)
        for item in result or []:
            text = str(item.get("text", "")).strip()
            if text:
                yield Hypothesis(
                    text, offset, offset + duration, language=self.language, final=True,
                    metadata={
                        "funasr_timestamp": item.get("timestamp"),
                        **(
                            self.hints.private_metadata("funasr-hotword")
                            if self.hints.active else {}
                        ),
                    },
                )

    def close(self) -> None:
        self.model = None
