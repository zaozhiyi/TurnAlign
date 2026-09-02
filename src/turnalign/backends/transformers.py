from __future__ import annotations

from collections.abc import Iterable

from ..hints import glm_transcription_prompt, whisper_initial_prompt
from ..models import AudioChunk, Hypothesis, Word
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from .common import (
    collect_pcm,
    local_model_files,
    pcm_to_float32,
    require_local_model_path,
)

_DEFAULT_WHISPER_MODEL = "openai/whisper-small"
_DEFAULT_WHISPER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
_DEFAULT_GLM_MODEL = "zai-org/GLM-ASR-Nano-2512"
_DEFAULT_GLM_REVISION = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"


def _model_revision(
    options: dict[str, object],
    *,
    default: str | None = None,
) -> str | None:
    revision = options.pop("revision", default)
    if revision is None:
        return None
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 200:
        raise ValueError("revision must be a non-empty string of at most 200 characters")
    return revision.strip()


def _pipeline_device(requested: str):
    requested = requested.lower()
    if requested.startswith(("cuda", "rocm")):
        parts = requested.split(":", 1)
        return int(parts[1]) if len(parts) == 2 else 0
    if requested == "mps":
        return "mps"
    return -1


class TransformersWhisperBackend:
    name = "transformers-whisper"
    session_hints = True
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=True,
        hotwords=True,
        context_prompt=True,
        accelerators=(Accelerator.CUDA, Accelerator.ROCM, Accelerator.MPS, Accelerator.CPU),
    )

    def __init__(self, config: AsrConfig):
        try:
            from transformers import pipeline
        except ModuleNotFoundError as error:
            if error.name == "transformers":
                raise RuntimeError("install this backend with: pip install 'turnalign[transformers]'") from error
            raise RuntimeError(f"Transformers dependency failed to initialize: {error}") from error
        except Exception as error:
            raise RuntimeError(f"Transformers dependency failed to initialize: {error}") from error
        options = dict(config.extra or {})
        self._local_model_path = None
        if config.require_local_model:
            self._local_model_path = require_local_model_path(
                config.model_path,
                directory=True,
            )
            model_id = str(self._local_model_path)
            options["local_files_only"] = True
        else:
            model_id = config.model or _DEFAULT_WHISPER_MODEL
        self.model_revision = _model_revision(
            options,
            default=(
                _DEFAULT_WHISPER_REVISION
                if model_id == _DEFAULT_WHISPER_MODEL
                else None
            ),
        )
        self.language = config.language
        self.hints = config.hints
        if config.compute_type:
            options.setdefault("dtype", config.compute_type)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=_pipeline_device(config.device),
            revision=self.model_revision,
            **options,
        )

    def loaded_model_files(self):
        return local_model_files(self._local_model_path)

    def set_hints(self, hints) -> None:
        self.hints = hints

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data:
            return
        audio = {"array": pcm_to_float32(data, channels), "sampling_rate": sample_rate}
        generate_kwargs = {"language": self.language} if self.language else {}
        initial_prompt = whisper_initial_prompt(self.hints)
        if initial_prompt:
            prompt_ids = self.pipe.tokenizer.get_prompt_ids(
                initial_prompt, return_tensors="pt"
            )
            model_device = getattr(getattr(self.pipe, "model", None), "device", None)
            if model_device is not None and hasattr(prompt_ids, "to"):
                prompt_ids = prompt_ids.to(model_device)
            generate_kwargs["prompt_ids"] = prompt_ids
        result = self.pipe(audio, return_timestamps="word", generate_kwargs=generate_kwargs)
        raw_words = result.get("chunks", []) if isinstance(result, dict) else []
        words = []
        for item in raw_words:
            timestamp = item.get("timestamp") or (0.0, 0.0)
            if timestamp[0] is None or timestamp[1] is None:
                continue
            words.append(Word(item.get("text", ""), offset + timestamp[0], offset + timestamp[1]))
        end = words[-1].end if words else offset + len(data) / (2 * channels * sample_rate)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        metadata = self.hints.private_metadata("whisper-prompt") if self.hints.active else {}
        yield Hypothesis(
            text.strip(), offset, end, words=words, language=self.language, final=True,
            metadata=metadata,
        )

    def close(self) -> None:
        self.pipe = None


class GlmAsrBackend:
    name = "glm-asr"
    session_hints = True
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=False,
        hotwords=True,
        context_prompt=True,
        languages=("zh", "en", "yue"),
        accelerators=(Accelerator.CUDA, Accelerator.ROCM, Accelerator.MPS, Accelerator.CPU),
    )

    def __init__(self, config: AsrConfig):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoProcessor
        except ModuleNotFoundError as error:
            if error.name == "transformers":
                raise RuntimeError("install this backend with: pip install 'turnalign[transformers]'") from error
            raise RuntimeError(f"Transformers dependency failed to initialize: {error}") from error
        except Exception as error:
            raise RuntimeError(f"Transformers dependency failed to initialize: {error}") from error
        kwargs = dict(config.extra or {})
        self._local_model_path = None
        if config.require_local_model:
            self._local_model_path = require_local_model_path(
                config.model_path,
                directory=True,
            )
            model_id = str(self._local_model_path)
            kwargs["local_files_only"] = True
        else:
            model_id = config.model or _DEFAULT_GLM_MODEL
        self.model_revision = _model_revision(
            kwargs,
            default=_DEFAULT_GLM_REVISION if model_id == _DEFAULT_GLM_MODEL else None,
        )
        processor_options: dict[str, object] = {"revision": self.model_revision}
        if "local_files_only" in kwargs:
            processor_options["local_files_only"] = kwargs["local_files_only"]
        # Built-in defaults are immutable; release-gate checks custom revisions.
        self.processor = AutoProcessor.from_pretrained(  # nosec B615
            model_id,
            **processor_options,
        )
        model_device = config.device
        if model_device.startswith("rocm"):
            model_device = model_device.replace("rocm", "cuda", 1)
        if model_device == "auto":
            kwargs.setdefault("device_map", "auto")
        kwargs.setdefault(
            "dtype",
            config.compute_type or ("float16" if model_device.startswith(("cuda", "mps")) else "auto"),
        )
        # Keep processor and model on the exact same operator-selected snapshot.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(  # nosec B615
            model_id,
            revision=self.model_revision,
            **kwargs,
        )
        if model_device != "auto":
            self.model = self.model.to(model_device)
        self.model.eval()
        self.language = config.language
        self.hints = config.hints
        self.prompt = glm_transcription_prompt(config.hints)

    def loaded_model_files(self):
        return local_model_files(self._local_model_path)

    def set_hints(self, hints) -> None:
        self.hints = hints
        self.prompt = glm_transcription_prompt(hints)

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data:
            return
        audio = pcm_to_float32(data, channels)
        if sample_rate != 16_000:
            raise ValueError("GLM-ASR backend expects 16 kHz audio")
        inputs = self.processor.apply_transcription_request(audio, prompt=self.prompt)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=500)
        text = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        end = offset + len(data) / (2 * channels * sample_rate)
        metadata = self.hints.private_metadata("glm-prompt") if self.hints.active else {}
        yield Hypothesis(
            text.strip(), offset, end, language=self.language, final=True, metadata=metadata
        )

    def close(self) -> None:
        self.model = None
        self.processor = None
