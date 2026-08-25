from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Iterable
from pathlib import Path

from ..models import AudioChunk, Hypothesis
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from ..hints import whisper_initial_prompt
from .common import collect_pcm


class WhisperCppBackend:
    name = "whisper-cpp"
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=False,
        hotwords=True,
        context_prompt=True,
        accelerators=(Accelerator.CUDA, Accelerator.MPS, Accelerator.CPU),
    )

    def __init__(self, config: AsrConfig):
        self.executable = config.executable or "whisper-cli"
        self.model_path = config.model_path or config.model
        self.language = config.language or "auto"
        self.no_gpu = config.device == "cpu"
        self.hints = config.hints
        self.initial_prompt = whisper_initial_prompt(config.hints)
        if not self.model_path:
            raise ValueError("whisper-cpp requires --model-path")
        if shutil.which(self.executable) is None and not Path(self.executable).is_file():
            raise RuntimeError(f"whisper.cpp executable not found: {self.executable}")

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data:
            return
        with tempfile.TemporaryDirectory(prefix="turnalign-") as directory:
            root = Path(directory)
            audio_path = root / "input.wav"
            output_base = root / "result"
            with wave.open(str(audio_path), "wb") as destination:
                destination.setnchannels(channels)
                destination.setsampwidth(2)
                destination.setframerate(sample_rate)
                destination.writeframes(data)
            command = [
                self.executable, "-m", str(self.model_path), "-f", str(audio_path),
                "-l", self.language, "-oj", "-of", str(output_base), "-np",
            ]
            if self.no_gpu:
                command.append("-ng")
            if self.initial_prompt:
                command.extend(["--prompt", self.initial_prompt])
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"whisper.cpp failed: {completed.stderr.strip()}")
            payload = json.loads(output_base.with_suffix(".json").read_text(encoding="utf-8"))
            items = payload.get("transcription") or payload.get("segments") or []
            if not items and payload.get("text"):
                end = offset + len(data) / (2 * channels * sample_rate)
                yield Hypothesis(
                    str(payload["text"]).strip(), offset, end, final=True,
                    metadata=(
                        self.hints.private_metadata("whisper-cpp-prompt")
                        if self.hints.active else {}
                    ),
                )
                return
            for item in items:
                offsets = item.get("offsets", {})
                start_ms = offsets.get("from", item.get("start", 0) * 1000)
                end_ms = offsets.get("to", item.get("end", 0) * 1000)
                yield Hypothesis(
                    str(item.get("text", "")).strip(),
                    offset + float(start_ms) / 1000,
                    offset + float(end_ms) / 1000,
                    final=True,
                    metadata=(
                        self.hints.private_metadata("whisper-cpp-prompt")
                        if self.hints.active else {}
                    ),
                )

    def close(self) -> None:
        return None
