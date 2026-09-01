from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Iterable
from pathlib import Path

from ..hints import whisper_initial_prompt
from ..models import AudioChunk, Hypothesis
from ..plugins import Accelerator, AsrConfig, BackendCapabilities
from .common import collect_pcm

_INDEXED_DEVICE = re.compile(r"^(?:cuda|rocm|vulkan):(\d+)$")
_UNINDEXED_DEVICES = {"auto", "cpu", "cuda", "rocm", "vulkan", "mps"}
_ALLOWED_OPTIONS = {"threads", "flash_attention", "allow_prompt_argv"}


def _bounded_integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"whisper-cpp {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"whisper-cpp {name} must be between {minimum} and {maximum}"
        )
    return value


def _device_index(device: str) -> int | None:
    normalized = device.strip().lower()
    match = _INDEXED_DEVICE.fullmatch(normalized)
    if match:
        return _bounded_integer(
            int(match.group(1)), name="device index", minimum=0, maximum=31
        )
    if normalized not in _UNINDEXED_DEVICES:
        raise ValueError(
            "whisper-cpp device must be auto, cpu, cuda, rocm, vulkan, mps, "
            "or cuda:N/rocm:N/vulkan:N"
        )
    return None


def _windows_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000))


class WhisperCppBackend:
    name = "whisper-cpp"
    session_hints = True
    capabilities = BackendCapabilities(
        streaming=False,
        word_timestamps=False,
        hotwords=True,
        context_prompt=True,
        accelerators=(
            Accelerator.CUDA,
            Accelerator.ROCM,
            Accelerator.VULKAN,
            Accelerator.MPS,
            Accelerator.CPU,
        ),
    )
    cancel_grace_seconds = 1.0

    def __init__(self, config: AsrConfig):
        self.executable = config.executable or "whisper-cli"
        self.model_path = config.model_path or config.model
        self.language = config.language or "auto"
        self.device = config.device.strip().lower()
        self.no_gpu = self.device == "cpu"
        self.device_index = _device_index(self.device)
        options = dict(config.extra or {})
        unknown = sorted(set(options) - _ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(
                "unsupported whisper-cpp backend option(s): " + ", ".join(unknown)
            )
        self.threads = None
        if "threads" in options:
            self.threads = _bounded_integer(
                options["threads"], name="threads", minimum=1, maximum=64
            )
        self.flash_attention = options.get("flash_attention")
        if self.flash_attention is not None and not isinstance(self.flash_attention, bool):
            raise ValueError("whisper-cpp flash_attention must be a boolean")
        self.allow_prompt_argv = options.get("allow_prompt_argv", False)
        if not isinstance(self.allow_prompt_argv, bool):
            raise TypeError("whisper-cpp allow_prompt_argv must be a boolean")
        self.hints = config.hints
        self.initial_prompt = whisper_initial_prompt(config.hints)
        if self.initial_prompt and not self.allow_prompt_argv:
            raise ValueError(
                "whisper-cpp only accepts prompts through process arguments; "
                "private hints are disabled by default because other local users may "
                "read the process list. Remove the hints or explicitly set backend "
                "option allow_prompt_argv=true after accepting that risk"
            )
        if not self.model_path:
            raise ValueError("whisper-cpp requires --model-path")
        if shutil.which(self.executable) is None and not Path(self.executable).is_file():
            raise RuntimeError(f"whisper.cpp executable not found: {self.executable}")
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = threading.Event()

    def set_hints(self, hints) -> None:
        initial_prompt = whisper_initial_prompt(hints)
        if initial_prompt and not self.allow_prompt_argv:
            raise ValueError(
                "whisper-cpp only accepts prompts through process arguments; "
                "private hints are disabled by default because other local users may "
                "read the process list. Remove the hints or explicitly set backend "
                "option allow_prompt_argv=true after accepting that risk"
            )
        self.hints = hints
        self.initial_prompt = initial_prompt

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterable[Hypothesis]:
        self._cancel_requested.clear()
        data, sample_rate, channels, offset = collect_pcm(chunks)
        if not data or self._cancel_requested.is_set():
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
            elif self.device_index is not None:
                command.extend(["-dev", str(self.device_index)])
            if self.threads is not None:
                command.extend(["-t", str(self.threads)])
            if self.flash_attention is not None:
                command.append("-fa" if self.flash_attention else "-nfa")
            if self.initial_prompt:
                command.extend(["--prompt", self.initial_prompt])
            with self._process_lock:
                if self._cancel_requested.is_set():
                    return
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=_windows_creationflags(),
                )
                self._process = process
            try:
                _, stderr = process.communicate()
            finally:
                with self._process_lock:
                    if self._process is process:
                        self._process = None
            if process.returncode:
                raise RuntimeError(f"whisper.cpp failed: {stderr.strip()}")
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

    def cancel(self) -> None:
        """Terminate the active CLI process so transport cancellation is real."""
        self._cancel_requested.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            escalation = threading.Timer(
                self.cancel_grace_seconds,
                self._kill_if_active,
                args=(process,),
            )
            escalation.daemon = True
            escalation.start()

    def _kill_if_active(self, process: subprocess.Popen[str]) -> None:
        """Force-kill a CLI process that ignored the graceful termination."""
        with self._process_lock:
            active = self._process is process
        if active and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def close(self) -> None:
        self.cancel()
