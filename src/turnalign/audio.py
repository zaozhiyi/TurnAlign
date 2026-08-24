from __future__ import annotations

import math
import queue
import shutil
import subprocess
import wave
from collections.abc import Iterator
from pathlib import Path

from .models import AudioChunk


def _chunk_bytes(sample_rate: int, channels: int, chunk_ms: int) -> int:
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    frames = max(1, round(sample_rate * chunk_ms / 1000))
    return frames * channels * 2


def wave_chunks(path: Path, chunk_ms: int = 500) -> Iterator[AudioChunk]:
    """Read PCM16 WAV without loading the full recording into memory."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("WAV input must be uncompressed signed 16-bit PCM")
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        frames = max(1, round(sample_rate * chunk_ms / 1000))
        start = 0.0
        while True:
            data = source.readframes(frames)
            if not data:
                break
            yield AudioChunk(data, start, sample_rate, channels)
            start += len(data) / (2 * channels * sample_rate)


def file_chunks(path: Path, chunk_ms: int = 500, ffmpeg: str = "ffmpeg") -> Iterator[AudioChunk]:
    """Decode audio to 16 kHz mono PCM; normalized PCM16 WAV needs no ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as source:
            ready = (
                source.getsampwidth() == 2
                and source.getcomptype() == "NONE"
                and source.getframerate() == 16_000
                and source.getnchannels() == 1
            )
        if ready:
            yield from wave_chunks(path, chunk_ms)
            return
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise RuntimeError("ffmpeg is required for compressed audio or WAV that is not 16 kHz mono PCM16")
    command = [
        executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    size = _chunk_bytes(16_000, 1, chunk_ms)
    start = 0.0
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            yield AudioChunk(data, start)
            start += len(data) / 32_000
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed with exit code {code}: {stderr.strip()}")


def microphone_chunks(
    *,
    device: int | str | None = None,
    sample_rate: int = 16_000,
    channels: int = 1,
    chunk_ms: int = 100,
    duration: float | None = None,
) -> Iterator[AudioChunk]:
    """Capture PCM16 from the default microphone until duration or Ctrl+C."""
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError("microphone input requires: pip install 'turnalign[microphone]'") from error

    blocksize = max(1, round(sample_rate * chunk_ms / 1000))
    pending: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=64)

    def callback(indata, frames, time_info, status):
        del frames, time_info
        if status and getattr(status, "input_overflow", False):
            try:
                pending.put_nowait(RuntimeError("microphone input overflow"))
            except queue.Full:
                pass
            return
        try:
            pending.put_nowait(bytes(indata))
        except queue.Full:
            pass

    start = 0.0
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=blocksize,
        device=device,
        channels=channels,
        dtype="int16",
        callback=callback,
    ):
        while duration is None or start < duration:
            item = pending.get()
            if isinstance(item, BaseException):
                raise item
            if duration is not None:
                remaining_frames = max(0, math.floor((duration - start) * sample_rate))
                item = item[: remaining_frames * channels * 2]
            if not item:
                break
            chunk = AudioChunk(item, start, sample_rate, channels)
            yield chunk
            start += chunk.duration


def input_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError("device listing requires: pip install 'turnalign[microphone]'") from error
    result = []
    for index, device in enumerate(sd.query_devices()):
        channels = int(device.get("max_input_channels", 0))
        if channels:
            result.append({
                "id": index,
                "name": str(device.get("name", "")),
                "channels": channels,
                "default_sample_rate": float(device.get("default_samplerate", 0)),
            })
    return result


def write_wave(path: Path, chunks: Iterator[AudioChunk]) -> None:
    iterator = iter(chunks)
    first = next(iterator, None)
    if first is None:
        raise ValueError("cannot write an empty audio stream")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(first.channels)
        destination.setsampwidth(2)
        destination.setframerate(first.sample_rate)
        destination.writeframes(first.pcm_s16le)
        for chunk in iterator:
            if (chunk.sample_rate, chunk.channels) != (first.sample_rate, first.channels):
                raise ValueError("audio format changed during stream")
            destination.writeframes(chunk.pcm_s16le)
