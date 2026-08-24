from __future__ import annotations

from collections.abc import Iterable

from ..models import AudioChunk


def collect_pcm(chunks: Iterable[AudioChunk]) -> tuple[bytes, int, int, float]:
    data = bytearray()
    sample_rate = 0
    channels = 0
    start = 0.0
    for index, chunk in enumerate(chunks):
        if index == 0:
            sample_rate, channels, start = chunk.sample_rate, chunk.channels, chunk.start
        elif (chunk.sample_rate, chunk.channels) != (sample_rate, channels):
            raise ValueError("audio format changed during transcription")
        data.extend(chunk.pcm_s16le)
    if not data:
        return b"", 16_000, 1, 0.0
    return bytes(data), sample_rate, channels, start


def pcm_to_float32(data: bytes, channels: int):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("this backend requires numpy") from error
    samples = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples
