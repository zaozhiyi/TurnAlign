from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from statistics import median

from ..models import AudioChunk, SpeechSegment
from ..session import pcm_rms


class EnergyVadBackend:
    """Dependency-free adaptive energy VAD with hysteresis and boundary padding."""

    name = "energy"

    def __init__(
        self,
        *,
        min_threshold: float = 0.003,
        noise_multiplier: float = 3.0,
        end_multiplier: float = 1.8,
        min_silence_seconds: float = 0.7,
        min_speech_seconds: float = 0.2,
        pre_roll_seconds: float = 0.3,
        max_segment_seconds: float = 20.0,
    ):
        if min_threshold < 0 or noise_multiplier <= 1 or end_multiplier <= 1:
            raise ValueError("invalid energy VAD thresholds")
        if min_silence_seconds < 0 or min_speech_seconds < 0 or pre_roll_seconds < 0:
            raise ValueError("invalid energy VAD durations")
        if max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds must be positive")
        self.min_threshold = float(min_threshold)
        self.noise_multiplier = float(noise_multiplier)
        self.end_multiplier = float(end_multiplier)
        self.min_silence_seconds = float(min_silence_seconds)
        self.min_speech_seconds = float(min_speech_seconds)
        self.pre_roll_seconds = float(pre_roll_seconds)
        self.max_segment_seconds = float(max_segment_seconds)

    @staticmethod
    def _noise_floor(samples: deque[float]) -> float:
        if not samples:
            return 0.001
        ordered = sorted(samples)
        quiet = ordered[: max(1, len(ordered) // 3)]
        return max(0.0001, median(quiet))

    def segment(self, chunks: Iterable[AudioChunk]) -> Iterator[SpeechSegment]:
        noise_samples: deque[float] = deque(maxlen=240)
        pre_roll: deque[AudioChunk] = deque()
        pre_roll_duration = 0.0
        active: list[AudioChunk] = []
        active_scores: list[float] = []
        active_duration = 0.0
        voiced_duration = 0.0
        trailing_silence = 0.0

        def remember(chunk: AudioChunk) -> None:
            nonlocal pre_roll_duration
            pre_roll.append(chunk)
            pre_roll_duration += chunk.duration
            while pre_roll and pre_roll_duration - pre_roll[0].duration >= self.pre_roll_seconds:
                pre_roll_duration -= pre_roll.popleft().duration

        def finish(forced: bool) -> SpeechSegment | None:
            nonlocal active, active_scores, active_duration, voiced_duration, trailing_silence
            result = None
            if active and voiced_duration >= self.min_speech_seconds:
                start = active[0].start
                end = active[-1].start + active[-1].duration
                result = SpeechSegment(
                    chunks=active,
                    start=start,
                    end=end,
                    confidence=min(1.0, max(active_scores, default=0.0)),
                    forced_split=forced,
                    metadata={
                        "noise_floor": round(self._noise_floor(noise_samples), 6),
                        "voiced_seconds": round(voiced_duration, 3),
                    },
                )
            active = []
            active_scores = []
            active_duration = 0.0
            voiced_duration = 0.0
            trailing_silence = 0.0
            return result

        for chunk in chunks:
            rms = pcm_rms(chunk)
            floor = self._noise_floor(noise_samples)
            start_threshold = max(self.min_threshold, floor * self.noise_multiplier)
            end_threshold = max(self.min_threshold * 0.7, floor * self.end_multiplier)

            if not active:
                if rms < start_threshold:
                    noise_samples.append(rms)
                    remember(chunk)
                    continue
                active = list(pre_roll)
                active.append(chunk)
                active_duration = sum(item.duration for item in active)
                voiced_duration = chunk.duration
                active_scores = [min(1.0, rms / max(start_threshold, 1e-9))]
                pre_roll.clear()
                pre_roll_duration = 0.0
            else:
                active.append(chunk)
                active_duration += chunk.duration
                voiced = rms >= end_threshold
                if voiced:
                    voiced_duration += chunk.duration
                    trailing_silence = 0.0
                    active_scores.append(min(1.0, rms / max(start_threshold, 1e-9)))
                else:
                    trailing_silence += chunk.duration
                    noise_samples.append(rms)

            forced = active_duration >= self.max_segment_seconds
            endpoint = trailing_silence >= self.min_silence_seconds
            if forced or endpoint:
                result = finish(forced)
                if result is not None:
                    yield result
                pre_roll.clear()
                pre_roll_duration = 0.0

        result = finish(False)
        if result is not None:
            yield result

    def close(self) -> None:
        return None
