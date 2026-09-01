from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher

from ..audio import AudioTimeline
from ..backends.common import pcm_to_float32
from ..models import AudioChunk, SpeakerTurn, SpeechSegment, Word


def _auto_model():
    try:
        from funasr import AutoModel
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "FunASR pipeline components require: pip install 'turnalign[funasr-pipeline]'"
        ) from error
    except Exception as error:
        raise RuntimeError(f"FunASR pipeline dependency failed to initialize: {error}") from error
    return AutoModel


def _pytorch_device(device: str) -> str:
    return device.replace("rocm", "cuda", 1) if device.startswith("rocm") else device


def _result_item(result) -> dict:
    if not result:
        return {}
    item = result[0]
    return item if isinstance(item, dict) else {}


def _materialization_limit(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_materialized_seconds must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("max_materialized_seconds must be finite and positive")
    return value


def _collect_pcm_bounded(
    chunks: Iterable[AudioChunk],
    *,
    max_seconds: float,
) -> tuple[bytes, int, int, float]:
    data = bytearray()
    sample_rate = 0
    channels = 0
    start = 0.0
    for index, chunk in enumerate(chunks):
        if index == 0:
            sample_rate, channels, start = chunk.sample_rate, chunk.channels, chunk.start
        elif (chunk.sample_rate, chunk.channels) != (sample_rate, channels):
            raise ValueError("audio format changed during diarization")
        maximum_bytes = round(max_seconds * sample_rate * channels * 2)
        if len(data) + len(chunk.pcm_s16le) > maximum_bytes:
            raise ValueError(
                "CAM++ input exceeds max_materialized_seconds; split the recording "
                "or explicitly raise the diarizer limit after sizing memory"
            )
        data.extend(chunk.pcm_s16le)
    if not data:
        return b"", 16_000, 1, 0.0
    return bytes(data), sample_rate, channels, start


class FsmnVadBackend:
    name = "fsmn-vad"

    def __init__(
        self,
        *,
        model: str = "fsmn-vad",
        device: str = "cpu",
        batch_size_s: int = 300,
        max_segment_seconds: float = 20.0,
        max_materialized_seconds: float = 10_800.0,
        hub: str = "ms",
        disable_update: bool = True,
        **model_options,
    ):
        if max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds must be positive")
        AutoModel = _auto_model()
        self.model = AutoModel(
            model=model, device=_pytorch_device(device), hub=hub,
            disable_update=disable_update, **model_options
        )
        self.model_id = model
        self.batch_size_s = int(batch_size_s)
        self.max_segment_seconds = float(max_segment_seconds)
        self.max_materialized_seconds = _materialization_limit(
            max_materialized_seconds
        )

    def segment(self, chunks: Iterable[AudioChunk]) -> Iterator[SpeechSegment]:
        with AudioTimeline.from_chunks(iter(chunks)) as timeline:
            if timeline.start is None or timeline.sample_rate is None or timeline.channels is None:
                return
            if timeline.sample_rate != 16_000:
                raise ValueError("FSMN-VAD expects 16 kHz audio")
            if timeline.duration > self.max_materialized_seconds:
                raise ValueError(
                    "FSMN-VAD input exceeds max_materialized_seconds; split the "
                    "recording or explicitly raise the VAD limit after sizing memory"
                )
            full_audio = timeline.slice(timeline.start, timeline.end)
            result = self.model.generate(
                input=pcm_to_float32(full_audio.pcm_s16le, timeline.channels),
                batch_size_s=self.batch_size_s,
            )
            regions = _result_item(result).get("value") or []
            duration = timeline.duration
            previous_end = 0.0
            for raw in sorted(regions, key=lambda item: float(item[0]) if item else 0.0):
                if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                    continue
                start = max(previous_end, min(duration, float(raw[0]) / 1000.0))
                end = max(start, min(duration, float(raw[1]) / 1000.0))
                while end > start:
                    piece_end = min(end, start + self.max_segment_seconds)
                    forced = piece_end < end
                    absolute_start = timeline.start + start
                    absolute_end = timeline.start + piece_end
                    audio = timeline.slice(absolute_start, absolute_end)
                    if not audio.pcm_s16le:
                        raise ValueError("component returned an empty audio region")
                    yield SpeechSegment(
                        chunks=[audio],
                        start=absolute_start,
                        end=absolute_end,
                        forced_split=forced,
                        metadata={"model": self.model_id, "source_region_ms": list(raw[:2])},
                    )
                    start = piece_end
                previous_end = max(previous_end, end)

    def close(self) -> None:
        self.model = None


_UNIT_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]|[^\s]")


def _units(text: str) -> list[str]:
    return _UNIT_PATTERN.findall(text)


def _meaningful(unit: str) -> bool:
    return any(character.isalnum() or "\u3400" <= character <= "\u9fff" for character in unit)


def _aligned_words(
    target_text: str,
    source_text: str,
    timestamps,
    *,
    offset: float,
    duration: float,
) -> list[Word]:
    target = _units(target_text)
    if not target:
        return []
    source = source_text.split() if " " in source_text.strip() else _units(source_text)
    source_times: list[tuple[float, float]] = []
    for raw in timestamps or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        start = min(duration, max(0.0, float(raw[0]) / 1000.0))
        end = min(duration, max(start, float(raw[1]) / 1000.0))
        source_times.append((offset + start, offset + end))
    source = source[: len(source_times)]
    source_times = source_times[: len(source)]

    target_indices = [index for index, unit in enumerate(target) if _meaningful(unit)]
    source_indices = [index for index, unit in enumerate(source) if _meaningful(unit)]
    target_norm = [target[index].casefold() for index in target_indices]
    source_norm = [source[index].casefold() for index in source_indices]
    anchors: dict[int, tuple[float, float]] = {}
    matcher = SequenceMatcher(a=source_norm, b=target_norm, autojunk=False)
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            source_index = source_indices[block.a + delta]
            target_index = target_indices[block.b + delta]
            anchors[target_index] = source_times[source_index]

    result_times: list[tuple[float, float] | None] = [anchors.get(index) for index in range(len(target))]
    cursor = 0
    absolute_end = offset + duration
    while cursor < len(result_times):
        if result_times[cursor] is not None:
            cursor += 1
            continue
        run_start = cursor
        while cursor < len(result_times) and result_times[cursor] is None:
            cursor += 1
        run_end = cursor
        left_anchor = result_times[run_start - 1] if run_start else None
        right_anchor = result_times[run_end] if run_end < len(result_times) else None
        if run_start and left_anchor is None:
            raise RuntimeError("alignment interpolation lost its left anchor")
        if run_end < len(result_times) and right_anchor is None:
            raise RuntimeError("alignment interpolation lost its right anchor")
        left = left_anchor[1] if left_anchor is not None else offset
        right = right_anchor[0] if right_anchor is not None else absolute_end
        right = max(left, right)
        width = (right - left) / max(1, run_end - run_start)
        for index in range(run_start, run_end):
            start = left + width * (index - run_start)
            result_times[index] = (start, left + width * (index - run_start + 1))

    words: list[Word] = []
    previous_end = offset
    for unit, raw_time in zip(target, result_times):
        if raw_time is None:
            raise RuntimeError("alignment interpolation left an unresolved timestamp")
        start = max(previous_end, raw_time[0])
        end = max(start, raw_time[1])
        end = min(absolute_end, end)
        start = min(start, end)
        words.append(Word(unit, start, end))
        previous_end = end
    return words


class ParaformerAlignmentBackend:
    name = "paraformer"

    def __init__(
        self,
        *,
        model: str = "paraformer-zh",
        device: str = "cpu",
        batch_size_s: int = 60,
        batch_size: int = 4,
        hub: str = "ms",
        disable_update: bool = True,
        **model_options,
    ):
        AutoModel = _auto_model()
        self.model = AutoModel(
            model=model, device=_pytorch_device(device), hub=hub,
            disable_update=disable_update, **model_options
        )
        self.batch_size_s = int(batch_size_s)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def align(self, audio: AudioChunk, text: str) -> list[Word]:
        return self.align_many([(audio, text)])[0]

    def align_many(self, items: list[tuple[AudioChunk, str]]) -> list[list[Word]]:
        if not items:
            return []
        if any(audio.sample_rate != 16_000 for audio, _ in items):
            raise ValueError("Paraformer alignment expects 16 kHz audio")
        result = self.model.generate(
            input=[pcm_to_float32(audio.pcm_s16le, audio.channels) for audio, _ in items],
            batch_size=self.batch_size,
            batch_size_s=self.batch_size_s,
            disable_pbar=True,
            sentence_timestamp=True,
            output_timestamp=True,
            return_time_stamps=True,
        )
        if len(result) != len(items):
            raise RuntimeError(
                f"Paraformer returned {len(result)} alignment results for {len(items)} inputs"
            )
        aligned = []
        for (audio, text), item in zip(items, result):
            aligned.append(_aligned_words(
                text,
                str(item.get("text", "")),
                item.get("timestamp") or item.get("timestamps") or [],
                offset=audio.start,
                duration=audio.duration,
            ))
        return aligned

    def close(self) -> None:
        self.model = None


class CamppDiarizationBackend:
    """Offline CAM++ diarization through FunASR's supported integrated pipeline."""

    name = "campp"

    def __init__(
        self,
        *,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        spk_model: str = "cam++",
        device: str = "cpu",
        batch_size_s: int = 300,
        hub: str = "ms",
        disable_update: bool = True,
        merge_gap_seconds: float = 0.2,
        max_materialized_seconds: float = 10_800.0,
        **model_options,
    ):
        AutoModel = _auto_model()
        self.model = AutoModel(
            model=model,
            vad_model=vad_model,
            spk_model=spk_model,
            device=_pytorch_device(device),
            hub=hub,
            disable_update=disable_update,
            **model_options,
        )
        self.batch_size_s = int(batch_size_s)
        self.merge_gap_seconds = float(merge_gap_seconds)
        self.max_materialized_seconds = _materialization_limit(
            max_materialized_seconds
        )

    def diarize(self, chunks: Iterable[AudioChunk]) -> Iterable[SpeakerTurn]:
        data, sample_rate, channels, offset = _collect_pcm_bounded(
            chunks,
            max_seconds=self.max_materialized_seconds,
        )
        if not data:
            return []
        if sample_rate != 16_000:
            raise ValueError("CAM++ diarization expects 16 kHz audio")
        result = self.model.generate(
            input=pcm_to_float32(data, channels),
            batch_size_s=self.batch_size_s,
            sentence_timestamp=True,
            output_timestamp=True,
            return_time_stamps=True,
        )
        sentence_info = _result_item(result).get("sentence_info") or []
        turns: list[SpeakerTurn] = []
        for item in sentence_info:
            if not isinstance(item, dict) or item.get("spk") is None:
                continue
            start = offset + max(0.0, float(item.get("start", 0))) / 1000.0
            end = offset + max(0.0, float(item.get("end", 0))) / 1000.0
            if end <= start:
                continue
            speaker = f"speaker-{int(item['spk']) + 1}"
            if turns and turns[-1].speaker == speaker and start - turns[-1].end <= self.merge_gap_seconds:
                turns[-1].end = max(turns[-1].end, end)
            else:
                turns.append(SpeakerTurn(start, end, speaker))
        return turns

    def close(self) -> None:
        self.model = None
