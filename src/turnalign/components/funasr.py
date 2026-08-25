from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from difflib import SequenceMatcher

from ..backends.common import collect_pcm, pcm_to_float32
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


def _audio_chunk(chunks: list[AudioChunk], start: float, end: float) -> AudioChunk:
    if not chunks:
        raise ValueError("cannot slice an empty audio stream")
    sample_rate, channels = chunks[0].sample_rate, chunks[0].channels
    frame_bytes = channels * 2
    output = bytearray()
    for chunk in chunks:
        if (chunk.sample_rate, chunk.channels) != (sample_rate, channels):
            raise ValueError("audio format changed before component processing")
        overlap_start = max(start, chunk.start)
        overlap_end = min(end, chunk.start + chunk.duration)
        if overlap_end <= overlap_start:
            continue
        first = round((overlap_start - chunk.start) * sample_rate) * frame_bytes
        last = round((overlap_end - chunk.start) * sample_rate) * frame_bytes
        output.extend(chunk.pcm_s16le[first:last])
    if not output:
        raise ValueError("component returned an empty audio region")
    return AudioChunk(bytes(output), start, sample_rate, channels)


def _result_item(result) -> dict:
    if not result:
        return {}
    item = result[0]
    return item if isinstance(item, dict) else {}


class FsmnVadBackend:
    name = "fsmn-vad"

    def __init__(
        self,
        *,
        model: str = "fsmn-vad",
        device: str = "cpu",
        batch_size_s: int = 300,
        max_segment_seconds: float = 20.0,
        hub: str = "ms",
        disable_update: bool = True,
        **model_options,
    ):
        if max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds must be positive")
        AutoModel = _auto_model()
        self.model = AutoModel(
            model=model, device=device, hub=hub, disable_update=disable_update, **model_options
        )
        self.model_id = model
        self.batch_size_s = int(batch_size_s)
        self.max_segment_seconds = float(max_segment_seconds)

    def segment(self, chunks: Iterable[AudioChunk]) -> Iterator[SpeechSegment]:
        items = list(chunks)
        data, sample_rate, channels, offset = collect_pcm(items)
        if not data:
            return
        if sample_rate != 16_000:
            raise ValueError("FSMN-VAD expects 16 kHz audio")
        result = self.model.generate(
            input=pcm_to_float32(data, channels), batch_size_s=self.batch_size_s
        )
        regions = _result_item(result).get("value") or []
        duration = len(data) / (2 * channels * sample_rate)
        previous_end = 0.0
        for raw in sorted(regions, key=lambda item: float(item[0]) if item else 0.0):
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            start = max(previous_end, min(duration, float(raw[0]) / 1000.0))
            end = max(start, min(duration, float(raw[1]) / 1000.0))
            while end > start:
                piece_end = min(end, start + self.max_segment_seconds)
                forced = piece_end < end
                absolute_start = offset + start
                absolute_end = offset + piece_end
                audio = _audio_chunk(items, absolute_start, absolute_end)
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
        left = result_times[run_start - 1][1] if run_start else offset
        right = result_times[run_end][0] if run_end < len(result_times) else absolute_end
        right = max(left, right)
        width = (right - left) / max(1, run_end - run_start)
        for index in range(run_start, run_end):
            start = left + width * (index - run_start)
            result_times[index] = (start, left + width * (index - run_start + 1))

    words: list[Word] = []
    previous_end = offset
    for unit, raw_time in zip(target, result_times):
        assert raw_time is not None
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
        hub: str = "ms",
        disable_update: bool = True,
        **model_options,
    ):
        AutoModel = _auto_model()
        self.model = AutoModel(
            model=model, device=device, hub=hub, disable_update=disable_update, **model_options
        )
        self.batch_size_s = int(batch_size_s)

    def align(self, audio: AudioChunk, text: str) -> list[Word]:
        if audio.sample_rate != 16_000:
            raise ValueError("Paraformer alignment expects 16 kHz audio")
        result = self.model.generate(
            input=pcm_to_float32(audio.pcm_s16le, audio.channels),
            batch_size_s=self.batch_size_s,
            sentence_timestamp=True,
            output_timestamp=True,
            return_time_stamps=True,
        )
        item = _result_item(result)
        return _aligned_words(
            text,
            str(item.get("text", "")),
            item.get("timestamp") or item.get("timestamps") or [],
            offset=audio.start,
            duration=audio.duration,
        )

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
        **model_options,
    ):
        AutoModel = _auto_model()
        self.model = AutoModel(
            model=model,
            vad_model=vad_model,
            spk_model=spk_model,
            device=device,
            hub=hub,
            disable_update=disable_update,
            **model_options,
        )
        self.batch_size_s = int(batch_size_s)
        self.merge_gap_seconds = float(merge_gap_seconds)

    def diarize(self, chunks: Iterable[AudioChunk]) -> Iterable[SpeakerTurn]:
        data, sample_rate, channels, offset = collect_pcm(chunks)
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
