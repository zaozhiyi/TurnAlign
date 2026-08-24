from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from time import perf_counter

from .fusion import assign_speakers
from .models import AudioChunk, Hypothesis, SpeakerTurn, TranscriptEvent, Word
from .plugins import AlignmentBackend, AsrBackend, DiarizationBackend


def pcm_rms(chunk: AudioChunk) -> float:
    samples = array("h")
    samples.frombytes(chunk.pcm_s16le)
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


def utterances(
    chunks: Iterable[AudioChunk],
    *,
    threshold: float = 0.012,
    silence_seconds: float = 0.7,
    max_seconds: float = 20.0,
) -> Iterator[list[AudioChunk]]:
    """Simple dependency-free endpointing for batch ASR backends."""
    if threshold < 0 or silence_seconds < 0 or max_seconds <= 0:
        raise ValueError("invalid utterance segmentation settings")
    active: list[AudioChunk] = []
    preroll: AudioChunk | None = None
    silence = 0.0
    duration = 0.0
    for chunk in chunks:
        voiced = pcm_rms(chunk) >= threshold
        if not active:
            if not voiced:
                preroll = chunk
                continue
            if preroll is not None:
                active.append(preroll)
                duration += preroll.duration
            active.append(chunk)
            duration += chunk.duration
            silence = 0.0
            preroll = None
        else:
            active.append(chunk)
            duration += chunk.duration
            silence = 0.0 if voiced else silence + chunk.duration
        if active and (silence >= silence_seconds or duration >= max_seconds):
            yield active
            active = []
            duration = 0.0
            silence = 0.0
            preroll = None
    if active:
        yield active


def live_windows(
    chunks: Iterable[AudioChunk],
    *,
    threshold: float = 0.012,
    silence_seconds: float = 0.7,
    max_seconds: float = 20.0,
    partial_seconds: float = 2.0,
) -> Iterator[tuple[list[AudioChunk], bool]]:
    """Yield growing partial windows and final endpointed utterances."""
    if partial_seconds <= 0:
        raise ValueError("partial_seconds must be positive")
    active: list[AudioChunk] = []
    preroll: AudioChunk | None = None
    silence = 0.0
    duration = 0.0
    next_partial = partial_seconds
    for chunk in chunks:
        voiced = pcm_rms(chunk) >= threshold
        if not active:
            if not voiced:
                preroll = chunk
                continue
            if preroll is not None:
                active.append(preroll)
                duration += preroll.duration
            active.append(chunk)
            duration += chunk.duration
            silence = 0.0
            preroll = None
        else:
            active.append(chunk)
            duration += chunk.duration
            silence = 0.0 if voiced else silence + chunk.duration
        if active and (silence >= silence_seconds or duration >= max_seconds):
            yield active, True
            active = []
            duration = 0.0
            silence = 0.0
            next_partial = partial_seconds
            preroll = None
        elif active and duration >= next_partial:
            yield list(active), False
            next_partial += partial_seconds
    if active:
        yield active, True


def _merge_hypotheses(hypotheses: Iterable[Hypothesis], final: bool) -> Hypothesis | None:
    items = [item for item in hypotheses if item.text]
    if not items:
        return None
    return Hypothesis(
        text=" ".join(item.text.strip() for item in items).strip(),
        start=items[0].start,
        end=items[-1].end,
        words=[word for item in items for word in item.words],
        language=items[-1].language,
        confidence=items[-1].confidence,
        final=final,
        metadata={"backend_segments": len(items)},
    )


def _event(hypothesis: Hypothesis, index: int, revision: int = 1) -> TranscriptEvent:
    return TranscriptEvent(
        kind="commit" if hypothesis.final else "partial",
        segment_id=f"seg-{index:06d}",
        revision=revision,
        start=hypothesis.start,
        end=hypothesis.end,
        text=hypothesis.text,
        words=hypothesis.words,
        metadata={
            **hypothesis.metadata,
            "language": hypothesis.language,
            "confidence": hypothesis.confidence,
        },
    )


def _slice_audio(chunks: list[AudioChunk], start: float, end: float) -> AudioChunk:
    if not chunks:
        return AudioChunk(b"", start)
    sample_rate, channels = chunks[0].sample_rate, chunks[0].channels
    output = bytearray()
    frame_bytes = channels * 2
    for chunk in chunks:
        if (chunk.sample_rate, chunk.channels) != (sample_rate, channels):
            raise ValueError("audio format changed before post-processing")
        overlap_start = max(start, chunk.start)
        overlap_end = min(end, chunk.start + chunk.duration)
        if overlap_end <= overlap_start:
            continue
        first = round((overlap_start - chunk.start) * sample_rate) * frame_bytes
        last = round((overlap_end - chunk.start) * sample_rate) * frame_bytes
        output.extend(chunk.pcm_s16le[first:last])
    return AudioChunk(bytes(output), start, sample_rate, channels)


def _speaker_for_interval(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    best = (0.0, "")
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best[0]:
            best = (overlap, turn.speaker)
    return best[1] or None


def _speaker_from_words(words: list[Word]) -> str | None:
    totals: dict[str, float] = {}
    for word in words:
        if word.speaker:
            totals[word.speaker] = totals.get(word.speaker, 0.0) + word.end - word.start
    return max(totals, key=totals.get) if totals else None


def transcribe_events(
    chunks: Iterable[AudioChunk],
    backend: AsrBackend,
    *,
    live: bool = False,
    vad: bool = True,
    vad_threshold: float = 0.012,
    silence_seconds: float = 0.7,
    max_utterance_seconds: float = 20.0,
    partial_seconds: float = 2.0,
    aligner: AlignmentBackend | None = None,
    diarizer: DiarizationBackend | None = None,
) -> Iterator[TranscriptEvent]:
    """Run either a native-streaming backend or endpointed batch backend."""
    started = perf_counter()
    index = 0
    revision = 1
    last_end = 0.0
    input_end = 0.0
    audio_seconds = 0.0
    recorded: list[AudioChunk] = []
    commits: list[TranscriptEvent] = []

    def observed_chunks() -> Iterator[AudioChunk]:
        nonlocal input_end, audio_seconds
        for chunk in chunks:
            audio_seconds += chunk.duration
            input_end = max(input_end, chunk.start + chunk.duration)
            if aligner is not None or diarizer is not None:
                recorded.append(chunk)
            yield chunk

    source = observed_chunks()
    try:
        if backend.capabilities.streaming:
            for hypothesis in backend.transcribe(source):
                if not hypothesis.text:
                    continue
                event = _event(hypothesis, index, revision)
                last_end = max(last_end, event.end)
                yield event
                if hypothesis.final:
                    commits.append(event)
                    index += 1
                    revision = 1
                else:
                    revision += 1
        else:
            if live:
                for group, final in live_windows(
                    source,
                    threshold=vad_threshold,
                    silence_seconds=silence_seconds,
                    max_seconds=max_utterance_seconds,
                    partial_seconds=partial_seconds,
                ):
                    hypothesis = _merge_hypotheses(backend.transcribe(group), final)
                    if hypothesis is None:
                        continue
                    event = _event(hypothesis, index, revision)
                    last_end = max(last_end, event.end)
                    yield event
                    if final:
                        commits.append(event)
                        index += 1
                        revision = 1
                    else:
                        revision += 1
            elif vad:
                groups: Iterable[Iterable[AudioChunk]] = utterances(
                    source,
                    threshold=vad_threshold,
                    silence_seconds=silence_seconds,
                    max_seconds=max_utterance_seconds,
                )
            else:
                groups = (source,)
            if not live:
                for group in groups:
                    for hypothesis in backend.transcribe(group):
                        if not hypothesis.text:
                            continue
                        hypothesis.final = True
                        event = _event(hypothesis, index)
                        last_end = max(last_end, event.end)
                        yield event
                        commits.append(event)
                        index += 1

        turns = list(diarizer.diarize(iter(recorded))) if diarizer is not None else []
        for event in commits:
            words = [
                Word(word.text, word.start, word.end, word.confidence, word.speaker)
                for word in event.words
            ]
            if aligner is not None:
                segment_audio = _slice_audio(recorded, event.start, event.end)
                words = aligner.align(segment_audio, event.text)
            if turns and words:
                words = assign_speakers(words, turns)
            speaker = _speaker_from_words(words) or _speaker_for_interval(event.start, event.end, turns)
            if words != event.words or speaker != event.speaker:
                yield TranscriptEvent(
                    kind="replace",
                    segment_id=event.segment_id,
                    revision=event.revision + 1,
                    start=event.start,
                    end=event.end,
                    text=event.text,
                    words=words,
                    speaker=speaker,
                    metadata={**event.metadata, "postprocessed": True},
                )

        elapsed = perf_counter() - started
        end_time = max(last_end, input_end)
        yield TranscriptEvent(
            kind="end",
            segment_id="session",
            revision=1,
            start=end_time,
            end=end_time,
            metadata={
                "segments": index,
                "backend": backend.name,
                "audio_seconds": round(audio_seconds, 3),
                "processing_seconds": round(elapsed, 3),
                "realtime_factor": round(elapsed / audio_seconds, 4) if audio_seconds else None,
                "speed_x": round(audio_seconds / elapsed, 3) if elapsed else None,
            },
        )
    finally:
        backend.close()
        for component in (aligner, diarizer):
            close = getattr(component, "close", None)
            if callable(close):
                close()
