from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from time import perf_counter

from .audio import AudioTimeline
from .fusion import assign_speakers
from .models import (
    AudioChunk,
    Hypothesis,
    SpeakerTurn,
    SpeechSegment,
    TranscriptEvent,
    Word,
)
from .plugins import (
    AlignmentBackend,
    AsrBackend,
    DiarizationBackend,
    OnlineDiarizationBackend,
    VadBackend,
)
from .stabilizer import LocalAgreement


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
    vad_backend: VadBackend | None = None,
    vad_audit: Callable[[dict[str, object]], None] | None = None,
    recorded_audio: list[AudioChunk] | None = None,
    recorded_timeline: AudioTimeline | None = None,
    parallel_diarization: bool = False,
    execution_profile: str | None = None,
    cancel_event: Event | None = None,
    close_backend: bool = True,
    emit_end: bool = True,
    online_diarizer: OnlineDiarizationBackend | None = None,
    segment_index_start: int = 0,
) -> Iterator[TranscriptEvent]:
    """Run either a native-streaming backend or endpointed batch backend."""
    started = perf_counter()
    if segment_index_start < 0:
        raise ValueError("segment_index_start must be non-negative")
    index = segment_index_start
    revision = 1
    last_end = 0.0
    input_end = 0.0
    audio_seconds = 0.0
    needs_recording = (
        aligner is not None
        or diarizer is not None
        or recorded_timeline is not None
    )
    timeline = recorded_timeline
    owns_timeline = False
    if needs_recording and timeline is None:
        timeline = AudioTimeline()
        owns_timeline = True
        for recorded_chunk in recorded_audio or []:
            timeline.append(recorded_chunk)
    record_from_source = needs_recording and recorded_audio is None and (
        recorded_timeline is None or recorded_timeline.chunk_count == 0
    )
    commits: list[TranscriptEvent] = []
    vad_speech_seconds = 0.0
    vad_regions = 0
    vad_forced_splits = 0
    vad_last_end = 0.0
    diarization_seconds = 0.0
    alignment_seconds = 0.0
    diarization_executor: ThreadPoolExecutor | None = None
    diarization_future: Future[list[SpeakerTurn]] | None = None
    online_session = online_diarizer.start_session() if online_diarizer is not None else None
    online_turns: list[SpeakerTurn] = []

    def observed_chunks() -> Iterator[AudioChunk]:
        nonlocal input_end, audio_seconds
        for chunk in chunks:
            if cancel_event is not None and cancel_event.is_set():
                return
            audio_seconds += chunk.duration
            input_end = max(input_end, chunk.start + chunk.duration)
            if record_from_source:
                assert timeline is not None
                timeline.append(chunk)
            if online_session is not None:
                online_turns.extend(online_session.accept_audio(chunk))
            yield chunk

    def decorate_online(event: TranscriptEvent) -> TranscriptEvent:
        if not online_turns:
            return event
        if event.words:
            event.words = assign_speakers(event.words, online_turns)
        event.speaker = (
            _speaker_from_words(event.words)
            or _speaker_for_interval(event.start, event.end, online_turns)
        )
        if event.speaker is not None:
            event.metadata["speaker_provisional"] = True
        return event

    source = observed_chunks()

    def run_diarizer() -> list[SpeakerTurn]:
        nonlocal diarization_seconds
        if diarizer is None:
            return []
        assert timeline is not None
        component_started = perf_counter()
        try:
            return list(diarizer.diarize(timeline.iter_chunks()))
        finally:
            diarization_seconds = perf_counter() - component_started

    if parallel_diarization:
        if diarizer is None or timeline is None or timeline.chunk_count == 0:
            raise ValueError("parallel diarization requires a diarizer and preloaded audio timeline")
        diarization_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turnalign-diarizer")
        diarization_future = diarization_executor.submit(run_diarizer)

    def audit_region(
        decision: str,
        start: float,
        end: float,
        *,
        segment: SpeechSegment | None = None,
    ) -> None:
        if vad_audit is None or end <= start:
            return
        payload: dict[str, object] = {
            "decision": decision,
            "start": round(start, 6),
            "end": round(end, 6),
            "duration": round(end - start, 6),
            "backend": getattr(vad_backend, "name", "energy"),
        }
        if segment is not None:
            payload.update({
                "confidence": segment.confidence,
                "forced_split": segment.forced_split,
                "metadata": segment.metadata,
            })
        vad_audit(payload)

    def audited_segments() -> Iterator[list[AudioChunk]]:
        nonlocal vad_speech_seconds, vad_regions, vad_forced_splits, vad_last_end
        assert vad_backend is not None
        for segment in vad_backend.segment(source):
            if segment.start < vad_last_end:
                raise ValueError("VAD returned overlapping or out-of-order speech segments")
            audit_region("silence", vad_last_end, segment.start)
            audit_region("speech", segment.start, segment.end, segment=segment)
            vad_speech_seconds += segment.end - segment.start
            vad_regions += 1
            vad_forced_splits += int(segment.forced_split)
            vad_last_end = segment.end
            yield segment.chunks
        audit_region("silence", vad_last_end, input_end)

    try:
        if backend.capabilities.streaming:
            streaming_session = None
            start_session = getattr(backend, "start_session", None)
            if callable(start_session):
                streaming_session = start_session()

            def streaming_hypotheses() -> Iterator[Hypothesis]:
                if streaming_session is None:
                    yield from backend.transcribe(source)
                    return
                try:
                    for source_chunk in source:
                        if cancel_event is not None and cancel_event.is_set():
                            streaming_session.cancel()
                            return
                        yield from streaming_session.accept_audio(source_chunk)
                    yield from streaming_session.finish()
                finally:
                    streaming_session.close()

            for hypothesis in streaming_hypotheses():
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not hypothesis.text:
                    continue
                event = decorate_online(_event(hypothesis, index, revision))
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
                agreement = LocalAgreement()
                for group, final in live_windows(
                    source,
                    threshold=vad_threshold,
                    silence_seconds=silence_seconds,
                    max_seconds=max_utterance_seconds,
                    partial_seconds=partial_seconds,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    hypothesis = _merge_hypotheses(backend.transcribe(group), final)
                    if hypothesis is None:
                        continue
                    stabilized = agreement.update(hypothesis.text, final=final)
                    if final:
                        hypothesis.text = stabilized.replace or agreement.committed
                    else:
                        hypothesis.text = agreement.committed + stabilized.partial
                    hypothesis.metadata["stabilized"] = True
                    event = decorate_online(_event(hypothesis, index, revision))
                    last_end = max(last_end, event.end)
                    yield event
                    if final:
                        commits.append(event)
                        index += 1
                        revision = 1
                        agreement.reset()
                    else:
                        revision += 1
            elif vad_backend is not None:
                groups = audited_segments()
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
                        event = decorate_online(_event(hypothesis, index))
                        last_end = max(last_end, event.end)
                        yield event
                        commits.append(event)
                        index += 1

        asr_seconds = perf_counter() - started
        if online_session is not None:
            online_turns.extend(online_session.finish())
        turns = (
            diarization_future.result()
            if diarization_future is not None
            else run_diarizer()
        )
        if not turns:
            turns = online_turns
        aligned_words: list[list[Word]] | None = None
        if aligner is not None:
            alignment_started = perf_counter()
            assert timeline is not None
            align_many = getattr(aligner, "align_many", None)
            if callable(align_many):
                aligned_words = []
                alignment_batch_size = int(getattr(aligner, "batch_size", 16))
                if alignment_batch_size <= 0:
                    raise ValueError("alignment batch size must be positive")
                for batch_start in range(0, len(commits), alignment_batch_size):
                    batch = commits[
                        batch_start:batch_start + alignment_batch_size
                    ]
                    inputs = [
                        (timeline.slice(event.start, event.end), event.text)
                        for event in batch
                    ]
                    results = list(align_many(inputs))
                    if len(results) != len(batch):
                        raise RuntimeError(
                            "batch aligner returned the wrong number of results"
                        )
                    aligned_words.extend(results)
            else:
                aligned_words = [
                    aligner.align(
                        timeline.slice(event.start, event.end),
                        event.text,
                    )
                    for event in commits
                ]
            alignment_seconds = perf_counter() - alignment_started

        for event_index, event in enumerate(commits):
            words = [
                Word(word.text, word.start, word.end, word.confidence, word.speaker)
                for word in event.words
            ]
            if aligned_words is not None:
                words = aligned_words[event_index]
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
        metadata = {
            "segments": index - segment_index_start,
            "segment_index_next": index,
            "backend": backend.name,
            "audio_seconds": round(audio_seconds, 3),
            "processing_seconds": round(elapsed, 3),
            "realtime_factor": round(elapsed / audio_seconds, 4) if audio_seconds else None,
            "speed_x": round(audio_seconds / elapsed, 3) if elapsed else None,
            "asr_seconds": round(asr_seconds, 3),
            "diarization_seconds": round(diarization_seconds, 3),
            "alignment_seconds": round(alignment_seconds, 3),
            "parallel_diarization": diarization_future is not None,
        }
        if execution_profile is not None:
            metadata["execution_profile"] = execution_profile
        if vad_backend is not None:
            metadata.update({
                "vad_backend": vad_backend.name,
                "vad_speech_seconds": round(vad_speech_seconds, 3),
                "vad_skipped_seconds": round(max(0.0, audio_seconds - vad_speech_seconds), 3),
                "vad_speech_regions": vad_regions,
                "vad_forced_splits": vad_forced_splits,
            })
        if emit_end and not (cancel_event is not None and cancel_event.is_set()):
            yield TranscriptEvent(
                kind="end",
                segment_id="session",
                revision=1,
                start=end_time,
                end=end_time,
                metadata=metadata,
            )
    finally:
        if diarization_executor is not None:
            diarization_executor.shutdown(wait=True, cancel_futures=True)
        if owns_timeline and timeline is not None:
            timeline.close()
        if online_session is not None:
            online_session.close()
        if online_diarizer is not None:
            online_diarizer.close()
        if close_backend:
            backend.close()
        for component in (vad_backend, aligner, diarizer):
            close = getattr(component, "close", None)
            if callable(close):
                close()
