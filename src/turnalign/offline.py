from __future__ import annotations

import logging
from dataclasses import dataclass

from .audio import AudioTimeline
from .fusion import assign_speakers
from .models import Hypothesis, SpeakerTurn, TranscriptEvent, Word
from .plugins import AlignmentBackend, AsrBackend, DiarizationBackend
from .resources import close_resources

LOGGER = logging.getLogger(__name__)


def _merged_hypothesis(items: list[Hypothesis], fallback: TranscriptEvent) -> Hypothesis:
    usable = [item for item in items if item.text]
    if not usable:
        return Hypothesis(
            fallback.text,
            fallback.start,
            fallback.end,
            words=list(fallback.words),
            final=True,
        )
    return Hypothesis(
        " ".join(item.text.strip() for item in usable).strip(),
        usable[0].start,
        usable[-1].end,
        words=[word for item in usable for word in item.words],
        language=usable[-1].language,
        confidence=usable[-1].confidence,
        final=True,
        metadata={"backend_segments": len(usable)},
    )


def _speaker_for_interval(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> str | None:
    overlaps = [
        (max(0.0, min(end, turn.end) - max(start, turn.start)), turn.speaker)
        for turn in turns
    ]
    best = max(overlaps, default=(0.0, ""))
    return best[1] if best[0] > 0 else None


@dataclass(slots=True)
class OfflineRefinementPipeline:
    """High-quality second pass that revises first-pass segment IDs in place."""

    backend: AsrBackend
    aligner: AlignmentBackend | None = None
    diarizer: DiarizationBackend | None = None
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("refinement batch_size must be positive")

    def refine(
        self,
        timeline: AudioTimeline,
        commits: list[TranscriptEvent],
    ):
        try:
            turns = (
                list(self.diarizer.diarize(timeline.iter_chunks()))
                if self.diarizer is not None else []
            )
            refined_speakers = [
                _speaker_for_interval(event.start, event.end, turns)
                if turns else event.speaker
                for event in commits
            ]
            mapping_candidates: dict[str, set[str]] = {}
            for event, speaker in zip(commits, refined_speakers):
                if event.speaker and speaker and event.speaker != speaker:
                    mapping_candidates.setdefault(event.speaker, set()).add(speaker)
            speaker_mappings = {
                source: next(iter(targets))
                for source, targets in mapping_candidates.items()
                if len(targets) == 1
            }
            emitted_mappings: set[str] = set()
            for batch_start in range(0, len(commits), self.batch_size):
                batch_commits = commits[batch_start:batch_start + self.batch_size]
                batch_speakers = refined_speakers[
                    batch_start:batch_start + self.batch_size
                ]
                audio = [
                    timeline.slice(event.start, event.end)
                    for event in batch_commits
                ]
                hypotheses = [
                    _merged_hypothesis(
                        list(self.backend.transcribe((item,))),
                        event,
                    )
                    for item, event in zip(audio, batch_commits)
                ]
                words: list[list[Word]] = [
                    list(item.words) for item in hypotheses
                ]
                if self.aligner is not None:
                    align_many = getattr(self.aligner, "align_many", None)
                    requests = [
                        (item, hypothesis.text)
                        for item, hypothesis in zip(audio, hypotheses)
                    ]
                    if callable(align_many):
                        words = list(align_many(requests))
                        if len(words) != len(batch_commits):
                            raise RuntimeError(
                                "batch aligner returned the wrong number of results"
                            )
                    else:
                        words = [
                            self.aligner.align(item, text)
                            for item, text in requests
                        ]

                for event, hypothesis, refined_words, speaker in zip(
                    batch_commits,
                    hypotheses,
                    words,
                    batch_speakers,
                ):
                    if turns and refined_words:
                        refined_words = assign_speakers(refined_words, turns)
                    changed = (
                        hypothesis.text != event.text
                        or refined_words != event.words
                        or speaker != event.speaker
                    )
                    if not changed:
                        continue
                    if (
                        event.speaker is not None
                        and speaker is not None
                        and event.speaker != speaker
                        and event.speaker in speaker_mappings
                        and event.speaker not in emitted_mappings
                    ):
                        emitted_mappings.add(event.speaker)
                        yield TranscriptEvent(
                            kind="speaker_merge",
                            segment_id=f"speaker-merge:{event.speaker}",
                            revision=1,
                            start=event.start,
                            end=event.end,
                            metadata={
                                "from_speaker": event.speaker,
                                "to_speaker": speaker,
                                "offline_refined": True,
                            },
                        )
                    yield TranscriptEvent(
                        kind="replace",
                        segment_id=event.segment_id,
                        revision=event.revision + 1,
                        start=event.start,
                        end=event.end,
                        text=hypothesis.text,
                        words=refined_words,
                        speaker=speaker,
                        metadata={
                            **event.metadata,
                            **hypothesis.metadata,
                            "offline_refined": True,
                            "refinement_backend": self.backend.name,
                        },
                    )
        finally:
            close_resources(
                (self.backend, self.aligner, self.diarizer),
                logger=LOGGER,
                reason="offline refinement shutdown",
            )
