from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise, permutations

from .exporters import final_segments
from .models import TranscriptEvent


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, 1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def _rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _distance(reference, hypothesis) / len(reference)


def character_error_rate(reference: str, hypothesis: str) -> float:
    return _rate(
        [character for character in reference if not character.isspace()],
        [character for character in hypothesis if not character.isspace()],
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    return _rate(reference.split(), hypothesis.split())


def diarization_error_rate(
    reference: list[TranscriptEvent],
    hypothesis: list[TranscriptEvent],
) -> float | None:
    ref_segments = [item for item in final_segments(reference) if item.speaker]
    hyp_segments = [item for item in final_segments(hypothesis) if item.speaker]
    boundaries = sorted({
        value
        for item in [*ref_segments, *hyp_segments]
        for value in (item.start, item.end)
    })
    if len(boundaries) < 2:
        return None

    def speaker_at(items: list[TranscriptEvent], point: float) -> str | None:
        for item in items:
            if item.start <= point < item.end:
                return item.speaker
        return None

    intervals: list[tuple[str | None, str | None, float]] = []
    for start, end in pairwise(boundaries):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        ref_speaker = speaker_at(ref_segments, midpoint)
        hyp_speaker = speaker_at(hyp_segments, midpoint)
        duration = end - start
        intervals.append((ref_speaker, hyp_speaker, duration))

    ref_labels = sorted({item[0] for item in intervals if item[0] is not None})
    hyp_labels = sorted({item[1] for item in intervals if item[1] is not None})
    mapping: dict[str, str | None] = {}
    if hyp_labels and len(hyp_labels) <= 8 and len(ref_labels) <= 8:
        candidates: list[str | None] = list(ref_labels)
        candidates.extend([None] * max(0, len(hyp_labels) - len(candidates)))
        best_score = -1.0
        for assignment in set(permutations(candidates, len(hyp_labels))):
            candidate_mapping = dict(zip(hyp_labels, assignment))
            score = sum(
                duration
                for ref_speaker, hyp_speaker, duration in intervals
                if hyp_speaker is not None
                and candidate_mapping.get(hyp_speaker) == ref_speaker
            )
            if score > best_score:
                best_score = score
                mapping = candidate_mapping
    else:
        for hyp_speaker in hyp_labels:
            overlaps = {
                ref_speaker: sum(
                    duration
                    for current_ref, current_hyp, duration in intervals
                    if current_ref == ref_speaker and current_hyp == hyp_speaker
                )
                for ref_speaker in ref_labels
            }
            mapping[hyp_speaker] = max(overlaps, key=overlaps.get) if overlaps else None

    reference_seconds = sum(
        duration for ref_speaker, _, duration in intervals if ref_speaker is not None
    )
    errors = sum(
        duration
        for ref_speaker, hyp_speaker, duration in intervals
        if (
            ref_speaker is not None
            or hyp_speaker is not None
        ) and (
            ref_speaker != (mapping.get(hyp_speaker) if hyp_speaker is not None else None)
        )
    )
    return errors / reference_seconds if reference_seconds else None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    character_error_rate: float
    word_error_rate: float
    diarization_error_rate: float | None
    reference_segments: int
    hypothesis_segments: int
    revision_updates_per_segment: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_events(
    reference: list[TranscriptEvent],
    hypothesis: list[TranscriptEvent],
) -> EvaluationReport:
    reference_final = final_segments(reference)
    hypothesis_final = final_segments(hypothesis)
    reference_text = " ".join(item.text for item in reference_final)
    hypothesis_text = " ".join(item.text for item in hypothesis_final)
    transcript_events = [
        item for item in hypothesis if item.kind in {"partial", "commit", "replace"}
    ]
    revisions = max(0, len(transcript_events) - len(hypothesis_final))
    return EvaluationReport(
        character_error_rate=character_error_rate(reference_text, hypothesis_text),
        word_error_rate=word_error_rate(reference_text, hypothesis_text),
        diarization_error_rate=diarization_error_rate(reference, hypothesis),
        reference_segments=len(reference_final),
        hypothesis_segments=len(hypothesis_final),
        revision_updates_per_segment=(
            revisions / len(hypothesis_final) if hypothesis_final else 0.0
        ),
    )
