from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import pairwise, permutations
from math import isfinite
from typing import Literal, cast
from unicodedata import category, normalize

from .exporters import final_segments
from .models import TranscriptEvent


@dataclass(frozen=True, slots=True)
class TextNormalization:
    """Explicit, serializable text policy used before CER/WER calculation."""

    unicode_form: str = "none"
    case_sensitive: bool = True
    punctuation_sensitive: bool = True

    def __post_init__(self) -> None:
        if self.unicode_form not in {"none", "NFC", "NFKC"}:
            raise ValueError("unicode_form must be one of: none, NFC, NFKC")
        if not isinstance(self.case_sensitive, bool):
            raise TypeError("case_sensitive must be a boolean")
        if not isinstance(self.punctuation_sensitive, bool):
            raise TypeError("punctuation_sensitive must be a boolean")

    def apply(self, text: str) -> str:
        if self.unicode_form != "none":
            form = cast(Literal["NFC", "NFKC"], self.unicode_form)
            text = normalize(form, text)
        if not self.punctuation_sensitive:
            text = "".join(
                " " if category(character).startswith("P") else character
                for character in text
            )
        return text if self.case_sensitive else text.casefold()


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
                if ref_speaker is not None
                and hyp_speaker is not None
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
            mapping[hyp_speaker] = (
                max(overlaps, key=overlaps.__getitem__) if overlaps else None
            )

    reference_seconds = sum(
        duration for ref_speaker, _, duration in intervals if ref_speaker is not None
    )
    errors = 0.0
    for ref_speaker, hyp_speaker, duration in intervals:
        if ref_speaker is None:
            # Any hypothesized speech during reference silence is a false alarm,
            # including a system speaker left unmapped by the optimal assignment.
            if hyp_speaker is not None:
                errors += duration
            continue
        if hyp_speaker is None or mapping.get(hyp_speaker) != ref_speaker:
            errors += duration
    return errors / reference_seconds if reference_seconds else None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    text_normalization: TextNormalization
    character_error_rate: float
    word_error_rate: float
    diarization_error_rate: float | None
    reference_segments: int
    hypothesis_segments: int
    revision_updates_per_segment: float
    reference_characters: int
    reference_words: int
    reference_speech_seconds: float
    reference_speakers: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_events(
    reference: list[TranscriptEvent],
    hypothesis: list[TranscriptEvent],
    *,
    text_normalization: TextNormalization | None = None,
) -> EvaluationReport:
    text_normalization = text_normalization or TextNormalization()
    reference_final = final_segments(reference)
    hypothesis_final = final_segments(hypothesis)
    reference_text = text_normalization.apply(
        " ".join(item.text for item in reference_final)
    )
    hypothesis_text = text_normalization.apply(
        " ".join(item.text for item in hypothesis_final)
    )
    transcript_events = [
        item for item in hypothesis if item.kind in {"partial", "commit", "replace"}
    ]
    revisions = max(0, len(transcript_events) - len(hypothesis_final))

    intervals = sorted(
        (item.start, item.end)
        for item in reference_final
        if item.end > item.start
    )
    reference_speech_seconds = 0.0
    if intervals:
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                reference_speech_seconds += current_end - current_start
                current_start, current_end = start, end
        reference_speech_seconds += current_end - current_start
    return EvaluationReport(
        text_normalization=text_normalization,
        character_error_rate=character_error_rate(reference_text, hypothesis_text),
        word_error_rate=word_error_rate(reference_text, hypothesis_text),
        diarization_error_rate=diarization_error_rate(reference, hypothesis),
        reference_segments=len(reference_final),
        hypothesis_segments=len(hypothesis_final),
        revision_updates_per_segment=(
            revisions / len(hypothesis_final) if hypothesis_final else 0.0
        ),
        reference_characters=sum(
            not character.isspace() for character in reference_text
        ),
        reference_words=len(reference_text.split()),
        reference_speech_seconds=reference_speech_seconds,
        reference_speakers=len({
            item.speaker for item in reference_final if item.speaker
        }),
    )


@dataclass(frozen=True, slots=True)
class QualityGateReport:
    status: str
    source_commit: str | None
    reference_sha256: str | None
    hypothesis_sha256: str | None
    created_at: str
    validity_seconds: float
    model: str | None
    model_revision: str | None
    evaluation: EvaluationReport
    max_character_error_rate: float | None
    max_word_error_rate: float | None
    max_diarization_error_rate: float | None
    max_revision_updates_per_segment: float | None
    min_reference_segments: int
    min_reference_characters: int
    min_reference_speech_seconds: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_quality_gate(
    reference: list[TranscriptEvent],
    hypothesis: list[TranscriptEvent],
    *,
    max_character_error_rate: float | None = None,
    max_word_error_rate: float | None = None,
    max_diarization_error_rate: float | None = None,
    max_revision_updates_per_segment: float | None = None,
    min_reference_segments: int = 1,
    min_reference_characters: int = 1,
    min_reference_speech_seconds: float = 0.0,
    text_normalization: TextNormalization | None = None,
    source_commit: str | None = None,
    reference_sha256: str | None = None,
    hypothesis_sha256: str | None = None,
    model: str | None = None,
    validity_seconds: float = 86400.0,
) -> QualityGateReport:
    maxima = {
        "max_character_error_rate": max_character_error_rate,
        "max_word_error_rate": max_word_error_rate,
        "max_diarization_error_rate": max_diarization_error_rate,
        "max_revision_updates_per_segment": max_revision_updates_per_segment,
    }
    if all(value is None for value in maxima.values()):
        raise ValueError("quality gate requires at least one maximum metric")
    for name, value in maxima.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be non-negative and finite")
    for name, value in (
        ("min_reference_segments", min_reference_segments),
        ("min_reference_characters", min_reference_characters),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(min_reference_speech_seconds, bool)
        or not isinstance(min_reference_speech_seconds, (int, float))
        or not isfinite(min_reference_speech_seconds)
        or min_reference_speech_seconds < 0
    ):
        raise ValueError(
            "min_reference_speech_seconds must be non-negative and finite"
        )
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, (int, float))
        or not isfinite(validity_seconds)
        or validity_seconds <= 0
    ):
        raise ValueError("validity_seconds must be finite and positive")

    evaluation = evaluate_events(
        reference,
        hypothesis,
        text_normalization=text_normalization,
    )
    model_revisions = {
        revision
        for event in hypothesis
        if isinstance((revision := event.metadata.get("model_revision")), str)
        and revision
    }
    hypothesis_model_revision = (
        next(iter(model_revisions)) if len(model_revisions) == 1 else None
    )
    failures: list[str] = []
    if evaluation.reference_segments < min_reference_segments:
        failures.append(
            f"reference segment count {evaluation.reference_segments} is below "
            f"minimum {min_reference_segments}"
        )
    if evaluation.reference_characters < min_reference_characters:
        failures.append(
            f"reference character count {evaluation.reference_characters} is below "
            f"minimum {min_reference_characters}"
        )
    if evaluation.reference_speech_seconds < min_reference_speech_seconds:
        failures.append(
            f"reference speech {evaluation.reference_speech_seconds:.3f}s is below "
            f"minimum {min_reference_speech_seconds:.3f}s"
        )
    for name, actual, maximum in (
        (
            "character error rate",
            evaluation.character_error_rate,
            max_character_error_rate,
        ),
        ("word error rate", evaluation.word_error_rate, max_word_error_rate),
        (
            "revision updates per segment",
            evaluation.revision_updates_per_segment,
            max_revision_updates_per_segment,
        ),
    ):
        if maximum is not None and actual > maximum:
            failures.append(f"{name} {actual:.6f} exceeds maximum {maximum:.6f}")
    if max_diarization_error_rate is not None:
        if evaluation.diarization_error_rate is None:
            failures.append(
                "diarization error rate is unavailable because labelled speaker "
                "intervals are missing"
            )
        elif evaluation.diarization_error_rate > max_diarization_error_rate:
            failures.append(
                "diarization error rate "
                f"{evaluation.diarization_error_rate:.6f} exceeds maximum "
                f"{max_diarization_error_rate:.6f}"
            )
    return QualityGateReport(
        status="passed" if not failures else "failed",
        source_commit=source_commit,
        reference_sha256=reference_sha256,
        hypothesis_sha256=hypothesis_sha256,
        created_at=(
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        ),
        validity_seconds=round(validity_seconds, 3),
        model=model,
        model_revision=hypothesis_model_revision,
        evaluation=evaluation,
        max_character_error_rate=max_character_error_rate,
        max_word_error_rate=max_word_error_rate,
        max_diarization_error_rate=max_diarization_error_rate,
        max_revision_updates_per_segment=max_revision_updates_per_segment,
        min_reference_segments=min_reference_segments,
        min_reference_characters=min_reference_characters,
        min_reference_speech_seconds=min_reference_speech_seconds,
        failures=tuple(failures),
    )
