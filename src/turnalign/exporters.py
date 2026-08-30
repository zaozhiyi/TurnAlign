from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from .models import TranscriptEvent


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1_000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def final_segments(events: Iterable[TranscriptEvent]) -> list[TranscriptEvent]:
    segments: dict[str, TranscriptEvent] = {}
    speaker_merges: dict[str, str] = {}
    for event in events:
        if event.kind == "speaker_merge":
            source = event.metadata.get("from_speaker")
            target = event.metadata.get("to_speaker")
            if isinstance(source, str) and isinstance(target, str):
                speaker_merges[source] = target
            continue
        if event.kind not in {"commit", "replace"}:
            continue
        previous = segments.get(event.segment_id)
        if previous is None or event.revision > previous.revision:
            segments[event.segment_id] = event
    result = []
    for event in sorted(segments.values(), key=lambda item: (item.start, item.end, item.segment_id)):
        speaker = event.speaker
        visited: set[str] = set()
        while speaker in speaker_merges and speaker not in visited:
            visited.add(speaker)
            speaker = speaker_merges[speaker]
        result.append(replace(event, speaker=speaker))
    return result


def render_text(events: Iterable[TranscriptEvent]) -> str:
    lines = []
    for event in final_segments(events):
        prefix = f"{event.speaker}: " if event.speaker else ""
        lines.append(f"{prefix}{event.text}".strip())
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(events: Iterable[TranscriptEvent]) -> str:
    blocks = []
    for index, event in enumerate(final_segments(events), 1):
        prefix = f"[{event.speaker}] " if event.speaker else ""
        blocks.append(
            f"{index}\n{_timestamp(event.start)} --> {_timestamp(event.end)}\n"
            f"{prefix}{event.text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
