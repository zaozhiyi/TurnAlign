from __future__ import annotations

from .models import SpeakerTurn, Word


def assign_speakers(words: list[Word], turns: list[SpeakerTurn]) -> list[Word]:
    """Assign each word to the speaker turn with the greatest time overlap."""
    for word in words:
        best_speaker = None
        best_overlap = 0.0
        best_confidence = float("-inf")
        for turn in turns:
            overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
            confidence = turn.confidence if turn.confidence is not None else 0.0
            if overlap > best_overlap or (overlap == best_overlap and overlap > 0 and confidence > best_confidence):
                best_overlap = overlap
                best_confidence = confidence
                best_speaker = turn.speaker
        word.speaker = best_speaker
    return words
