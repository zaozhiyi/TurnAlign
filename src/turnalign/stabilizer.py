from __future__ import annotations

import unicodedata
from collections import deque
from dataclasses import dataclass, field


def common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def _normalized(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    boundaries: list[int] = []
    for index, character in enumerate(text):
        if character.isspace() or unicodedata.category(character)[0] in {"P", "Z"}:
            continue
        for folded in character.casefold():
            characters.append(folded)
            boundaries.append(index + 1)
    return "".join(characters), boundaries


def _suffix_after_normalized(text: str, length: int) -> str:
    if length <= 0:
        return text
    _, boundaries = _normalized(text)
    if length > len(boundaries):
        return ""
    return text[boundaries[length - 1]:]


def _prefix_through_normalized(text: str, length: int) -> str:
    if length <= 0:
        return ""
    _, boundaries = _normalized(text)
    if length > len(boundaries):
        return text
    return text[:boundaries[length - 1]]


@dataclass(slots=True)
class AgreementResult:
    committed_delta: str = ""
    partial: str = ""
    replace: str | None = None


@dataclass(slots=True)
class LocalAgreement:
    """Commit the common prefix shared by the latest N hypotheses."""

    min_confirmations: int = 2
    committed: str = ""
    _history: deque[str] = field(default_factory=deque)
    _committed_key: str = ""

    def __post_init__(self) -> None:
        if self.min_confirmations < 2:
            raise ValueError("min_confirmations must be at least 2")
        self._history = deque(self._history, maxlen=self.min_confirmations)

    def update(self, hypothesis: str, final: bool = False) -> AgreementResult:
        hypothesis = hypothesis.strip()
        hypothesis_key, _ = _normalized(hypothesis)
        if final:
            if not hypothesis_key.startswith(self._committed_key):
                self.committed = hypothesis
                self._committed_key = hypothesis_key
                self._history.clear()
                return AgreementResult(replace=hypothesis)
            delta = _suffix_after_normalized(hypothesis, len(self._committed_key))
            self.committed += delta
            self._committed_key = hypothesis_key
            self._history.clear()
            return AgreementResult(committed_delta=delta)

        self._history.append(hypothesis)
        agreed_key = ""
        agreed_source = ""
        if len(self._history) == self.min_confirmations:
            agreed_source = self._history[0]
            agreed_key, _ = _normalized(agreed_source)
            for item in list(self._history)[1:]:
                item_key, _ = _normalized(item)
                agreed_key = common_prefix(agreed_key, item_key)
        delta = ""
        if agreed_key.startswith(self._committed_key) and len(agreed_key) > len(self._committed_key):
            agreed_prefix = _prefix_through_normalized(agreed_source, len(agreed_key))
            delta = _suffix_after_normalized(agreed_prefix, len(self._committed_key))
        if delta:
            self.committed += delta
            self._committed_key = agreed_key
        partial = (
            _suffix_after_normalized(hypothesis, len(self._committed_key))
            if hypothesis_key.startswith(self._committed_key)
            else ""
        )
        return AgreementResult(committed_delta=delta, partial=partial)

    def reset(self) -> None:
        self.committed = ""
        self._committed_key = ""
        self._history.clear()
