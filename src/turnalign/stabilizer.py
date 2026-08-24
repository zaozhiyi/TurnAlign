from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


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

    def __post_init__(self) -> None:
        if self.min_confirmations < 2:
            raise ValueError("min_confirmations must be at least 2")
        self._history = deque(self._history, maxlen=self.min_confirmations)

    def update(self, hypothesis: str, final: bool = False) -> AgreementResult:
        hypothesis = hypothesis.strip()
        if final:
            if not hypothesis.startswith(self.committed):
                self.committed = hypothesis
                self._history.clear()
                return AgreementResult(replace=hypothesis)
            delta = hypothesis[len(self.committed) :]
            self.committed = hypothesis
            self._history.clear()
            return AgreementResult(committed_delta=delta)

        self._history.append(hypothesis)
        agreed = ""
        if len(self._history) == self.min_confirmations:
            agreed = self._history[0]
            for item in list(self._history)[1:]:
                agreed = common_prefix(agreed, item)
        delta = agreed[len(self.committed) :] if agreed.startswith(self.committed) else ""
        if delta:
            self.committed += delta
        partial = hypothesis[len(self.committed) :] if hypothesis.startswith(self.committed) else hypothesis
        return AgreementResult(committed_delta=delta, partial=partial)

    def reset(self) -> None:
        self.committed = ""
        self._history.clear()
