from __future__ import annotations

import math
from dataclasses import dataclass

MAX_HOTWORDS = 128
MAX_HOTWORD_CHARS = 80
MAX_CONTEXT_CHARS = 2_000


@dataclass(frozen=True, slots=True)
class AsrHints:
    """Backend-neutral user vocabulary and topic context for ASR decoding."""

    hotwords: tuple[str, ...] = ()
    context: str | None = None
    boost: float | None = None

    def __post_init__(self) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in self.hotwords:
            value = str(raw).strip()
            if not value or value.startswith("#"):
                continue
            if "\n" in value or "\r" in value:
                raise ValueError("hotwords must contain one phrase per item")
            if len(value) > MAX_HOTWORD_CHARS:
                raise ValueError(f"hotword exceeds {MAX_HOTWORD_CHARS} characters")
            key = value.casefold()
            if key not in seen:
                normalized.append(value)
                seen.add(key)
        if len(normalized) > MAX_HOTWORDS:
            raise ValueError(f"at most {MAX_HOTWORDS} hotwords are allowed per request")
        if self.context is not None and not isinstance(self.context, str):
            raise TypeError("context must be a string")
        context = self.context.strip() if self.context else None
        if context and len(context) > MAX_CONTEXT_CHARS:
            raise ValueError(f"context exceeds {MAX_CONTEXT_CHARS} characters")
        boost = self.boost
        if boost is not None:
            if isinstance(boost, bool) or not isinstance(boost, (int, float)):
                raise TypeError("hotword boost must be a number")
            boost = float(boost)
            if not math.isfinite(boost) or not 0 < boost <= 1_000:
                raise ValueError("hotword boost must be finite and between 0 and 1000")
        object.__setattr__(self, "hotwords", tuple(normalized))
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "boost", boost)

    @property
    def active(self) -> bool:
        return bool(self.hotwords or self.context)

    def private_metadata(self, method: str) -> dict[str, object]:
        """Report hint use without copying private phrases into output events."""
        return {
            "method": method,
            "hotword_count": len(self.hotwords),
            "context_used": self.context is not None,
            "boost_requested": self.boost is not None,
        }


def glm_transcription_prompt(hints: AsrHints) -> str | None:
    if not hints.active:
        return None
    parts = ["Transcribe the input speech faithfully and output only what was spoken."]
    if hints.context:
        parts.append(f"Topic context, for reference only: {hints.context}")
    if hints.hotwords:
        parts.append("Possible phrases: " + ", ".join(hints.hotwords))
    parts.append(
        "Use a phrase only when acoustically supported. Do not add explanations or unspoken content."
    )
    return "\n".join(parts)


def whisper_initial_prompt(hints: AsrHints) -> str | None:
    """Whisper prompts behave as prior transcript text, not instructions."""
    parts = []
    if hints.context:
        parts.append(hints.context)
    if hints.hotwords:
        parts.append(", ".join(hints.hotwords))
    return "\n".join(parts) or None
