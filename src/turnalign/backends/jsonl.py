from __future__ import annotations

from pathlib import Path

from ..jsonutil import strict_json_object
from ..models import Hypothesis
from ..plugins import Accelerator, BackendCapabilities


class JsonlBackend:
    """Replay existing JSONL transcripts through the common backend shape."""

    name = "jsonl"
    capabilities = BackendCapabilities(
        streaming=True,
        word_timestamps=False,
        accelerators=(Accelerator.CPU,),
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def hypotheses(self):
        with self.path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    item = strict_json_object(
                        line,
                        label=f"{self.path}:{line_number}",
                    )
                    yield Hypothesis(
                        text=str(item.get("text", "")).strip(),
                        start=float(str(item["start"])),
                        end=float(str(item["end"])),
                        final=True,
                        metadata={key: value for key, value in item.items() if key not in {"text", "start", "end"}},
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{self.path}:{line_number}: {error}") from error

    def transcribe(self, chunks):
        # Replay is deterministic; audio chunks are intentionally ignored.
        del chunks
        yield from self.hypotheses()

    def close(self) -> None:
        return None
