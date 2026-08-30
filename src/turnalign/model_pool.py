from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any

from .plugins import AsrConfig


@dataclass(slots=True)
class _Entry:
    backend: Any = None
    busy: bool = True
    loading: bool = True
    last_used: float = 0.0


class BackendPool:
    """Reuse one loaded model per configuration and serialize unsafe inference."""

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._condition = threading.Condition()
        self._entries: dict[str, _Entry] = {}
        self._closed = False

    @staticmethod
    def key(name: str, config: AsrConfig) -> str:
        return f"{name}:{json.dumps(asdict(config), sort_keys=True, default=str)}"

    def acquire(
        self,
        name: str,
        config: AsrConfig,
        factory: Callable[[], Any],
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, Any]:
        key = self.key(name, config)
        creator = False
        evicted_backend = None
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("backend pool is closed")
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("backend acquisition cancelled")
                entry = self._entries.get(key)
                if entry is None:
                    if len(self._entries) >= self.max_entries:
                        candidates = [
                            (candidate_key, candidate)
                            for candidate_key, candidate in self._entries.items()
                            if not candidate.busy and not candidate.loading
                        ]
                        if not candidates:
                            raise RuntimeError("backend pool capacity reached")
                        evicted_key, evicted = min(
                            candidates,
                            key=lambda item: item[1].last_used,
                        )
                        evicted_backend = evicted.backend
                        self._entries.pop(evicted_key)
                    entry = _Entry()
                    self._entries[key] = entry
                    creator = True
                    break
                if not entry.loading and not entry.busy:
                    entry.busy = True
                    entry.last_used = monotonic()
                    return key, entry.backend
                self._condition.wait(timeout=0.1)
        if evicted_backend is not None:
            evicted_backend.close()
        if creator:
            try:
                backend = factory()
            except BaseException:
                with self._condition:
                    self._entries.pop(key, None)
                    self._condition.notify_all()
                raise
            with self._condition:
                entry.backend = backend
                entry.loading = False
                entry.last_used = monotonic()
                return key, backend
        raise AssertionError("unreachable backend pool state")

    def release(self, key: str) -> None:
        with self._condition:
            entry = self._entries.get(key)
            if entry is not None:
                entry.busy = False
                entry.last_used = monotonic()
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            backends = [
                entry.backend
                for entry in self._entries.values()
                if entry.backend is not None
            ]
            self._entries.clear()
            self._condition.notify_all()
        for backend in backends:
            backend.close()
