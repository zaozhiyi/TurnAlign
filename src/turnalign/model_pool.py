from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any

from .plugins import AsrConfig
from .resources import close_resources

LOGGER = logging.getLogger(__name__)


def _close_backend(backend: Any, *, reason: str) -> None:
    """Isolate third-party cleanup failures after a pool entry is detached."""
    close_resources((backend,), logger=LOGGER, reason=reason)


class BackendPoolCapacityError(RuntimeError):
    """Raised when every bounded backend slot is actively leased."""


@dataclass(slots=True)
class _Entry:
    config_key: str = ""
    backend: Any = None
    busy: bool = True
    loading: bool = True
    last_used: float = 0.0


class BackendPool:
    """Reuse bounded model replicas and serialize each unsafe instance."""

    def __init__(self, max_entries: int = 8, max_entries_per_key: int = 1) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_entries_per_key <= 0:
            raise ValueError("max_entries_per_key must be positive")
        if max_entries_per_key > max_entries:
            raise ValueError("max_entries_per_key cannot exceed max_entries")
        self.max_entries = max_entries
        self.max_entries_per_key = max_entries_per_key
        self._condition = threading.Condition()
        self._entries: dict[str, _Entry] = {}
        self._next_slot = 0
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
        config_key = self.key(name, config)
        creator = False
        slot_key = ""
        evicted_backend = None
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("backend pool is closed")
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("backend acquisition cancelled")
                matching = [
                    (candidate_key, candidate)
                    for candidate_key, candidate in self._entries.items()
                    if candidate.config_key == config_key
                ]
                for candidate_key, candidate in matching:
                    if not candidate.loading and not candidate.busy:
                        candidate.busy = True
                        candidate.last_used = monotonic()
                        return candidate_key, candidate.backend
                if len(matching) < self.max_entries_per_key:
                    if len(self._entries) >= self.max_entries:
                        candidates = [
                            (candidate_key, candidate)
                            for candidate_key, candidate in self._entries.items()
                            if not candidate.busy and not candidate.loading
                        ]
                        if not candidates:
                            raise BackendPoolCapacityError(
                                "backend pool capacity reached"
                            )
                        evicted_key, evicted = min(
                            candidates,
                            key=lambda item: item[1].last_used,
                        )
                        evicted_backend = evicted.backend
                        self._entries.pop(evicted_key)
                    self._next_slot += 1
                    slot_key = f"{config_key}#{self._next_slot}"
                    entry = _Entry(config_key=config_key)
                    self._entries[slot_key] = entry
                    creator = True
                    break
                self._condition.wait(timeout=0.1)
        if evicted_backend is not None:
            _close_backend(evicted_backend, reason="eviction")
        if creator:
            try:
                backend = factory()
            except BaseException:
                with self._condition:
                    self._entries.pop(slot_key, None)
                    self._condition.notify_all()
                raise
            close_after_load = False
            with self._condition:
                if self._closed or self._entries.get(slot_key) is not entry:
                    close_after_load = True
                else:
                    entry.backend = backend
                    entry.loading = False
                    entry.last_used = monotonic()
                    return slot_key, backend
            if close_after_load:
                _close_backend(backend, reason="late initialization")
                raise RuntimeError("backend pool closed during initialization")
        raise AssertionError("unreachable backend pool state")

    def release(self, key: str) -> None:
        with self._condition:
            entry = self._entries.get(key)
            if entry is not None:
                entry.busy = False
                entry.last_used = monotonic()
                self._condition.notify_all()

    def discard(self, key: str) -> None:
        """Remove and close one lease instead of retaining sensitive session state."""
        with self._condition:
            entry = self._entries.pop(key, None)
            self._condition.notify_all()
        if entry is not None and entry.backend is not None:
            _close_backend(entry.backend, reason="discard")

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
            _close_backend(backend, reason="pool shutdown")
