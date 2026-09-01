from __future__ import annotations

import subprocess
from typing import BinaryIO, TypeVar

_ProcessData = TypeVar("_ProcessData", str, bytes)
PROCESS_EXIT_TIMEOUT_SECONDS = 5
PROCESS_ERROR_LIMIT_BYTES = 64 * 1024


def process_error_tail(
    output: BinaryIO,
    *,
    limit_bytes: int = PROCESS_ERROR_LIMIT_BYTES,
) -> str:
    """Read a bounded tail of subprocess diagnostics from a seekable file."""
    output.flush()
    output.seek(0, 2)
    length = output.tell()
    output.seek(max(0, length - limit_bytes))
    message = output.read(limit_bytes).decode("utf-8", errors="replace").strip()
    if length > limit_bytes:
        return f"[earlier output truncated] {message}"
    return message


def terminate_process(
    process: subprocess.Popen[_ProcessData],
    *,
    timeout_seconds: float = PROCESS_EXIT_TIMEOUT_SECONDS,
) -> None:
    """Reap a child process, escalating to kill when graceful exit stalls."""
    if process.poll() is not None:
        process.wait()
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()
