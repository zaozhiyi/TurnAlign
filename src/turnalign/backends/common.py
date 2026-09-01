from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from pathlib import Path

from ..models import AudioChunk

_LOCAL_MODEL_ROOT = Path("/var/lib/turnalign/models")
_MAX_MODEL_FILES = 65_536
_MAX_MODEL_BYTES = 64 * 1024 * 1024 * 1024


def require_local_model_path(path: object, *, directory: bool) -> Path:
    """Return a canonical, immutable local model path bound to the retained root."""

    if not isinstance(path, str) or not path:
        raise ValueError("local model loading requires an explicit --model-path")
    candidate = Path(path)
    if not candidate.is_absolute() or Path(os.path.normpath(str(candidate))) != candidate:
        raise ValueError("local model path must be absolute and normalized")
    try:
        root = _LOCAL_MODEL_ROOT.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"local model path cannot be resolved: {candidate}") from error
    if root != _LOCAL_MODEL_ROOT or resolved != candidate:
        raise ValueError("local model path and root must not contain symbolic links")
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(
            f"local model path must be under {_LOCAL_MODEL_ROOT}: {candidate}"
        )
    # Every ancestor controls whether the service account can replace the
    # retained tree. Validating only the leaf is insufficient when a writable
    # parent permits rename-based substitution.
    ancestors = tuple(reversed(resolved.parents)) + (resolved,)
    metadata = None
    for ancestor in ancestors:
        try:
            ancestor_metadata = os.lstat(ancestor)
        except OSError as error:
            raise ValueError(f"local model path is unavailable: {candidate}") from error
        if (
            stat.S_ISLNK(ancestor_metadata.st_mode)
            or ancestor_metadata.st_uid != 0
            or stat.S_IMODE(ancestor_metadata.st_mode) & 0o022
        ):
            raise ValueError(
                "local model path and all ancestors must be root-owned, "
                f"non-writable by group/others, and non-symlink: {ancestor}"
            )
        if ancestor != resolved and not stat.S_ISDIR(ancestor_metadata.st_mode):
            raise ValueError(f"local model ancestor must be a directory: {ancestor}")
        metadata = ancestor_metadata
    if metadata is None:
        raise RuntimeError("local model path validation lost its metadata")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"local model path must be a directory: {candidate}")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"local model path must be a regular file: {candidate}")
    return resolved


def local_model_files(path: Path | None) -> tuple[Path, ...]:
    """List every immutable file retained under a validated local model root."""

    if path is None:
        return ()
    root = require_local_model_path(str(path), directory=True)
    entries: list[Path] = []
    total_size = 0
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != 0
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise ValueError(f"unsafe model directory: {child}")
        for name in file_names:
            child = directory_path / name
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != 0
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise ValueError(f"unsafe model file: {child}")
            total_size += metadata.st_size
            if len(entries) >= _MAX_MODEL_FILES or total_size > _MAX_MODEL_BYTES:
                raise ValueError("local model tree exceeds production evidence limits")
            entries.append(child)
    if not entries:
        raise ValueError(f"local model directory contains no files: {root}")
    return tuple(sorted(entries))


def collect_pcm(chunks: Iterable[AudioChunk]) -> tuple[bytes, int, int, float]:
    data = bytearray()
    sample_rate = 0
    channels = 0
    start = 0.0
    for index, chunk in enumerate(chunks):
        if index == 0:
            sample_rate, channels, start = chunk.sample_rate, chunk.channels, chunk.start
        elif (chunk.sample_rate, chunk.channels) != (sample_rate, channels):
            raise ValueError("audio format changed during transcription")
        data.extend(chunk.pcm_s16le)
    if not data:
        return b"", 16_000, 1, 0.0
    return bytes(data), sample_rate, channels, start


def pcm_to_float32(data: bytes, channels: int):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("this backend requires numpy") from error
    samples = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples
