from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

from .backends.common import require_local_model_path


class ModelRevisionError(RuntimeError):
    pass


_MODEL_EVIDENCE_ROOT = Path("/var/lib/turnalign/models")
_MAX_MODEL_EVIDENCE_FILES = 65_536


def _digest_regular_file(path: Path) -> tuple[str, int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError(f"model evidence is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            for block in iter(lambda: source.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
            final_metadata = os.fstat(source.fileno())
        current_metadata = os.lstat(path)
        if (
            (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            != (
                final_metadata.st_dev,
                final_metadata.st_ino,
                final_metadata.st_size,
                final_metadata.st_mtime_ns,
            )
            or (final_metadata.st_dev, final_metadata.st_ino)
            != (current_metadata.st_dev, current_metadata.st_ino)
        ):
            raise ValueError(f"model evidence changed while being hashed: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), size


def observed_model_files(resource: object) -> tuple[dict[str, object], ...]:
    """Return content-bound evidence only for backends that loaded local models."""

    provider = getattr(resource, "loaded_model_files", None)
    if not callable(provider):
        return ()
    raw_paths = provider()
    if not isinstance(raw_paths, (list, tuple)):
        raise TypeError("backend loaded_model_files must return a sequence")
    entries: list[dict[str, object]] = []
    for path in raw_paths:
        if not isinstance(path, Path):
            raise TypeError("backend loaded_model_files entries must be Path objects")
        resolved = require_local_model_path(str(path), directory=False)
        root = _MODEL_EVIDENCE_ROOT
        if resolved != root and not resolved.is_relative_to(root):
            raise ValueError(
                f"loaded model path must be under {_MODEL_EVIDENCE_ROOT}: {path}"
            )
        try:
            metadata = os.lstat(resolved)
        except OSError as error:
            raise ValueError(f"loaded model file is unavailable: {path}") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError(
                f"loaded model path must be root-owned, immutable, and regular: {path}"
            )
        digest, size = _digest_regular_file(resolved)
        entries.append({
            "path": str(resolved),
            "sha256": digest,
            "bytes": size,
        })
    if len(entries) > _MAX_MODEL_EVIDENCE_FILES:
        raise ValueError("backend returned too many loaded model files")
    return tuple(entries)


def model_revision(resource: object) -> str | None:
    value = getattr(resource, "model_revision", None)
    return value if isinstance(value, str) and value else None


def is_immutable_model_revision(resource: object) -> bool:
    revision = model_revision(resource)
    return bool(
        revision
        and re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision)
    )


def require_immutable_model_revision(resource: object) -> str:
    revision = model_revision(resource)
    if not is_immutable_model_revision(resource):
        raise ModelRevisionError(
            "model revision is not pinned to an immutable 40- or "
            "64-character commit hash"
        )
    if revision is None:
        raise RuntimeError("immutable model revision validation lost its value")
    return revision


def close_resources(
    resources: Iterable[object | None],
    *,
    logger: logging.Logger,
    reason: str,
) -> None:
    """Close every plugin resource without allowing one failure to stop cleanup."""
    for resource in resources:
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            logger.warning(
                "resource close failed during %s resource_type=%s",
                reason,
                type(resource).__name__,
                exc_info=True,
            )
