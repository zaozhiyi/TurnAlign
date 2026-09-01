from __future__ import annotations

import logging
import re
from collections.abc import Iterable


class ModelRevisionError(RuntimeError):
    pass


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
