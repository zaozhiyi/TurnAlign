from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_SOURCE_COMMIT_ENVIRONMENT = "TURNALIGN_SOURCE_COMMIT"
_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_UNBOUND_SOURCE = "unbound"


class CustomBuildHook(BuildHookInterface):
    """Embed the exact CI source revision in every built wheel."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        source_commit = os.environ.get(
            _SOURCE_COMMIT_ENVIRONMENT,
            _UNBOUND_SOURCE,
        )
        if (
            source_commit != _UNBOUND_SOURCE
            and _SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None
        ):
            raise ValueError(
                f"{_SOURCE_COMMIT_ENVIRONMENT} must be a lowercase 40-character "
                "Git commit"
            )
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            prefix="turnalign-source-commit-",
            suffix=".txt",
            delete=False,
        ) as temporary:
            temporary.write(f"{source_commit}\n")
        generated = Path(temporary.name)
        self._generated_source = generated
        build_data["force_include"][str(generated)] = "turnalign/_source_commit.txt"

    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact_path: str,
    ) -> None:
        del version, build_data, artifact_path
        generated = getattr(self, "_generated_source", None)
        if generated is not None:
            generated.unlink(missing_ok=True)
