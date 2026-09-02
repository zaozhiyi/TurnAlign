import logging
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turnalign.backends import common as backend_common
from turnalign.resources import (
    ModelRevisionError,
    close_resources,
    is_immutable_model_revision,
    model_revision,
    require_immutable_model_revision,
)


class Resource:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.closed = False

    def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


class ResourceCleanupTests(unittest.TestCase):
    def test_model_revision_helpers_require_exact_commit_hashes(self):
        missing = Resource()
        mutable = Resource()
        mutable.model_revision = "main"
        pinned = Resource()
        pinned.model_revision = "a" * 40

        self.assertIsNone(model_revision(missing))
        self.assertFalse(is_immutable_model_revision(mutable))
        with self.assertRaisesRegex(ModelRevisionError, "not pinned"):
            require_immutable_model_revision(mutable)
        self.assertTrue(is_immutable_model_revision(pinned))
        self.assertEqual(require_immutable_model_revision(pinned), "a" * 40)

    def test_one_close_failure_does_not_block_remaining_resources(self):
        first = Resource(fail=True)
        second = Resource()
        with self.assertLogs("turnalign.test.resources", level="WARNING") as captured:
            close_resources(
                (None, object(), first, second),
                logger=logging.getLogger("turnalign.test.resources"),
                reason="test cleanup",
            )
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertIn("resource_type=Resource", captured.output[0])

    @unittest.skipUnless(os.name == "posix", "model ownership is a POSIX invariant")
    def test_local_model_path_rejects_symlinks_and_writable_ancestors(self):
        original_lstat = os.lstat
        with tempfile.TemporaryDirectory() as directory:
            retained_root = (Path(directory) / "models").resolve()

            def trusted_root_owned(path):
                metadata = original_lstat(path)
                mode = metadata.st_mode
                if Path(path) != retained_root:
                    mode &= ~0o022
                return SimpleNamespace(st_mode=mode, st_uid=0)

            model = retained_root / "model"
            model.mkdir(parents=True)
            weights = model / "weights.bin"
            weights.write_bytes(b"weights")
            link = retained_root / "linked-model"
            link.symlink_to(model, target_is_directory=True)
            with patch.object(
                backend_common,
                "_LOCAL_MODEL_ROOT",
                retained_root,
            ), patch.object(
                backend_common.os,
                "lstat",
                side_effect=trusted_root_owned,
            ):
                self.assertEqual(
                    backend_common.require_local_model_path(
                        str(model), directory=True
                    ),
                    model,
                )
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    backend_common.require_local_model_path(
                        str(link), directory=True
                    )

                retained_root.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)
                with self.assertRaisesRegex(ValueError, "group/others"):
                    backend_common.require_local_model_path(
                        str(model), directory=True
                    )


if __name__ == "__main__":
    unittest.main()
