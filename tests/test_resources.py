import logging
import unittest

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


if __name__ == "__main__":
    unittest.main()
