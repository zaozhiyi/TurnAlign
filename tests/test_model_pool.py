import unittest

from turnalign.model_pool import BackendPool
from turnalign.plugins import AsrConfig


class FakeBackend:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class BackendPoolTests(unittest.TestCase):
    def test_idle_backend_is_reused_then_evicted_at_capacity(self):
        pool = BackendPool(max_entries=1)
        first = FakeBackend()
        key, acquired = pool.acquire("a", AsrConfig(model="one"), lambda: first)
        self.assertIs(acquired, first)
        pool.release(key)

        same_key, reused = pool.acquire(
            "a",
            AsrConfig(model="one"),
            lambda: self.fail("reused backend must not reload"),
        )
        self.assertIs(reused, first)
        pool.release(same_key)

        second = FakeBackend()
        second_key, acquired = pool.acquire(
            "a",
            AsrConfig(model="two"),
            lambda: second,
        )
        self.assertTrue(first.closed)
        self.assertIs(acquired, second)
        pool.release(second_key)
        pool.close()
        self.assertTrue(second.closed)

    def test_busy_pool_fails_instead_of_loading_unbounded_models(self):
        pool = BackendPool(max_entries=1)
        key, _ = pool.acquire("a", AsrConfig(model="one"), FakeBackend)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            pool.acquire("a", AsrConfig(model="two"), FakeBackend)
        pool.release(key)
        pool.close()


if __name__ == "__main__":
    unittest.main()
