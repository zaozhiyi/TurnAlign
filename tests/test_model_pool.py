import unittest
from threading import Event

from turnalign.hints import AsrHints
from turnalign.model_pool import BackendPool, BackendPoolCapacityError
from turnalign.plugins import AsrConfig


class FakeBackend:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FailingCloseBackend(FakeBackend):
    def close(self):
        self.closed = True
        raise RuntimeError("close failed")


class BlockingCloseBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.finished = Event()

    def close(self):
        self.started.set()
        self.release.wait()
        super().close()
        self.finished.set()


class BackendPoolTests(unittest.TestCase):
    def test_capacity_requires_strict_positive_integers(self):
        for name in ("max_entries", "max_entries_per_key"):
            for value in (True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    ValueError, name
                ):
                    BackendPool(**{name: value})

    def test_pool_keys_are_strict_canonical_private_fingerprints(self):
        first = AsrConfig(
            model="model",
            extra={"nested": {"b": 2, "a": 1}},
            hints=AsrHints(context="PRIVATE_CONTEXT"),
        )
        reordered = AsrConfig(
            model="model",
            extra={"nested": {"a": 1, "b": 2}},
            hints=AsrHints(context="PRIVATE_CONTEXT"),
        )
        key = BackendPool.key("fake", first)
        self.assertRegex(key, r"^fake:[0-9a-f]{64}$")
        self.assertNotIn("PRIVATE_CONTEXT", key)
        self.assertEqual(key, BackendPool.key("fake", reordered))
        self.assertNotEqual(
            key,
            BackendPool.key("fake", AsrConfig(model="another")),
        )

        for config, expected in (
            (AsrConfig(extra={"value": float("nan")}), "Out of range"),
            (AsrConfig(extra={"value": object()}), "JSON serializable"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(
                (TypeError, ValueError), expected
            ):
                BackendPool.key("fake", config)
        for name in ("", " fake ", 1):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "backend name"
            ):
                BackendPool.key(name, AsrConfig())

    def test_per_config_replicas_allow_parallel_leases_then_reuse(self):
        pool = BackendPool(max_entries=2, max_entries_per_key=2)
        created = []

        def factory():
            backend = FakeBackend()
            created.append(backend)
            return backend

        first_key, first = pool.acquire("a", AsrConfig(model="one"), factory)
        second_key, second = pool.acquire("a", AsrConfig(model="one"), factory)
        self.assertIsNot(first, second)
        self.assertEqual(len(created), 2)
        pool.release(first_key)
        reused_key, reused = pool.acquire("a", AsrConfig(model="one"), factory)
        self.assertIs(reused, first)
        self.assertEqual(len(created), 2)
        pool.release(reused_key)
        pool.release(second_key)
        pool.close()

    def test_replica_limit_must_fit_total_capacity(self):
        with self.assertRaisesRegex(ValueError, "max_entries_per_key"):
            BackendPool(max_entries=1, max_entries_per_key=2)

    def test_backend_finishing_after_pool_close_is_not_leaked(self):
        pool = BackendPool()
        backend = FakeBackend()

        def close_pool_during_load():
            pool.close()
            return backend

        with self.assertRaisesRegex(RuntimeError, "closed during initialization"):
            pool.acquire("a", AsrConfig(model="one"), close_pool_during_load)
        self.assertTrue(backend.closed)

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
        with self.assertRaisesRegex(BackendPoolCapacityError, "capacity"):
            pool.acquire("a", AsrConfig(model="two"), FakeBackend)
        pool.release(key)
        pool.close()

    def test_discard_closes_and_removes_a_sensitive_lease(self):
        pool = BackendPool(max_entries=1)
        backend = FakeBackend()
        key, _ = pool.acquire("a", AsrConfig(model="one"), lambda: backend)

        pool.discard(key)

        self.assertTrue(backend.closed)
        replacement = FakeBackend()
        replacement_key, acquired = pool.acquire(
            "a", AsrConfig(model="one"), lambda: replacement
        )
        self.assertIs(acquired, replacement)
        pool.release(replacement_key)
        pool.close()

    def test_detach_removes_without_running_untrusted_close(self):
        pool = BackendPool(max_entries=1)
        backend = FakeBackend()
        key, _ = pool.acquire("a", AsrConfig(model="one"), lambda: backend)

        self.assertIs(pool.detach(key), backend)
        self.assertFalse(backend.closed)
        replacement_key, replacement = pool.acquire(
            "a", AsrConfig(model="one"), FakeBackend
        )
        self.assertIsInstance(replacement, FakeBackend)
        pool.release(replacement_key)
        pool.close()

    def test_nonblocking_shutdown_detaches_before_untrusted_close_returns(self):
        pool = BackendPool()
        backend = BlockingCloseBackend()
        key, _ = pool.acquire("a", AsrConfig(model="one"), lambda: backend)
        pool.release(key)

        pool.close(wait=False)

        self.assertTrue(backend.started.wait(1))
        self.assertFalse(backend.closed)
        backend.release.set()
        self.assertTrue(backend.finished.wait(1))
        self.assertTrue(backend.closed)

    def test_close_failures_do_not_block_discard_replacement_or_pool_shutdown(self):
        pool = BackendPool(max_entries=2)
        first = FailingCloseBackend()
        first_key, _ = pool.acquire("a", AsrConfig(model="one"), lambda: first)
        with self.assertLogs("turnalign.model_pool", level="WARNING"):
            pool.discard(first_key)
        self.assertTrue(first.closed)

        second = FailingCloseBackend()
        third = FakeBackend()
        second_key, _ = pool.acquire("a", AsrConfig(model="two"), lambda: second)
        pool.release(second_key)
        third_key, _ = pool.acquire("a", AsrConfig(model="three"), lambda: third)
        pool.release(third_key)
        with self.assertLogs("turnalign.model_pool", level="WARNING"):
            pool.close()
        self.assertTrue(second.closed)
        self.assertTrue(third.closed)

    def test_late_initialization_preserves_pool_error_when_close_fails(self):
        pool = BackendPool()
        backend = FailingCloseBackend()

        def close_pool_during_load():
            pool.close()
            return backend

        with self.assertLogs("turnalign.model_pool", level="WARNING"), self.assertRaisesRegex(
            RuntimeError, "closed during initialization"
        ):
            pool.acquire("a", AsrConfig(model="one"), close_pool_during_load)
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
