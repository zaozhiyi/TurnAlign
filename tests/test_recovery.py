import unittest
from array import array

from turnalign.models import AudioChunk, TranscriptEvent
from turnalign.recovery import (
    RecoveryCapacityError,
    RecoveryEventLimitError,
    RecoveryStore,
)


class RecoveryStoreTests(unittest.TestCase):
    def test_resume_preserves_audio_event_and_segment_sequences(self):
        store = RecoveryStore()
        try:
            session, resumed = store.open("config")
            self.assertFalse(resumed)
            audio = AudioChunk(array("h", [1] * 1_600).tobytes(), 0)
            self.assertEqual(store.append_audio(session, audio), 0)
            payload = store.append_event(
                session,
                TranscriptEvent("partial", "seg-000000", 1, 0, 0.1, "draft"),
            )
            self.assertEqual(payload["sequence"], 0)
            self.assertEqual(payload["acknowledged_sequence"], 0)
            store.release(session)

            recovered, resumed = store.open(
                "config", session.session_id, session.resume_token
            )
            self.assertTrue(resumed)
            self.assertEqual(recovered.timeline.end, 0.1)
            self.assertEqual(recovered.next_segment_index, 1)
            self.assertEqual(len(store.replay_after(recovered, -1)), 2)
            replay = store.replay_after(recovered, 0)
            self.assertEqual(len(replay), 1)
            self.assertEqual(replay[0]["kind"], "commit")
            self.assertTrue(replay[0]["metadata"]["recovery_committed"])
            store.release(recovered, completed=True)
            with self.assertRaisesRegex(ValueError, "completed"):
                store.open("config", session.session_id, session.resume_token)
        finally:
            store.close()

    def test_resume_rejects_configuration_change(self):
        store = RecoveryStore()
        try:
            session, _ = store.open("config-a")
            store.release(session)
            with self.assertRaisesRegex(ValueError, "configuration"):
                store.open("config-b", session.session_id, session.resume_token)
        finally:
            store.close()

    def test_resume_requires_the_session_secret(self):
        store = RecoveryStore()
        try:
            session, _ = store.open("config")
            store.release(session)
            self.assertGreaterEqual(len(session.resume_token), 32)
            self.assertNotIn(session.resume_token, repr(session))
            for token in (None, "", "wrong-token", "错误令牌🔒"):
                with self.subTest(token=token), self.assertRaisesRegex(
                    PermissionError, "authentication"
                ):
                    store.open("config", session.session_id, token)
            recovered, resumed = store.open(
                "config", session.session_id, session.resume_token
            )
            self.assertTrue(resumed)
            store.release(recovered, completed=True)
        finally:
            store.close()

    def test_event_replay_window_is_bounded_and_rejects_stale_ack(self):
        store = RecoveryStore(max_events_per_session=2)
        try:
            session, _ = store.open("config")
            for index in range(3):
                store.append_event(
                    session,
                    TranscriptEvent(
                        "commit",
                        f"seg-{index:06d}",
                        1,
                        index,
                        index + 1,
                        str(index),
                    ),
                )
            self.assertEqual(len(session.events), 2)
            self.assertEqual(session.first_retained_event_sequence, 1)
            self.assertEqual(
                [event["sequence"] for event in store.replay_after(session, 0)],
                [1, 2],
            )
            with self.assertRaisesRegex(ValueError, "retained recovery window"):
                store.replay_after(session, -1)
        finally:
            store.close()

    def test_recovery_limits_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_sessions"):
            RecoveryStore(max_sessions=0)
        with self.assertRaisesRegex(ValueError, "max_events_per_session"):
            RecoveryStore(max_events_per_session=0)
        with self.assertRaisesRegex(ValueError, "max_event_bytes"):
            RecoveryStore(max_event_bytes=0)
        with self.assertRaisesRegex(ValueError, "max_event_bytes_per_session"):
            RecoveryStore(max_event_bytes_per_session=0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            RecoveryStore(max_event_bytes=10, max_event_bytes_per_session=9)
        with self.assertRaisesRegex(ValueError, "max_audio_bytes_per_session"):
            RecoveryStore(max_audio_bytes_per_session=0)
        with self.assertRaisesRegex(ValueError, "max_total_audio_bytes"):
            RecoveryStore(max_total_audio_bytes=0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            RecoveryStore(
                max_audio_bytes_per_session=10,
                max_total_audio_bytes=8,
            )

    def test_audio_capacity_is_bounded_and_completed_audio_is_released(self):
        store = RecoveryStore(
            max_sessions=2,
            max_audio_bytes_per_session=4,
            max_total_audio_bytes=6,
        )
        try:
            first, _ = store.open("first")
            second, _ = store.open("second")
            store.append_audio(first, AudioChunk(array("h", [1, 2]).tobytes(), 0))
            with self.assertRaisesRegex(RecoveryCapacityError, "session audio capacity"):
                store.append_audio(first, AudioChunk(array("h", [3]).tobytes(), 0.1))
            store.append_audio(second, AudioChunk(array("h", [4]).tobytes(), 0))
            with self.assertRaisesRegex(RecoveryCapacityError, "store audio capacity"):
                store.append_audio(second, AudioChunk(array("h", [5]).tobytes(), 0.1))

            store.release(first, completed=True)
            self.assertEqual(first.audio_bytes, 0)
            self.assertTrue(first.timeline._closed)
            store.append_audio(second, AudioChunk(array("h", [5]).tobytes(), 0.1))
        finally:
            store.close()

    def test_closed_store_rejects_new_sessions_and_audio(self):
        store = RecoveryStore()
        session, _ = store.open("config")
        store.close()
        store.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            store.open("config")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            store.append_audio(session, AudioChunk(array("h", [1]).tobytes(), 0))

    def test_event_byte_limits_reject_single_payload_and_roll_retained_window(self):
        limited = RecoveryStore(max_event_bytes=100, max_event_bytes_per_session=200)
        try:
            session, _ = limited.open("config")
            with self.assertRaisesRegex(RecoveryEventLimitError, "event payload"):
                limited.append_event(
                    session,
                    TranscriptEvent("commit", "seg-000000", 1, 0, 1, "x" * 500),
                )
            self.assertEqual(session.next_event_sequence, 0)
            self.assertEqual(session.events, [])
        finally:
            limited.close()

        rolling = RecoveryStore(
            max_events_per_session=10,
            max_event_bytes=2_048,
            max_event_bytes_per_session=2_048,
        )
        try:
            session, _ = rolling.open("config")
            rolling.append_event(
                session,
                TranscriptEvent("commit", "seg-000000", 1, 0, 1, "first"),
            )
            first_size = session.retained_event_bytes
            rolling.max_event_bytes_per_session = first_size * 2 - 1
            rolling.append_event(
                session,
                TranscriptEvent("commit", "seg-000001", 1, 1, 2, "second"),
            )
            self.assertEqual([event["sequence"] for event in session.events], [1])
            self.assertEqual(len(session.event_sizes), 1)
            self.assertEqual(
                session.retained_event_bytes,
                session.event_sizes[0],
            )
            with self.assertRaisesRegex(ValueError, "retained recovery window"):
                rolling.replay_after(session, -1)
        finally:
            rolling.close()

    def test_prune_expired_removes_only_inactive_sessions_and_reclaims_audio(self):
        store = RecoveryStore(max_audio_bytes_per_session=8, max_total_audio_bytes=8)
        try:
            expired, _ = store.open("expired")
            store.append_audio(expired, AudioChunk(array("h", [1, 2]).tobytes(), 0))
            store.release(expired)
            active, _ = store.open("active")
            baseline = max(expired.updated_at, active.updated_at)

            self.assertEqual(store.prune_expired(10, now=baseline + 9), 0)
            self.assertEqual(store.prune_expired(10, now=baseline + 10), 1)
            self.assertTrue(expired.timeline._closed)
            self.assertFalse(active.timeline._closed)
            store.append_audio(active, AudioChunk(array("h", [3, 4]).tobytes(), 0))
            with self.assertRaisesRegex(PermissionError, "authentication"):
                store.open("expired", expired.session_id, expired.resume_token)
        finally:
            store.close()

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            RecoveryStore().prune_expired(float("nan"))

    def test_prune_expired_honors_exact_deadline_despite_float_rounding(self):
        store = RecoveryStore()
        try:
            session, _ = store.open("config")
            store.release(session)
            session.updated_at = 9.8765431209876

            self.assertLess((session.updated_at + 10) - session.updated_at, 10)
            self.assertEqual(
                store.prune_expired(10, now=session.updated_at + 10),
                1,
            )
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
