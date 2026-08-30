import unittest
from array import array

from turnalign.models import AudioChunk, TranscriptEvent
from turnalign.recovery import RecoveryStore


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

            recovered, resumed = store.open("config", session.session_id)
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
                store.open("config", session.session_id)
        finally:
            store.close()

    def test_resume_rejects_configuration_change(self):
        store = RecoveryStore()
        try:
            session, _ = store.open("config-a")
            store.release(session)
            with self.assertRaisesRegex(ValueError, "configuration"):
                store.open("config-b", session.session_id)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
