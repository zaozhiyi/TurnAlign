import unittest

from turnalign.exporters import final_segments, render_srt, render_text
from turnalign.models import TranscriptEvent


class ExporterTests(unittest.TestCase):
    def test_replace_and_speaker_merge_produce_one_final_segment(self):
        events = [
            TranscriptEvent("partial", "s1", 1, 1, 2, "draft"),
            TranscriptEvent("commit", "s1", 2, 1, 2, "draft", speaker="temp-1"),
            TranscriptEvent(
                "speaker_merge",
                "merge:temp-1",
                1,
                1,
                2,
                metadata={"from_speaker": "temp-1", "to_speaker": "speaker-1"},
            ),
            TranscriptEvent("replace", "s1", 3, 1, 2, "final", speaker="speaker-1"),
            TranscriptEvent("end", "session", 1, 2, 2),
        ]
        segments = final_segments(events)
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].text, segments[0].speaker), ("final", "speaker-1"))
        self.assertIn("00:00:01,000 --> 00:00:02,000", render_srt(events))
        self.assertEqual(render_text(events), "speaker-1: final\n")


if __name__ == "__main__":
    unittest.main()
