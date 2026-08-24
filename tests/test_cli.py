import json
import tempfile
import unittest
from pathlib import Path

from turnalign.cli import replay, validate_events


class CliIntegrationTests(unittest.TestCase):
    def test_replay_creates_parent_and_valid_end_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "nested" / "events.jsonl"
            source.write_text(
                json.dumps({"start": 0, "end": 1.25, "text": "你好"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(replay(source, output), 0)
            events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["kind"] for event in events], ["commit", "end"])
            self.assertEqual(events[-1]["metadata"]["segments"], 1)
            self.assertEqual(validate_events(output), 0)

    def test_validate_requires_end(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                json.dumps({
                    "kind": "commit", "segment_id": "s", "revision": 1,
                    "start": 0, "end": 1, "text": "x"
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing end"):
                validate_events(source)


if __name__ == "__main__":
    unittest.main()
