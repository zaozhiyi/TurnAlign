import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turnalign import cli
from turnalign.cli import replay, validate_events


class CliIntegrationTests(unittest.TestCase):
    def test_private_hint_files_are_loaded_without_becoming_output_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hotwords = root / "hotwords.txt"
            context = root / "context.txt"
            hotwords.write_text("TERM_A\n# ignored\nTERM_B\n", encoding="utf-8")
            context.write_text("topic", encoding="utf-8")
            hints = cli._hints(SimpleNamespace(
                hotword=[], hotwords_file=[hotwords], context=None,
                context_file=context, hotword_boost=None,
            ))
            self.assertEqual(hints.hotwords, ("TERM_A", "TERM_B"))
            self.assertEqual(hints.context, "topic")

    def test_file_transcription_defaults_to_energy_vad(self):
        captured = {}

        def fake_transcribe(args):
            captured["vad_backend"] = args.vad_backend
            return 0

        with patch.object(cli, "transcribe_file", fake_transcribe), patch(
            "sys.argv", ["turnalign", "transcribe", "sample.wav"]
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured["vad_backend"], "energy")

    def test_no_vad_overrides_safe_default(self):
        captured = {}

        def fake_transcribe(args):
            captured["vad_backend"] = args.vad_backend
            return 0

        with patch.object(cli, "transcribe_file", fake_transcribe), patch(
            "sys.argv", ["turnalign", "transcribe", "sample.wav", "--no-vad"]
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured["vad_backend"], "none")

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
