import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turnalign import cli
from turnalign.cli import replay, validate_events
from turnalign.devices import Device
from turnalign.plugins import Accelerator
from turnalign.profiles import select_execution_profile


class CliIntegrationTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_whisper_cpp_vulkan_uses_cpu_profile_without_hiding_asr_device(self):
        devices = [Device(
            accelerator=Accelerator.CPU,
            available=True,
            runtime="native",
            device="cpu",
            vendor="test",
            name="test CPU",
            dtype="float32",
        )]
        requested = cli._profile_requested_device("whisper-cpp", "vulkan:1")
        self.assertEqual(requested, "cpu")
        self.assertEqual(cli._profile_requested_device("whisper-cpp", "vulkan:0"), "cpu")
        self.assertEqual(
            cli._profile_requested_device("whisper-cpp", " Vulkan:1 "), "cpu"
        )
        self.assertEqual(cli._profile_requested_device("whisper-cpp", "vulkan"), "cpu")
        self.assertEqual(
            cli._profile_name("whisper-cpp", "auto", "vulkan:1"),
            "cpu-low-memory",
        )
        self.assertEqual(cli._resolved_device("vulkan:1"), "vulkan:1")
        self.assertEqual(
            select_execution_profile(
                "auto", requested_device=requested, devices=devices, system="Windows"
            ).name,
            "cpu-low-memory",
        )
        with self.assertRaisesRegex(RuntimeError, "does not support requested accelerator"):
            select_execution_profile(
                "auto",
                requested_device=cli._profile_requested_device(
                    "transformers-whisper", "vulkan:1"
                ),
                devices=devices,
                system="Windows",
            )

    def test_whisper_cpp_vulkan_environment_reaches_backend_without_generic_probe(self):
        captured = {}

        def fake_profile(name, *, requested_device):
            captured["profile_name"] = name
            captured["profile_device"] = requested_device
            return SimpleNamespace(asr_device="cpu")

        def fake_create(name, config):
            captured["backend"] = name
            captured["asr_device"] = config.device
            raise RuntimeError("backend-created")

        args = SimpleNamespace(
            execution_profile="auto",
            backend="whisper-cpp",
            device="auto",
            model=None,
            language="zh",
            compute_type=None,
            executable="whisper-cli",
            model_path="model.bin",
            backend_option=[],
        )
        with patch.dict(os.environ, {"TURNALIGN_DEVICE": "vulkan:1"}), patch.object(
            cli, "select_execution_profile", side_effect=fake_profile
        ), patch.object(cli, "create_asr", side_effect=fake_create), self.assertRaisesRegex(
            RuntimeError, "backend-created"
        ):
            cli.transcribe_file(args)

        self.assertEqual(captured["profile_name"], "cpu-low-memory")
        self.assertEqual(captured["profile_device"], "cpu")
        self.assertEqual(captured["backend"], "whisper-cpp")
        self.assertEqual(captured["asr_device"], "vulkan:1")

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
