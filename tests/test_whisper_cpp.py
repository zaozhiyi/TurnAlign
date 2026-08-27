import json
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turnalign.backends.whisper_cpp import WhisperCppBackend, _windows_creationflags
from turnalign.models import AudioChunk
from turnalign.plugins import Accelerator, AsrConfig


class WhisperCppConfigurationTests(unittest.TestCase):
    def _config(self, root: Path, **overrides) -> AsrConfig:
        executable = root / "whisper-cli.exe"
        model = root / "model.bin"
        executable.touch()
        model.touch()
        values = {
            "executable": str(executable),
            "model_path": str(model),
            "device": "vulkan:1",
            "extra": {"threads": 2, "flash_attention": False},
        }
        values.update(overrides)
        return AsrConfig(**values)

    def test_vulkan_is_declared_and_indexed_configuration_is_bounded(self):
        self.assertIn(Accelerator.VULKAN, WhisperCppBackend.capabilities.accelerators)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = WhisperCppBackend(self._config(root))
            self.assertEqual(backend.device_index, 1)
            self.assertEqual(
                WhisperCppBackend(self._config(root, device="vulkan:0")).device_index,
                0,
            )
            self.assertEqual(backend.threads, 2)
            self.assertFalse(backend.flash_attention)

            for device in ("vulkan:-1", "vulkan:32", "vulkan:x", "mps:1"):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, "device"):
                    WhisperCppBackend(self._config(root, device=device))

    def test_only_typed_whitelisted_options_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = (
                {"threads": 0},
                {"threads": 65},
                {"threads": True},
                {"threads": "2"},
                {"flash_attention": "false"},
                {"arguments": ["--no-gpu"]},
            )
            for options in invalid:
                with (
                    self.subTest(options=options),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    WhisperCppBackend(self._config(root, extra=options))

    def test_vulkan_command_maps_only_validated_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = WhisperCppBackend(self._config(root))
            captured = {}

            def fake_run(command, **options):
                captured["command"] = command
                captured["options"] = options
                output_base = Path(command[command.index("-of") + 1])
                output_base.with_suffix(".json").write_text(
                    json.dumps({"text": "ok", "transcription": []}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stderr="")

            pcm = array("h", [1000] * 1600).tobytes()
            with patch("turnalign.backends.whisper_cpp.subprocess.run", side_effect=fake_run):
                result = list(backend.transcribe([AudioChunk(pcm, 0.0)]))

            command = captured["command"]
            self.assertEqual(command[command.index("-dev") + 1], "1")
            self.assertEqual(command[command.index("-t") + 1], "2")
            self.assertIn("-nfa", command)
            self.assertNotIn("-ng", command)
            self.assertEqual(
                captured["options"]["creationflags"], _windows_creationflags()
            )
            self.assertEqual(result[0].text, "ok")

    def test_windows_runs_below_normal_when_the_platform_supports_it(self):
        if _windows_creationflags():
            self.assertEqual(_windows_creationflags(), 0x00004000)


if __name__ == "__main__":
    unittest.main()
