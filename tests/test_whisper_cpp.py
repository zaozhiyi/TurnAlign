import json
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from turnalign.backends.whisper_cpp import WhisperCppBackend, _windows_creationflags
from turnalign.hints import AsrHints
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
                {"allow_prompt_argv": "true"},
                {"arguments": ["--no-gpu"]},
            )
            for options in invalid:
                with (
                    self.subTest(options=options),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    WhisperCppBackend(self._config(root, extra=options))

    def test_private_prompt_requires_explicit_process_argument_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hinted = self._config(
                root,
                hints=AsrHints(("PRIVATE_TERM",), context="topic"),
            )
            with self.assertRaisesRegex(ValueError, "process arguments"):
                WhisperCppBackend(hinted)

            allowed = self._config(
                root,
                hints=hinted.hints,
                extra={"allow_prompt_argv": True},
            )
            backend = WhisperCppBackend(allowed)
            self.assertEqual(backend.initial_prompt, "topic\nPRIVATE_TERM")
            self.assertTrue(backend.allow_prompt_argv)

    def test_vulkan_command_maps_only_validated_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = WhisperCppBackend(self._config(root))
            captured = {}

            def fake_popen(command, **options):
                captured["command"] = command
                captured["options"] = options
                output_base = Path(command[command.index("-of") + 1])
                output_base.with_suffix(".json").write_text(
                    json.dumps({"text": "ok", "transcription": []}),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    returncode=0,
                    wait=lambda: 0,
                )

            pcm = array("h", [1000] * 1600).tobytes()
            with patch("turnalign.backends.whisper_cpp.subprocess.Popen", side_effect=fake_popen):
                result = list(backend.transcribe([AudioChunk(pcm, 0.0)]))

            command = captured["command"]
            self.assertEqual(command[command.index("-dev") + 1], "1")
            self.assertEqual(command[command.index("-t") + 1], "2")
            self.assertIn("-nfa", command)
            self.assertNotIn("-ng", command)
            self.assertEqual(
                captured["options"]["creationflags"], _windows_creationflags()
            )
            self.assertEqual(captured["options"]["stdout"], subprocess.DEVNULL)
            self.assertEqual(result[0].text, "ok")

    def test_failure_diagnostics_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCppBackend(self._config(Path(directory)))

            def fake_popen(_command, **options):
                options["stderr"].write(bytes(70_000) + b"diagnostic-tail")
                return SimpleNamespace(returncode=1, wait=lambda: 1)

            with (
                patch(
                    "turnalign.backends.whisper_cpp.subprocess.Popen",
                    side_effect=fake_popen,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "earlier output truncated.*diagnostic-tail",
                ) as error,
            ):
                list(backend.transcribe([AudioChunk(bytes(3_200), 0.0)]))
            self.assertLess(len(str(error.exception)), 66_000)

    def test_result_json_is_bounded_strict_and_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCppBackend(self._config(Path(directory)))
            pcm = array("h", [1000] * 1600).tobytes()

            def transcribe_output(content: bytes, *, limit: int = 1024):
                def fake_popen(command, **_options):
                    output_base = Path(command[command.index("-of") + 1])
                    output_base.with_suffix(".json").write_bytes(content)
                    return SimpleNamespace(returncode=0, wait=lambda: 0)

                with (
                    patch(
                        "turnalign.backends.whisper_cpp.subprocess.Popen",
                        side_effect=fake_popen,
                    ),
                    patch(
                        "turnalign.backends.whisper_cpp._MAX_OUTPUT_JSON_BYTES",
                        limit,
                    ),
                ):
                    return list(backend.transcribe([AudioChunk(pcm, 0.0)]))

            for content, expected in (
                (b'{"text":"first","text":"second"}', "duplicate JSON key"),
                (b'{"text":"ok","value":NaN}', "non-standard JSON number"),
                (b'{"text":{"nested":true}}', "text must be a string"),
                (
                    b'{"segments":[{"text":"bad","start":true,"end":1}]}',
                    "start timestamp must be non-negative",
                ),
                (b"x" * 33, "JSON output exceeds 32 bytes"),
            ):
                limit = 32 if content == b"x" * 33 else 1024
                with (
                    self.subTest(expected=expected),
                    self.assertRaisesRegex((TypeError, ValueError), expected),
                ):
                    transcribe_output(content, limit=limit)

            result = transcribe_output(json.dumps({
                "transcription": [{
                    "text": "typed",
                    "offsets": {"from": 250, "to": 750},
                }],
            }).encode())
            self.assertEqual(result[0].text, "typed")
            self.assertEqual((result[0].start, result[0].end), (0.25, 0.75))

    def test_cancel_terminates_the_active_process(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCppBackend(self._config(Path(directory)))
            process = Mock()
            process.poll.return_value = None
            backend._process = process

            backend.cancel()

            process.terminate.assert_called_once_with()

    def test_cancel_escalation_kills_the_same_active_process(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCppBackend(self._config(Path(directory)))
            process = Mock()
            process.poll.return_value = None
            backend._process = process

            backend._kill_if_active(process)

            process.kill.assert_called_once_with()

    def test_cancel_during_audio_collection_prevents_process_start(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCppBackend(self._config(Path(directory)))
            pcm = array("h", [1000] * 1600).tobytes()

            def collect_then_cancel(_chunks):
                backend.cancel()
                return pcm, 16_000, 1, 0.0

            with (
                patch(
                    "turnalign.backends.whisper_cpp.collect_pcm",
                    side_effect=collect_then_cancel,
                ),
                patch("turnalign.backends.whisper_cpp.subprocess.Popen") as popen,
            ):
                self.assertEqual(list(backend.transcribe([])), [])
            popen.assert_not_called()

    def test_windows_runs_below_normal_when_the_platform_supports_it(self):
        if _windows_creationflags():
            self.assertEqual(_windows_creationflags(), 0x00004000)


if __name__ == "__main__":
    unittest.main()
