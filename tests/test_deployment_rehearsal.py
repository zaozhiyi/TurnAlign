import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from turnalign import deployment_rehearsal as rehearsal
from turnalign.deployment_rehearsal import (
    ReadinessEvidence,
    RehearsalProbeConfig,
    ServiceRestartEvidence,
    run_deployment_activation,
    run_deployment_rehearsal,
)


class FakeWebSocketReport:
    def __init__(self, source_commit: str, *, passed: bool = True):
        self.passed = passed
        self.source_commit = source_commit

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "passed" if self.passed else "failed",
            "source_commit": self.source_commit,
            "failures": [] if self.passed else ["probe failed"],
        }


@unittest.skipUnless(os.name == "posix", "deployment rehearsal is Linux/POSIX-only")
class DeploymentRehearsalTests(unittest.IsolatedAsyncioTestCase):
    previous = "a" * 40
    candidate = "b" * 40
    boot_id = "12345678-1234-4234-8234-123456789abc"

    @staticmethod
    def _restart(*_args, **_kwargs) -> ServiceRestartEvidence:
        return ServiceRestartEvidence(0, 0, 0.1, None)

    @staticmethod
    def _ready(uri: str, **_kwargs) -> ReadinessEvidence:
        return ReadinessEvidence(uri, 200, True, True, 1, 0.1, None)

    def _patches(self):
        runtime_prefix = f"/opt/turnalign/releases/{self.candidate}/venv"
        return (
            patch.object(rehearsal.platform, "system", return_value="Linux"),
            patch.object(rehearsal.os, "geteuid", return_value=0),
            patch.object(
                rehearsal,
                "_installed_runtime_identity",
                return_value={
                    "python_executable": f"{runtime_prefix}/bin/python",
                    "python_prefix": runtime_prefix,
                    "turnalign_source_commit": self.candidate,
                    "turnalign_version": "0.1.0",
                },
            ),
            patch.object(rehearsal, "_validate_release_directory"),
            patch.object(rehearsal, "_read_linux_boot_id", return_value=self.boot_id),
            patch.object(rehearsal, "_restart_service", side_effect=self._restart),
            patch.object(rehearsal, "_wait_readiness", side_effect=self._ready),
            patch.object(
                rehearsal,
                "_acquire_deployment_lock",
                side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
            ),
        )

    def test_deployment_lock_is_exclusive_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "deployment.lock"
            with patch.object(rehearsal, "_DEPLOYMENT_LOCK_PATH", lock_path):
                first = rehearsal._acquire_deployment_lock()
                try:
                    with self.assertRaisesRegex(RuntimeError, "operation is active"):
                        rehearsal._acquire_deployment_lock()
                finally:
                    os.close(first)
                second = rehearsal._acquire_deployment_lock()
                os.close(second)

    async def test_activation_switches_from_previous_to_candidate(self):
        observed_commits = []

        async def websocket_gate(_uri, **options):
            observed_commits.append(options["source_commit"])
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.previous)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_activation(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertTrue(report.passed)
            self.assertEqual(report.activation.from_commit, self.previous)
            self.assertIsNone(report.rollback)
            self.assertEqual(report.final_active_commit, self.candidate)
            self.assertEqual(os.readlink(current), str(releases / self.candidate))
            self.assertEqual(observed_commits, [self.candidate])

    async def test_failed_activation_is_probed_after_verified_rollback(self):
        calls = 0

        async def websocket_gate(_uri, **options):
            nonlocal calls
            calls += 1
            return FakeWebSocketReport(
                options["source_commit"],
                passed=calls != 1,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.previous)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_activation(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertFalse(report.passed)
            self.assertIsNotNone(report.rollback)
            self.assertTrue(report.rollback.passed)
            self.assertEqual(report.final_active_commit, self.previous)
            self.assertEqual(os.readlink(current), str(releases / self.previous))
            self.assertEqual(calls, 2)

    async def test_cancelled_activation_still_restores_previous_release(self):
        calls = 0

        async def websocket_gate(_uri, **options):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.previous)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ), self.assertRaises(asyncio.CancelledError):
                await run_deployment_activation(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertEqual(calls, 2)
            self.assertEqual(os.readlink(current), str(releases / self.previous))

    async def test_boot_change_after_candidate_probe_triggers_verified_rollback(self):
        observed_commits = []

        async def websocket_gate(_uri, **options):
            observed_commits.append(options["source_commit"])
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.previous)
            changed_boot = "87654321-4321-4321-8321-cba987654321"
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patch.object(
                rehearsal,
                "_read_linux_boot_id",
                side_effect=[self.boot_id, changed_boot, changed_boot],
            ), patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_activation(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertFalse(report.passed)
            self.assertIsNotNone(report.rollback)
            self.assertTrue(report.rollback.passed)
            self.assertEqual(report.final_active_commit, self.previous)
            self.assertEqual(os.readlink(current), str(releases / self.previous))
            self.assertEqual(observed_commits, [self.candidate, self.previous])

    async def test_atomically_rolls_back_and_restores_candidate(self):
        observed_commits = []

        async def websocket_gate(_uri, **options):
            observed_commits.append(options["source_commit"])
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.candidate)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_rehearsal(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertTrue(report.passed)
            self.assertEqual(report.rollback.from_commit, self.candidate)
            self.assertEqual(report.restore.from_commit, self.previous)
            self.assertEqual(report.final_active_commit, self.candidate)
            self.assertEqual(os.readlink(current), str(releases / self.candidate))
            self.assertEqual(observed_commits, [self.previous, self.candidate])

    async def test_failed_rollback_probe_still_restores_candidate(self):
        calls = 0

        async def websocket_gate(_uri, **options):
            nonlocal calls
            calls += 1
            return FakeWebSocketReport(
                options["source_commit"],
                passed=calls != 1,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.candidate)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_rehearsal(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertFalse(report.passed)
            self.assertTrue(report.restore.passed)
            self.assertEqual(os.readlink(current), str(releases / self.candidate))

    async def test_cancellation_still_attempts_candidate_restore(self):
        calls = 0

        async def websocket_gate(_uri, **options):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.candidate)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ), self.assertRaises(asyncio.CancelledError):
                await run_deployment_rehearsal(
                    self.previous,
                    self.candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertEqual(calls, 2)
            self.assertEqual(os.readlink(current), str(releases / self.candidate))

    async def test_preflight_rejects_non_public_probe_without_switching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.candidate)
            with patch.object(rehearsal.platform, "system", return_value="Linux"), patch.object(
                rehearsal.os,
                "geteuid",
                return_value=0,
            ), patch.object(
                rehearsal,
                "_acquire_deployment_lock",
                side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
            ), self.assertRaisesRegex(ValueError, "public wss"):
                await run_deployment_rehearsal(
                    self.previous,
                    self.candidate,
                    "ws://127.0.0.1:8765/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )
            self.assertEqual(os.readlink(current), str(releases / self.candidate))
