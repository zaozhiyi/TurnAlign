import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from turnalign import deployment_rehearsal as rehearsal
from turnalign.deployment_rehearsal import (
    PendingDeploymentTransaction,
    ReadinessEvidence,
    RehearsalProbeConfig,
    ServiceRestartEvidence,
    run_deployment_activation,
    run_deployment_recovery,
    run_deployment_rehearsal,
)
from turnalign.production_gate import write_json_report


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

    def setUp(self):
        transaction_write = patch.object(rehearsal, "_write_pending_transaction")
        pending_rejection = patch.object(rehearsal, "_reject_pending_transaction")
        transaction_write.start()
        pending_rejection.start()
        self.addCleanup(transaction_write.stop)
        self.addCleanup(pending_rejection.stop)

    @staticmethod
    def _restart(*_args, **_kwargs) -> ServiceRestartEvidence:
        return ServiceRestartEvidence(0, 0, 0.1, None)

    @staticmethod
    def _ready(uri: str, **_kwargs) -> ReadinessEvidence:
        return ReadinessEvidence(uri, 200, True, True, 1, 0.1, None)

    def _transaction(
        self,
        releases: Path,
        current: Path,
        *,
        operation: str = "activation",
    ) -> PendingDeploymentTransaction:
        return PendingDeploymentTransaction(
            schema_version=2,
            transaction_id="d" * 64,
            operation=operation,
            previous_commit=self.previous,
            candidate_commit=self.candidate,
            boot_id=self.boot_id,
            release_root=str(releases),
            current_link=str(current),
            service="turnalign.service",
            systemctl="/usr/bin/systemctl",
            ready_uri="http://127.0.0.1:8765/readyz",
            websocket_uri="wss://asr.example.com/ws",
            created_at="2026-09-01T08:00:00.000Z",
        )

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
        marker_active_commits = []

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
            ), patch.object(
                rehearsal,
                "_write_pending_transaction",
                side_effect=lambda _transaction: marker_active_commits.append(
                    rehearsal._active_commit(releases, current)
                ),
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
            self.assertEqual(marker_active_commits, [self.previous])

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

    async def test_interrupted_activation_recovery_restores_and_probes_previous(self):
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
            transaction = self._transaction(releases, current)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "_read_pending_transaction",
                return_value=transaction,
            ), patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_recovery(
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertTrue(report.passed)
            self.assertEqual(report.transaction_id, transaction.transaction_id)
            self.assertEqual(report.initial_active_commit, self.candidate)
            self.assertEqual(report.final_active_commit, self.previous)
            self.assertEqual(os.readlink(current), str(releases / self.previous))
            self.assertEqual(observed_commits, [self.previous])

    async def test_interrupted_rehearsal_recovery_restores_and_probes_candidate(self):
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
            transaction = self._transaction(
                releases,
                current,
                operation="rehearsal",
            )
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "_read_pending_transaction",
                return_value=transaction,
            ), patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_recovery(
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertTrue(report.passed)
            self.assertEqual(report.operation, "rehearsal")
            self.assertEqual(report.recovery_commit, self.candidate)
            self.assertEqual(report.initial_active_commit, self.previous)
            self.assertEqual(report.final_active_commit, self.candidate)
            self.assertEqual(os.readlink(current), str(releases / self.candidate))
            self.assertEqual(observed_commits, [self.candidate])

    async def test_interrupted_recovery_is_idempotent_when_previous_is_active(self):
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
            transaction = self._transaction(releases, current)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "_read_pending_transaction",
                return_value=transaction,
            ), patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ):
                report = await run_deployment_recovery(
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertTrue(report.passed)
            self.assertEqual(report.initial_active_commit, self.previous)
            self.assertEqual(report.recovery.from_commit, self.previous)
            self.assertEqual(os.readlink(current), str(releases / self.previous))
            self.assertEqual(observed_commits, [self.previous])

    async def test_recovery_refuses_an_unrelated_active_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            unrelated = "e" * 40
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            (releases / unrelated).mkdir()
            current.symlink_to(releases / unrelated)
            transaction = self._transaction(releases, current)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "_read_pending_transaction",
                return_value=transaction,
            ), self.assertRaisesRegex(RuntimeError, "does not match the active"):
                await run_deployment_recovery(
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                )

            self.assertEqual(os.readlink(current), str(releases / unrelated))

    async def test_rehearsal_persists_transaction_before_switching(self):
        observed_transactions = []

        async def websocket_gate(_uri, **options):
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            (releases / self.previous).mkdir(parents=True)
            (releases / self.candidate).mkdir()
            current.symlink_to(releases / self.candidate)
            patches = self._patches()

            def capture(transaction):
                observed_transactions.append((
                    transaction.operation,
                    rehearsal._active_commit(releases, current),
                ))

            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ), patch.object(
                rehearsal,
                "_write_pending_transaction",
                side_effect=capture,
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
            self.assertEqual(observed_transactions, [("rehearsal", self.candidate)])
            self.assertEqual(report.transaction_path, str(rehearsal._DEPLOYMENT_TRANSACTION_PATH))

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

    async def test_rehearsal_refuses_a_pending_activation_before_switching(self):
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
            ), patch.object(
                rehearsal,
                "_reject_pending_transaction",
                side_effect=RuntimeError("pending deployment must be recovered"),
            ), self.assertRaisesRegex(RuntimeError, "must be recovered"):
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
            self.assertEqual(os.readlink(current), str(releases / self.candidate))


@unittest.skipUnless(os.name == "posix", "deployment release validation is POSIX-only")
class DeploymentReleaseValidationTests(unittest.TestCase):
    commit = "b" * 40

    def _release(self, root: Path) -> tuple[Path, Path]:
        releases = root / "releases"
        release = releases / self.commit
        package = (
            release
            / "venv"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "turnalign"
        )
        package.mkdir(parents=True)
        (package / "_source_commit.txt").write_text(
            f"{self.commit}\n",
            encoding="ascii",
        )
        dependency = package.parent / "dependency.py"
        dependency.write_text("VALUE = 1\n", encoding="ascii")
        binary = release / "venv" / "bin" / "python"
        binary.parent.mkdir()
        binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        return releases, dependency

    def test_release_validation_covers_the_complete_immutable_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            releases, _dependency = self._release(Path(directory))
            with patch.object(
                rehearsal,
                "_required_release_owner",
                return_value=os.getuid(),
            ):
                self.assertEqual(
                    rehearsal._validate_release_directory(releases, self.commit),
                    releases / self.commit,
                )

    def test_release_validation_rejects_a_mutable_dependency_file(self):
        with tempfile.TemporaryDirectory() as directory:
            releases, dependency = self._release(Path(directory))
            dependency.chmod(0o666)
            with patch.object(
                rehearsal,
                "_required_release_owner",
                return_value=os.getuid(),
            ), self.assertRaisesRegex(ValueError, "unsafe or mutable entry"):
                rehearsal._validate_release_directory(releases, self.commit)

    def test_release_validation_rejects_a_link_to_a_mutable_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases, dependency = self._release(root)
            target = root / "mutable-target.py"
            target.write_text("VALUE = 2\n", encoding="ascii")
            target.chmod(0o666)
            dependency.unlink()
            dependency.symlink_to(target)

            def immutable_mode(metadata):
                return not stat.S_IMODE(metadata.st_mode) & 0o022

            with patch.object(
                rehearsal,
                "_required_release_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal,
                "_release_entry_is_immutable",
                side_effect=immutable_mode,
            ), self.assertRaisesRegex(ValueError, "symbolic link target is unsafe"):
                rehearsal._validate_release_directory(releases, self.commit)

    def test_release_validation_accepts_a_secure_venv_python_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases, _dependency = self._release(root)
            binary = releases / self.commit / "venv" / "bin" / "python"
            target = binary.with_name("python3")
            binary.replace(target)
            binary.symlink_to(target.name)

            def immutable_mode(metadata):
                return not stat.S_IMODE(metadata.st_mode) & 0o022

            with patch.object(
                rehearsal,
                "_required_release_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal,
                "_release_entry_is_immutable",
                side_effect=immutable_mode,
            ):
                self.assertEqual(
                    rehearsal._validate_release_directory(releases, self.commit),
                    releases / self.commit,
                )

    def test_release_validation_rejects_an_external_directory_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases, dependency = self._release(root)
            external = root / "external-package"
            external.mkdir()
            (external / "mutable.py").write_text("VALUE = 3\n", encoding="ascii")
            dependency.unlink()
            dependency.symlink_to(external, target_is_directory=True)

            def immutable_mode(metadata):
                return not stat.S_IMODE(metadata.st_mode) & 0o022

            with patch.object(
                rehearsal,
                "_required_release_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal,
                "_release_entry_is_immutable",
                side_effect=immutable_mode,
            ), self.assertRaisesRegex(ValueError, "external directory"):
                rehearsal._validate_release_directory(releases, self.commit)


@unittest.skipUnless(os.name == "posix", "deployment transaction is POSIX-only")
class DeploymentTransactionFileTests(unittest.TestCase):
    def _transaction(self, root: Path) -> PendingDeploymentTransaction:
        return PendingDeploymentTransaction(
            schema_version=2,
            transaction_id="d" * 64,
            operation="activation",
            previous_commit="a" * 40,
            candidate_commit="b" * 40,
            boot_id="12345678-1234-4234-8234-123456789abc",
            release_root=str(root / "releases"),
            current_link=str(root / "current"),
            service="turnalign.service",
            systemctl="/usr/bin/systemctl",
            ready_uri="http://127.0.0.1:8765/readyz",
            websocket_uri="wss://asr.example.com/ws",
            created_at="2026-09-01T08:00:00.000Z",
        )

    def test_root_only_transaction_round_trip_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deployment-state" / "pending-activation.json"
            transaction = self._transaction(root)
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ):
                rehearsal._write_pending_transaction(transaction)
                self.assertEqual(rehearsal._read_pending_transaction(), transaction)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                with self.assertRaisesRegex(RuntimeError, "must be recovered"):
                    rehearsal._write_pending_transaction(transaction)
                with self.assertRaisesRegex(RuntimeError, "must be recovered"):
                    rehearsal._reject_pending_transaction()
                rehearsal._remove_pending_transaction(transaction.transaction_id)
                with self.assertRaisesRegex(RuntimeError, "no recoverable"):
                    rehearsal._read_pending_transaction()

    def test_legacy_activation_transaction_remains_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deployment-state" / "pending-activation.json"
            payload = self._transaction(root).to_dict()
            payload["schema_version"] = 1
            payload.pop("operation")
            path.parent.mkdir()
            path.parent.chmod(0o700)
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            path.chmod(0o600)
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ):
                transaction = rehearsal._read_pending_transaction()
            self.assertEqual(transaction.schema_version, 2)
            self.assertEqual(transaction.operation, "activation")

    def test_transaction_rejects_mutable_or_non_strict_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deployment-state" / "pending-activation.json"
            transaction = self._transaction(root)
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ):
                rehearsal._write_pending_transaction(transaction)
                path.chmod(0o640)
                with self.assertRaisesRegex(RuntimeError, "root-only regular file"):
                    rehearsal._read_pending_transaction()
                path.chmod(0o600)
                payload = transaction.to_dict()
                raw = json.dumps(payload, separators=(",", ":"))
                path.write_text(
                    raw[:-1] + ',"transaction_id":"' + "e" * 64 + '"}\n',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "not strict JSON"):
                    rehearsal._read_pending_transaction()
                payload = transaction.to_dict()
                payload["operation"] = "unknown"
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "invalid identities"):
                    rehearsal._read_pending_transaction()

    def test_transaction_write_handles_short_writes_and_cleans_up_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deployment-state" / "pending-activation.json"
            transaction = self._transaction(root)
            original_write = os.write

            def short_write(descriptor, data):
                return original_write(descriptor, bytes(data[: max(1, len(data) // 2)]))

            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal.os,
                "write",
                side_effect=short_write,
            ):
                rehearsal._write_pending_transaction(transaction)
                self.assertEqual(rehearsal._read_pending_transaction(), transaction)
            path.unlink()
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal.os,
                "write",
                side_effect=OSError("disk failure"),
            ), self.assertRaisesRegex(OSError, "disk failure"):
                rehearsal._write_pending_transaction(transaction)
            self.assertFalse(path.exists())

            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal.os,
                "replace",
                side_effect=OSError("rename failure"),
            ), self.assertRaisesRegex(OSError, "rename failure"):
                rehearsal._write_pending_transaction(transaction)
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_transaction_directory_must_not_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attacker = root / "attacker"
            attacker.mkdir()
            parent = root / "deployment-state"
            parent.symlink_to(attacker)
            path = parent / "pending-activation.json"
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ), self.assertRaisesRegex(RuntimeError, "root-only regular directory"):
                rehearsal._write_pending_transaction(self._transaction(root))

    def test_finalization_requires_matching_identity_and_active_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deployment-state" / "pending-activation.json"
            transaction = self._transaction(root)
            releases = Path(transaction.release_root)
            releases.mkdir()
            current = Path(transaction.current_link)
            current.symlink_to(releases / transaction.candidate_commit)
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal.platform,
                "system",
                return_value="Linux",
            ), patch.object(
                rehearsal.os,
                "geteuid",
                return_value=0,
            ), patch.object(
                rehearsal,
                "_acquire_deployment_lock",
                side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
            ):
                rehearsal._write_pending_transaction(transaction)
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    rehearsal.finalize_deployment_transaction(
                        "e" * 64,
                        transaction.candidate_commit,
                    )
                self.assertTrue(path.exists())
                rehearsal.finalize_deployment_transaction(
                    transaction.transaction_id,
                    transaction.candidate_commit,
                )
                self.assertFalse(path.exists())

                payload = transaction.to_dict()
                payload["operation"] = "rehearsal"
                rehearsal_transaction = rehearsal._transaction_from_payload(payload)
                rehearsal._write_pending_transaction(rehearsal_transaction)
                with self.assertRaisesRegex(RuntimeError, "unrelated release"):
                    rehearsal.finalize_deployment_transaction(
                        rehearsal_transaction.transaction_id,
                        rehearsal_transaction.previous_commit,
                    )
                rehearsal.finalize_deployment_transaction(
                    rehearsal_transaction.transaction_id,
                    rehearsal_transaction.candidate_commit,
                )
                self.assertFalse(path.exists())

    def test_activation_report_and_marker_finalize_end_to_end(self):
        async def websocket_gate(_uri, **options):
            return FakeWebSocketReport(options["source_commit"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            previous = "a" * 40
            candidate = "b" * 40
            (releases / previous).mkdir(parents=True)
            (releases / candidate).mkdir()
            current.symlink_to(releases / previous)
            path = root / "deployment-state" / "pending-activation.json"
            report_path = root / "activation-report.json"
            runtime_prefix = f"/opt/turnalign/releases/{candidate}/venv"
            with patch.object(
                rehearsal,
                "_DEPLOYMENT_TRANSACTION_PATH",
                path,
            ), patch.object(
                rehearsal,
                "_required_transaction_owner",
                return_value=os.getuid(),
            ), patch.object(
                rehearsal.platform,
                "system",
                return_value="Linux",
            ), patch.object(
                rehearsal.os,
                "geteuid",
                return_value=0,
            ), patch.object(
                rehearsal,
                "_installed_runtime_identity",
                return_value={
                    "python_executable": f"{runtime_prefix}/bin/python",
                    "python_prefix": runtime_prefix,
                    "turnalign_source_commit": candidate,
                    "turnalign_version": "0.1.0",
                },
            ), patch.object(
                rehearsal,
                "_validate_release_directory",
            ), patch.object(
                rehearsal,
                "_read_linux_boot_id",
                return_value="12345678-1234-4234-8234-123456789abc",
            ), patch.object(
                rehearsal,
                "_restart_service",
                return_value=ServiceRestartEvidence(0, 0, 0.1, None),
            ), patch.object(
                rehearsal,
                "_wait_readiness",
                side_effect=lambda uri, **_options: ReadinessEvidence(
                    uri,
                    200,
                    True,
                    True,
                    1,
                    0.1,
                    None,
                ),
            ), patch.object(
                rehearsal,
                "run_websocket_gate",
                side_effect=websocket_gate,
            ), patch.object(
                rehearsal,
                "_acquire_deployment_lock",
                side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
            ):
                report = asyncio.run(run_deployment_activation(
                    previous,
                    candidate,
                    "wss://asr.example.com/ws",
                    release_root=releases,
                    current_link=current,
                    probe=RehearsalProbeConfig(
                        backend="fake",
                        model="model-a",
                    ),
                ))
                self.assertTrue(report.passed)
                self.assertTrue(path.exists())
                write_json_report(report_path, report.to_dict())
                rehearsal.finalize_deployment_transaction(
                    report.transaction_id,
                    candidate,
                )
            self.assertTrue(report_path.exists())
            self.assertFalse(path.exists())
            self.assertEqual(os.readlink(current), str(releases / candidate))
