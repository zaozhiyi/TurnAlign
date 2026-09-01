import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "deploy" / "systemd" / "turnalign.service"
NGINX = ROOT / "deploy" / "nginx" / "turnalign.conf.example"
ENVIRONMENT = ROOT / "deploy" / "systemd" / "turnalign.env.example"


def option_number(content: str, option: str) -> float:
    match = re.search(rf"--{re.escape(option)}\s+(\d+(?:\.\d+)?)", content)
    if match is None:
        raise AssertionError(f"missing deployment option: --{option}")
    return float(match.group(1))


class DeploymentArtifactTests(unittest.TestCase):
    def test_service_is_local_authenticated_and_fail_closed(self):
        content = SERVICE.read_text(encoding="utf-8")
        for setting in (
            "--host 127.0.0.1",
            "--preload",
            "--warmup-file /var/lib/turnalign/warmup.wav",
            "--require-immutable-model-revision",
            "--auth-token-env TURNALIGN_AUTH_TOKEN",
            "ExecStartPre=/usr/bin/test -r /var/lib/turnalign/warmup.wav",
        ):
            self.assertIn(setting, content)
        self.assertNotIn("--allow-remote", content)

    def test_service_shutdown_and_recovery_limits_are_consistent(self):
        content = SERVICE.read_text(encoding="utf-8")
        grace = option_number(content, "shutdown-grace-timeout")
        stop = float(re.search(r"TimeoutStopSec=(\d+(?:\.\d+)?)s", content).group(1))
        self.assertGreater(stop, grace)
        per_session = option_number(content, "max-recovery-audio-mib")
        total = option_number(content, "max-recovery-total-mib")
        self.assertLessEqual(per_session, total)
        raw_audio_bytes = option_number(content, "max-session-seconds") * 32_000
        self.assertLessEqual(raw_audio_bytes, per_session * 1024 * 1024)

    def test_service_uses_a_restricted_cpu_security_profile(self):
        content = SERVICE.read_text(encoding="utf-8")
        for setting in (
            "User=turnalign",
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "PrivateDevices=true",
            "ProtectSystem=strict",
            "ProtectHome=true",
            "CapabilityBoundingSet=",
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        ):
            self.assertIn(setting, content)

    def test_nginx_preserves_websocket_upgrade_and_disables_retry(self):
        content = NGINX.read_text(encoding="utf-8")
        for setting in (
            "proxy_set_header Upgrade $http_upgrade;",
            "proxy_set_header Connection $turnalign_connection_upgrade;",
            "proxy_buffering off;",
            "proxy_request_buffering off;",
            "proxy_next_upstream off;",
            "limit_req zone=turnalign_handshake",
            "limit_conn turnalign_connections",
        ):
            self.assertIn(setting, content)

    def test_nginx_timeouts_exceed_application_idle_timeout(self):
        service = SERVICE.read_text(encoding="utf-8")
        nginx = NGINX.read_text(encoding="utf-8")
        idle = option_number(service, "client-idle-timeout")
        read_timeout = float(re.search(r"proxy_read_timeout\s+(\d+(?:\.\d+)?)s", nginx).group(1))
        send_timeout = float(re.search(r"proxy_send_timeout\s+(\d+(?:\.\d+)?)s", nginx).group(1))
        self.assertGreater(read_timeout, idle)
        self.assertGreater(send_timeout, idle)
        self.assertIn("proxy_pass http://turnalign_backend/healthz;", nginx)
        self.assertIn("proxy_pass http://turnalign_backend/readyz;", nginx)
        self.assertRegex(nginx, r"location = /metrics\s*\{[^}]*return 404;")
        self.assertNotIn("proxy_pass http://turnalign_backend/metrics", nginx)

    def test_secret_example_is_a_placeholder_and_deploy_is_packaged(self):
        environment = ENVIRONMENT.read_text(encoding="utf-8")
        self.assertIn("TURNALIGN_AUTH_TOKEN=replace-", environment)
        self.assertNotRegex(environment, r"sk-[A-Za-z0-9_-]{16,}")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"/deploy"', pyproject)


if __name__ == "__main__":
    unittest.main()
