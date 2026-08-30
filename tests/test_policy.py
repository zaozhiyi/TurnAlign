import unittest

from turnalign.policy import ServerPolicy


class ServerPolicyTests(unittest.TestCase):
    def test_default_policy_allows_only_server_selected_backend_and_model(self):
        policy = ServerPolicy.defaults("funasr", "approved-model")
        self.assertEqual(
            policy.validate_start(
                {"type": "start", "backend": "funasr", "model": "approved-model"},
                default_backend="funasr",
                default_model="approved-model",
            ),
            ("funasr", "approved-model"),
        )
        with self.assertRaisesRegex(ValueError, "backend"):
            policy.validate_start(
                {"backend": "whisper-cpp"},
                default_backend="funasr",
                default_model="approved-model",
            )
        with self.assertRaisesRegex(ValueError, "model"):
            policy.validate_start(
                {"model": "unapproved-model"},
                default_backend="funasr",
                default_model="approved-model",
            )

    def test_default_policy_rejects_client_paths_and_components(self):
        policy = ServerPolicy.defaults("whisper-cpp")
        with self.assertRaisesRegex(ValueError, "paths"):
            policy.validate_start(
                {"executable": "/tmp/untrusted"},
                default_backend="whisper-cpp",
                default_model=None,
            )
        with self.assertRaisesRegex(ValueError, "component"):
            policy.validate_start(
                {"aligner": "paraformer"},
                default_backend="whisper-cpp",
                default_model=None,
            )
        component_policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            allowed_components=frozenset({"paraformer"}),
        )
        with self.assertRaisesRegex(ValueError, "component options"):
            component_policy.validate_start(
                {"aligner": "paraformer", "aligner_options": {"device": "cuda"}},
                default_backend="fake",
                default_model=None,
            )

    def test_remote_bind_requires_explicit_opt_in(self):
        policy = ServerPolicy.defaults("fake")
        policy.validate_bind("127.0.0.1")
        policy.validate_bind("::1")
        with self.assertRaisesRegex(ValueError, "allow_remote"):
            policy.validate_bind("0.0.0.0")

    def test_authentication_uses_public_generic_error(self):
        policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            auth_token="secret",
        )
        with self.assertRaises(PermissionError) as raised:
            policy.validate_start({}, default_backend="fake", default_model=None)
        self.assertEqual(policy.public_error(raised.exception)["code"], "unauthorized")
        self.assertNotIn("secret", str(policy.public_error(raised.exception)))

    def test_internal_paths_are_redacted(self):
        policy = ServerPolicy.defaults("fake")
        payload = policy.public_error(RuntimeError("failed at /Users/private/model.bin"))
        self.assertEqual(payload["code"], "session_error")
        self.assertNotIn("/Users/private", payload["message"])


if __name__ == "__main__":
    unittest.main()
