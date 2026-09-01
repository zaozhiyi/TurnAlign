import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryGovernanceTests(unittest.TestCase):
    def test_codeql_uses_immutable_actions_and_fork_safe_upload(self):
        content = (ROOT / ".github" / "workflows" / "codeql.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("security-events: write", content)
        self.assertIn("queries: security-extended", content)
        self.assertIn("github.event_name == 'pull_request'", content)
        self.assertIn("'never' || 'always'", content)
        for line in content.splitlines():
            if "uses:" in line:
                revision = line.split("@", 1)[1].split()[0]
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_codeql_covers_main_pull_requests_and_schedule(self):
        content = (ROOT / ".github" / "workflows" / "codeql.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("push:", content)
        self.assertIn("pull_request:", content)
        self.assertIn("schedule:", content)
        self.assertGreaterEqual(content.count("branches: [main]"), 2)

    def test_dependabot_covers_python_and_actions(self):
        content = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        self.assertIn("package-ecosystem: pip", content)
        self.assertIn("package-ecosystem: github-actions", content)
        self.assertEqual(content.count("interval: weekly"), 2)
        self.assertEqual(content.count('patterns: ["*"]'), 2)

    def test_security_policy_avoids_public_exploit_reports(self):
        content = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("Do not publish exploit details", content)
        self.assertIn("private vulnerability reporting", content)
        self.assertIn("contact requested", content)


if __name__ == "__main__":
    unittest.main()
