# Security policy

## Supported versions

TurnAlign is pre-1.0. Security fixes are made on `main` and included in the next
reviewed release. Older commits and unmaintained forks are not supported. A
deployment is supported only when it uses an immutable TurnAlign artifact,
locked dependencies, pinned model revisions, and the security controls described
in `deploy/README.md`.

## Repository governance before release

Green workflow checks are evidence, but GitHub does not enforce them unless the
upstream owner configures repository rules. Before merging a production-bound
change, protect `main` with pull requests, require the complete CI and CodeQL
check set, dismiss stale approvals, and restrict direct pushes and force-pushes.
Enable Dependabot alerts and security updates (and secret scanning/push
protection where available). Re-check these settings on the upstream repository
and its fork; a checked-in workflow or `dependabot.yml` file alone does not
enable the corresponding GitHub security service.

## Reporting a vulnerability

Do not publish exploit details, credentials, private audio, transcripts, or
model artifacts in a GitHub issue.

The upstream repository does not currently have GitHub private vulnerability
reporting enabled. Until an upstream administrator enables it, contact the
repository owner through the contact method on
<https://github.com/GuanZhengPM> and request a private security channel. Share
only a short, non-exploitable summary in that first contact. If no private
contact method is available, open an issue containing only the words “Security
contact requested” and no technical details.

Once private vulnerability reporting is enabled, use the repository Security
tab's “Report a vulnerability” action instead.

A useful private report includes:

- affected TurnAlign commit, package version, backend, and deployment topology;
- impact and the trust boundary that is crossed;
- minimal reproduction steps using synthetic, non-sensitive data;
- whether the issue is already being exploited or publicly known;
- suggested mitigation, if available.

Keep the report private until a fix and disclosure plan are agreed. Ordinary
hardening ideas without sensitive exploit details may use a public issue.
