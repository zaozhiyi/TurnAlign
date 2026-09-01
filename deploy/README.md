# Production reference deployment

This directory contains a deliberately narrow reference topology:

```text
Internet -> TLS/rate limits (Nginx) -> 127.0.0.1:8765 -> one TurnAlign process
```

The supplied systemd unit is for a **Linux CPU host running the built-in pinned
`funasr-streaming` model**. It is not a generic GPU unit. `PrivateDevices=true`
is intentional; create and validate a platform-specific unit before exposing
CUDA, ROCm, Vulkan, or other accelerator devices. Run TurnAlign natively on
macOS because Linux containers and services cannot expose Metal/MPS.

## 1. Prepare an immutable installation

Build the wheel in CI with `TURNALIGN_SOURCE_COMMIT` set to the exact checked-out
40-character commit, retain its `SHA256SUMS` and generated CycloneDX SBOM, and
install the exact reviewed artifact with a deployment-owned dependency lock.
Regenerate the SBOM from the final model-specific environment before the
production gate. Do not install the current Git branch or download an
unreviewed wheel during service startup.

Development builds without that environment variable remain installable but
carry an `unbound` marker and cannot pass `production-gate`.

For the reference CPU profile, create a target-specific, hash-locked dependency
set and install each source commit into a new release directory before the
wheel. Never update the active virtual environment in place:

```bash
uv pip compile pyproject.toml --extra server --extra funasr \
  --generate-hashes --output-file requirements.lock
sudo install -d -m 0755 /opt/turnalign/releases/<source-commit>
sudo python3 -m venv --without-pip \
  /opt/turnalign/releases/<source-commit>/venv
sudo /usr/local/bin/uv pip sync \
  --python /opt/turnalign/releases/<source-commit>/venv/bin/python \
  --no-compile-bytecode \
  requirements.lock
sudo /usr/local/bin/uv pip install \
  --python /opt/turnalign/releases/<source-commit>/venv/bin/python \
  --no-compile-bytecode --no-deps dist/turnalign-0.1.0-py3-none-any.whl
sudo chown -R root:root /opt/turnalign/releases/<source-commit>
sudo chmod -R go-w /opt/turnalign/releases/<source-commit>
```

Activation, recovery, and rollback rehearsal recursively inspect the complete
candidate and preceding release trees before changing `current`. Every regular
file and directory must be root-owned and non-writable by group or others;
symbolic links must themselves be root-owned and resolve through an equally
immutable root-owned path. Directory links may resolve only inside the same
release tree; an external link may target only a regular file, such as the
system interpreter selected by a normal virtual environment. This covers the
interpreter, TurnAlign, and all installed dependencies rather than trusting only
the top-level release directory or embedded source identity.

Install a reviewed, pinned `uv` binary at the root-owned path shown above, or
replace it with its actual absolute path. `uv pip sync` removes packages not
present in the complete lock, while `--without-pip` avoids seeding packaging
tools into the runtime environment. The final
runtime SBOM must not contain pip, Twine, CycloneDX, linters, auditors, or other
build-only tools. If a backend genuinely imports `setuptools`/`pkg_resources`,
retain an exact hash-pinned setuptools entry in the runtime lock; otherwise
remove the bootstrap copy before generating the SBOM.

Run the SBOM generator from a separately installed, pinned `cyclonedx-bom`
tool, not from inside the production environment:

```bash
cyclonedx-py environment /opt/turnalign/releases/<source-commit>/venv/bin/python \
  --pyproject pyproject.toml --mc-type application --sv 1.6 \
  --output-reproducible --of JSON -o sbom.cdx.json
```

The production gate requires target-resolved exact `==` pins with SHA-256 hashes
and no environment markers, and verifies both directions: every lock entry is present at that version,
and every installed SBOM runtime component has a matching lock entry. Editable,
VCS, local-path, unhashed, unlocked and build-only components fail.

The example paths assume:

- executable: `/opt/turnalign/current/venv/bin/python -I -B -u -m turnalign.cli`
- service account: `turnalign`
- writable process state: `/var/lib/turnalign-state`
- immutable model and warm-up evidence: `/var/lib/turnalign`
- root-only authentication credential: `/etc/turnalign/auth-token`
- normalized warm-up audio: `/var/lib/turnalign/warmup.wav`

Install the `server` and `funasr` extras in the image or virtual environment.
Pin the complete transitive dependency set for the target CPU architecture.
Download model weights to a staging directory, verify their retained manifest,
then install them under a root-controlled ancestor that the service account
cannot rename or modify:

```bash
sudo install -d -m 0755 -o root -g root \
  /var/lib/turnalign/models/paraformer-zh-streaming
sudo cp -a /path/to/verified-model/. \
  /var/lib/turnalign/models/paraformer-zh-streaming/
sudo install -m 0644 -o root -g root /path/to/warmup.wav \
  /var/lib/turnalign/warmup.wav
sudo chown -R root:root /var/lib/turnalign
sudo chmod -R go-w /var/lib/turnalign
```

Do not make `/var/lib/turnalign` a `StateDirectory`; the supplied unit uses the
separate `/var/lib/turnalign-state` path for writable runtime state. The unit
starts with `--preload`, `--require-local-model`, a real warm-up inference, and
`--require-immutable-model-revision`; missing dependencies, weights, unsafe
ancestors, or an unreadable warm-up file therefore fail before the socket is ready.
The reference unit also applies `IPAddressDeny=any` with
`IPAddressAllow=localhost`: the model process can accept Nginx and host-local
monitoring traffic but cannot fetch weights, call third-party APIs, or otherwise
open Internet connections. Complete every download and cache validation before
starting the service. A backend that genuinely requires remote inference is not
compatible with this reference profile and needs a separately reviewed unit.

The systemd unit and production validator require `current` to be an absolute
symlink to one immutable commit directory. Do not use ad-hoc `ln`/`mv` during a
normal rollout: step 6 runs the serialized `deployment-activate` transaction.
For a brand-new host with no `current` link, bootstrap a reviewed preceding
release before any user traffic so the first candidate has a real rollback
target:

```bash
sudo ln -s /opt/turnalign/releases/<source-commit> \
  /opt/turnalign/.current-<source-commit>
sudo mv -Tf /opt/turnalign/.current-<source-commit> /opt/turnalign/current
```

This manual operation is bootstrap-only. Do not remove the preceding release
directory until the candidate has passed activation, every gate, and the
rollback rehearsal. No package is installed into or removed from a running
release.

## 2. Install the service

Copy `systemd/turnalign.service` to `/etc/systemd/system/`. Create a random
credential readable only by root; systemd copies it into a protected per-service
credentials directory instead of exposing it through the service environment:

```bash
sudo install -d -m 0700 -o root -g root /etc/turnalign
sudo install -m 0600 -o root -g root /dev/null /etc/turnalign/auth-token
openssl rand -hex 32 | sudo tee /etc/turnalign/auth-token >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now turnalign
curl --fail --silent http://127.0.0.1:8765/readyz
curl --fail --silent http://127.0.0.1:8765/metrics
```

Do not commit, print, or pass the generated token on a command line. The unit
uses `LoadCredential=` and `--auth-token-file`; TurnAlign rejects symlinks,
non-regular files, files readable by group/others, embedded newlines, NUL bytes,
and credentials larger than 8 KiB. Rotate the token by atomically replacing the
root-owned source file and restarting the service.
Keep `TimeoutStopSec` greater than TurnAlign's `--shutdown-grace-timeout`.
Capacity values in the unit are conservative starting points, not measured
production targets. Size model memory, temporary disk, concurrency, and
session duration on the deployment host.

Gate reports are accepted for at most 24 hours, tolerate at most five minutes
of forward clock skew, and the live deployment-state snapshot is capped at five
minutes. Keep target-host time synchronized and regenerate expired evidence.

`/metrics` exposes label-free Prometheus counters for active/admitted/rejected
sessions, incomplete or failed work, recovery, audio volume, flow control and
output pressure. It contains no transcript text, session IDs, backend/model
labels or credentials. Scrape it from a host-local collector on
`127.0.0.1:8765`; the example Nginx server deliberately returns 404 for the
public `/metrics` path.

## 3. Terminate TLS at Nginx

Copy `nginx/turnalign.conf.example` to a dedicated self-contained HTTP-context
file such as `/etc/nginx/conf.d/turnalign.conf`, replace the hostname and
certificate paths, then validate and reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

The application is intentionally bound only to `127.0.0.1`. Do not expose port
8765 through a host firewall or container publish rule. The public WebSocket
endpoint in this example is `wss://asr.example.com/ws`; clients still put the
application token in the first protocol message, never in the URI or proxy
logs. Browser clients remain denied until an exact `--allow-origin` value is
added to the service after reviewing the frontend origin.

Keep Nginx's `proxy_read_timeout` greater than the application's initialization,
client-idle, and finalization deadlines. The example uses 330 seconds because
the service allows initialization for 300 seconds and finalization for 120
seconds. Application deadlines remain the inner resource limits; the proxy
deadline must not terminate a valid slow phase first.

## 4. Release gates

Before routing users, retain all of the following with the release artifact:

1. `/readyz` succeeds only after preload and warm-up.
2. `turnalign websocket-gate` passes against the **public** `wss://` endpoint,
   including the fault-resume option and intended concurrency/soak duration;
   every ready response identifies the same requested backend, backend
   implementation, model, device and immutable model revision.
3. `turnalign quality-gate` passes on the versioned, human-labelled production
   corpus using project-owned thresholds.
4. All three reports name the exact release source commit; the release audio,
   quality reference and quality hypothesis digests match the retained input
   artifacts, and the quality/release/public-WebSocket reports identify the same
   backend implementation and immutable model revision.
5. The exact wheel hash, dependency lock, validated CycloneDX SBOM, model
   commit, model-file checksums, Nginx configuration, service unit, deployment
   activation, rollback rehearsal, host profile, and gate reports are saved. Treat labelled
   references and hypotheses as potentially sensitive data and restrict their
   storage.
6. While the preceding release is active, run the command from the inactive
   candidate as root. It owns the actual atomic switch, restart, and probes:

   ```bash
   sudo /opt/turnalign/releases/<candidate-source-commit>/venv/bin/python \
     -I -B -u -m turnalign.cli deployment-activate \
     wss://asr.example.com/ws \
     --previous-commit <preceding-source-commit> \
     --candidate-commit <candidate-source-commit> \
     --backend funasr-streaming --model paraformer-zh-streaming \
     --probe-audio /var/lib/turnalign/probe/probe-60s.wav \
     --auth-token-file /etc/turnalign/auth-token \
     --report deployment-activation.json
   ```

   The command refuses an already-active candidate. It takes the root-only
   non-blocking lock at `/run/lock/turnalign-deployment.lock`, first persists a
   bounded root-only marker at
   `/var/lib/turnalign-deployment/pending-activation.json`, switches and
   restarts exactly once, then requires preloaded readiness and a real-time
   concurrent public recovery probe. Failure or cancellation switches back to
   the preceding release, restarts it, and repeats both probes before returning
   control. The marker is removed only after the report is durably written and
   the expected release is active. Retain the report; only a successful
   activation report can pass the aggregate production gate.

   `finally` cannot handle SIGKILL, power loss, or a host reboot. If activation
   ends without a report, do not edit `current` or delete the marker manually.
   Run the candidate-installed recovery command:

   ```bash
   sudo /opt/turnalign/releases/<candidate-source-commit>/venv/bin/python \
     -I -B -u -m turnalign.cli deployment-recover \
     --backend funasr-streaming --model paraformer-zh-streaming \
     --probe-audio /var/lib/turnalign/probe/probe-60s.wav \
     --auth-token-file /etc/turnalign/auth-token \
     --report deployment-recovery.json
   ```

   It accepts only the immutable release pair recorded before activation,
   restores the preceding release, restarts it, and repeats readiness plus the
   public recovery probe. A successful recovery report is operational evidence,
   not one of the fifteen final artifacts; rerun activation to produce a new
   final report. A pending marker blocks another activation, rehearsal, and
   `host-profile`.
7. With the candidate now active, exercise the rollback and restore path:

   ```bash
   sudo /opt/turnalign/current/venv/bin/python -I -B -u -m turnalign.cli \
     deployment-rehearsal \
     wss://asr.example.com/ws \
     --previous-commit <preceding-source-commit> \
     --candidate-commit <candidate-source-commit> \
     --backend funasr-streaming --model paraformer-zh-streaming \
     --probe-audio /var/lib/turnalign/probe/probe-60s.wav \
     --auth-token-file /etc/turnalign/auth-token \
     --report rollback-rehearsal.json
   ```

   After both live phases pass, capture a fresh target-host state snapshot that
   `production-gate` uses as the live deployment check:

   ```bash
   sudo /opt/turnalign/current/venv/bin/python -I -B -u -m turnalign.cli \
     deployment-status --validity-seconds 300 --output deployment-state.json
   ```

   Keep the exact `-I -B -u -m turnalign.cli` prefix for all deployment commands:
   it ignores environment/user
   import paths, bypasses the installer-generated console launcher, disables
   bytecode generation, and keeps journal output unbuffered. The final gate
   requires the installed package tree to
   equal the reviewed Wheel exactly and therefore rejects added
   `__pycache__`/`.pyc` files. The command refuses
   non-Linux/non-root execution, an unbound candidate
   runtime, mutable or non-root-owned release directories, a non-public probe,
   and any initial active release other than the candidate. A root-only
   same non-blocking lock prevents activation, recovery, or rehearsal from racing
   a symlink transition. The rehearsal switches to the
   preceding release, restarts and checks `systemctl is-active`, requires
   preloaded readiness plus a real-time concurrent public recovery probe, then
   restores and reprobes the candidate. Probe failure or cancellation still
   triggers candidate restoration, and the command fails unless both exact
   transitions pass. Retain its report rather than writing a success summary.
   Before switching, rehearsal persists an operation-tagged transaction at the
   same root-only marker path. If SIGKILL, power loss, or reboot interrupts it,
   run candidate-installed `deployment-recover`; it restores and reprobes the
   candidate, after which the rehearsal must be rerun. Legacy schema-1 markers
   are treated as interrupted activations.
8. For tagged upstream distributions,
   `gh attestation verify FILE --repo GuanZhengPM/TurnAlign` verifies the signed
   GitHub/Sigstore build provenance.
   This supplements rather than replaces the final fifteen-artifact deployment
   gate, whose model, host and production-corpus evidence is environment-specific.

Persist each gate with `--report`, generate retained-model provenance with
`turnalign model-manifest`, and run
`sudo /opt/turnalign/current/venv/bin/python -I -B -u -m turnalign.cli host-profile` on the
target host after
activating the candidate and finalizing the other fourteen artifact classes. The
command holds the root-only deployment lock for the complete capture and reads
the commit embedded in the installed Wheel, so the production
host does not need a Git checkout. It refuses non-Linux hosts, source checkouts,
unbound Wheels, mismatched explicitly supplied commits, and non-versioned Python
environments. Its schema 6 evidence also proves that the root-owned canonical
`current` link selects the candidate and hashes the complete active `turnalign/`
tree; the aggregate gate requires that tree to match the retained Wheel exactly.
Installed dependency content is hashed file by file into bounded deterministic
tree digests, so large runtimes such as Torch do not overflow the profile JSON.
Then run `turnalign
production-gate` with the source commit and all fifteen required artifact kinds
shown in the root README. Keep
the resulting aggregate report beside the release artifact; it rejects local
`ws://`, missing recovery/latency controls, mutable model revisions, undersized
quality evidence, incomplete artifact sets, activation or rollback reports with
forged or out-of-order transitions, mismatched release pairs, a different Linux
boot or restored model identity,
an active package loaded outside the versioned environment, installed files
that are missing, modified, symlinked, writable or absent from the retained
Wheel, a retained Nginx file containing
unresolved includes or weakened TLS/WebSocket/upstream controls, and a retained
systemd unit that
weakens loopback binding, credential handling, resource limits, network
isolation, version-switchable activation, graceful lifecycle behavior, or
least-privilege controls.

WebSocket recovery is process-local. Keep one process behind this reference
upstream. A multi-process or multi-host topology must either provide sticky
routing for the recovery window or implement a shared durable recovery store;
the repository does not currently provide the latter.
