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
  requirements.lock
sudo /usr/local/bin/uv pip install \
  --python /opt/turnalign/releases/<source-commit>/venv/bin/python \
  --no-deps dist/turnalign-0.1.0-py3-none-any.whl
sudo chown -R root:root /opt/turnalign/releases/<source-commit>
```

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

The production gate requires exact `==` pins with SHA-256 hashes and verifies
both directions: every unconditional lock entry is present at that version,
and every installed SBOM runtime component has a matching lock entry. Editable,
VCS, local-path, unhashed, unlocked and build-only components fail.

The example paths assume:

- executable: `/opt/turnalign/current/venv/bin/turnalign`
- service account: `turnalign`
- state/model cache: `/var/lib/turnalign`
- root-only authentication credential: `/etc/turnalign/auth-token`
- normalized warm-up audio: `/var/lib/turnalign/warmup.wav`

Install the `server` and `funasr` extras in the image or virtual environment.
Pin the complete transitive dependency set for the target CPU architecture.
Model weights must be downloaded ahead of deployment under the `turnalign`
account. The unit starts with `--preload`, a real warm-up inference, and
`--require-immutable-model-revision`; missing dependencies, weights, or an
unreadable warm-up file therefore fail before the socket is ready.
The reference unit also applies `IPAddressDeny=any` with
`IPAddressAllow=localhost`: the model process can accept Nginx and host-local
monitoring traffic but cannot fetch weights, call third-party APIs, or otherwise
open Internet connections. Complete every download and cache validation before
starting the service. A backend that genuinely requires remote inference is not
compatible with this reference profile and needs a separately reviewed unit.

Activate a fully prepared release by atomically replacing the `current`
symlink. The systemd unit and production validator require this exact layout:

```bash
sudo ln -s /opt/turnalign/releases/<source-commit> \
  /opt/turnalign/.current-<source-commit>
sudo mv -Tf /opt/turnalign/.current-<source-commit> /opt/turnalign/current
```

Do not remove the preceding release directory until the new release has passed
all gates and the rollback rehearsal. A failed activation is reversed by the
same atomic link replacement; no package is installed into or removed from the
running release.

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
   including the fault-resume option and intended concurrency/soak duration.
3. `turnalign quality-gate` passes on the versioned, human-labelled production
   corpus using project-owned thresholds.
4. All three reports name the exact release source commit; the release audio,
   quality reference and quality hypothesis digests match the retained input
   artifacts, and the quality/release reports identify the same immutable model.
5. The exact wheel hash, dependency lock, validated CycloneDX SBOM, model
   commit, model-file checksums, Nginx configuration, service unit, host
   profile, and gate reports are saved. Treat labelled references and
   hypotheses as potentially sensitive data and restrict their storage.
6. Rollback is tested by atomically pointing `/opt/turnalign/current` at the
   preceding immutable release directory, restarting the service, and rerunning
   readiness plus the public WebSocket gate. Atomically restore the candidate,
   restart, and rerun both probes before approval. Retain the two source commits,
   model revisions, activation times, command exit status and probe reports as
   deployment-system evidence; a hand-written success JSON is not sufficient.
7. For tagged upstream distributions,
   `gh attestation verify FILE --repo GuanZhengPM/TurnAlign` verifies the signed
   GitHub/Sigstore build provenance.
   This supplements rather than replaces the final eleven-artifact deployment
   gate, whose model, host and production-corpus evidence is environment-specific.

Persist each gate with `--report`, generate retained-model provenance with
`turnalign model-manifest`, and run
`/opt/turnalign/current/venv/bin/turnalign host-profile` on the target host after
activating the candidate and finalizing the other ten artifact classes. The
command reads the commit embedded in the installed Wheel, so the production
host does not need a Git checkout. It refuses non-Linux hosts, source checkouts,
unbound Wheels, mismatched explicitly supplied commits, and non-versioned Python
environments. Then run `turnalign
production-gate` with the source commit and all eleven required artifact kinds
shown in the root README. Keep
the resulting aggregate report beside the release artifact; it rejects local
`ws://`, missing recovery/latency controls, mutable model revisions, undersized
quality evidence, incomplete artifact sets, a retained Nginx file containing
unresolved includes or weakened TLS/WebSocket/upstream controls, and a retained
systemd unit that
weakens loopback binding, credential handling, resource limits, network
isolation, version-switchable activation, graceful lifecycle behavior, or
least-privilege controls.

WebSocket recovery is process-local. Keep one process behind this reference
upstream. A multi-process or multi-host topology must either provide sticky
routing for the recovery window or implement a shared durable recovery store;
the repository does not currently provide the latter.
