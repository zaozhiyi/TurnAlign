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

Build the wheel in CI, retain its `SHA256SUMS`, and install the exact reviewed
artifact with a deployment-owned dependency lock. Do not install the current
Git branch or download an unreviewed wheel during service startup.

The example paths assume:

- executable: `/opt/turnalign/venv/bin/turnalign`
- service account: `turnalign`
- state/model cache: `/var/lib/turnalign`
- configuration: `/etc/turnalign/turnalign.env`
- normalized warm-up audio: `/var/lib/turnalign/warmup.wav`

Install the `server` and `funasr` extras in the image or virtual environment.
Pin the complete transitive dependency set for the target CPU architecture.
Model weights must be downloaded ahead of deployment under the `turnalign`
account. The unit starts with `--preload`, a real warm-up inference, and
`--require-immutable-model-revision`; missing dependencies, weights, or an
unreadable warm-up file therefore fail before the socket is ready.

## 2. Install the service

Copy `systemd/turnalign.service` to `/etc/systemd/system/`. Copy
`systemd/turnalign.env.example` to `/etc/turnalign/turnalign.env`, replace the
placeholder with a random secret, and restrict the file to root and the service
group:

```bash
sudo install -d -m 0750 -o root -g turnalign /etc/turnalign
sudo install -m 0640 -o root -g turnalign \
  deploy/systemd/turnalign.env.example /etc/turnalign/turnalign.env
openssl rand -hex 32
sudo systemctl daemon-reload
sudo systemctl enable --now turnalign
curl --fail --silent http://127.0.0.1:8765/readyz
```

Write the generated value into `TURNALIGN_AUTH_TOKEN` without committing it.
Keep `TimeoutStopSec` greater than TurnAlign's `--shutdown-grace-timeout`.
Capacity values in the unit are conservative starting points, not measured
production targets. Size model memory, temporary disk, concurrency, and
session duration on the deployment host.

## 3. Terminate TLS at Nginx

Copy `nginx/turnalign.conf.example` into the Nginx `http` context, replace the
hostname and certificate paths, then validate and reload:

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

## 4. Release gates

Before routing users, retain all of the following with the release artifact:

1. `/readyz` succeeds only after preload and warm-up.
2. `turnalign websocket-gate` passes against the **public** `wss://` endpoint,
   including the fault-resume option and intended concurrency/soak duration.
3. `turnalign quality-gate` passes on the versioned, human-labelled production
   corpus using project-owned thresholds.
4. The exact wheel hash, dependency lock, model commit, model-file checksums,
   Nginx configuration, service unit, host profile, and gate reports are saved.
5. Rollback is tested by restoring the preceding immutable artifact and model
   bundle, not by mutating the running environment.

WebSocket recovery is process-local. Keep one process behind this reference
upstream. A multi-process or multi-host topology must either provide sticky
routing for the recovery window or implement a shared durable recovery store;
the repository does not currently provide the latter.
