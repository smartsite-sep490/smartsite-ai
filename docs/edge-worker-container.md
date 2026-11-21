# Run the camera worker on one operator machine

The operator machine owns inference. All browser clients use the same SmartSite
Backend URL, Site and authorized accounts. Browsers never receive the AI service
token. Local Web must point to that Backend as well to share events.

## Build

```sh
docker build -f Dockerfile.worker --build-arg SOURCE_REVISION=YOUR_SOURCE_SHA -t smartsite-ai-worker:YOUR_SOURCE_SHA .
```

The default locked `vision` extra is CPU-only. An explicitly selected build with
`--build-arg VISION_EXTRA=cuda126` uses the existing CUDA12.6 dependencies; it also
needs a compatible NVIDIA driver/container runtime and GPU device reservation.
The default Compose file does not request a GPU. A CPU image must use a model spec
with `device: cpu`; declaring CUDA does not silently fall back to CPU.

This image contains neither detector weights nor identity models. It starts no
camera/model during build or module import. No Face capability is activated.
Keep the original model provenance, class map, checksum, evaluation reference
and license review with the mounted artifact. Ultralytics use still requires the
applicable AGPL/commercial license review; packaging does not change that license.

## Configure and run

1. Put reviewed `runtime.json` and `model-spec.json` in a private config directory.
   Use `deploy/runtime.example.json` as a shape only: its IDs/hash are placeholders.
   The artifact path inside the model spec must use `/models/...`, not a Windows
   drive path. Preserve the approved checksum and class map.
2. Mount the approved weights directory read-only. Supply an RTSP source through
   the private environment file; secrets must not appear in CLI arguments or Git.
3. Obtain a real scoped generation credential from the trusted Backend operator.
   Place it in the private config directory as `generation.json` for the explicit
   state-volume initialization below. Never fabricate a generation.
4. Set the following Compose variables in your shell or a private Compose env file:

```text
SMARTSITE_AI_WORKER_IMAGE=YOUR_REGISTRY/smartsite-ai-worker@sha256:REVIEWED_DIGEST
SMARTSITE_AI_WORKER_ENV_FILE=/absolute/private/worker.env
SMARTSITE_AI_CONFIG_DIR=/absolute/private/config
SMARTSITE_AI_MODEL_DIR=/absolute/private/models
```

Copy `deploy/worker.env.example` outside Git and replace its fake values. The
ingestion URL must identify the Backend shared by the browsers. An operator on
another machine cannot reach your Backend using `127.0.0.1`.

```sh
docker compose -f deploy/compose.worker.yaml pull
docker compose -f deploy/compose.worker.yaml run --rm --no-deps --entrypoint python camera-worker -c "from pathlib import Path; import shutil; p=Path('/state/evidence-camera-01'); p.mkdir(mode=0o700,exist_ok=True); p.chmod(0o700); shutil.copyfile('/config/generation.json','/state/generation.json'); Path('/state/generation.json').chmod(0o600)"
docker compose -f deploy/compose.worker.yaml up -d
docker compose -f deploy/compose.worker.yaml logs --tail=40
docker compose -f deploy/compose.worker.yaml down
```

Stopping does not delete the state volume. It retains the SQLite retry outbox and
local evidence. Do not run two workers with the same camera/generation/state.
The initialization command runs as UID10001 and must run only while that camera
worker is stopped. Create each reviewed camera's evidence directory the same way.
Replacing a generation while a worker is active is not a supported restart path;
use the trusted source/generation handover instead.
The runtime supports one to three reviewed cameras; this is not a throughput claim.
Authentication/configuration/model failure stops the worker rather than creating
fake observations. Preview remains localhost-only on host port8001; Backend access
from elsewhere requires the reviewed authenticated HTTPS gateway, never a public
unauthenticated camera port. Outbound RTSP must obey the host network policy.

## Delivery limits

Windows Docker Desktop does not automatically expose a laptop webcam as `/dev/video0`.
Keep its current host capture until the bridge is integrated. A Linux camera can
be mapped explicitly as a device after operator review. Approved video mounts can
use the existing runtime source path with `live:false`; tracked-v1.7 LIVE delivery
cannot be reused as a fabricated replay generation.

Events and images are separate: the current local evidence writer writes under
`/state`, which a cloud Backend cannot read as its own filesystem. Shared media,
source-control gateway and real two-client acceptance must be completed before
calling the deployed MF05/MF06 flow ready. API health alone is not camera health.
