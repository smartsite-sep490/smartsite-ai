# SmartSite AI

SmartSite AI is the independent computer-vision and AI service for the SmartSite construction-site management platform.

The service is responsible for camera ingestion, visual detection, tracking, PPE monitoring, restricted-zone monitoring, identity experiments, evidence generation, and supplementary OpenAI analysis. It does **not** own construction-site business authorization.

Before contributing or using a coding assistant, read [AGENTS.md](AGENTS.md) for the mandatory AI architecture, model, privacy, testing, Git, and code-quality rules, then follow [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow.

## Responsibilities

SmartSite AI is intended to provide:

- RTSP camera ingestion;
- person and PPE detection;
- multi-frame tracking;
- restricted-zone entry detection;
- optional identity-candidate generation;
- evidence generation;
- technical detection events;
- supplementary OpenAI evidence analysis.

The service does not independently decide whether a worker is allowed into a Site, whether a worker has permission for a Zone, whether an alert is an official violation, whether an Incident should be closed, or whether an identity prediction is sufficient for business action. Those decisions remain under the SmartSite backend and authorized human operators.

## Architecture

```text
IP Camera / RTSP
       |
       v
Camera Ingestion
       |
       v
Person / PPE Detection
       |
       +------> PPE Pipeline (MF05)
       |
       v
Tracking
       |
       +------> Restricted-Zone Pipeline (MF06)
       |                    |
       |                    v
       |              Identity Candidate
       |              InsightFace / ArcFace
       |
       v
Evidence Builder
       |
       +------> OpenAI Evidence Analysis
       |
       v
AI Detection Event
       |
       v
SmartSite Backend
       |
       +-- business validation
       +-- Zone authorization
       +-- deduplication
       +-- Safety Alert
       +-- Incident workflow
```

## Technology Direction

| Responsibility | Technology |
| --- | --- |
| API | Python, FastAPI |
| Detection | YOLO11s baseline |
| Tracking | Deterministic IoU tracker at the provider-neutral pipeline boundary |
| PPE monitoring | Trained PPE detection weights |
| Zone monitoring | Tracking + configured geometry |
| Identity experiment | InsightFace / ArcFace |
| Evidence analysis | OpenAI API |
| Business authorization | SmartSite Backend |
| Business database | Neon PostgreSQL through Backend |

YOLO11s is the selected implementation baseline for MF05/MF06. RF-DETR Nano/Small and YOLO26s remain optional benchmark challengers; they do not block the first implementation. Roboflow Workflows and NVIDIA DeepStream have been researched but are not part of the initial implementation baseline.

## Design Principles

### Detection is not a business conclusion

A model prediction is supporting evidence. Safety Officers remain responsible for verification where required.

### Track ID is not Worker ID

A tracker identifier represents an object across frames. It is not a worker identity.

### Unknown identity is valid

When identity confidence is insufficient, the system keeps the person unidentified. It must not fabricate or infer identity from clothing, prior attendance, or camera presence.

### Identity and authorization are separate

For MF06:

```text
Person detected
      |
      v
Identity candidate
      |
      v
SmartSite Backend
      |
      v
Zone permission evaluation
      |
      v
Allowed / Denied / Unavailable
```

The AI service does not own Site/Zone authorization rules.

## Current Status

**Current phase: FastAPI and technical MF05/MF06 pipeline foundation.**

Implemented:

- FastAPI application factory;
- validated environment configuration;
- strict, versioned MF05/MF06 observation models and vendored JSON Schema provenance;
- RFC 8785 compatible hashing with cross-runtime safe-integer guards;
- authenticated Backend ingestion client with bounded retries and strict response validation;
- authenticated, bounded camera-configuration polling with strong ETag/304 support and stale-snapshot fail-closed behavior;
- an explicit one-camera headless worker that connects ingestion, verified YOLO11 inference, tracking, MF05/MF06 pipelines, and Backend delivery without a browser session;
- a crash-tolerant SQLite observation outbox with canonical payload conflict detection, retry scheduling, terminal 4xx classification, and Backend idempotency compatibility;
- camera ingestion worker foundation (typed `FrameEnvelope`, `FrameSource` protocol boundary, `BoundedFrameQueue` with drop-stale backpressure, bounded exponential backoff with jitter and cancellation, `FakeFrameSource` for deterministic testing, and URL credential sanitization);
- optional OpenCV video source for local video files, camera indexes, and RTSP URLs, plus a worker smoke-test command;
- verified local model-artifact metadata, a lazy Ultralytics YOLO runner, and normalized `DetectionBatch` output;
- local annotated-video validation with an optional MF05/MF06 UI timeline export;
- deterministic IoU person tracking with stream/session-scoped track IDs;
- PPE-to-person association with technical `PRESENT`/observable `MISSING` observations;
- configured polygon restricted-zone transition detection with geometry-version handling;
- MF05/MF06 orchestration into the locked technical observation event, with synthetic fixtures and behavior tests;
- liveness endpoint;
- readiness endpoint;
- capability endpoint;
- production documentation disabling;
- automated tests;
- Python package build;
- Docker image;
- non-root runtime;
- GitHub Actions CI.

Not yet implemented:

- live RTSP hardware/network validation;
- production/container GPU deployment and 1–3 camera end-to-end capacity validation;
- PPE model weights;
- production tracker/model calibration and evaluation on representative site data;
- InsightFace integration;
- dynamic camera discovery;
- an operational retention/pruning policy for delivered outbox rows;
- Safety Alert list/detail/review APIs and Web integration;
- exact processed-frame synchronization for live Web overlays;
- OpenAI adapter;
- benchmark results.

The API explicitly reports `inference_ready: false` until a configured worker has a verified local
artifact, active camera source, and tested runtime; the local video command does not change API
readiness.

## Requirements

### Guided enrollment quality (demo policy v1)

Authenticated verification JSON can include `enrollmentTarget: "front" | "left" | "right"`
with `templates: []`. This is a quality-only operation, never identity matching or permission.
It returns `UNKNOWN` + `FACE_QUALITY_ACCEPTED` on success, `QUALITY_FAILED` with a safe reason
on rejection, or `AI_UNAVAILABLE`. Enrollment independently rechecks all three ordered poses
and checks embedding consistency before encrypting the template.

Versioned demo checks: one detected face, detector confidence >=0.70, face dimensions >=160px,
width/frame ratio >=0.18 and <=0.75, height/frame <=0.90, horizontal/vertical center offsets
<=0.22/0.25, grayscale mean 45..215, face-crop Laplacian variance >=45, eye distance >=20px,
head roll <=18 degrees. Nose displacement projected onto the eye axis, normalized by eye
distance, is within +/-0.10 for front, +0.12..+0.50 for the user's left, -0.50..-0.12 for right.
Raw frames are unmirrored; only Web previews are mirrored. These are tunable, uncalibrated
heuristics, not yaw-angle measurements, anti-spoofing or production biometric validation.
Calibrate on permitted real-camera captures before making accuracy claims.

- Python **3.12.x**
- uv **0.11.6**

Install the configured Python version and dependencies:

```sh
uv python install 3.12.13
uv sync --frozen
```

Run the service:

```sh
uv run --frozen smartsite-ai
```

### Video source smoke test

The video files used for local validation must stay outside Git. Install the
optional vision dependencies, then pass an absolute path from Downloads:

```powershell
uv sync --frozen --extra vision
uv run --frozen --extra vision python -m smartsite_ai.ingestion.video_smoke `
  "$env:USERPROFILE\Downloads\hazard_restricted_zone_test.mp4" `
  --stream-id hazard-demo --camera-external-id hazard-demo --max-frames 120

uv run --frozen --extra vision python -m smartsite_ai.ingestion.video_smoke `
  "$env:USERPROFILE\Downloads\morteza_ppe_test_video.mp4" `
  --stream-id ppe-demo --camera-external-id ppe-demo --max-frames 120
```

The command validates OpenCV open/decode, BGR24 frame envelope creation, EOF,
worker queue delivery, and cleanup. It does not run YOLO inference or claim PPE
or zone accuracy.

### Headless MF05/MF06 worker

`smartsite-ai-camera-worker` runs one configured video, laptop camera, or RTSP stream without an
open Web page. It fetches the Backend-owned region snapshot before loading the model or opening the
source, polls with ETag, applies newer geometry at a frame boundary, and writes every deliverable
event to a local SQLite outbox before sending it. A crash after Backend acceptance but before the
local acknowledgement safely resends the same event ID; the Backend's ingestion idempotency avoids
duplicating the business event.

The Backend must already contain the camera and active regions, and the camera UUID must be present
in its `AI_CONFIGURATION_CAMERA_IDS` allowlist. The AI process uses the same server-side service
credential for configuration and event ingestion:

```powershell
$env:SMARTSITE_AI_BACKEND_INGESTION_URL = 'http://127.0.0.1:3000'
$env:SMARTSITE_AI_BACKEND_SERVICE_TOKEN = '<local-service-token>'

uv sync --frozen --extra cuda126
uv run --frozen --extra cuda126 smartsite-ai-camera-worker `
  --source 'C:\SmartSiteData\recordings\ppe-demo.mp4' `
  --stream-id 'site-gate-01' `
  --camera-id '<backend-camera-uuid>' `
  --camera-external-id 'CAM-GATE-01' `
  --ppe-region-id '<active-ppe-region-uuid>' `
  --model-spec 'C:\SmartSiteData\config\yolo11s-ppe-artifact.json' `
  --outbox 'C:\SmartSiteData\runtime\smartsite-ai-outbox.sqlite3' `
  --target-fps 10
```

For a laptop webcam, use `--source 0 --live`. For credentialed RTSP, set
`SMARTSITE_AI_WORKER_SOURCE` and omit `--source`, then add `--live`; never commit camera credentials
or put them in command history. A finite video exits after EOF. Exit code `0` means the run ended
with no pending or terminal outbox entries; exit code `2` means events remain pending or a
non-retryable Backend response requires operator review.

### Multi-camera runtime

`smartsite-ai-camera-runtime` supervises one to three cameras declared in a strict manifest. It
loads one verified YOLO11s artifact and closes that runner once. Each camera keeps its own region
store, configuration poller, tracker, PPE temporal gate, outbox, and optional evidence directory.
Ultralytics inference is serialized: `UltralyticsYoloRunner.concurrent_inference_safe` is false, so
one `predict` runs at a time. A failed stream is recorded and does not stop a camera that is still
healthy. Credentialed RTSP belongs in an environment reference such as
`SMARTSITE_AI_CAMERA_YARD_02_SOURCE`; the manifest stores the variable name, not the URL.

Exit `0` means every camera finished with no pending or terminal outbox rows. Exit `2` means every
camera finished and at least one outbox still needs operator review. Exit `1` means a camera failed
during preflight or while running, including a camera left `not_started` after an earlier preflight
failure. The manifest shape is `examples/camera-runtime-manifest.example.json`.

```powershell
uv run --frozen --extra cuda126 smartsite-ai-camera-runtime `
  --manifest 'C:\SmartSiteData\config\camera-runtime.json'
```

### Source probe

`smartsite-ai-source-probe` checks one video or camera source before the runtime opens a model.
Pass the environment variable name, never the URI. The variable name must match
`SMARTSITE_AI_[A-Z0-9_]{1,80}`. `--max-frames` is 1..300 and `--timeout-seconds` is 1..120.
A successful run prints one JSON line with `status`, `framesRead`, `width`, `height`, and
`elapsedMs`. Failures print `error`, `code`, and an allowlisted `message`. Exit `0` is success,
`1` is a closed failure, and `130` is interruption. The installed command supervises the read in
a child process. The child inherits the environment and its command receives only the variable
name and numeric bounds. The parent waits for `--timeout-seconds` plus 1 second of startup
allowance. If the child is still alive, the parent terminates it, kills it when terminate does
not finish, and waits until that process has exited. A normal worker exit closes the capture.
Forced timeout or Ctrl-C ends the worker process so the operating system reclaims it; that path
does not gracefully release the capture. The parent repeats the child's stdout only when it is
exactly one allowlisted JSON object, and it never prints the child's stderr. A real OpenCV
source needs the `vision` extra; this command does not load a model or claim that a camera is
healthy beyond the frames it decoded.

```powershell
$env:SMARTSITE_AI_CAMERA_TAPO_SOURCE = '<rtsp-url>'
uv run --frozen --extra vision smartsite-ai-source-probe `
  --source-env SMARTSITE_AI_CAMERA_TAPO_SOURCE `
  --max-frames 30 `
  --timeout-seconds 15
```

Vision import and configuration smoke. This checks that the runner is not marked thread-safe and
that the example manifest parses. It does not download weights, open a camera, or measure GPU
capacity:

```powershell
uv sync --frozen --extra vision
uv run --frozen --extra vision python -c "from pathlib import Path; from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner; from smartsite_ai.runtime.manifest import load_manifest_bytes; assert UltralyticsYoloRunner.concurrent_inference_safe is False; manifest = load_manifest_bytes(Path('examples/camera-runtime-manifest.example.json').read_bytes()); assert len(manifest.cameras) == 2 and manifest.cameras[1].source.kind == 'env'"
```

This slice provides durable technical event production. It does not yet provide the Safety Alert
review screen or frame-perfect browser streaming.

Default development address: `http://127.0.0.1:8000`.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health/live` | HTTP process liveness |
| `GET /health/ready` | FastAPI lifecycle readiness |
| `GET /v1/capabilities` | Runtime AI capability status |
| `GET /docs` | Development Swagger UI |
| `GET /openapi.json` | Development OpenAPI schema |

Documentation endpoints are disabled in production.

## Configuration

Environment variables use the `SMARTSITE_AI_` prefix.

| Variable | Default | Description |
| --- | --- | --- |
| `SMARTSITE_AI_ENVIRONMENT` | `development` | Runtime environment |
| `SMARTSITE_AI_HOST` | `127.0.0.1` | API bind address |
| `SMARTSITE_AI_PORT` | `8000` | API port |
| `SMARTSITE_AI_LOG_LEVEL` | `info` | Logging level |
| `SMARTSITE_AI_BACKEND_INGESTION_URL` | unset | Backend origin or exact AI ingestion endpoint |
| `SMARTSITE_AI_BACKEND_SERVICE_TOKEN` | unset | Bearer credential for Backend ingestion |
| `SMARTSITE_AI_WORKER_SOURCE` | unset | Secret worker video/camera/RTSP source; CLI `--source` overrides it |
| `SMARTSITE_AI_REALTIME_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or a CUDA index such as `cuda:0` |

Invalid configuration prevents worker startup. The ingestion clients fail closed when their URL or
token is absent, while the FastAPI health/capability foundation can run without a camera worker.
Camera sources and verified local model artifact paths are supported by the explicit headless worker;
OpenAI credentials are not yet an implemented runtime capability.

## Vision Dependencies

Vision dependencies are isolated from the core API environment.

```sh
uv sync --frozen --extra vision
```

On an NVIDIA deployment compatible with CUDA 12.6 wheels, install the pinned GPU profile instead:

```sh
uv sync --frozen --extra cuda126
uv run --frozen --extra cuda126 python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`SMARTSITE_AI_REALTIME_DEVICE=auto` selects `cuda:0` only when this runtime reports CUDA as
available. An explicit unavailable CUDA device fails worker initialization rather than silently
falling back to CPU. The CUDA extra changes only the local/deployment environment; API imports and
startup still do not initialize a model or GPU.

The optional vision group currently pins Ultralytics and Supervision. YOLO11s selects the detector architecture/size, not a validated PPE checkpoint. GPU support must be validated against the selected PyTorch, CUDA, hardware, and trained weights before it is treated as a supported runtime. Ultralytics artifacts are AGPL-3.0 by default; a proprietary or commercial deployment must complete a license review before release.

## Local PPE reference-video validation

The annotated-video command provides a local, explicit validation path. It does not download a
model or video, and it requires an existing local model, exact SHA-256, HTTPS source URL, declared
license, and JSON class map. Acquire and verify the two smoke-only reference inputs as described in
[models/README.md](models/README.md), then install the optional dependencies and run:

```powershell
uv sync --frozen --extra vision
uv run --frozen smartsite-ai-detect-video --input recordings/ppe-reference.mp4 --model models/ppe-yolov8-reference.pt --model-artifact-id ansarimajid-construction-ppe-yolov8-reference --model-version 8139436e91aecb109362e13cacfea44a16e08358 --model-family yolov8 --model-sha256 5c981fd81432236cd6c88fa336697370f110383a62cc967f7759debf3c2b147e --class-map .cache/ppe-yolov8-reference.class-map.json --output runs/ppe-reference.annotated.mp4 --metadata-output runs/ppe-reference.run.json --model-source-url https://raw.githubusercontent.com/Ansarimajid/Construction-PPE-Detection/8139436e91aecb109362e13cacfea44a16e08358/Model/ppe.pt --model-license 'MIT (repository declaration; checkpoint terms unverified)' --confidence-threshold 0.25 --iou-threshold 0.45 --image-size 640 640 --device cpu
```

The command writes an annotated MP4 to `runs/ppe-reference.annotated.mp4` and atomic run metadata
to `runs/ppe-reference.run.json`. Both paths are ignored and must remain local. A successful local
run proves only that this checkpoint, class map, local clip, and runtime completed the bounded
video command. It does not establish the checkpoint's training-data provenance, license,
accuracy, PPE-policy validity, production suitability, YOLO11s performance, or GPU support. Run
with a GPU device only after `torch.cuda.is_available()` is true in the installed vision
environment, and record the resulting hardware/runtime evidence separately.

This YOLOv8 command is a **smoke reference only**. It proves that the local video, provider, and
annotation path can run; it is not the selected production model and its output must not be reported
as YOLO11s evaluation evidence.

## Official YOLO11s PPE evaluation gate

The selected gate evaluates a locally fine-tuned **official Ultralytics YOLO11s detector** against
versioned, representative PPE data. The command never downloads weights or datasets. Keep model
weights, dataset media, labels, episode indexes, provider data configuration, annotated evidence,
and run outputs outside Git. Commit only non-secret example contracts and reviewed aggregate
results when the team explicitly decides they belong in the repository.

Copy [examples/yolo11s-ppe-artifact.example.json](examples/yolo11s-ppe-artifact.example.json) to a
local ignored path and replace every placeholder. `artifactPath` must be an absolute path to the
fine-tuned `.pt` file, `sha256` must be the checksum of that exact file, and the public source URL
and license must describe the actual artifact. The example uses a Windows absolute path because the
artifact contract rejects relative paths; it does not refer to a bundled or downloadable model.

An evaluation run requires all of these existing local inputs:

- a dataset manifest plus the selected split index and referenced images/labels;
- the verified YOLO11s artifact specification;
- a ground-truth episode index for candidate-level alert scoring;
- a camera-region configuration and the UUID of its PPE region;
- a provider data configuration used for independent provider validation.

Create a new report directory name for every run. The CLI rejects an existing report directory so a
previous result cannot be silently overwritten.

The evaluation CLI starts a temporary Ultralytics runtime sandbox before importing the provider.
Ultralytics settings, Matplotlib/provider caches, temporary files, and local fallback fonts stay in
that sandbox; offline mode is enabled and auto-install is disabled. Independent provider validation
also writes run output only to a unique temporary workspace. Both locations are removed on success
and failure. The provider YAML must declare an absolute dataset root and resolve its selected,
training, and validation image directories inside one bounded, local, non-symlink dataset root.
Validation snapshots pre-existing `*.cache` files, removes only caches created by that invocation,
and discards the result if cleanup or the integrity of a pre-existing cache cannot be verified.

```powershell
uv sync --frozen --extra vision
uv run --frozen --extra vision smartsite-ai-evaluate `
  --dataset-manifest C:\SmartSite\local-data\ppe-evaluation-manifest.json `
  --artifact-spec C:\SmartSite\local-config\yolo11s-ppe-artifact.json `
  --episodes-index C:\SmartSite\local-data\indexes\test-episodes.jsonl `
  --region-configuration C:\SmartSite\local-config\ppe-evaluation-regions.json `
  --ppe-region-id f81d4fae-7dec-11d0-a765-00a0c91e6bf6 `
  --provider-data-config C:\SmartSite\local-data\data.yaml `
  --split test `
  --match-iou 0.50 `
  --report-dir C:\SmartSite\local-runs\yolo11s-ppe-test-2026-09-24 `
  --annotated
```

The gate must produce predictions, detection accuracy, candidate/episode metrics, and a summary from
the same verified artifact and dataset split. Annotated media is optional evidence and is generated
outside the timed evaluation path. Until a real fine-tuned artifact and the required local inputs
have completed this gate, SmartSite must describe YOLO11s as the selected architecture and training
target, not as a validated PPE checkpoint.

### Local YOLO11s PPE fine-tuning

Before fine-tuning, prepare the pinned local Roboflow **Construction Site Safety v27** YOLO export
with `smartsite-ai-prepare-ppe-dataset`. The command performs no network access and never mutates
the downloaded export. It validates the export's Roboflow metadata against the reviewed project
and version, requires exactly 2,603 train, 114 validation, and 82 test image/label pairs, decodes
every bounded image, validates every normalized YOLO row, and remaps only these five classes:

```text
source 5 Person          -> 0 Person
source 0 Hardhat         -> 1 Hardhat
source 2 NO-Hardhat      -> 2 NO-Hardhat
source 7 Safety Vest     -> 3 Safety Vest
source 4 NO-Safety Vest  -> 4 NO-Safety Vest
```

`Mask`, `NO-Mask`, `Safety Cone`, `machinery`, and `vehicle` annotations are counted and removed.
The metadata inside an export is self-declared and does not by itself authenticate its publisher.
Download version 27 from the linked official project page, then run `--inspect-only`. Review and
retain the printed SHA-256 locally; it pins the exact downloaded bytes. Preparation requires that
value and publishes a new directory only after validation succeeds. The result contains canonical
`data.yaml` plus `preparation.manifest.json` with every file hash, class/split counts, license,
attribution, and input/output aggregate hashes. Keep both source and prepared datasets outside Git
and retain attribution to
Roboflow Universe Projects with the [version 27 dataset page](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The provider page and archive README currently advertise 2,801 images, while the reviewed
YOLOv11 ZIP downloaded on 2026-09-25 contains 2,799 image/label pairs: 2,603 train, 114 validation,
and 82 test. The preparation gate uses the files actually present in that reviewed artifact. If a
later provider export changes these counts, treat it as a new source artifact and review it instead
of weakening the validation to make it pass. The reviewed ZIP checksum, extracted-content
aggregate, and actual split counts are recorded in
[`provenance/construction-site-safety-v27-yolov11.json`](provenance/construction-site-safety-v27-yolov11.json).
The generated `data.yaml` pins the absolute prepared-dataset root because Ultralytics resolves a
relative `path` through its machine-level dataset setting. Move the prepared dataset only by
running preparation again into the new destination so its manifest and data configuration remain
consistent.

```powershell
uv run --frozen smartsite-ai-prepare-ppe-dataset `
  --input-dir C:\SmartSite\local-data\construction-site-safety-27 `
  --inspect-only

uv run --frozen smartsite-ai-prepare-ppe-dataset `
  --input-dir C:\SmartSite\local-data\construction-site-safety-27 `
  --expected-source-aggregate <64-lowercase-hex-from-inspection> `
  --output-dir C:\SmartSite\local-data\smartsite-ppe-5class
```

Use the prepared directory's `data.yaml` as the training input below. Do not delete class names in
the provider YAML manually: the numeric IDs in every label file must be remapped as this command
does.

`smartsite-ai-train-ppe` fine-tunes an explicit local **official Ultralytics YOLO11s** base
checkpoint. It does not download weights or datasets. Keep the base checkpoint, dataset, labels,
training runs, and resulting weights in ignored local directories. The launcher requires absolute
paths, rejects an existing final run directory, resolves `auto` to an available CUDA device or CPU,
pins deterministic training, disables Ultralytics AMP so its safety check cannot download an
unreviewed auxiliary model, uses in-process data loading to avoid unstable Windows worker
subprocesses, and writes `training.manifest.json` atomically only after a non-empty
`weights/best.pt` exists. The manifest records the command, normalized configuration, requested
and resolved device, exact `data.yaml` and base-checkpoint SHA-256 values, runtime/GPU facts, Git
SHA and dirty state, verified prepared-dataset aggregate, and fine-tuned checkpoint SHA-256. It
recomputes every prepared file before and after provider execution; a changed, missing, extra,
linked, or unmanifested file fails the run. A failed or interrupted run has no `COMPLETE` manifest.

The input `data.yaml` must reference existing local `train`, `val`, and `test` inputs and already
use this exact class ID map:

```text
0 Person
1 Hardhat
2 NO-Hardhat
3 Safety Vest
4 NO-Safety Vest
```

The recommended source research dataset currently exports more classes. **Deleting extra entries
from `names` is invalid** because existing label IDs would then describe different objects. Prepare
and review a canonical five-class dataset by filtering/remapping both the labels and the class map
before using this launcher. Dataset acquisition, license review, and label transformation are
deliberately outside this command. [The example data configuration](examples/yolo11s-ppe-data.example.yaml)
shows only the required final shape.

After placing an official `yolo11s.pt` checkpoint and the prepared dataset outside Git, create a
new run name and execute:

```powershell
uv sync --frozen --extra cuda126
New-Item -ItemType Directory -Force C:\SmartSite\local-runs | Out-Null
uv run --frozen --extra cuda126 smartsite-ai-train-ppe `
  --data C:\SmartSite\local-data\ppe\data.yaml `
  --base-weights C:\SmartSite\local-models\yolo11s.pt `
  --output-root C:\SmartSite\local-runs `
  --name yolo11s-ppe-2026-09-25 `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --patience 30 `
  --seed 42 `
  --device cuda:0
```

Use `--device cpu` on a machine without supported CUDA. Training time and feasible batch size vary
by GPU and available VRAM; the RTX 4060 is a local validation target, not a runtime requirement.
After training, create the ignored artifact specification using the exact `best.pt` checksum from
the manifest, then run the evaluation gate above on the held-out test split. A completed training
manifest proves reproducibility of the run inputs and artifact identity; it does not by itself
prove PPE accuracy or production readiness.

Build the evaluation inputs from the verified prepared dataset instead of manually translating
YOLO labels. The converter re-verifies every prepared byte, requires the canonical five-class map,
copies media into a new self-contained local directory, converts normalized YOLO boxes into strict
evaluation annotations, and atomically publishes loader-validated train, validation, and test
JSONL indexes. `conversion.manifest.json` links the resulting evaluation aggregate to the exact
prepared-dataset aggregate and preparation manifest. Keep the output outside Git or under an
ignored path such as `.cache/`.

```powershell
uv run --frozen smartsite-ai-build-evaluation-dataset `
  --data C:\SmartSiteData\smartsite-ppe-5class-v27\data.yaml `
  --output-dir C:\SmartSiteData\smartsite-ppe-evaluation-v27 `
  --dataset-id construction-site-safety `
  --dataset-version 27-smartsite-5class-v1
```

The source dataset contains independent annotated images. It does not contain reviewed person
track identities or temporal missing-PPE episodes, so the converter does not invent an episode
index. Candidate-level alert scoring still requires a separately reviewed video episode JSONL via
`--episodes-index`; use the converted held-out images for detection metrics and provider validation.

### Reviewed video corpus for event-level PPE metrics

Build temporal evaluation inputs only from videos and labels that a person has reviewed. Copy
[`examples/ppe-video-corpus-source.example.json`](examples/ppe-video-corpus-source.example.json),
[`examples/ppe-video-reviewed-frames.example.jsonl`](examples/ppe-video-reviewed-frames.example.jsonl),
and [`examples/ppe-video-episodes.example.jsonl`](examples/ppe-video-episodes.example.jsonl) to an
ignored local directory, then replace every example value. Frame labels use stable
`personInstanceId` values throughout a clip and connect every PPE object to that frame's Person
annotation with `relatedPersonAnnotationId`. `observablePpeItems` records which body-area evidence
was actually reviewable; absence of a PPE box is not a missing-PPE label.

The builder runs no model inference and makes no network request. It verifies source hashes,
probes actual video dimensions/frame count/duration, decodes each selected frame to verify its
timestamp, and enforces the reviewed `maxFrameGapSeconds` cadence (at most one second). It rejects
orphan, conflicting, or duplicate-person relationships and requires every reviewed negative-PPE
frame to belong to exactly one non-overlapping ground-truth episode with no positive-PPE evidence
inside that interval. It copies the reviewed videos and labels into a new atomic, self-contained
evaluation directory and loads the result through the official dataset and episode loaders before
publishing `COMPLETE`.

```powershell
uv sync --frozen --extra vision
uv run --frozen --extra vision smartsite-ai-build-video-evaluation-corpus `
  --source-manifest C:\SmartSiteData\ppe-video-labels\source.json `
  --output-dir C:\SmartSiteData\ppe-video-evaluation-v1
```

The generated `corpus.manifest.json` records input hashes, reviewer metadata, evaluated camera
seconds, and rate-gate eligibility. The official false-candidate-rate gate requires at least 1,800
reviewed camera-seconds. A smaller corpus remains useful for smoke evaluation but is explicitly
marked ineligible. Tool validation proves structural consistency and traceability; it does not
replace a second-person review of the visual labels. Pass the generated
`evaluation.manifest.json` and `indexes/test-episodes.jsonl` to `smartsite-ai-evaluate`.

### DRAFT temporal PPE review package

Use the verified YOLO11s detector to reduce the first pass of manual video labeling. This command
samples and decodes real local frames at a cadence no greater than one second, resets provisional
tracking at every clip boundary, and exports original JPEGs, clearly marked overlay JPEGs, and a
strict `proposals.jsonl` worklist. It runs in the offline Ultralytics sandbox and never treats a
missing detection as proof that PPE is missing.

```powershell
uv run --frozen --extra cuda126 smartsite-ai-prepare-video-review `
  --input C:\SmartSiteData\review-source\ppe-shift-a.mp4 `
  --artifact-spec C:\SmartSiteData\runtime\yolo11s-ppe-artifact.json `
  --output-dir C:\SmartSiteData\review-packages\ppe-shift-a-draft `
  --cadence-seconds 0.5
```

The output directory must be new and either outside the repository or ignored by Git. Every
manifest and proposal row is permanently marked `DRAFT`; provisional track IDs are clip-local and
are not worker identities. The package deliberately omits reviewer identity, review time,
ground-truth annotations, and PPE episodes, so it cannot be passed to the official reviewed-corpus
builder. A person must correct the boxes, identities, PPE labels, and temporal episodes in a
separate review step before `smartsite-ai-build-video-evaluation-corpus` can accept the data. The
manifest records the clean Git commit and exact before/after hashes of the artifact spec,
checkpoint, and every source video. Failure or interruption removes the staged package.
The hidden `.smartsite-publication-owner` file is internal ownership metadata used only to make
interruption cleanup safe; it is not an annotation, review result, or ground-truth record.

### Shared-runner multistream benchmark

Measure the verified local YOLO11s detector with one shared, serialized model runner and one to
three concurrent local video replays. The command performs no rendering, output-video writing,
Backend calls, or network requests. Its JSON is a report-only baseline: `COMPLETE` means the run
finished and its provenance is intact; it does not mean a performance threshold passed.

```powershell
uv sync --frozen --extra cuda126
uv run --frozen --extra cuda126 smartsite-ai-benchmark-multistream `
  --input C:\SmartSiteData\benchmark\ppe.mp4 `
  --artifact-spec C:\SmartSiteData\runtime\artifact.json `
  --output C:\SmartSiteData\benchmark\yolo11s-multistream.json `
  --stream-counts 1 2 3 `
  --warmup-seconds 10 `
  --measurement-seconds 60 `
  --target-fps 10
```

Run the command from a clean Git commit; a dirty or unavailable repository is rejected so a
`COMPLETE` report always identifies the exact implementation. One input is replayed as independent
logical streams and is marked `syntheticConcurrentReplay: true` for the two- and three-stream
scenarios. Supply three distinct `--input` values to benchmark three different videos. The report
records processed/dropped frames, effective FPS, p50/p95/p99 decode-and-pack, queue, full detector,
pipeline and scheduled-replay-to-detection
latency, sampled RSS/CPU, synchronized Torch allocator VRAM, NVIDIA driver and hardware/runtime
versions, Git commit, and exact input/artifact/checkpoint hashes. A frame completed after the fixed
measurement window is counted as dropped instead of inflating throughput, and skipped schedules
also advance the replay source. The report records the actual execution time separately when a
slow inference drains beyond the measurement window. It deliberately does not claim
physical-camera-to-alert latency or total process VRAM outside the Torch allocator.

After a training run reaches `COMPLETE`, generate the artifact spec from its manifest. The command
verifies `weights/best.pt`, its exact path, size and SHA-256, the prepared dataset aggregate, and the
canonical class map. It requires an explicit public HTTPS artifact URL and an explicitly confirmed
license review; it does not upload the checkpoint or test the URL over the network. Use `--device
cpu` when evaluating on a CPU host; the default `trained` value reuses the resolved device recorded
by training.

```powershell
uv run --frozen smartsite-ai-build-artifact-spec `
  --training-manifest C:\SmartSiteData\runs\yolo11s-ppe-run\training.manifest.json `
  --output C:\SmartSiteData\local-config\yolo11s-ppe-artifact.json `
  --artifact-id smartsite-yolo11s-ppe `
  --source-url https://github.com/smartsite-sep490/model-releases/releases/download/v1/best.pt `
  --license AGPL-3.0-only `
  --license-reviewed
```

### Local MF05/MF06 UI test export

The local command can pass normalized YOLO batches through the technical MF05/MF06 pipeline and
write a timeline file consumed by the web UI. It stays local, does not call the Backend, and does
not decide an authorized/unauthorized result. The output paths below are ignored by Git.

```powershell
uv run --frozen --extra vision smartsite-ai-detect-video --input ..\smartsite\apps\web\public\assets\morteza_ppe_test_video.mp4 --camera-external-id ppe-demo --model models\ppe-yolov8-reference.pt --model-artifact-id ansarimajid-construction-ppe-yolov8-reference --model-version 8139436e91aecb109362e13cacfea44a16e08358 --model-family yolov8 --model-sha256 5c981fd81432236cd6c88fa336697370f110383a62cc967f7759debf3c2b147e --class-map .cache\ppe-yolov8-reference.class-map.json --output runs\ppe-ui.annotated.mp4 --metadata-output runs\ppe-ui.run.json --ui-timeline-output ..\smartsite\apps\web\public\assets\ppe-ai.timeline.json --region-configuration examples\ppe-ui-region-configuration.example.json --ppe-region-id f81d4fae-7dec-11d0-a765-00a0c91e6bf6 --model-source-url https://raw.githubusercontent.com/Ansarimajid/Construction-PPE-Detection/8139436e91aecb109362e13cacfea44a16e08358/Model/ppe.pt --model-license 'MIT (repository declaration; checkpoint terms unverified)' --device cpu

uv run --frozen --extra vision smartsite-ai-detect-video --input ..\smartsite\apps\web\public\assets\hazard_restricted_zone_test.mp4 --camera-external-id zone-demo --model models\ppe-yolov8-reference.pt --model-artifact-id ansarimajid-construction-ppe-yolov8-reference --model-version 8139436e91aecb109362e13cacfea44a16e08358 --model-family yolov8 --model-sha256 5c981fd81432236cd6c88fa336697370f110383a62cc967f7759debf3c2b147e --class-map .cache\ppe-yolov8-reference.class-map.json --output runs\zone-ui.annotated.mp4 --metadata-output runs\zone-ui.run.json --ui-timeline-output ..\smartsite\apps\web\public\assets\zone-ai.timeline.json --region-configuration examples\zone-ui-region-configuration.example.json --ppe-region-id f81d4fae-7dec-11d0-a765-00a0c91e6bf6 --model-source-url https://raw.githubusercontent.com/Ansarimajid/Construction-PPE-Detection/8139436e91aecb109362e13cacfea44a16e08358/Model/ppe.pt --model-license 'MIT (repository declaration; checkpoint terms unverified)' --device cpu
```

## Testing

Run the validation suite:

```sh
uv lock --check
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
uv build
```

Tests must not depend on paid external API calls, production credentials, live cameras, or model downloads.

## Docker

Build and run:

```sh
docker build -t smartsite-ai:dev .
docker run --rm --name smartsite-ai -p 127.0.0.1:8000:8000 smartsite-ai:dev
```

The runtime image installs only core runtime dependencies, runs as a non-root user, excludes datasets/model artifacts/environment secrets, and exposes a health check.

## MF05 — PPE Monitoring

```text
RTSP Frame
   -> Person / PPE Detection
   -> Tracking
   -> PPE Observation / Missing-PPE Signal
   -> Evidence
   -> Detection Event
   -> SmartSite Backend
   -> Safety Alert
   -> Safety Officer Verification
```

The initial PPE scope focuses on missing safety helmets and high-visibility vests. Exact model classes and weights must be validated against representative project data. Required-PPE business policy and violation determination belong to the SmartSite Backend.

## MF06 — Restricted-Zone Monitoring

```text
RTSP Frame
   -> Person Detection
   -> Tracking
   -> Configured Zone Entry
   -> Identity Candidate
   -> SmartSite Backend
   -> Zone Permission Evaluation
```

For zones prohibited to everyone, identity may not be required to generate a detection. For zones controlled by individual permission, identity and authorization remain separate. If identity or authorization cannot be resolved, the event remains unverified rather than becoming a false definitive conclusion.

## OpenAI Integration

OpenAI is a supplementary evidence-analysis capability. Its role includes assisting with visible-evidence description, PPE observations, Safety Officer review, and incident context.

OpenAI does not replace detection, tracking, worker identity, Zone authorization, or human verification. Original detections and evidence remain preserved even when OpenAI analysis is unavailable or disagrees.

## Model and Artifact Management

Model artifacts and datasets should not be committed directly to Git. Traceable model metadata should include model family/version, weights checksum, training/evaluation dataset versions, input resolution, confidence thresholds, benchmark hardware, and benchmark date.

See [models/README.md](models/README.md).

## Performance Validation

The current prototype target is an NVIDIA RTX 4060 with approximately 1–3 camera streams.

No FPS, latency, throughput, or accuracy claim is considered guaranteed until benchmarked using representative SmartSite footage. See [docs/feasibility.md](docs/feasibility.md).

## Backend Integration

### Database-backed face templates

The opt-in face demo no longer stores templates in a local file. Enrollment returns a Fernet-encrypted
template over the authenticated service endpoint; the Backend persists it in PostgreSQL together
with an explicit account/worker link. The AI runtime retains the encryption key, not DB credentials.
`SMARTSITE_AI_IDENTITY_TEMPLATE_STORE_PATH` is no longer used.

The existing verification endpoint also accepts authenticated JSON containing `jpegBase64` and
`templates` (`profileReferenceHash`, `encryptedTemplate`). The Backend selects active, site-scoped
account profiles from PostgreSQL. AI decrypts them transiently and returns only technical match
evidence. Corruption, wrong keys, incompatible dimensions and ambiguous matches fail closed.
JPEG requests remain supported for quality assessment, with no database candidates.

Deploy with the paired Backend account-template migration. Legacy local profiles require explicit
account linking and reenrollment; old files are preserved. Configure the same
`SMARTSITE_AI_IDENTITY_TEMPLATE_ENCRYPTION_KEY` and compatible model on every AI instance. Startup
still does not load or download a model; missing model files return unavailable during inference.

Integration notes live in [docs/integration.md](docs/integration.md).

The AI service emits technical detection evidence through `POST /api/v1/integrations/ai/events`. The client validates both request and acknowledgement contracts, retries only transport/408/429/5xx failures, and checks that the returned `eventId` matches the submitted event. The SmartSite backend is responsible for validating events, enforcing business rules, resolving access permission, deduplicating detections, creating Safety Alerts, and maintaining Incident lifecycle.

## Security and Privacy

AI and biometric data must be treated as sensitive. Important principles include secrets outside source control, least-privilege service authentication, encrypted transport, controlled evidence access, explicit biometric retention rules, auditability, and no identity assignment below the approved confidence policy.

## Related Repository

Main application platform: [smartsite-sep490/smartsite](https://github.com/smartsite-sep490/smartsite).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review conventions.

## Academic Context

SmartSite AI is developed as part of the SmartSite SEP490 capstone project. The service is currently a prototype and engineering platform; it must not be interpreted as a certified safety system or as a replacement for qualified safety personnel.
