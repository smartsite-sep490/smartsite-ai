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
- production detector-to-pipeline worker wiring;
- production tracker/model calibration and evaluation on representative site data;
- InsightFace integration;
- camera/inference event producer loop;
- OpenAI adapter;
- benchmark results.

The API explicitly reports `inference_ready: false` until a configured worker has a verified local
artifact, active camera source, and tested runtime; the local video command does not change API
readiness.

## Requirements

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
| `SMARTSITE_AI_REALTIME_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or a CUDA index such as `cuda:0` |

Invalid configuration prevents startup. The ingestion client fails closed when its URL or token is absent, but the FastAPI health/capability foundation can run before a camera worker is enabled. Camera credentials, model paths, and OpenAI credentials are intentionally not treated as implemented capabilities yet.

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
aggregate, actual split counts, and prepared-content aggregate are recorded in
[`provenance/construction-site-safety-v27-yolov11.json`](provenance/construction-site-safety-v27-yolov11.json).

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
pins deterministic training, and writes `training.manifest.json` atomically only after a non-empty
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
