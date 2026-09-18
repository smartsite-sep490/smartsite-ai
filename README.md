# SmartSite AI

SmartSite AI is the independent computer-vision and AI service for the SmartSite construction-site management platform.

The service is responsible for camera ingestion, visual detection, tracking, PPE monitoring, restricted-zone monitoring, identity experiments, evidence generation, and supplementary OpenAI analysis. It does **not** own construction-site business authorization.

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
| Detection | YOLO26 |
| Tracking | Supervision + tracker |
| PPE monitoring | Trained PPE detection weights |
| Zone monitoring | Tracking + configured geometry |
| Identity experiment | InsightFace / ArcFace |
| Evidence analysis | OpenAI API |
| Business authorization | SmartSite Backend |
| Business database | Neon PostgreSQL through Backend |

RF-DETR remains a comparison/fallback candidate. Roboflow Workflows and NVIDIA DeepStream have been researched but are not part of the initial implementation baseline.

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

**Current phase: FastAPI foundation.**

Implemented:

- FastAPI application factory;
- validated environment configuration;
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

- RTSP ingestion;
- GPU runtime configuration;
- YOLO inference;
- PPE model weights;
- tracking pipeline;
- Zone geometry processing;
- InsightFace integration;
- AI event producer;
- OpenAI adapter;
- benchmark results.

The API explicitly reports `inference_ready: false` until inference capabilities actually exist.

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

Invalid configuration prevents startup. Camera credentials, model paths, and OpenAI credentials are intentionally not treated as implemented capabilities yet.

## Vision Dependencies

Vision dependencies are isolated from the core API environment.

```sh
uv sync --frozen --extra vision
```

The optional vision group currently pins Ultralytics and Supervision. GPU support must be validated against the selected PyTorch, CUDA, hardware, and model versions before it is treated as a supported runtime.

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

The AI service emits technical detection evidence. The SmartSite backend is responsible for validating events, enforcing business rules, resolving access permission, deduplicating detections, creating Safety Alerts, and maintaining Incident lifecycle.

## Security and Privacy

AI and biometric data must be treated as sensitive. Important principles include secrets outside source control, least-privilege service authentication, encrypted transport, controlled evidence access, explicit biometric retention rules, auditability, and no identity assignment below the approved confidence policy.

## Related Repository

Main application platform: [smartsite-sep490/smartsite](https://github.com/smartsite-sep490/smartsite).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review conventions.

## Academic Context

SmartSite AI is developed as part of the SmartSite SEP490 capstone project. The service is currently a prototype and engineering platform; it must not be interpreted as a certified safety system or as a replacement for qualified safety personnel.
