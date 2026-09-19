# SmartSite AI Engineering Contract

This file is the mandatory engineering guide for people and coding agents working in the SmartSite AI repository. `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative. A passing build does not override these rules.

## 1. Read before changing code

1. Read the issue, its MF/FR/UC references when applicable, this file, `README.md`, `docs/feasibility.md`, and `docs/integration.md`.
2. Inspect the existing implementation, contracts, and tests before proposing a new pattern.
3. State the observable behavior and acceptance criteria before implementation.
4. Keep code and documentation consistent. Never claim a camera, GPU, model, benchmark, or external integration works without current evidence.
5. Ask for a product decision when requirements conflict. Do not silently invent identity, safety, or authorization rules.

Instruction priority is: explicit task requirements, this file, repository documentation, then nearby code conventions. Security and data-boundary rules remain mandatory.

## 2. Service responsibility and trust boundary

This repository owns technical camera processing for MF05 PPE Monitoring and MF06 Restricted Zone Monitoring:

```text
camera/video -> ingestion -> detector -> tracker/pipeline
             -> evidence/event builder -> SmartSite Backend
```

- FastAPI is the control/API surface; long-running camera and inference work belongs in explicit workers.
- YOLO11s is the selected detector baseline. RF-DETR Nano/Small and YOLO26s are optional benchmark challengers, not implementation blockers.
- AI emits observations and evidence. The Backend resolves Site/Zone context, permissions, business violations, alerts, incidents, and lifecycle state.
- A track ID is not a Worker ID. An identity prediction is only a candidate. Unknown identity is valid and must remain unknown.
- OpenAI is supplementary and asynchronous. It must not replace deterministic detection, identity, authorization, or Safety Officer decisions.

## 3. Module boundaries

Add modules only when implementing real behavior. Keep these responsibilities independent:

- `api/control`: health, readiness, capability, and future worker control endpoints.
- `ingestion`: source configuration, RTSP/video connection, decode, sampling, reconnect, and backpressure.
- `inference`: detector interface, YOLO11s adapter, artifact loading, preprocessing, and raw normalized detections.
- `tracking`: temporal association and stable per-stream track identifiers.
- `pipelines/ppe`: associate observable PPE with a person and produce technical PPE observations.
- `pipelines/zones`: evaluate tracks against configured region geometry and produce entry observations.
- `identity`: optional identity candidates and explicit unknown results; never permission decisions.
- `evidence`: approved frame/crop references and metadata; no business conclusion.
- `integrations`: authenticated Backend/OpenAI/storage clients and their transport policies.

Dependencies point inward. Domain models and protocols must not import FastAPI, Ultralytics, OpenCV, HTTP clients, or storage SDKs.

## 4. Camera and concurrency rules

- Camera ingestion must not import or construct a YOLO model. It publishes a small typed frame envelope to a consumer boundary.
- Detector adapters consume frames and return normalized detections. They must not own RTSP reconnect, Zone authorization, or Backend persistence.
- Every frame includes stream/session identity, camera external ID, capture time, dimensions, and a monotonic sequence where available.
- Bound queues and define the overload policy. Live monitoring should drop stale frames according to policy rather than grow memory without limit.
- Reconnect uses bounded exponential backoff with jitter and cancellation. A failed camera must not terminate unrelated streams or the API process.
- Blocking decode/inference must not block the FastAPI event loop.
- Resource ownership is explicit: create, start, stop, drain, and release streams/models deterministically.
- Never open a live camera, download a model, or initialize GPU state during module import or FastAPI application startup. Enabled inference starts under an explicit worker lifecycle after configuration validation; the API observes and reports that worker's state.

## 5. Model and artifact rules

- Model artifacts are configured by local/runtime path with version, source, checksum, license, class map, and evaluation reference.
- Do not commit `.pt`, `.onnx`, `.engine`, embeddings, or other generated model artifacts.
- Do not rely on implicit network downloads. Missing or mismatched artifacts make the inference worker/capability unavailable with a safe actionable reason. Base API readiness remains independent, unless a deployment profile explicitly declares that worker as a required dependency and exposes that requirement through a reviewed readiness contract.
- Keep preprocessing, image size, thresholds, class mapping, and postprocessing versioned with the model configuration.
- A generic COCO YOLO11s checkpoint is not a validated PPE model. MF05 requires trained/evaluated classes for the agreed PPE scope.
- Do not infer that absent PPE means a violation when the relevant body area is unobservable. Omit or retain an internal unknown result; the current external PPE contract only permits reviewed `PRESENT` or `MISSING` observations.
- Compare models on the same held-out site/video split. Report per-class precision/recall, missed violations, false alerts per camera-hour, p95 end-to-end latency, FPS/stream, dropped frames, RAM/VRAM, and exact hardware/runtime.
- Ultralytics artifacts are AGPL-3.0 by default. Record the artifact license and complete the release license review before proprietary/commercial deployment.

## 6. MF05 PPE rules

- Associate PPE with the correct person; detecting a vest or helmet somewhere in the frame is insufficient.
- Keep configured region ID and geometry version with observations so the Backend can validate context.
- Handle occlusion, partial visibility, small objects, and multiple people explicitly in evaluation.
- Temporal confirmation, cooldown, and deduplication policy must be deterministic and tested. The AI service produces evidence; the Backend owns durable alert grouping and final state.
- Store only the evidence required by the approved retention policy. Prefer references/metadata in events rather than embedding large images.

## 7. MF06 Zone and identity rules

- Zone processing uses configured region geometry and version. Do not invent or trust a business `zoneId` from model output.
- Emit technical entry/track observations. Do not emit `allowed`, `denied`, or final violation state from the AI service.
- The Backend decides among allowed, prohibited, unauthorized, and authorization unavailable using current business data.
- Identity and tracking are separate. Track continuity does not prove identity, and one identity result must not be attached to a distant/new track without evidence.
- Thresholds and enrollment artifacts require evaluation. Below threshold, conflicting, unavailable, or poor-quality results remain unknown.
- Never match by clothing, previous attendance, camera presence, or other unsupported shortcuts.

## 8. Contracts and Backend integration

- `contracts/` is a vendored, immutable copy of the Backend-owned machine-readable contract with provenance metadata.
- Never hand-edit the vendored schema, golden vectors, or provenance to make a local test pass.
- Contract updates use two phases because squash merge changes the source commit SHA. First merge the canonical contract PR in `smartsite`; then vendor the exact bytes from the resulting immutable `main` SHA, update provenance/golden vectors, run paired checks, and merge the AI PR.
- Linked PRs may be developed and reviewed in parallel, but provenance is finalized only after the canonical Backend PR merges. Breaking changes require a new version and a staged compatibility/deployment plan.
- Preserve canonical hashing behavior across Python and TypeScript. Numeric ranges must remain interoperable with JSON, JavaScript, JCS, and PostgreSQL.
- Events are authenticated, idempotent, bounded in size, and retryable according to the integration policy.
- Retry only classified transient failures, use bounded attempts/backoff, and never retry a semantic conflict as if it were transient.
- Do not log service tokens, signed URLs, raw credentials, face embeddings, or unrestricted evidence.

## 9. Python engineering rules

- Use Python 3.12 and the pinned `uv` environment/lockfile.
- Public and cross-module interfaces require precise type hints. Use Pydantic at external boundaries and focused dataclasses/protocols for internal domain interfaces where appropriate.
- Prefer small pure functions for geometry, mapping, filtering, and policy-free transformations.
- Use dependency injection for clocks, transports, model adapters, and external services when behavior must be tested.
- Do not catch `Exception` without either re-raising, translating at a boundary, or recording safe actionable context. Never silently continue after corrupted state.
- Cancellation is control flow. Workers must propagate cancellation and release resources.
- Use structured logging with event/camera/session identifiers. Do not log every frame or high-cardinality payload by default.
- Keep import side effects minimal. Importing a module must not open files, network connections, cameras, model weights, or GPU contexts.
- Avoid generic managers, global registries, deep inheritance, and speculative plugin systems. Introduce an abstraction only for an active boundary with at least one real implementation/test seam.
- New runtime dependencies require a documented reason, compatible license, pinned version, and lockfile update.

## 10. Security and data handling

- Secrets belong in runtime environment or an approved secret manager. Commit only fake `.env.example` values.
- Never commit real camera recordings, evidence, face images, embeddings, restricted datasets, or model weights.
- Use synthetic or explicitly permitted fixtures and remove identifying metadata.
- Validate URLs, schemes, file paths, payload sizes, media type, and ownership at boundaries. Avoid server-side request forgery and path traversal.
- OpenAI calls stay in a server-side adapter, are bounded by time/size/cost, and are never made by automated tests.
- A model result cannot bypass Backend authentication, scope checks, or human verification.

## 11. Testing and quality gates

Use the smallest meaningful test first, then run the complete suite. Tests verify behavior and failure modes, not private implementation details.

Before a PR, run:

```sh
uv lock --check
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
uv build
```

- Bug fixes require a regression test that fails before the fix.
- Camera code requires deterministic fake sources for reconnect, timeout, cancellation, sampling, and backpressure tests.
- Inference adapters require fake model adapters for unit tests; real-model/GPU tests are a separate marked validation layer.
- Contract changes require anti-drift, schema, canonical hash, and paired Backend integration checks.
- External clients require timeout, authentication, retry classification, idempotency, invalid response, and secret-redaction tests.
- Tests must not require production credentials, paid APIs, live cameras, network model downloads, or a GPU unless explicitly marked and excluded from the default CI suite.
- Report hardware validation with exact device, driver, CUDA/Torch/runtime, artifact checksum, input, duration, and measurements.
- Changes to vision/inference code must also run `uv sync --frozen --extra vision` and a documented import/configuration smoke test. GPU execution remains a separately reported hardware validation when unavailable in CI.

## 12. Git and parallel work

- Start from current `main`; do not commit directly to `main`.
- One issue describes one reviewable behavior with MF/FR/UC references when applicable, scope, acceptance criteria, and contract impact.
- Use short-lived branches such as `feat/camera-worker`, `feat/yolo11s-adapter`, or `fix/backend-retry`. Coding agents may use their required prefix.
- One person owns a branch. Never share an uncoordinated working tree.
- Keep PRs focused and use Conventional Commit-style messages.
- Do not force-push shared branches, bypass failing checks, merge without required review/authorization, or delete branches unless the team explicitly requests cleanup.
- Shared hotspots require coordination: `contracts/`, `uv.lock`, configuration, CI, public domain types, and application startup.
- Cross-repository work uses the same issue identifier and linked PRs. State the required merge/deploy order.

For parallel MF05/MF06 work, Camera Worker owns ingestion/reconnect/backpressure, while YOLO11s Adapter owns artifact loading/preprocessing/inference/normalization. Agree on typed frame and detection boundaries first. Neither task edits the other's implementation without coordination.

## 13. Definition of Done

A change is done only when all applicable statements are true:

- Acceptance criteria and technical boundaries are implemented.
- The service still reports unavailable capabilities honestly when camera/model/GPU configuration is absent.
- Contract, producer behavior, Backend expectations, and documentation agree.
- Happy path and material failure paths have meaningful tests.
- Required local checks and CI pass with no unexplained failure.
- No secret or sensitive/model artifact is present in the diff or Git history.
- Shutdown, cancellation, bounded retries/queues, logging, and recovery are covered where applicable.
- The PR states what was verified and what still requires camera, GPU, dataset, credentials, or deployment.
- Another team member reviews the change.

## 14. Coding-agent rules

- Read this entire file before editing. If the AI tool does not automatically load `AGENTS.md`, attach or paste it into the task context.
- Inspect code and tests; do not guess APIs, package versions, model behavior, or hardware capability.
- Follow existing conventions unless the issue explicitly changes them.
- Do not broaden scope, rewrite unrelated code, create placeholder layers, or generate speculative abstractions.
- Do not suppress type, lint, contract, or test errors merely to obtain a green check.
- Do not report completion from code inspection alone. Run the required commands and cite actual results.
- AI output always requires human review. Never provide production credentials or unrestricted personal/site data to a coding assistant.
