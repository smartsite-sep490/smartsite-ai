# SmartSite Vision Pipeline Contracts v1

Status: **Approved design baseline for parallel implementation**  
Date: 2026-09-20  
Scope: MF05 PPE Monitoring and MF06 Restricted Zone Monitoring

## Purpose

This document freezes the three boundaries that must remain stable while the team implements
video ingestion, YOLO11s inference, Backend camera/region configuration, tracking, PPE, and zone
pipelines in parallel.

The boundaries are deliberately technical. The AI service observes frames, detections, tracks,
PPE, and region entries. The Backend owns Site/Zone records, current authorization, violation
decisions, alert lifecycle, and durable persistence.

## Contract ownership

| Contract | Canonical owner | Consumers |
| --- | --- | --- |
| `FrameEnvelope` | `smartsite-ai/ingestion` | detector adapters and evidence pipeline |
| `NormalizedDetection` / `DetectionBatch` | `smartsite-ai/inference` | tracking, PPE, and zone pipelines |
| `CameraRegionConfiguration` wire schema | `smartsite` Backend | `smartsite-ai` configuration client |
| `TechnicalObservationEvent` | existing `smartsite/contracts` schema | Backend ingestion and `smartsite-ai` producer |

The existing `TechnicalObservationEvent` v1.0.0 remains unchanged. A shared wire-contract change
starts in `smartsite`, is merged there, and is then vendored byte-for-byte into `smartsite-ai`.

## 1. FrameEnvelope v1

`FrameEnvelope` represents one immutable decoded frame. Version 1 uses packed OpenCV-compatible
BGR bytes so Video File Source and RTSP Source produce the same input without JPEG encode/decode.

```python
class FrameEnvelope:
    stream_id: str                    # 1..128 characters; runtime stream instance name
    session_id: UUID                  # new UUID for every successful source connection/open
    camera_external_id: str           # 1..128; matches Backend Camera.externalId
    captured_at: datetime             # timezone-aware; normalized to UTC
    width: int                        # 1..16384
    height: int                       # 1..16384
    sequence_number: int              # 0..2^63-1; strictly increasing inside one session
    pixel_format: Literal["BGR24"]
    payload: bytes                    # exactly width * height * 3 bytes
```

Required invariants:

- The model is frozen and forbids unknown fields.
- The source transfers payload ownership. Queued bytes cannot refer to a mutable/reused OpenCV
  buffer.
- `session_id` is a UUID string on serialization because the existing Backend event contract
  requires `streamSessionId` to be a UUID.
- A source creates a new `session_id` when a file is opened or a live source reconnects. Sequence
  starts at zero and strictly increases for accepted frames in that session.
- `captured_at` for a video file is derived from the run start plus media presentation timestamp.
  RTSP uses the best source timestamp available and otherwise a documented UTC capture clock.
- Version 1 does not accept path strings, arbitrary descriptors, RGB guessing, encoded JPEG, NV12,
  CUDA tensors, or mutable arrays. A future format requires an explicit versioned extension.
- The ingestion worker may drop stale envelopes under load, so consumers must tolerate gaps in
  `sequence_number`.

This tightens the current foundation, whose `session_id` and `bytes | str` payload validation are
too permissive for the Backend UUID contract and for deterministic image decoding. The tightening
must merge before Video Source and detector implementation branches.

## 2. NormalizedDetection and DetectionBatch v1

The detector returns technical model output. It does not assign tracks, associate PPE with a
person, decide that missing PPE is a violation, identify a worker, or decide Zone permission.

```python
class NormalizedBoundingBox:
    x1: float                         # 0.0 <= x1 < x2 <= 1.0
    y1: float                         # 0.0 <= y1 < y2 <= 1.0
    x2: float
    y2: float
    coordinate_space: Literal["NORMALIZED_0_1"]

class NormalizedDetection:
    class_id: int                     # >= 0; artifact-specific class index
    class_name: str                   # 1..128; resolved from pinned artifact class map
    confidence: float                 # 0.0..1.0
    bounding_box: NormalizedBoundingBox

class DetectionBatch:
    stream_id: str
    session_id: UUID
    camera_external_id: str
    captured_at: datetime             # copied from FrameEnvelope
    frame_width: int
    frame_height: int
    sequence_number: int
    model_artifact_id: str            # stable configured artifact ID
    model_version: str
    model_sha256: str                 # lowercase 64-character hex digest
    detections: tuple[NormalizedDetection, ...]
```

Required invariants:

- All models are immutable and forbid unknown fields.
- The adapter clips finite model coordinates to the frame boundary, converts them to normalized
  coordinates, and rejects non-finite or zero-area boxes.
- Empty `detections` is a valid successful result.
- Detection order is deterministic: descending confidence, then class ID, then box coordinates.
- `class_name` comes only from the configured and verified artifact class map. Generic COCO
  YOLO11s is not claimed to be a validated PPE model.
- Model metadata is recorded once on the batch, not duplicated on every detection.
- The YOLO adapter accepts `FrameEnvelope.pixel_format == "BGR24"`; unsupported formats fail with
  a typed safe error.
- Loading and inference are explicit lifecycle operations and run outside the FastAPI event loop.
  Importing modules and starting the API cannot download or initialize a model/GPU.

The detector protocol is provider-neutral. `Yolo11Detector` implements it through an injected
runner/model factory so unit tests use a fake runner and default CI requires no weights, GPU, live
camera, OpenCV window, or network download.

## 3. CameraRegionConfiguration snapshot v1

The Backend publishes camera-specific observation geometry. The wire contract uses `regionId`
because the existing event contract and `camera_observation_region` entity resolve observations by
`(cameraExternalId, regionId, geometryVersion)`. The AI service must never receive or emit a
business permission decision.

```json
{
  "schemaVersion": "1.0.0",
  "cameraExternalId": "CAM-GATE-01",
  "regions": [
    {
      "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
      "geometryVersion": 1,
      "coordinateSpace": "NORMALIZED_0_1",
      "polygon": {
        "coordinates": [[0.10, 0.15], [0.80, 0.15], [0.80, 0.90], [0.10, 0.90]]
      }
    }
  ]
}
```

Field rules:

- `schemaVersion` is exactly `1.0.0`.
- `cameraExternalId` is 1..128 characters and identifies an existing active Backend camera.
- `regionId` is the UUID primary key of `camera_observation_region`, not `zone.id`.
- `geometryVersion` is an integer >= 1 and increments whenever polygon geometry or coordinate
  space changes.
- `coordinateSpace` is exactly `NORMALIZED_0_1`.
- `polygon.coordinates` contains at least three distinct non-collinear `[x, y]` vertices. Each
  coordinate is finite and inside `[0, 1]`; origin is the frame's top-left, `x` increases right and
  `y` increases down. The first vertex is not repeated at the end; consumers close the polygon.
  Adjacent duplicates, zero-area polygons, holes, and self-intersections are rejected in v1.
- The response is an atomic full snapshot containing active regions only. A disabled or deleted
  region disappears from the next snapshot; consumers replace the prior snapshot atomically.

The contract deliberately excludes `siteId`, `zoneId`, `ZoneRestrictionPolicy`, required PPE,
allowed/denied state, worker authorization, and alert state. The Backend retains those values and
maps AI observations back to the current Site/Zone context. MF05 configuration for which technical
PPE classes a particular model can observe belongs to the model/pipeline configuration, not this
Zone geometry contract.

`regionId` remains stable for one logical camera projection. Geometry edits increment
`geometryVersion` atomically. Moving the projection to another camera or associating it with a
different business Zone creates a new `regionId`. Management writes must transactionally ensure
that the Camera and Zone belong to the same Site. The current database stores only the latest row,
so stale in-flight observations fail closed; retained geometry history is a later operational
feature rather than an implicit fallback.

## Data flow

```text
Video file / RTSP source
  -> FrameEnvelope
  -> Detector.detect(frame)
  -> DetectionBatch[NormalizedDetection]
  -> tracker
  -> PPE pipeline / region-entry pipeline using CameraRegionConfiguration
  -> existing TechnicalObservationEvent v1.0.0
  -> Backend context resolution, authorization, alerts, incidents
```

## Error and lifecycle rules

- Invalid frames, artifact metadata, detections, or region configurations fail at their boundary;
  callers never receive a partially valid model.
- One bad camera or frame cannot terminate unrelated streams.
- A stale `(regionId, geometryVersion)` is never silently mapped to the current geometry.
- Unknown detector classes are retained as artifact classes but ignored by pipelines lacking an
  explicit semantic mapping.
- Missing model weights/checksum mismatch makes detector capability unavailable with a safe reason;
  API liveness remains independent.
- Logs may contain stream/session/camera/region/model artifact identifiers, but never RTSP
  credentials, signed URLs, raw frame payloads, unrestricted evidence, or face embeddings.

## Parallel implementation boundaries

1. **Contract hardening PR first:** update `FrameEnvelope` to this v1 contract and add the inference
   domain models/protocol without Ultralytics imports.
2. **YOLO11s branch:** owns `smartsite_ai/inference/`; consumes the frozen envelope and returns
   `DetectionBatch`.
3. **Video/RTSP source branch:** owns `smartsite_ai/ingestion/adapters/`; produces the frozen
   envelope and does not import YOLO.
4. **Backend camera/region branch:** owns the canonical configuration JSON schema, validation,
   persistence/API, and mapping from Backend entities to `CameraRegionConfiguration`.
5. **Pipeline branch:** owns `tracking/` and `pipelines/`; tests against fake `DetectionBatch` and
   configuration fixtures, without importing Ultralytics or opening media sources.

Only the contract-hardening owner edits shared domain models. Changes to these contracts require a
separate reviewed PR and coordinated rebases. `pyproject.toml`, `uv.lock`, application startup,
vendored contracts, and CI are shared hotspots with one owner at a time.

## Acceptance criteria before parallel coding

- Python and JSON-schema models express every invariant above and reject unknown fields.
- Golden examples serialize identically across Python and TypeScript where the contract crosses
  repositories.
- The current Backend event schema remains byte-identical.
- Core tests run without the vision extra, weights, GPU, network, or production credentials.
- The contract PR passes both repositories' required checks and is reviewed before the four feature
  branches start implementation.
