# Tích hợp SmartSite Backend

## Opt-in empty LIVE read budget

`OpenCvFrameSource` defaults to one native read. An explicitly composed worker may
set `live_read_attempts=3` and `live_read_retry_delay=0.1` to tolerate transient
empty reads on the **same open capture**. Attempts are bounded to 1–5 and delay is
finite in (0, 1] seconds. No network/camera reopen, new session or synthetic frame is
created. Real frames retain their actual capture time and consecutive sequences.
Replay EOF remains immediate; native exceptions and corrupt frames still fail.
Exhaustion returns `None` through the existing worker failure path. Cancellation
propagates, and native reads still drain before release. The retry count/backoff
does not bound the duration of a blocked native read; that existing limitation
requires process-level supervision. No worker opts in automatically, and this is
not evidence of physical-camera, reconnection or production acceptance.

Nguồn chuẩn: `contracts/schemas/v1/technical-observation-event.json` tại commit bất biến ghi trong `contracts/metadata.json` của repository `smartsite`.

Phiên bản event contract đang dùng: **1.0.0**. Schema và golden vectors được vendor nguyên bytes; CI tải đúng source commit và fail nếu có drift.

## Contract foundation đang chạy

Service mặc định ở `127.0.0.1:8000`; trong Compose, các container cùng mạng dùng `http://ai:8000`. Backend consumer hiện là `POST http://backend:3000/api/v1/integrations/ai/events`, xác thực bằng Bearer service token.

- `GET /health/live`: 200, `{"status":"ok","service":"smartsite-ai"}`.
- `GET /health/ready`: 200 sau startup, `{"status":"ready","service":"smartsite-ai","scope":"api","inference_ready":false}`. Ngoài lifecycle trả 503 với `status: "not_ready"`.
- `GET /v1/capabilities`: `api_version: "v1"`, `inference_ready: false`, `capabilities` gồm camera, detector, zone, identity, openai. Mỗi capability có `status: "not_configured"`, `provider` và `reason`. Các ghi chú không chứa secret hoặc model path.

Không dùng API readiness để bật nghiệp vụ PPE/Zone. Headless/multi-camera worker và inference có lifecycle opt-in riêng; không tự mở nguồn hay tải model khi API khởi động. Readiness API không chứng minh worker, camera hoặc chất lượng model đã được nghiệm thu. OpenAPI development mô tả các HTTP endpoint hiện có.

Mọi thay đổi contract phải bắt đầu ở repo `smartsite`, commit schema trước, sau đó vendor bytes và cập nhật source SHA/checksums ở repo này. Không sửa schema vendored trước.

## Explicit Camera PPE policy bundle v2

The default legacy event and configuration routes are unchanged. The separately opted-in
`BackendConfigurationClient(policy_v2=True)` reads `configuration-bundle/v2` and requires
its ETag to match the validated immutable bundle hash. `CameraPolicyConfigurationBundle`
preserves the Camera default, Region INHERIT/EXPLICIT/DISABLED policies, geometry and Zone
provenance without changing signed wire content. Schema/example provenance is recorded
separately in `contracts/metadata.json`; CI compares the exact bytes with the immutable
Platform commit, including the existing referenced v1 Region schema.

Consumers must explicitly select `RegionConfigurationStore(allow_policy_v2=True)` and
`CameraConfigurationPoller(require_bundle=True)`. Default stores refuse policy v2. Once v2
is installed, a legacy bundle cannot replace it. Installation retains policy and geometry
as one snapshot. `resolve_ppe_policy` accepts the supplied containing Region IDs; it does
not establish geometric containment, wearing, identity, or a business violation. An empty
selection applies the Camera policy without fabricating a Region.

The poller's existing definitive generation ACK, freshness/expiry, serialized retry,
in-flight snapshot checks and finite replay draining remain in force for v2. Delivery
context source kind remains an explicit caller selection under the existing runtime
boundary; the decoder does not prove a source is live. A receipt confirms configuration
loading, not detection accuracy or authorization. Decoder/storage alone do not activate
a producer. The explicit v2 manifest/worker and tracked v1.7 encoder are described below;
camera/model activation, qualification and service deployment remain separate gates.

## Thiết kế dự kiến

- Backend cấp cấu hình camera/Zone theo phạm vi AI được phép.
- AI quan sát video, tạo ID sự kiện ổn định và tham chiếu bằng chứng.
- Backend xác thực producer, kiểm tra payload và chống trùng khi AI retry.
- Backend quyết định quyền vào theo identity, Zone và thời gian; AI không truy cập trực tiếp database nghiệp vụ.
- Mất quyền truy cập hoặc không định danh được phải có trạng thái riêng.

Client có timeout riêng cho connect/read/write/pool; retry tối đa ba lần chỉ với lỗi transport, 408, 429 và 5xx. 4xx nghiệp vụ không retry. Mọi request dict được Pydantic validate trước khi gửi; acknowledgement phải có status hợp lệ, `alertIds` và cùng `eventId`.

## Local diagnostic camera preview

The authenticated `/ws/realtime` demo endpoint sends `previewVersion: 1` messages containing
the exact inference frame as `imageDataUrl` (JPEG), its detections, and the configured restricted
`zonePolygons`. Each message includes `cameraExternalId`, `sessionId`, decimal-string
`sequenceNumber`, `capturedAt`, and encoded image `width`/`height`. JPEG output is bounded to
1280 pixels per dimension and 1 MiB before Base64 encoding. Normalized boxes and polygons use
the same image coordinate space. Empty detection frames still carry pixels and source identity.

Clients must display the image and its boxes together, bound image decoding, and discard stale
work after disconnect/session changes. A separately playing MP4 cannot be used as the background
for realtime boxes. These transient diagnostic pixels do not replace authenticated retained
evidence or change `TechnicalObservationEvent` v1.0.0. The controlled laptop/Tapo profiles remain
exclusive to the durable worker; shared live-camera preview fan-out is not implemented here.
Model/artifact initialization runs in a background thread after the client connects, so it does
not occupy the API event loop. Cancellation preserves ownership of the loading task and closes
a model that finishes loading after cancellation, including repeated cancellation requests.
Provider construction serializes its process-wide offline guards with a threading lock:
a replacement connection cannot restore download/auto-install defaults while a cancelled
connection's native model loader is still constructing a checkpoint.

Cancelling a YOLO detection waits for its already-started native prediction to finish before
propagating cancellation. This preserves the shared inference lane and prevents single-camera,
multicamera, or preview cleanup from closing the runner while prediction is active. Repeated
cancellation does not abandon that work; a late provider error is consumed while cancellation
remains the caller outcome. This is graceful ownership, not a hard shutdown timeout: a permanently
hung native provider still requires process-level recovery.

## Release

### Explicit tracked Camera/Region PPE producer

The multi-camera `smartsite-ai-camera-runtime` manifest has an optional per-camera
`trackedV17` object. This selects policy-bundle v2, tracked event v1.7 and the tracked
part/result endpoints together. Defaults and the existing `partitionedFrameDelivery`
and `generationBoundEventDelivery` profiles remain unchanged. The new option cannot
be combined with either legacy delivery flag.

```json
{
  "trackedV17": {
    "siteId": "00000000-0000-4000-8000-000000000077",
    "profileSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "pendingSeconds": 30,
    "predecessorWaitSeconds": 1,
    "maxPending": 2,
    "maxBytes": 100000
  }
}
```

This is a synthetic fragment, not a deployable or qualified model profile. A real
entry additionally requires `live: true`, private absolute `evidenceDir` and
`generationCredentialFile` paths, the existing source/outbox fields, and a Backend
profile/Camera allowlist that matches the approved producer. The profile digest must
come from release qualification, not be invented to bypass a Backend rejection.
Generation/session claims, installed-bundle identity, ACK freshness and source
cancellation remain enforced. No model, camera or inference starts on import.

A tracked generation is pinned to one immutable bundle/view and producer scope.
Its poller refuses to install or acknowledge a changed bundle, invalidates its
cached delivery context and reports `TrackedScopeTransitionRequired`. The worker
and pipeline also refuse cross-scope continuation, retirement or model/session/
dimension changes. Receipt refresh for the identical bundle remains valid. A
trusted Backend source-transition owner must finish the prior scope and authorize
the replacement source before new settings can be used. Automatic source handoff
for live configuration changes is a separate integration gate still open; do not
reset tracker IDs or re-label old captures to bypass it. A saved policy is not
proof that the stopped worker applied that policy.

The producer applies Camera PPE outside all configured polygons. Inside a polygon,
the issued Region INHERIT/EXPLICIT/DISABLED policy applies; DISABLED does not fall
back to Camera PPE. Temporal confirmation is bound to the Region geometry or Camera
configuration and cannot borrow a streak across those boundaries. Tracked delivery
also requires the candidate's exact session and captured instant; old manually
constructed legacy candidates are not valid tracked confirmations.

Each complete frame's original wire bytes and expected subject references enter
SQLite atomically. A `RAW_STORED` receipt confirms storage only. A continuation can
use the prior frame only after its exact `ADMITTED_TRACKED_FRAME` result is checked.
Unresolved/expired work does not become a predecessor or a successful source drain.
Final-source draining uses the earliest outstanding frame's original stored expiry,
not a fresh lease and not the shorter predecessor-polling timeout.
Evidence keeps the original frame/part identity and JPEG SHA. Empty captured frames
may retain lifecycle data; failed inference is not an empty safe scene.

This tracked worker mode requires LIVE admission. Existing legacy REPLAY handling
is preserved; Backend review-only replay receipts must not be treated as operational
admitted seals or real safety notifications. Backend rollout must coordinate its
explicit v1.7 profile and forward schema fence; a v1.6 runtime cannot be used as a
mixed-version canary against a widened v1.7 schema. Local unit/mock transport proofs
do not qualify a model, physical camera, production rollout or end-to-end alert flow.

AI có version/image riêng. Mỗi release ghi model version, cấu hình có ảnh hưởng, phiên bản contract và Backend đã kiểm thử cùng. Thay model không mặc nhiên đồng nghĩa API thay đổi; cần chạy lại đo chất lượng và ca nghiệm thu.

## Shared capture preview and setup images (explicit runtime opt-in)

`smartsite-ai-camera-runtime --manifest /absolute/private/runtime.json --preview-port 8123`
opts into the media host. Without `--preview-port`, the runtime keeps its previous inference-only
construction. Configure `SMARTSITE_AI_PREVIEW_SERVICE_TOKEN` server-side to match the Backend's
`SMARTSITE_AI_SERVICE_TOKEN`. The token is never a browser credential or a CLI argument.
The host uses `SMARTSITE_AI_HOST` (default127.0.0.1). Its socket is bound before runtime preflight
or model/camera activation, so a port conflict cannot leave a second camera owner running.

Configure the Backend's existing `SMARTSITE_AI_PREVIEW_SOURCES` allowlist for the camera's external
ID and this host's origin. Loopback is appropriate only when reachable from the Backend host;
remote operation still requires reviewed TLS/network routing and secret management. This source
addition does not configure a tunnel, change the website, or deploy either service.

The opt-in assembly gives each StreamWorker a `PreviewFrameQueue()` and one CameraPreviewObserver.
Validated immutable frames enter this single-slot lane before inference sampling. A slow encoder
or viewer cannot steal inference frames, block capture on a consumer, or open another camera.
Source release replaces the preview generation; stopped/failed streams invalidate cached images.
An encoding that completes after invalidation is discarded. Runtime completion/cancellation drains
owned preview encoding after worker/poller cleanup. Previously delivered pixels remain client-held
bytes; the API and Backend recheck freshness for each new response.

The observer-only routes use the server Bearer token and accept exactly one cameraExternalId query:

- `/v1/realtime/video-stream`: existing Backend-compatible9-key CameraMediaFrame NDJSON, at most640px
  per side and180000characters per image data URL. Images contain original session/sequence/capture
  metadata and no detections, Worker identity or business conclusion. The API reserves at most2
  viewers per camera/4per runtime; response lifetime includes stalled ASGI sends and is bounded by
  600seconds plus at most50ms for a final empty body. Browser disconnect releases a viewer slot,
  not the capture owner.
- `/v1/realtime/configuration-image`: the same capture observer provides a fresh image with purpose
  REGION_CONFIGURATION, explicit LIVE/REPLAY source kind and SHA256 of the exact JPEG bytes. It
  does not start the separate capture-only configuration runtime. Manager/Site authorization and
  version/reference persistence remain in the Backend's existing configuration-image flow.

Encoding is off the event loop, at most10encodes/second per camera, with at most one pending frame.
When no current raw frame is available, the route returns503; a request never activates a camera/model. Raw display pixels
are not retained evidence and are not an AI overlay. The analyzed-frame `/preview/latest` surface is now supplied by the same opt-in host.
Physical camera/browser acceptance, independent detection quality and rollout remain separate gates.
### Exact analyzed-frame overlay

`/v1/realtime/preview/latest?cameraExternalId=...` requires the same server token and camera scope
as the raw stream. It returns the existing `{sourceKind,frame}` response expected by the Backend.
The processing worker offers an immutable packet after detector/frame binding, configuration guards
and temporal processing. One pending packet and one off-loop encoder retain original pixels,
capture time, sequence, observations and confirmed Zone occupancy. They do not borrow a newer raw
frame. Configuration/session/freshness are checked before/after encoding and again on API reads.
When no current analyzed frame is available, the route returns503, including during warmup. Raw preview remains independent.

Expanded v2 policies resolve each person's bottom-center location using the same geometry helper
and canonical Camera/Region PPE resolver. Required PPE with no observation remains UNKNOWN;
COMPLIANT requires a nonempty required set with every item observed PRESENT. DISABLED does not
become a safety-compliance conclusion. Track IDs remain local technical tracks. TemporalCONFIRMED
preview is technical evidence status, not a Backend-approved case, Worker identity or Zone permission.
The immutable occupancy snapshot keeps people visibly inside between entry transitions.

A preview sink failure is logged using a fixed safe diagnostic and the sink is retired until restart;
it does not interrupt durable event handling. Preview/API availability does not prove successful
Backend ingestion, persistence, recipient notification or independent model accuracy.
