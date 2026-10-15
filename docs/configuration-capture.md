# Region configuration image: capture-only software boundary

This purpose is independent of YOLO, tracking, Region configuration and event
ingestion. It does not make a PPE observation or identify a Worker. The existing
inference application, startup and preview routes are unchanged.

## Explicit ownership

`ConfigurationSourceLease` binds one existing `FrameSource`, its stream identity,
one camera external ID and explicit `LIVE` or `REPLAY` kind. Bindings are read-only.
The runtime must provide **one shared lease for one physical resource**. Separate
leases, device aliases and other processes are not excluded by this in-memory
lease. Physical/source exclusivity is an activation prerequisite, not established
by passing these unit tests. No global registry or new source URL API is added.

`ConfigurationCapture(lease)` performs no I/O at construction. Only an explicit
`await owner.start()` opens that source. Repeated concurrent starts on that owner
are idempotent; a competing owner using its lease is refused. Each capture has a
single-use lifecycle. EOF, source/encoder failure and stop clear pixels. Recovery
requires an explicit new owner after proven release; observer requests never do it.

`await owner.stop()` immediately removes published/pending pixels and waits at
most two seconds by default. A `False` result means **release is unresolved**,
not that the source is closed. Native connect/read and encoding are allowed to
drain before close. The lease remains claimed during that drain. A failed close
or unexpected cancellation poisons ownership instead of permitting a duplicate
source. Do not call the private worker task's `cancel()` or close its source
underneath a native read. An unresponsive decoder requires explicit runtime/process
recovery; the API does not promise to terminate a blocked native driver.

## Bounded pixels and observer route

One encoder and at most one pending raw frame are retained. New incoming frames
replace the pending frame. Encoding is capped at two attempts per second;
requests cannot increase it. The default Pillow encoder runs outside the API event
loop, accepts at most 16,777,216 input pixels, never upscales, and preserves aspect
ratio within 1280x1280. Each output side must be at least 64px, JPEG at most 1MiB.
Extreme aspect ratios that become too narrow are refused.

Source camera/stream/session/sequence are checked. A connection begins at sequence
zero, remains in that session and increases its sequence. Capture timestamps are
preserved. Issue-time freshness is age <=2s and future skew <=1s, for both source
kinds. Replay must be paced by its source; it is not labelled live or retimestamped
to make stale pixels pass. Invalidation prevents a late encode resurrecting an
older image after a newer invalid timestamp clears the buffer.

`configuration_image_router(owner, SecretStr(service_token))` can be explicitly
mounted by a runtime that already owns the capture. The router has no lifespan
hook and never calls start/read/stop. It provides:

`GET /v1/realtime/configuration-image?cameraExternalId=<exact-bound-id>`

Server-only Bearer auth is required; do not expose the token to browsers. Missing
or incorrect auth is 401, malformed/duplicate query 400, unknown camera 404, missing
or stale pixels 503. Responses include no-store/nosniff. The payload contains only
schemaVersion1, purpose REGION_CONFIGURATION, explicit sourceKind and original
frame lineage/JPEG/SHA. No source URL or model/Zone/identity fields are returned.

## Qualification and open gates

Run the four `test_configuration_capture*` / `test_configuration_image*` test files
with the existing Python 3.12 test environment. They use fake sources and synthetic
pixels only; API checks use in-memory ASGI transport. No camera, native decoder,
model, GPU, network server or application startup is exercised.

The standalone modules are not activated or mounted in the current website or
inference app. Physical ownership/alias exclusion, an explicitly approved runtime
entrypoint, native driver deadlines, real camera/browser/operator verification and
production acceptance remain open. AI/capture execution stays paused until the
human directly authorizes it.

Canonical wire ownership remains in SmartSite Main. Paired local parser checks
are compatibility evidence; they do not finalize immutable merged-Main provenance.
Do not hand-edit the existing vendored `contracts/` or its metadata.
