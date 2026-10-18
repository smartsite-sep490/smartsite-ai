# Region configuration image: capture-only software boundary

## Explicit local operator command

`smartsite-ai-configuration-capture --manifest /private/capture.json` hosts the
existing authenticated observer route on127.0.0.1 only. It binds the socket before
constructing/starting one capture owner. It never loads YOLO or changes the
existing inference app. Import/help/config validation do not open a source.

Private manifest (source/token values remain in the named environment variables):

```json
{
  "schemaVersion": "1.0.0",
  "streamId": "configuration-source",
  "cameraExternalId": "configured-camera-external-id",
  "sourceKind": "LIVE",
  "sourceEnv": "SMARTSITE_AI_CONFIGURATION_SOURCE",
  "tokenEnv": "SMARTSITE_AI_CONFIGURATION_TOKEN",
  "resourceId": "aa732b3e-73f8-4e85-b914-0b529f78a181",
  "lockDirectory": "/private/shared-physical-source-locks",
  "port": 5091,
  "durationSeconds": 120
}
```

Use one operator-assigned physical-resource UUID/shared private local directory
for every alias; the sample UUID is not discovered hardware identity. Local video
requires REPLAY and is paced; camera indices require LIVE. Network kind/rights
still require operator commissioning. Source/token arguments, raw config/error
values and viewer-controlled source URLs are not supported. JSON is <=16KiB,
regular/non-symlink, strict fields and unique keys; token is bounded ASCII.

SIGINT/SIGTERM/SIGBREAK requests server exit, as does the finite duration (<=1h)
or terminal capture state. Uvicorn signal re-raising is disabled so the original
capture can drain first. Bounded stop returningFalse emits DRAIN_PENDING, then
the host waits for the same native task without cancelling it or repeating close.
Repeated cancellation of a waiter does not release ownership. Failed close emits
STOP_UNRESOLVED/exit1; it is never graceful/native-release proof. A stuck native
driver can outlive the requested duration and require explicit verified-process
recovery. Do not automatically relaunch while that process/its native work lives.

For hidden Windows processes, optionally pass `--stop-file /private/STOP` with
an absolute path in the manifest's directory. It must be named `STOP`, absent
and not a symlink at startup; an existing file refuses activation. Creating it
requests the same graceful drain as a signal, with polling bounded at four checks
per second. A filesystem read failure requests drain and returns a safe failure.
The watcher never cancels native capture, resets ownership or deletes the file.
This operator-local control is not a viewer endpoint and does not extend the 1h cap.

Startup reports BOUND_NOT_READY with its local port, not pixel availability.
Poll the existing authenticated endpoint for current pixels; requests cannot
start/recover capture. Binding/protocol failures use fixed safe codes. Default
tests use fake capture/server; actual camera/Backend/browser evidence is separate.

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

`ConfigurationCaptureRuntime` now assembles a standalone observer-only FastAPI
application with a deferred source factory. Construction, ASGI lifespan and GET
requests do not construct/connect sources. Explicit `start()` first acquires a
nonblocking local OS lock, then starts one capture worker. A stop requested before
the worker's first step avoids the source factory entirely. The factory must
return a new disconnected source with the exact expected stream identity; the
runtime refuses to adopt or close another owner's connected/mismatched source.

`ConfigurationProcessLease` requires an existing private absolute local directory
and an operator-assigned physical-resource UUID. All aliases of that physical
source must share this UUID and directory across cooperating processes. On Windows
it uses `msvcrt.LK_NBLCK` for one byte beyond EOF; on POSIX it uses nonblocking
`flock`. The dedicated empty file is not truncated/deleted. Lock failure, a changed
inode, symlink leaf, nonempty file or resolved UNC directory refuses startup before
the source factory. Configure only a local trusted directory; network filesystems,
hostile directory modification and cross-host distributed locking are unsupported.

The process lock remains held until explicit `await runtime.stop()` confirms
capture drain/release, including after natural EOF or source failure. A cancelled
stop waiter does not release it. Failed native release keeps it held for explicit
process recovery. Do not externally release this private guard to bypass a hung
source. OS process exit releases its file descriptors; that is not graceful native
camera cleanup proof. Older/noncooperating inference/capture workers must be
excluded independently before activation. Locks do not discover physical aliases.

Runtime assembly and same-host lock contention are qualified using fake sources,
in-memory ASGI and local test files only. Actual second-Python-process contention
was tested on Windows; POSIX lock execution is not certified by that test. Source
kind must match the stream and REPLAY must opt into pacing. The replay source must
still supply timestamps within the issue-time freshness budget.

These modules are not activated or mounted in the current website or inference
app; no new CLI/production rollout is claimed. Physical resource mapping/exclusive
ownership, native driver deadlines, real camera/browser/operator verification and
production acceptance remain open. AI/capture execution stays paused until the
human directly authorizes it.

Canonical wire ownership remains in SmartSite Main. Paired local parser checks
are compatibility evidence; they do not finalize immutable merged-Main provenance.
Do not hand-edit the existing vendored `contracts/` or its metadata.
