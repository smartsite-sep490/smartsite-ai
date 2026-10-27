# Shared preview transport (opt-in)

`realtime_snapshot_enabled`, `realtime_live_latest_frame` and `realtime_live_media_enabled` default false. Raw LIVE media requires latest-frame capture. No dependency or model artifact changes.

Backend-only bearer service authentication and an exact cameraExternalId query protect `/v1/realtime/preview/latest` and `/v1/realtime/video-stream`. Never put the service token in browser code or query parameters. The Backend must authorize the user's Site/camera before proxying. API process startup does not start capture/model work.

An authenticated snapshot request may start one lazy preview producer under the existing realtime resource lock. It uses settings with ingestion disabled: viewing does not create alert side effects. A raw media request observes that producer only and cannot start/restart it. At most two raw viewers, bounded NDJSON lifetime and no-store responses. Native source shutdown still depends on decoder read timeout.

For LIVE latest-frame mode, one capture feeds inference and the one-slot raw encoder. Raw JPEG display is not inference evidence and must not be paired with boxes from a different frame. Original capture/session/exact sequence remain attached. Source end, error and shutdown clear pixels; stale snapshots and raw frames are unavailable. A terminal producer is not automatically respawned by polling.

Recorded-file media, source reconnect, geometry reload, experimental model profiles, R2 persistence and production deployment are separate work. Source tests use fakes; no hardware latency or browser acceptance is claimed.
