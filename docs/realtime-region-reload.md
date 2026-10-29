# Realtime region snapshot refresh

`SMARTSITE_AI_REALTIME_REGION_CONFIGURATION_RELOAD=false` by default. Initial configuration uses the existing bounded wire reader and validates the selected PPE region before model allocation. Opt-in reload reads the operator snapshot before each inference step through that parser and RealtimeConfigurationState.

Camera identity, configuration/geometry versions and retained region-history checks remain authoritative. Same-version content changes, rollback, missing selected region, malformed/oversized data or another camera stop the producer. A valid selected-region geometry revision clears temporal confirmation before replacement. Unrelated geometry changes preserve selected PPE confirmation; Zone state retains geometry-aware domain handling.

Publish complete snapshots atomically. A partially written file fails closed and needs explicit operator restart after repair. This does not implement website-to-worker delivery/ACK or historical policy authority.

Camera movement is separate: normalized polygons do not follow the physical scene. This flag does not detect view changes or warp/revalidate geometry. No PTZ/drift/hardware/production acceptance. Tests use strict fixtures, real domain pipeline and fake detector/source, no camera/model loading.
