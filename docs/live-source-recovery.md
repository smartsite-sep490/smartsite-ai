# Realtime source-only recovery

`SMARTSITE_AI_REALTIME_LIVE_RECONNECT_ATTEMPTS` is0 by default, validated0..3. LIVE read failures or end-of-source after the existing failure budget can trigger bounded jittered reconnect within the same producer/model/provider lifetime. REPLAY uses its existing loop and never this recovery.

Before reconnect, clear analysis/raw previews and temporal confirmation. OpenCV reconnect starts a new session/sequence origin and tracker/PPE state follows that session boundary; a track is not a Worker identity. Thirty successful source reads replenish the outage budget; view requests do not. Cancellation does not start another attempt. Exhaustion remains terminal for the producer; polling cannot respawn it.

The capture cleanup latch remains authoritative: failed native release cannot be fixed by retrying. Native calls that never return still need driver timeout/process supervision. This is source recovery for the existing realtime surface; it does not renew trusted camera-worker generations or grant issuer/database permissions. No geometry hot reload, model profile change, production rollout or live hardware qualification follows from this flag.

Tests use synthetic frames, fake source/native gates and a mocked detector/provider. They verify session/reset/order and no model reload, not scene accuracy or camera latency.
