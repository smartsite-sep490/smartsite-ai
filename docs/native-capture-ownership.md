# Native capture ownership and live latest-frame adapter

OpenCvFrameSource owns each native open/read/FPS/timestamp/release operation until that call finishes. Cancelling the Python waiter does not release a capture while a native call still uses it. Repeated cancellation drains the same owned operation; a capture opened after cancellation is discarded exactly once. A failed native release is sticky and prevents implicit close retry or reopening on that same source instance.

Every existing OpenCvFrameSource caller receives this ownership behavior: the explicit worker/runtime, source probes, video smoke tool and diagnostic realtime surface. Importing the source module still does not import cv2 or open hardware. Replay timestamp, sampling and pacing policies remain unchanged.

The diagnostic realtime path can explicitly enable SMARTSITE_AI_REALTIME_LIVE_LATEST_FRAME=true. It then wraps one existing LIVE capture in LiveLatestFrameSource: one reader retains at most one original frame while the consumer performs inference, replacing stale queued frames. The default remains false. File replay never selects this live adapter and retains ordered source behavior. This does not change the main ingestion queue policy or start another capture.

On source error, EOF or shutdown the adapter discards queued pixels and wakes readers; error text cannot expose source credentials. Close waits for the active read before releasing the capture, including repeated cancellation. Native decoder calls that never return still require driver/read timeouts or process recovery; this is resource ownership, not a hard shutdown deadline or a measured FPS claim.

Tests use gated fake native calls, fake frames, fake sources and a mocked realtime stack/encoder. They establish cancellation/order/cleanup and backlog selection only. No real camera, cv2 capture, GPU/model, retained evidence, network service or database is exercised. Shared preview hubs/media transport, replay preview, source reconnection and model warmup remain separate source units.
