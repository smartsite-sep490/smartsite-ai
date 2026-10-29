# Native capture lifecycle

OpenCV open/read/FPS/position/release are blocking native operations. Cancelling an asyncio to_thread waiter does not stop its native thread. One source IO lock now remains held until each operation ends, including repeated cancellation. A capture returned after cancelled open is released exactly once. No concurrent release/read or reopen is admitted.

Cancellation remains cancellation after an ordinary native failure; a failed release is different: cleanup is unproven and a safe SourceConnectionError stays visible. The source latches that state so later close/connect cannot silently retry/reopen. Error messages contain exception types, never native credential-bearing text.

This change cannot interrupt a native decoder that never returns. Configure a driver/read timeout or process supervision before claiming bounded shutdown on hardware. Tests use gated fake native threads and do not prove physical camera drivers or latency. No model, reconnect strategy, configuration or runtime activation changes.
