# Recorded video display

`SMARTSITE_AI_REALTIME_RECORDED_MEDIA_ENABLED=false` by default. When explicitly enabled, `/v1/realtime/recorded-video?cameraExternalId=...` serves only the configured local `.mp4` source, never a query-selected path or remote URL. A server bearer token and exact configured camera are required; extra/duplicate query or authorization fields are refused.

Reads are bounded to2MB plus one byte for overflow detection. Missing/invalid/oversized files return unavailable. Response includes MP4 content type, no-store, nosniff, configured camera and SHA256 for Backend verification. Header sniffing is not a full video integrity/codec validation.

This endpoint neither starts capture/inference nor emits alerts. It is display playback and is not synchronized AI evidence. The Backend must authorize user Site/camera and verify source/SHA before serving its existing protected recorded-video route. Browser codec playback, real media timing and production acceptance require actual separate evidence.
