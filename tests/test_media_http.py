import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from smartsite_ai.app import create_app
from smartsite_ai.config import Settings


def make_app():
    return create_app(
        Settings(
            _env_file=None,
            backend_service_token="fake-media-service-token",
            realtime_camera_external_id="cam",
            realtime_live_media_enabled=True,
            realtime_live_latest_frame=True,
            realtime_snapshot_enabled=True,
        )
    )


def test_media_http_rejects_missing_auth_without_starting_camera():
    app = make_app()
    with TestClient(app) as client:
        assert client.get("/v1/realtime/video-stream?cameraExternalId=cam").status_code == 401
        assert not app.state.preview_hub.started
        assert app.state.media_viewers == 0


def test_media_http_rejects_wrong_camera_and_extra_query():
    app = make_app()
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer fake-media-service-token"}
        assert (
            client.get(
                "/v1/realtime/video-stream?cameraExternalId=other", headers=headers
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/v1/realtime/video-stream?cameraExternalId=cam&token=secret", headers=headers
            ).status_code
            == 400
        )


def test_media_http_raw_pixels_stream_without_inference_or_respawn():
    app = make_app()
    raw = {
        "type": "media-frame",
        "mediaVersion": 1,
        "cameraExternalId": "cam",
        "sessionId": str(uuid4()),
        "sequenceNumber": "7",
        "capturedAt": datetime.now(UTC).isoformat(),
        "width": 2,
        "height": 2,
        "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
    }
    with TestClient(app) as client:
        media = getattr(app.state, "media_hub", None)
        assert media is not None, "No independent media hub"
        frames = iter([raw, None])
        media.latest = lambda: next(frames, None)
        response = client.get(
            "/v1/realtime/video-stream?cameraExternalId=cam",
            headers={"Authorization": "Bearer fake-media-service-token"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert response.headers["cache-control"] == "no-store"
        assert json.loads(response.text) == raw
        assert app.state.media_viewers == 0
        assert not app.state.preview_hub.started


def test_media_http_viewer_limit_does_not_start_or_replace_the_producer():
    app = make_app()
    with TestClient(app) as client:
        app.state.media_hub.latest = lambda: {"sessionId": "test", "sequenceNumber": "1"}
        app.state.media_viewers = 2
        response = client.get(
            "/v1/realtime/video-stream?cameraExternalId=cam",
            headers={"Authorization": "Bearer fake-media-service-token"},
        )
        assert response.status_code == 429
        assert response.headers["retry-after"] == "2"
        assert app.state.media_viewers == 2
        assert not app.state.preview_hub.started
        app.state.media_viewers = 0
