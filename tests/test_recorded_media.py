import hashlib

import pytest
from fastapi.testclient import TestClient

from smartsite_ai.app import create_app
from smartsite_ai.config import Settings


def app_for(path, enabled=True):
    return create_app(
        Settings(
            _env_file=None,
            realtime_source=str(path),
            realtime_camera_external_id="replay-a",
            backend_service_token="service-fixture",
            realtime_recorded_media_enabled=enabled,
        )
    )


def test_recorded_source_is_scoped_authenticated_and_exact(tmp_path):
    video = tmp_path / "clip.mp4"
    payload = b"\x00\x00\x00\x18ftypisom" + b"fixture"
    video.write_bytes(payload)
    with TestClient(app_for(video)) as client:
        assert (
            client.get("/v1/realtime/recorded-video?cameraExternalId=replay-a").status_code == 401
        )
        h = {"Authorization": "Bearer service-fixture"}
        assert (
            client.get("/v1/realtime/recorded-video?cameraExternalId=other", headers=h).status_code
            == 404
        )
        assert (
            client.get(
                "/v1/realtime/recorded-video?cameraExternalId=replay-a&path=other", headers=h
            ).status_code
            == 400
        )
        response = client.get("/v1/realtime/recorded-video?cameraExternalId=replay-a", headers=h)
        assert response.status_code == 200
        assert response.content == payload
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "video/mp4"
        assert not client.app.state.preview_hub.started
        assert response.headers["x-source-sha256"] == hashlib.sha256(payload).hexdigest()
        assert response.headers["x-camera-external-id"] == "replay-a"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_recorded_source_refuses_unconfigured_live_or_large_payload(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"1234ftyp" + b"x" * 2000000)
    h = {"Authorization": "Bearer service-fixture"}
    for path, enabled in [(video, False), ("rtsp://fixture/stream", True), (video, True)]:
        with TestClient(app_for(path, enabled)) as client:
            assert (
                client.get(
                    "/v1/realtime/recorded-video?cameraExternalId=replay-a", headers=h
                ).status_code
                == 503
            )


@pytest.mark.parametrize("payload", [b"", b"not-an-mp4-header", b"1234ftyp"])
def test_invalid_or_short_source_is_unavailable(tmp_path, payload):
    video = tmp_path / "clip.mp4"
    video.write_bytes(payload)
    with TestClient(app_for(video)) as client:
        response = client.get(
            "/v1/realtime/recorded-video?cameraExternalId=replay-a",
            headers={"Authorization": "Bearer service-fixture"},
        )
        assert response.status_code == 503
        assert not client.app.state.preview_hub.started


def test_duplicate_auth_query_and_missing_file_are_rejected(tmp_path):
    app = app_for(tmp_path / "missing.mp4")
    with TestClient(app) as client:
        path = "/v1/realtime/recorded-video?cameraExternalId=replay-a"
        assert (
            client.get(
                path,
                headers=[
                    ("Authorization", "Bearer service-fixture"),
                    ("Authorization", "Bearer service-fixture"),
                ],
            ).status_code
            == 401
        )
        h = {"Authorization": "Bearer service-fixture"}
        assert client.get(path + "&cameraExternalId=replay-a", headers=h).status_code == 400
        assert client.get(path, headers=h).status_code == 503
        assert not app.state.preview_hub.started
