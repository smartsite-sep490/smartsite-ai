import base64
import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smartsite_ai.configuration_capture_view import configuration_capture_view_router


class RawBuffer:
    camera = "camera-a"
    reads = 0
    frame = None

    def latest(self):
        self.reads += 1
        return self.frame


def client_for(buffer, enabled=True, kind="LIVE"):
    app = FastAPI()
    app.include_router(
        configuration_capture_view_router(
            buffer,
            SecretStr("configuration-capture-service-token"),
            source_kind=kind,
            enabled=enabled,
        )
    )
    return TestClient(app)


def test_unauthorized_or_invalid_scope_never_reads_capture():
    buffer = RawBuffer()
    with client_for(buffer) as client:
        assert (
            client.get("/v1/realtime/configuration-image?cameraExternalId=camera-a").status_code
            == 401
        )
        headers = {"Authorization": "Bearer configuration-capture-service-token"}
        assert (
            client.get(
                "/v1/realtime/configuration-image?cameraExternalId=other", headers=headers
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/v1/realtime/configuration-image?cameraExternalId=camera-a&token=x",
                headers=headers,
            ).status_code
            == 400
        )
    assert buffer.reads == 0


@pytest.mark.parametrize("kind", ["LIVE", "REPLAY"])
def test_original_capture_jpeg_is_hashed_without_monitoring_fields(kind):
    buffer = RawBuffer()
    jpeg = b"\xff\xd8fixture\xff\xd9"
    buffer.frame = {
        "type": "media-frame",
        "mediaVersion": 1,
        "cameraExternalId": "camera-a",
        "sessionId": str(uuid4()),
        "sequenceNumber": "4",
        "capturedAt": datetime.now(UTC).isoformat(),
        "width": 640,
        "height": 480,
        "imageDataUrl": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
    }
    with client_for(buffer, kind=kind) as client:
        response = client.get(
            "/v1/realtime/configuration-image?cameraExternalId=camera-a",
            headers={"Authorization": "Bearer configuration-capture-service-token"},
        )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["purpose"] == "REGION_CONFIGURATION" and body["sourceKind"] == kind
    assert body["frame"]["imageSha256"] == hashlib.sha256(jpeg).hexdigest()
    assert body["frame"]["imageDataUrl"] == buffer.frame["imageDataUrl"]
    assert "detections" not in body["frame"] and "type" not in body["frame"]


@pytest.mark.parametrize("enabled", [True, False])
def test_empty_or_disabled_capture_fails_closed(enabled):
    buffer = RawBuffer()
    with client_for(buffer, enabled=enabled) as client:
        response = client.get(
            "/v1/realtime/configuration-image?cameraExternalId=camera-a",
            headers={"Authorization": "Bearer configuration-capture-service-token"},
        )
    assert response.status_code == 503
    assert buffer.reads == (1 if enabled else 0)


def test_weak_service_token_is_refused():
    with pytest.raises(ValueError, match="service token"):
        configuration_capture_view_router(
            RawBuffer(), SecretStr(""), source_kind="LIVE", enabled=True
        )


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "frame", "mediaVersion": 1},
        {"type": "media-frame", "mediaVersion": 1, "cameraExternalId": "other"},
        {"type": "media-frame", "mediaVersion": 1, "cameraExternalId": "camera-a"},
    ],
)
def test_monitoring_or_malformed_frame_is_not_configuration_reference(frame):
    buffer = RawBuffer()
    buffer.frame = frame
    with client_for(buffer) as client:
        response = client.get(
            "/v1/realtime/configuration-image?cameraExternalId=camera-a",
            headers={"Authorization": "Bearer configuration-capture-service-token"},
        )
    assert response.status_code == 503
