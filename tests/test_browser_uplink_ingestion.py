import base64
import io
from copy import deepcopy
from uuid import UUID

import pytest
from PIL import Image

from smartsite_ai.ingestion.browser_uplink import decode_browser_uplink_frame
from smartsite_ai.ingestion.source import SourceReadError

LEASE = UUID("55555555-5555-4555-8555-555555555555")


def sample_packet() -> dict:
    jpeg = io.BytesIO()
    Image.new("RGB", (8, 4), (220, 30, 10)).save(jpeg, format="JPEG")
    return {
        "type": "media-frame",
        "mediaVersion": 1,
        "cameraExternalId": "browser-camera",
        "sessionId": str(LEASE),
        "sequenceNumber": "42",
        "capturedAt": "2026-11-27T00:00:00.123456Z",
        "width": 8,
        "height": 4,
        "imageDataUrl": "data:image/jpeg;base64," + base64.b64encode(jpeg.getvalue()).decode(),
    }


def decode(packet: dict):
    return decode_browser_uplink_frame(
        packet,
        stream_id="browser-camera",
        camera_external_id="browser-camera",
        session_id=LEASE,
    )


def test_browser_uplink_decodes_jpeg_without_losing_capture_identity() -> None:
    packet = sample_packet()
    frame = decode(packet)
    assert frame.session_id == LEASE and frame.sequence_number == 42
    assert frame.captured_at.microsecond == 123456
    assert frame.camera_external_id == "browser-camera"
    assert (frame.width, frame.height) == (8, 4)
    assert len(frame.payload) == 96
    blue, green, red = frame.payload[:3]
    assert red > green > blue


@pytest.mark.parametrize(
    "replacement",
    [
        {"sessionId": "55555555-5555-4555-8555-555555555556"},
        {"cameraExternalId": "outside-scope"},
        {"sequenceNumber": "0"},
        {"sequenceNumber": "042"},
        {"width": 9},
        {"height": True},
        {"capturedAt": "2026-11-27T00:00:00"},
        {"capturedAt": "2026-11-27T00:00:00+00:99"},
        {"imageDataUrl": "data:image/jpeg;base64,AAAA"},
        {"imageDataUrl": "data:image/jpeg;base64,!" + "A" * 180000},
        {"extra": 1},
    ],
)
def test_browser_uplink_rejects_invalid_or_cross_scope_frames(replacement: dict) -> None:
    packet = deepcopy(sample_packet())
    packet.update(replacement)
    with pytest.raises(SourceReadError, match="invalid browser uplink frame"):
        decode(packet)
