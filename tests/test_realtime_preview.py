import asyncio
import base64
import io
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from fastapi import WebSocketDisconnect
from PIL import Image

from smartsite_ai import realtime
from smartsite_ai.config import Settings
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.envelope import FrameEnvelope


def frame(width: int = 4, height: int = 2) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="preview",
        session_id=UUID("00000000-0000-4000-8000-000000000001"),
        camera_external_id="CAM-01",
        captured_at=datetime(2026, 9, 30, tzinfo=UTC),
        sequence_number=9_223_372_036_854_775_807,
        width=width,
        height=height,
        payload=bytes([0, 0, 255]) * width * height,
    )


def test_preview_binds_even_an_empty_detection_frame_to_its_source_pixels() -> None:
    builder = getattr(realtime, "build_realtime_preview", None)
    assert callable(builder), "realtime must supply exact-frame preview pixels"
    envelope = frame()
    payload = builder(None, envelope)
    assert payload["cameraExternalId"] == envelope.camera_external_id
    assert payload["sessionId"] == str(envelope.session_id)
    assert payload["sequenceNumber"] == str(envelope.sequence_number)
    assert payload["capturedAt"] == envelope.captured_at_iso
    assert payload["detections"] == []
    assert payload["zoneDetections"] == []
    image = Image.open(io.BytesIO(base64.b64decode(payload["imageDataUrl"].split(",")[1])))
    assert image.size == (4, 2)
    assert image.getpixel((0, 0))[0] > 240
    assert image.getpixel((0, 0))[2] < 10  # BGR is converted correctly to RGB


def test_preview_bounds_image_dimensions_without_changing_normalized_geometry() -> None:
    builder = getattr(realtime, "build_realtime_preview", None)
    assert callable(builder), "realtime must bound preview image size"
    box = {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9}
    event = MagicMock()
    event.to_wire_dict.return_value = {
        "cameraExternalId": "CAM-01",
        "observations": [{"type": "PERSON", "trackId": 1, "confidence": 0.8, "boundingBox": box}],
    }
    payload = builder(event, frame(2000, 1000))
    image = Image.open(io.BytesIO(base64.b64decode(payload["imageDataUrl"].split(",")[1])))
    assert image.size == (1280, 640)
    assert (payload["width"], payload["height"]) == image.size
    assert len(payload["imageDataUrl"]) <= 1_400_000
    assert payload["detections"][0]["boundingBox"] == box
    assert payload["zoneDetections"][0]["boundingBox"] == box


def test_preview_uses_inference_configuration_instead_of_a_browser_default_polygon() -> None:
    configuration = CameraRegionConfiguration.model_validate_json("""{
      "schemaVersion":"1.0.0", "configurationVersion":5, "cameraExternalId":"CAM-01",
      "regions":[{"regionId":"00000000-0000-4000-8000-000000000010","geometryVersion":3,
      "coordinateSpace":"NORMALIZED_0_1","polygon":{"coordinates":[[0.1,0.1],[0.9,0.1],[0.9,0.9]]}}]
    }""")
    payload = realtime.build_realtime_preview(None, frame(), configuration=configuration)
    assert payload["zonePolygons"] == [
        {
            "regionId": "00000000-0000-4000-8000-000000000010",
            "geometryVersion": 3,
            "coordinates": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
        }
    ]


def test_socket_loop_transmits_pixels_from_the_same_frame_given_to_detector(monkeypatch) -> None:
    envelope = frame()
    detector = MagicMock()
    detector.detect = AsyncMock(
        return_value=DetectionBatch.from_frame(
            envelope,
            model_artifact_id="fake",
            model_version="test",
            model_sha256="a" * 64,
            detections=(),
        )
    )
    runner = MagicMock()
    pipeline = MagicMock()
    pipeline.process.return_value = None
    pipeline.occupied_zone_track_regions.return_value = frozenset()
    source = MagicMock()
    source.connect = AsyncMock()
    source.read_frame = AsyncMock(return_value=envelope)
    source.close = AsyncMock()
    monkeypatch.setattr(
        realtime, "_load_realtime_stack", lambda settings: (detector, runner, pipeline, None)
    )
    monkeypatch.setattr(realtime, "OpenCvFrameSource", lambda config: source)
    sent = []

    async def send(payload):
        sent.append(payload)
        raise WebSocketDisconnect()

    socket = MagicMock()
    socket.accept = AsyncMock()
    socket.close = AsyncMock()
    socket.send_json = send
    asyncio.run(realtime.stream_realtime(socket, Settings(realtime_source="0")))
    assert len(sent) == 1
    assert sent[0]["sequenceNumber"] == str(envelope.sequence_number)
    assert sent[0]["imageDataUrl"].startswith("data:image/jpeg;base64,")
    assert detector.detect.call_args.args[0] is envelope
    source.close.assert_awaited_once()
    runner.close.assert_called_once()
