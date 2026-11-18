import asyncio
import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from test_camera_media_observer import until
from test_policy_configuration_bundle import content, packet
from test_processing_worker import _configuration

from smartsite_ai.camera_media_api import camera_media_router
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.ingestion.queue import PreviewFrameQueue
from smartsite_ai.ingestion.testing import make_test_frame
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.runtime.analysis_preview import AnalysisPreviewObserver, encode_analysis_frame
from smartsite_ai.runtime.media_preview import CameraPreviewObserver


async def wait_payload(observer):
    for _ in range(200):
        payload = await observer.latest()
        if payload is not None:
            return payload
        await asyncio.sleep(0.001)
    raise AssertionError("Analysis preview unavailable")


def test_expanded_required_ppe_unobserved_gloves_cannot_become_compliant():
    value = content()
    value["cameraPpePolicy"] = {
        "mode": "EXPLICIT",
        "requiredPpe": ["HARD_HAT", "SAFETY_VEST", "GLOVES"],
    }
    value["regions"] = []
    configuration = CameraPolicyConfigurationBundle.from_wire_bytes(packet(value))
    frame = make_test_frame("analysis", 0, camera_external_id=configuration.camera_external_id)
    event = TechnicalObservationEvent.create(
        event_id="11111111-1111-4111-8111-111111111111",
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at_iso,
        frame_dimensions={"width": 2, "height": 2},
        observations=[
            {
                "type": "PERSON",
                "trackId": 7,
                "boundingBox": {
                    "coordinateSpace": "NORMALIZED_0_1",
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.8,
                    "y2": 0.9,
                },
            },
            *[
                {
                    "type": "PPE",
                    "trackId": 7,
                    "ppeItem": item,
                    "status": "PRESENT",
                    "regionId": "22222222-2222-4222-8222-222222222222",
                    "geometryVersion": 1,
                }
                for item in ("HARD_HAT", "SAFETY_VEST")
            ],
        ],
    )
    payload = encode_analysis_frame(AnalysisFrame(frame, event, configuration, "ppe", frozenset()))
    assert payload["detections"][0]["alertState"] == "UNKNOWN"
    assert payload["detections"][0]["ppeStatus"]["GLOVES"] == "UNKNOWN"


@pytest.mark.anyio
async def test_latest_analysis_http_returns_exact_person_and_frame_and_rechecks_configuration():
    now = datetime.now(UTC)
    frame = make_test_frame("analysis", 19, camera_external_id="CAM-GATE-01", captured_at=now)
    event = TechnicalObservationEvent.create(
        event_id="11111111-1111-4111-8111-111111111111",
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at_iso,
        frame_dimensions={"width": 2, "height": 2},
        observations=[
            {
                "type": "PERSON",
                "trackId": 7,
                "confidence": 0.91,
                "boundingBox": {
                    "coordinateSpace": "NORMALIZED_0_1",
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.8,
                    "y2": 0.9,
                },
            }
        ],
    )
    configuration_current = True

    async def current(_):
        return configuration_current

    analysis = AnalysisPreviewObserver("CAM-GATE-01", current=current, clock=lambda: now)
    raw = CameraPreviewObserver(
        "CAM-GATE-01",
        PreviewFrameQueue(),
        source_kind="LIVE",
        is_streaming=lambda: True,
        analysis=analysis,
    )
    app = FastAPI()
    app.include_router(camera_media_router({"CAM-GATE-01": raw}, SecretStr("x" * 32)))
    endpoint = "/v1/realtime/preview/latest?cameraExternalId=CAM-GATE-01"
    headers = {"Authorization": "Bearer " + "x" * 32}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get(endpoint)).status_code == 401
        assert (await client.get(endpoint, headers=headers)).status_code == 503
        await raw.start()
        try:
            # No new entry observation on this frame: the same person remains
            # in a confirmed occupied region after the earlier entry transition.
            analysis.offer(
                AnalysisFrame(
                    frame,
                    event,
                    _configuration(),
                    "ppe",
                    frozenset(),
                    occupied_zone_regions=frozenset({(7, "22222222-2222-4222-8222-222222222222")}),
                )
            )
            await wait_payload(analysis)
            response = await client.get(endpoint, headers=headers)
            assert response.status_code == 200
            payload = response.json()
            assert payload["sourceKind"] == "LIVE"
            assert payload["frame"]["sequenceNumber"] == "19"
            assert payload["frame"]["sessionId"] == str(frame.session_id)
            assert payload["frame"]["detections"][0]["trackId"] == 7
            assert payload["frame"]["detections"][0]["ppeStatus"]["HARD_HAT"] == "UNKNOWN"
            assert payload["frame"]["detections"][0]["active"] is False
            assert payload["frame"]["zoneDetections"][0]["active"] is True
            assert payload["frame"]["zoneDetections"][0]["label"] == "IN RESTRICTED ZONE"
            # The old result cannot be relabeled with another capture's pixels.
            wrong_frame = frame.model_copy(update={"captured_at": now + timedelta(microseconds=1)})
            analysis.offer(AnalysisFrame(wrong_frame, event, _configuration(), "ppe", frozenset()))
            assert await analysis.latest() is None
            assert analysis.reason == "analysis_frame_binding_invalid"
            configuration_current = False
            assert (await client.get(endpoint, headers=headers)).status_code == 503
        finally:
            if not await raw.stop():
                await raw.wait_for_release()


@pytest.mark.anyio
async def test_analyzed_frame_keeps_exact_pixels_and_expires_without_changing_identity():
    now = datetime.now(UTC)
    frame = make_test_frame("analysis", 37, camera_external_id="CAM-GATE-01", captured_at=now)
    item = AnalysisFrame(frame, None, _configuration(), "ppe", frozenset())
    current = True

    async def valid(_):
        return current

    observer = AnalysisPreviewObserver("CAM-GATE-01", current=valid, clock=lambda: now)
    await observer.start()
    try:
        observer.offer(item)
        payload = await wait_payload(observer)
        assert payload["sequenceNumber"] == "37"
        assert payload["sessionId"] == str(frame.session_id)
        assert payload["detections"] == []
        assert payload["zoneDetections"] == []
        assert payload["imageDataUrl"].startswith("data:image/jpeg;base64,")
        now += timedelta(seconds=2)
        assert await observer.latest() is None
        now = frame.captured_at
        current = False
        assert await observer.latest() is None
    finally:
        assert await observer.stop()


@pytest.mark.anyio
async def test_completed_analysis_encoder_cannot_publish_after_configuration_or_session_change():
    entered, release = threading.Event(), threading.Event()
    current = True
    now = datetime.now(UTC)
    item = AnalysisFrame(
        make_test_frame("analysis", 0, camera_external_id="CAM-GATE-01", captured_at=now),
        None,
        _configuration(),
        "ppe",
        frozenset(),
    )

    def encode(packet):
        entered.set()
        assert release.wait(2)
        return encode_analysis_frame(packet)

    async def valid(_):
        return current

    observer = AnalysisPreviewObserver(
        "CAM-GATE-01", current=valid, clock=lambda: now, encoder=encode
    )
    await observer.start()
    try:
        observer.offer(item)
        await until(entered.is_set)
        current = False
        release.set()
        await asyncio.sleep(0.03)
        assert await observer.latest() is None
        assert observer.reason is None
    finally:
        release.set()
        await observer.stop()
