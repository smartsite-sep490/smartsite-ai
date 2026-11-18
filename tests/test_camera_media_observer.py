import asyncio
import base64
import hashlib
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from smartsite_ai.camera_media_api import camera_media_router
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import PreviewFrameQueue
from smartsite_ai.ingestion.testing import make_test_frame
from smartsite_ai.runtime.media_preview import (
    CameraMediaFrame,
    CameraPreviewObserver,
    encode_media_frame,
)


async def until(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Preview did not reach expected state")


def encoded(frame: FrameEnvelope) -> CameraMediaFrame:
    return {
        "type": "media-frame",
        "mediaVersion": 1,
        "cameraExternalId": frame.camera_external_id,
        "sessionId": str(frame.session_id),
        "sequenceNumber": str(frame.sequence_number),
        "capturedAt": frame.captured_at_iso,
        "width": 2,
        "height": 2,
        "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
    }


@pytest.mark.anyio
async def test_observer_fences_encoding_from_old_source_and_hides_stale_pixels() -> None:
    queue = PreviewFrameQueue()
    now = datetime.now(UTC)
    entered, release = threading.Event(), threading.Event()

    def blocked(frame: FrameEnvelope) -> CameraMediaFrame:
        entered.set()
        assert release.wait(2)
        return encoded(frame)

    observer = CameraPreviewObserver(
        "ext-media",
        queue,
        source_kind="LIVE",
        is_streaming=lambda: True,
        clock=lambda: now,
        encoder=blocked,
    )
    await observer.start()
    queue.put(make_test_frame("media", 0, captured_at=now))
    try:
        await until(entered.is_set)
        queue.clear()
        release.set()
        await until(lambda: observer.encodes_completed == 1)
        assert observer.latest() is None
        queue.put(make_test_frame("media", 1, captured_at=now))
        await until(lambda: observer.latest() is not None)
        assert observer.latest()["sequenceNumber"] == "1"
        now += timedelta(seconds=2)
        assert observer.latest() is None
    finally:
        release.set()
        assert await observer.stop()


@pytest.mark.anyio
async def test_observer_stop_drains_native_encoder_without_publishing_or_cancelling_it() -> None:
    queue = PreviewFrameQueue()
    entered, release = threading.Event(), threading.Event()

    def blocked(frame: FrameEnvelope) -> CameraMediaFrame:
        entered.set()
        assert release.wait(2)
        return encoded(frame)

    observer = CameraPreviewObserver(
        "ext-media",
        queue,
        source_kind="LIVE",
        is_streaming=lambda: True,
        encoder=blocked,
        stop_timeout=0.01,
    )
    await observer.start()
    queue.put(make_test_frame("media", 0))
    try:
        await until(entered.is_set)
        assert await observer.stop() is False
        assert observer.latest() is None
    finally:
        release.set()
        await observer.wait_for_release()
    assert observer.state == "STOPPED"
    assert observer.latest() is None


@pytest.mark.anyio
async def test_media_route_auth_scope_and_original_frame_lineage() -> None:
    queue = PreviewFrameQueue()
    observer = CameraPreviewObserver(
        "ext-media", queue, source_kind="LIVE", is_streaming=lambda: True, encoder=encoded
    )
    app = FastAPI()
    app.include_router(
        camera_media_router({"ext-media": observer}, SecretStr("x" * 32), max_stream_seconds=0.02)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        endpoint = "/v1/realtime/video-stream?cameraExternalId=ext-media"
        assert (await client.get(endpoint)).status_code == 401
        headers = {"Authorization": "Bearer " + "x" * 32}
        assert (await client.get(endpoint, headers=headers)).status_code == 503
        assert (
            await client.get(endpoint + "&cameraExternalId=ext-media", headers=headers)
        ).status_code == 400
        assert (
            await client.get(endpoint.replace("ext-media", "another"), headers=headers)
        ).status_code == 404
        assert observer.state == "IDLE"
        await observer.start()
        frame = make_test_frame("media", 7)
        queue.put(frame)
        try:
            await until(lambda: observer.latest() is not None)
            response = await client.get(endpoint, headers=headers)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["content-type"].startswith("application/x-ndjson")
            assert len(response.text.splitlines()) == 1
            assert response.json() == encoded(frame)
            payload = observer.latest()
            payload["sequenceNumber"] = "999"
            assert observer.latest()["sequenceNumber"] == "7"
            queue.clear()
            assert (await client.get(endpoint, headers=headers)).status_code == 503
        finally:
            await observer.stop()


@pytest.mark.anyio
async def test_configuration_image_uses_the_existing_capture_observer_without_another_reader() -> (
    None
):
    queue = PreviewFrameQueue()
    observer = CameraPreviewObserver(
        "ext-media", queue, source_kind="LIVE", is_streaming=lambda: True
    )
    app = FastAPI()
    app.include_router(camera_media_router({"ext-media": observer}, SecretStr("x" * 32)))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/v1/realtime/configuration-image?cameraExternalId=ext-media"
        headers = {"Authorization": "Bearer " + "x" * 32}
        assert (await client.get(path)).status_code == 401
        assert (await client.get(path, headers=headers)).status_code == 503
        assert observer.state == "IDLE"
        frame = make_test_frame("media", 0).model_copy(
            update={"width": 64, "height": 64, "payload": bytes(64 * 64 * 3)}
        )
        await observer.start()
        queue.put(frame)
        try:
            await until(lambda: observer.latest() is not None)
            response = await client.get(path, headers=headers)
            assert response.status_code == 200
            payload = response.json()
            assert payload["schemaVersion"] == 1
            assert payload["purpose"] == "REGION_CONFIGURATION"
            assert payload["sourceKind"] == "LIVE"
            image = payload["frame"]
            assert image["sessionId"] == str(frame.session_id)
            assert image["capturedAt"] == frame.captured_at_iso
            jpeg = base64.b64decode(image["imageDataUrl"].split(",")[1], validate=True)
            assert image["imageSha256"] == hashlib.sha256(jpeg).hexdigest()
            assert "detections" not in image
            queue.clear()
            assert (await client.get(path, headers=headers)).status_code == 503
        finally:
            await observer.stop()


@pytest.mark.anyio
async def test_stalled_media_send_cannot_hold_the_viewer_slot_past_deadline() -> None:
    queue = PreviewFrameQueue()
    observer = CameraPreviewObserver(
        "ext-media", queue, source_kind="LIVE", is_streaming=lambda: True, encoder=encoded
    )
    app = FastAPI()
    app.include_router(
        camera_media_router({"ext-media": observer}, SecretStr("x" * 32), max_stream_seconds=0.02)
    )
    await observer.start()
    queue.put(make_test_frame("media", 0))
    blocked = asyncio.Event()

    async def receive():
        await blocked.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            await blocked.wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v1/realtime/video-stream",
        "query_string": b"cameraExternalId=ext-media",
        "headers": [(b"authorization", b"Bearer " + b"x" * 32)],
        "server": ("test", 80),
        "client": ("client", 123),
    }
    try:
        await until(lambda: observer.latest() is not None)
        await asyncio.wait_for(
            asyncio.gather(app(scope.copy(), receive, send), app(scope.copy(), receive, send)), 0.3
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.get(
                        "/v1/realtime/video-stream?cameraExternalId=ext-media",
                        headers={"Authorization": "Bearer " + "x" * 32},
                    )
                    for _ in range(2)
                ]
            )
            assert [response.status_code for response in responses] == [200, 200]
    finally:
        blocked.set()
        await observer.stop()


@pytest.mark.anyio
async def test_media_viewer_limits_release_after_streams_finish() -> None:
    queue = PreviewFrameQueue()
    observer = CameraPreviewObserver(
        "ext-media", queue, source_kind="LIVE", is_streaming=lambda: True, encoder=encoded
    )
    app = FastAPI()
    app.include_router(
        camera_media_router({"ext-media": observer}, SecretStr("x" * 32), max_stream_seconds=0.03)
    )
    await observer.start()
    queue.put(make_test_frame("media", 0))
    try:
        await until(lambda: observer.latest() is not None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:

            async def read() -> httpx.Response:
                return await client.get(
                    "/v1/realtime/video-stream?cameraExternalId=ext-media",
                    headers={"Authorization": "Bearer " + "x" * 32},
                )

            responses = await asyncio.gather(read(), read(), read())
            assert sorted(response.status_code for response in responses) == [200, 200, 429]
            assert (await read()).status_code == 200
            assert observer.state == "RUNNING"
    finally:
        await observer.stop()


def test_media_encoder_bounds_dimensions_and_keeps_exact_capture_metadata() -> None:
    pytest.importorskip("PIL")
    frame = make_test_frame("media", 9223372036854775807).model_copy(
        update={"width": 1280, "height": 720, "payload": bytes(1280 * 720 * 3)},
    )
    result = encode_media_frame(frame)
    assert (result["width"], result["height"]) == (640, 360)
    assert result["sequenceNumber"] == "9223372036854775807"
    assert result["capturedAt"] == frame.captured_at_iso
    assert len(result["imageDataUrl"]) <= 180000
    assert set(result) == set(encoded(frame))
