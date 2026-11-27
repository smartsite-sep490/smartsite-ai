import asyncio
import base64
import io
from copy import deepcopy
from uuid import UUID

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from smartsite_ai.ingestion.browser_uplink import (
    BrowserUplinkFrameSource,
    decode_browser_uplink_frame,
)
from smartsite_ai.ingestion.source import SourceReadError
from smartsite_ai.integrations.backend_client import BackendClient

LEASE = UUID("55555555-5555-4555-8555-555555555555")
SITE = UUID(int=30)
CAMERA = UUID(int=40)


@pytest.mark.anyio
async def test_backend_webcam_source_keeps_cursor_after_reconnect() -> None:
    cursors = []
    secret = SecretStr("a" * 43)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == (
            f"/api/v1/integrations/ai/sites/{SITE}/cameras/{CAMERA}/browser-uplink/{LEASE}/frame"
        )
        assert request.headers["authorization"] == "Bearer test-service-token"
        assert request.headers["x-browser-frame-reader-secret"] == secret.get_secret_value()
        cursors.append(request.url.params["afterSequence"])
        if len(cursors) == 2:
            return httpx.Response(503)
        if len(cursors) == 3:
            return httpx.Response(204)
        return httpx.Response(
            200,
            json=sample_packet()
            | {
                "sequenceNumber": "42" if len(cursors) == 1 else "44",
            },
        )

    async with BackendClient(
        "https://backend.test", "test-service-token", transport=httpx.MockTransport(handle)
    ) as client:

        async def read_packet(after_sequence: int):
            return await client.read_browser_frame(SITE, CAMERA, LEASE, after_sequence, secret)

        source = BrowserUplinkFrameSource.from_frame_reader(
            read_packet, stream_id="browser", camera_external_id="browser-camera", session_id=LEASE
        )
        await source.connect()
        try:
            first = await source.read_frame()
            assert first is not None and first.sequence_number == 42
            with pytest.raises(SourceReadError):
                await source.read_frame()
        finally:
            await source.close()
        await source.connect()
        try:
            next_frame = await source.read_frame()
            assert next_frame is not None and next_frame.sequence_number == 44
        finally:
            await source.close()
    assert cursors == ["0", "42", "42", "42"]


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["redirect", "oversize", "duplicate"])
async def test_browser_frame_http_guard(failure: str) -> None:
    requests = []
    secret = SecretStr("b" * 43)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.host)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://unexpected.test/frame"})
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b" " * 190_000 if failure == "oversize" else b'{"a":1,"a":2}',
        )

    async with BackendClient(
        "https://backend.test", "test-service-token", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ValueError) as error:
            await client.read_browser_frame(SITE, CAMERA, LEASE, 0, secret)
    assert secret.get_secret_value() not in str(error.value)
    assert requests == ["backend.test"]


@pytest.mark.anyio
async def test_browser_uplink_source_cancelled_by_close() -> None:
    entered_read = asyncio.Event()
    generator_closed = asyncio.Event()

    async def packets():
        try:
            yield sample_packet()
            entered_read.set()
            await asyncio.Event().wait()
        finally:
            generator_closed.set()

    source = BrowserUplinkFrameSource(
        packets, stream_id="browser", camera_external_id="browser-camera", session_id=LEASE
    )
    await source.connect()
    first = await source.read_frame()
    assert first is not None and first.sequence_number == 42
    read_task = asyncio.create_task(source.read_frame())
    await asyncio.wait_for(entered_read.wait(), timeout=1)
    await asyncio.wait_for(source.close(), timeout=1)
    assert await read_task is None
    assert generator_closed.is_set() and not source.is_connected
    await source.close()


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
