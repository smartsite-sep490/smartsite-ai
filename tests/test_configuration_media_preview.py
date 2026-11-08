"""Raw display must observe an explicitly started owner, never create one."""

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from configuration_capture_fakes import NOW, SESSION, FakeSource, frame, until
from pydantic import SecretStr

from smartsite_ai.configuration_runtime import ConfigurationCaptureRuntime
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.media_preview import encode_media_frame

TOKEN = "synthetic-configuration-media-preview-service-token"
PATH = "/v1/realtime/video-stream?cameraExternalId=camera-a"


def test_raw_encoder_rejects_oversize_input_before_image_allocation(monkeypatch):
    from PIL import Image

    allocations = []
    monkeypatch.setattr(Image, "frombytes", lambda *a, **k: allocations.append(True))
    with pytest.raises(ValueError, match="pixel budget"):
        encode_media_frame(SimpleNamespace(width=5000, height=4000, payload=b""))
    assert allocations == []


def runtime_pair(directory, *, enabled=None):
    source = FakeSource()
    builds = []

    def factory(config):
        builds.append(config)
        return source

    options = {} if enabled is None else {"raw_preview_enabled": enabled}
    runtime = ConfigurationCaptureRuntime(
        stream=StreamConfig(
            stream_id="configuration-stream",
            camera_external_id="camera-a",
            source_url="test://camera",
        ),
        source_kind="LIVE",
        resource_id=UUID("e98f6475-0660-4c71-a466-4c93d1739ee0"),
        lock_directory=directory,
        service_token=SecretStr(TOKEN),
        source_factory=factory,
        clock=lambda: NOW,
        stop_timeout=0.03,
        **options,
    )
    return runtime, source, builds


def test_existing_configuration_runtime_does_not_enable_raw_preview_by_default(tmp_path):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(runtime.app), base_url="http://test"
        ) as client:
            response = await client.get(PATH, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 404
        assert builds == [] and source.connects == source.reads == source.closes == 0
        assert runtime.state == "IDLE"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": f"Bearer {TOKEN}"}, 503),
    ],
)
def test_raw_preview_requests_cannot_start_a_configuration_source(tmp_path, headers, status):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path, enabled=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(runtime.app), base_url="http://test"
        ) as client:
            response = await client.get(PATH, headers=headers)
        assert response.status_code == status
        assert response.headers["Cache-Control"] == "no-store"
        assert builds == [] and source.connects == source.reads == source.closes == 0
        assert runtime.state == "IDLE"

    asyncio.run(scenario())


def test_two_http_viewers_share_one_owner_and_eof_revokes_display(tmp_path):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path, enabled=True)
        bodies = 0
        viewers_received = asyncio.Event()

        async def observed_app(scope, receive, send):
            streaming = False

            async def observed_send(message):
                nonlocal bodies, streaming
                await send(message)
                if message["type"] == "http.response.start":
                    streaming = message["status"] == 200
                if streaming and message["type"] == "http.response.body" and message.get("body"):
                    bodies += 1
                    if bodies >= 2:
                        viewers_received.set()

            await runtime.app(scope, receive, observed_send)

        await runtime.start()
        source.items.put_nowait(frame())
        await until(lambda: runtime._media_hub.latest() is not None)
        headers = {"Authorization": f"Bearer {TOKEN}"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(observed_app), base_url="http://test"
        ) as client:
            tasks = [asyncio.create_task(client.get(PATH, headers=headers)) for _ in range(2)]
            try:
                await asyncio.wait_for(viewers_received.wait(), timeout=2)
                assert (await client.get(PATH, headers=headers)).status_code == 429
                assert len(builds) == source.connects == 1
                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks.pop(0)
                assert runtime.state == "RUNNING" and source.closes == 0
                tasks.append(asyncio.create_task(client.get(PATH, headers=headers)))
                await until(lambda: bodies >= 3)
                assert (await client.get(PATH, headers=headers)).status_code == 429
                source.items.put_nowait(None)
                responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
                for response in responses:
                    assert response.status_code == 200
                    lines = response.text.splitlines()
                    assert len(lines) == 1
                    item = json.loads(lines[0])
                    assert item["type"] == "media-frame" and item["mediaVersion"] == 1
                    assert item["cameraExternalId"] == "camera-a"
                    assert item["sessionId"] == str(SESSION) and item["sequenceNumber"] == "0"
                    assert item["capturedAt"] == frame().captured_at_iso
                    assert item["width"] == item["height"] == 64
                    assert len(lines[0].encode()) < 256 * 1024
                    assert response.headers["Cache-Control"] == "no-store"
                    assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert (await client.get(PATH, headers=headers)).status_code == 503
                assert runtime._media_hub.latest() is None
            finally:
                source.items.put_nowait(None)
                await runtime.stop()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        assert source.closes == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/v1/realtime/video-stream", None, 400),
        (PATH + "&cameraExternalId=camera-a", None, 400),
        (PATH + "&unexpected=1", None, 400),
        (PATH.replace("camera-a", "camera-b"), None, 404),
        (PATH, [("Authorization", f"Bearer {TOKEN}")] * 2, 401),
    ],
)
def test_raw_preview_rejects_invalid_scope_before_observing(tmp_path, path, headers, status):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path, enabled=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(runtime.app), base_url="http://test"
        ) as client:
            response = await client.get(
                path, headers=headers or {"Authorization": f"Bearer {TOKEN}"}
            )
        assert response.status_code == status
        assert TOKEN not in response.text and "test://camera" not in response.text
        assert builds == [] and source.connects == source.reads == source.closes == 0

    asyncio.run(scenario())


def test_optional_observer_failure_revokes_display_without_skipping_source_teardown(tmp_path):
    class BrokenObserver:
        def offer(self, frame):
            raise RuntimeError("synthetic observer failure")

        def clear(self):
            raise RuntimeError("synthetic observer clear failure")

    async def scenario():
        runtime, source, _ = runtime_pair(tmp_path, enabled=True)
        runtime._capture._frame_observer = BrokenObserver()
        await runtime.start()
        source.items.put_nowait(frame())
        try:
            await until(lambda: runtime._capture.latest() is not None)
            assert not runtime._capture.observer_available
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(runtime.app), base_url="http://test"
            ) as client:
                response = await client.get(PATH, headers={"Authorization": f"Bearer {TOKEN}"})
            assert response.status_code == 503
        finally:
            source.items.put_nowait(None)
            assert await runtime.stop()
        assert source.closes == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_failed_response_start_releases_viewer_without_stopping_owner(tmp_path, failure):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path, enabled=True)
        statuses = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])
                raise failure("synthetic response-start disconnect")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/realtime/video-stream",
            "raw_path": b"/v1/realtime/video-stream",
            "query_string": b"cameraExternalId=camera-a",
            "root_path": "",
            "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
            "server": ("test", 80),
            "client": ("test", 1234),
        }
        await runtime.start()
        source.items.put_nowait(frame())
        try:
            await until(lambda: runtime._media_hub.latest() is not None)
            for _ in range(3):
                with pytest.raises(failure, match="response-start disconnect"):
                    await runtime.app(dict(scope), receive, send)
            assert statuses == [200, 200, 200]
            assert runtime.state == "RUNNING" and source.closes == 0
            assert len(builds) == source.connects == 1
        finally:
            source.items.put_nowait(None)
            assert await runtime.stop()

    asyncio.run(scenario())
