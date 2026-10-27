import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.mark.anyio
async def test_two_viewers_share_producer_and_stale_frames_are_unavailable():
    from smartsite_ai.preview_snapshot import LatestPreviewHub

    starts = 0
    now = 10.0
    ready = asyncio.Event()

    async def produce(sink):
        nonlocal starts
        starts += 1
        await sink.send_json({"type": "frame", "cameraExternalId": "camera-a"})
        ready.set()
        await asyncio.Event().wait()

    hub = LatestPreviewHub(produce, "camera-a", asyncio.Lock(), clock=lambda: now)
    assert hub.latest() is None
    await ready.wait()
    assert hub.latest()["cameraExternalId"] == "camera-a"
    assert hub.latest()["cameraExternalId"] == "camera-a"
    assert starts == 1
    now += 5.01
    assert hub.latest() is None
    await hub.close()
    assert hub.latest() is None
    assert starts == 1


@pytest.mark.anyio
async def test_failure_clears_buffer_and_never_restarts_for_a_viewer():
    from smartsite_ai.preview_snapshot import LatestPreviewHub

    starts = 0
    complete = asyncio.Event()

    async def produce(sink):
        nonlocal starts
        starts += 1
        await sink.send_json({"type": "frame", "cameraExternalId": "camera-a"})
        await sink.send_json({"type": "error", "message": "fake-private-secret"})
        complete.set()

    hub = LatestPreviewHub(produce, "camera-a", asyncio.Lock())
    assert hub.latest() is None
    await complete.wait()
    await asyncio.sleep(0)
    for _ in range(3):
        assert hub.latest() is None
    assert starts == 1
    await hub.close()


@pytest.mark.anyio
async def test_wrong_camera_and_resource_contention_never_publish_frames():
    from smartsite_ai.preview_snapshot import LatestPreviewHub

    starts = 0

    async def produce(sink):
        nonlocal starts
        starts += 1
        await sink.send_json({"type": "frame", "cameraExternalId": "camera-b"})

    lock = asyncio.Lock()
    await lock.acquire()
    hub = LatestPreviewHub(produce, "camera-a", lock)
    assert hub.latest() is None
    await asyncio.sleep(0)
    assert starts == 0
    lock.release()
    assert hub.latest() is None
    await asyncio.sleep(0)
    assert hub.latest() is None
    assert starts == 1
    await hub.close()


def test_http_latest_authentication_source_binding_and_shutdown(monkeypatch):
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    starts = 0
    received_settings = []

    async def produce(sink, settings):
        nonlocal starts
        starts += 1
        received_settings.append(settings)
        await sink.send_json({"type": "frame", "cameraExternalId": "camera-a"})
        await asyncio.Event().wait()

    monkeypatch.setattr("smartsite_ai.app.stream_realtime", produce)
    app = create_app(
        Settings(
            _env_file=None,
            realtime_snapshot_enabled=True,
            realtime_source="rtsp://fake.invalid/live",
            realtime_model_path="fake-model-never-loaded.pt",
            realtime_camera_external_id="camera-a",
            backend_service_token="fake-server-only-token",
            backend_ingestion_url="http://fake.invalid/events",
        )
    )
    path = "/v1/realtime/preview/latest?cameraExternalId=camera-a"
    headers = {"Authorization": "Bearer fake-server-only-token"}
    with TestClient(app) as client:
        assert client.get(path).status_code == 401
        assert client.get(path + "&token=fake-server-only-token").status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get(path, headers=[(b"authorization", b"Bearer \xff")]).status_code == 401
        assert client.get(path + "&cameraExternalId=camera-a", headers=headers).status_code == 400
        assert client.get(path.replace("camera-a", "camera-b"), headers=headers).status_code == 404
        assert starts == 0
        first = client.get(path, headers=headers)
        assert first.status_code in (200, 503)
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert response.json()["sourceKind"] == "LIVE"
        assert response.json()["frame"]["cameraExternalId"] == "camera-a"
        assert response.headers["cache-control"] == "no-store"
        assert "fake-server-only-token" not in response.text
        assert client.get(path, headers=headers).status_code == 200
        assert starts == 1
        assert received_settings[0].backend_ingestion_url is None
    assert app.state.preview_hub.latest() is None


def test_snapshot_opt_in_is_required():
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    with TestClient(
        create_app(Settings(_env_file=None, backend_service_token="fake-token"))
    ) as client:
        response = client.get(
            "/v1/realtime/preview/latest?cameraExternalId=ppe-demo",
            headers={"Authorization": "Bearer fake-token"},
        )
    assert response.status_code == 503
