import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smartsite_ai.source_control_api import source_control_router
from smartsite_ai.source_control_models import SourcePolicy


def test_scoped_probe_is_authenticated_bounded_and_does_not_apply(tmp_path: Path):
    site = str(uuid4())
    calls = []

    async def probe(source, live, scope):
        calls.append((source, live))
        return {
            "width": 640,
            "height": 480,
            "capturedAt": "2026-11-20T00:00:00Z",
            "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
        }

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge-a",
            label="AI host",
            scopes={(site, "camera-a")},
            policy=SourcePolicy({"usb-front": 0}, {}, ()),
            service_token=SecretStr("a" * 48),
            probe=probe,
        )
    )
    client = TestClient(app)
    url = f"/v1/source-control/{site}/camera-a"
    assert client.get(url + "/capabilities").status_code == 401
    headers = {"Authorization": "Bearer " + "a" * 48}
    caps = client.get(url + "/capabilities", headers=headers)
    assert caps.status_code == 200
    assert caps.json()["browserCameraSupported"] is False
    assert caps.json()["applySupported"] is False
    result = client.post(
        url + "/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb-front"}
    )
    assert result.status_code == 200
    assert result.json()["state"] == "TESTED"
    assert result.json()["sourceKind"] == "LIVE"
    assert result.json()["receiptId"]
    assert calls == [("0", True)]
    assert client.post(url + "/apply", headers=headers, json={}).status_code == 404
    foreign = client.get(f"/v1/source-control/{uuid4()}/camera-a/capabilities", headers=headers)
    assert foreign.status_code == 404


def test_invalid_or_secret_input_never_reaches_capture(tmp_path):
    site = str(uuid4())

    async def forbidden(source, live, scope):
        raise AssertionError("must not capture")

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={(site, "cam")},
            policy=SourcePolicy({}, {}, ()),
            service_token=SecretStr("b" * 48),
            probe=forbidden,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + "b" * 48}
    for body in [
        {"kind": "RTSP", "endpoint": "rtsp://u:super-secret@127.0.0.1/x"},
        {"kind": "WEBCAM", "deviceId": "unknown"},
        {"kind": "WEBCAM", "password": "super-secret"},
    ]:
        r = client.post(f"/v1/source-control/{site}/cam/probe", headers=headers, json=body)
        assert r.status_code == 400
        assert "super-secret" not in r.text
        assert "127.0.0.1" not in r.text


def test_http_disconnect_cancels_probe_and_releases_owner_without_waiting_timeout():
    import json

    async def run():
        site = str(uuid4())
        started = asyncio.Event()
        released = asyncio.Event()

        async def probe(source, live, scope):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        app = FastAPI()
        app.include_router(
            source_control_router(
                host_id="edge",
                label="Host",
                scopes={(site, "cam")},
                policy=SourcePolicy({"usb0": 0}, {}, ()),
                service_token=SecretStr("a" * 48),
                probe=probe,
            )
        )
        body = json.dumps({"kind": "WEBCAM", "deviceId": "usb0"}).encode()
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await started.wait()
            return {"type": "http.disconnect"}

        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": f"/v1/source-control/{site}/cam/probe",
            "raw_path": b"",
            "query_string": b"",
            "headers": [
                (b"authorization", b"Bearer " + b"a" * 48),
                (b"content-type", b"application/json"),
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 80),
            "root_path": "",
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=1)
        assert released.is_set()
        assert next(m["status"] for m in messages if m["type"] == "http.response.start") == 499

    asyncio.run(run())
