import asyncio
import json
from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smartsite_ai import source_control_probe
from smartsite_ai.tools import run_source_control as control

SITE = "11111111-1111-4111-8111-111111111111"


def test_existing_owned_webcam_reuses_only_its_matching_preview(monkeypatch):
    captures = []

    async def capture(source, live):
        captures.append(source)
        raise AssertionError("must not open the active device twice")

    def respond(request):
        assert request.url.path == "/v1/realtime/configuration-image"
        assert request.url.params["cameraExternalId"] == "camera-a"
        return httpx.Response(
            200,
            json={
                "sourceKind": "LIVE",
                "frame": {
                    "cameraExternalId": "camera-a",
                    "width": 640,
                    "height": 480,
                    "capturedAt": datetime.now(UTC).isoformat(),
                    "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
                },
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(control, "capture_source", capture)
    monkeypatch.setattr(
        control.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    manifest = control.HostManifest(
        host_id="edge",
        label="Host",
        scopes=[(SITE, "camera-a"), (SITE, "other-camera")],
        webcams={"usb0": 0},
        active_sources=[
            control.ActiveSource(
                site_id=SITE,
                camera_external_id="camera-a",
                source=SecretStr("0"),
                preview_origin="http://127.0.0.1:62034",
            )
        ],
    )
    client = TestClient(control.build_source_control_app(manifest, SecretStr("a" * 48)))
    headers = {"Authorization": "Bearer " + "a" * 48}
    result = client.post(
        f"/v1/source-control/{SITE}/camera-a/probe",
        headers=headers,
        json={"kind": "WEBCAM", "deviceId": "usb0"},
    )
    assert result.status_code == 200
    assert result.json()["frame"]["width"] == 640
    other = client.post(
        f"/v1/source-control/{SITE}/other-camera/probe",
        headers=headers,
        json={"kind": "WEBCAM", "deviceId": "usb0"},
    )
    assert other.status_code == 503
    assert captures == []


def test_capture_cancellation_reaps_child_and_never_puts_source_in_command(monkeypatch):
    class Output:
        async def read(self, size):
            await asyncio.Event().wait()

    class Process:
        stdout = Output()
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True
            self.returncode = -1

        def kill(self):
            self.returncode = -1

        async def wait(self):
            return self.returncode

    process = Process()

    async def run():
        started = asyncio.Event()

        async def spawn(*args, **kwargs):
            assert all("secret-credential" not in str(arg) for arg in args)
            assert (
                kwargs["env"]["SMARTSITE_AI_SOURCE_TEST_SOURCE"]
                == "rtsp://secret-credential@192.168.1.5/x"
            )
            assert "PGPASSWORD" not in kwargs["env"]
            started.set()
            return process

        monkeypatch.setenv("PGPASSWORD", "never-forward-database-password")
        monkeypatch.setattr(source_control_probe.asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(
            source_control_probe.capture_source("rtsp://secret-credential@192.168.1.5/x", True)
        )
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation must propagate")
        assert process.terminated

    asyncio.run(run())


def test_split_child_stdout_is_read_until_eof(monkeypatch):
    payload = json.dumps({"imageDataUrl": "x" * 180000}).encode()
    chunks = [payload[:50000], payload[50000:120000], payload[120000:], b""]

    class Output:
        async def read(self, size):
            return chunks.pop(0)

    class Process:
        stdout = Output()
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -1

        def kill(self):
            self.returncode = -1

    async def spawn(*args, **kwargs):
        return Process()

    monkeypatch.setattr(source_control_probe.asyncio, "create_subprocess_exec", spawn)
    result = asyncio.run(source_control_probe.capture_source("0", True))
    assert result == {"imageDataUrl": "x" * 180000}
    assert chunks == []
