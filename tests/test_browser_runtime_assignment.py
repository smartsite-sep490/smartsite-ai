import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from test_browser_uplink_ingestion import sample_packet
from test_tracked_runtime_selection import document

from smartsite_ai.config import Settings
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.runtime.supervisor import CameraOutcome, RuntimeReport
from smartsite_ai.tools import run_camera_runtime as cli


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["assigned", "missing", "wrong-site", "wrong-lease"])
async def test_camera_runtime_selects_only_the_supervised_browser_source(
    tmp_path, monkeypatch, mode
):
    manifest, bundle = document(tmp_path)
    camera = manifest["cameras"][0]
    camera["source"] = {"kind": "browser"}
    camera["sampledContinuity"] = {"maxSourceSequenceGap": 64, "maxCaptureGapMs": 2000}
    generation = json.loads((tmp_path / "generation.json").read_text())
    assignment = {
        "siteId": camera["trackedV17"]["siteId"],
        "cameraId": camera["cameraId"],
        "leaseId": generation["streamSessionId"],
        "readerSecret": "r" * 43,
    }
    if mode == "wrong-site":
        assignment["siteId"] = generation["cameraId"]
    if mode == "wrong-lease":
        assignment["leaseId"] = generation["workerGenerationId"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assignment_path = tmp_path / "browser.json"
    assignment_path.write_text(json.dumps(assignment))
    assignment_path.chmod(0o600)
    calls = []

    def http(request):
        calls.append(request.url.path)
        assert request.headers["x-browser-frame-reader-secret"] == assignment["readerSecret"]
        if request.url.params["afterSequence"] == "42":
            return httpx.Response(204)
        return httpx.Response(
            200,
            json=sample_packet()
            | {
                "sessionId": assignment["leaseId"],
                "cameraExternalId": camera["cameraExternalId"],
            },
        )

    def runtime_backend(_settings):
        return BackendClient("https://backend.test", "service", transport=httpx.MockTransport(http))

    def runtime_config(_settings, **options):
        assert options == {"policy_v2": True}
        return BackendConfigurationClient(
            "https://backend.test",
            "service",
            policy_v2=True,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json=bundle.model_dump(mode="json", by_alias=True),
                    headers={"ETag": f'"sha256:{bundle.bundle_sha256}"'},
                )
            ),
        )

    async def execute(plan):
        assert await plan.preflight() is None
        if mode != "assigned":
            pytest.fail("invalid assignment must not construct a runtime")
        session = plan.build_sessions(SimpleNamespace())[0]
        stream = session.worker._stream
        await stream.start()
        try:
            frame = await asyncio.wait_for(stream.get_frame(), timeout=1)
            assert frame.sequence_number == 42 and str(frame.session_id) == assignment["leaseId"]
        finally:
            await stream.stop()
        return RuntimeReport(
            (CameraOutcome(camera["streamId"], camera["cameraExternalId"], "stopped"),),
            model_loaded=True,
            model_closed=True,
        )

    def fail_camera_open(_):
        pytest.fail("browser source must not use OpenCV")

    monkeypatch.setattr(cli.BackendClient, "from_settings", runtime_backend)
    monkeypatch.setattr(cli.BackendConfigurationClient, "from_settings", runtime_config)
    monkeypatch.setattr(cli, "OpenCvFrameSource", fail_camera_open)
    monkeypatch.setattr(cli, "execute_runtime", execute)
    options = {} if mode == "missing" else {"browser_reader_assignment_file": assignment_path}
    if mode == "assigned":
        await cli.run_manifest(manifest_path, settings=Settings(_env_file=None), **options)
        assert calls and all(path.endswith(f"/{assignment['leaseId']}/frame") for path in calls)
    else:
        with pytest.raises(CameraRuntimeError):
            await cli.run_manifest(manifest_path, settings=Settings(_env_file=None), **options)
        assert calls == []
