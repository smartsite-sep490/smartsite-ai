import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from test_tracked_policy_pipeline import policy_bundle

from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.runtime.assembly import (
    assemble_sessions,
    close_configuration_clients,
    preflight_manifest,
)
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest


def document(tmp_path):
    bundle = policy_bundle(outside=True)
    model = tmp_path / "model.json"
    model.write_text("{}")
    credential = tmp_path / "generation.json"
    credential.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "cameraId": bundle.camera_id,
                "workerGenerationId": "00000000-0000-4000-8000-000000000031",
                "streamSessionId": "00000000-0000-4000-8000-000000000032",
                "secret": "x" * 43,
                "expiresAt": (datetime.now(UTC) + timedelta(minutes=10))
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            }
        )
    )
    value = {
        "schemaVersion": "1",
        "modelSpec": str(model),
        "cameras": [
            {
                "streamId": "synthetic",
                "cameraId": bundle.camera_id,
                "cameraExternalId": bundle.camera_external_id,
                "ppeRegionId": bundle.regions[0].region_id,
                "source": {"kind": "path", "path": "0"},
                "live": True,
                "outbox": str(tmp_path / "outbox.sqlite"),
                "evidenceDir": str(tmp_path),
                "generationCredentialFile": str(credential),
                "trackedV17": {
                    "siteId": "00000000-0000-4000-8000-000000000077",
                    "profileSha256": "b" * 64,
                    "pendingSeconds": 30,
                    "predecessorWaitSeconds": 1,
                    "maxPending": 2,
                    "maxBytes": 100_000,
                },
            }
        ],
    }
    return value, bundle


@pytest.mark.parametrize(
    "invalid",
    ["replay", "no-evidence", "no-generation", "partitioned", "legacy-bound", "wait", "capacity"],
)
def test_tracked_runtime_refuses_ambiguous_or_incomplete_configuration_before_resources(
    tmp_path, invalid
):
    value, _ = document(tmp_path)
    camera = value["cameras"][0]
    if invalid == "replay":
        camera["live"] = False
    elif invalid == "no-evidence":
        camera.pop("evidenceDir")
    elif invalid == "no-generation":
        camera.pop("generationCredentialFile")
    elif invalid == "partitioned":
        camera["partitionedFrameDelivery"] = True
    elif invalid == "legacy-bound":
        camera["generationBoundEventDelivery"] = True
    elif invalid == "wait":
        camera["trackedV17"]["predecessorWaitSeconds"] = 30
    else:
        camera["trackedV17"]["maxPending"] = True
    with pytest.raises(CameraRuntimeError):
        parse_manifest(value)


@pytest.mark.anyio
async def test_explicit_tracked_manifest_assembles_v2_worker_without_loading_model_or_camera(
    tmp_path,
):
    from types import SimpleNamespace

    value, bundle = document(tmp_path)
    bound = bind_manifest(parse_manifest(value), {})
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=bundle.model_dump(mode="json", by_alias=True),
            headers={"ETag": f'"sha256:{bundle.bundle_sha256}"'},
        )

    def legacy():
        pytest.fail("Tracked source must not fetch the legacy configuration endpoint")

    def v2():
        return BackendConfigurationClient(
            "https://backend.example",
            "synthetic-service",
            policy_v2=True,
            transport=httpx.MockTransport(response),
        )

    failed, plans = await preflight_manifest(
        bound, configuration_client_factory=legacy, policy_v2_client_factory=v2
    )
    assert failed is None and len(plans) == 1
    backends = []

    def backend():
        client = BackendClient("https://backend.example", "synthetic-service")
        backends.append(client)
        return client

    try:
        sessions = assemble_sessions(
            plans,
            detector=SimpleNamespace(),
            source_factory=lambda _: SimpleNamespace(),
            backend_client_factory=backend,
            evidence_max_bytes=1_048_576,
            delivery_interval_seconds=0.05,
        )
        worker = sessions[0].worker
        assert worker._tracked_capture.profile_sha256 == "b" * 64
        assert worker._pipeline._tracked_v17 is True
        assert worker._dispatcher._tracked_batch_limit == 2
        assert worker._delivery_context_provider.__self__ is plans[0].poller
        assert worker._evidence_publisher._bind_content_hash is True
        assert plans[0].poller._require_bundle is True
        assert plans[0].poller._pin_bundle_scope is True
        assert len(requests) == 1 and requests[0].url.path.endswith("/configuration-bundle/v2")
    finally:
        await close_configuration_clients(plans)
        for client in backends:
            await client.aclose()
