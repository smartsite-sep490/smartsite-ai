import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from test_mf05_mf06_bundle_policy import detection, scene_batch
from test_tracked_policy_pipeline import pipeline, policy_bundle
from test_tracked_transport import packet

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, TrackedCapturePolicy

SITE = "00000000-0000-4000-8000-000000000077"


@pytest.mark.parametrize("field", ["pending_seconds", "predecessor_wait_seconds"])
def test_direct_tracked_policy_rejects_boolean_timeouts(field):
    _, _, credential, *_ = packet()
    values = dict(
        credential=credential,
        site_id=SITE,
        profile_sha256="b" * 64,
        pending_seconds=30,
        predecessor_wait_seconds=0.5,
        max_pending=2,
        max_bytes=100_000,
    )
    values[field] = True
    with pytest.raises(ValueError, match="limits"):
        TrackedCapturePolicy(**values)


@pytest.mark.anyio
async def test_staging_failure_cancels_pending_live_capture_and_stops_stream():
    import asyncio
    from unittest.mock import AsyncMock

    from smartsite_ai.domain.tracked_observations import TrackedScopeTransitionRequired

    _, _, credential, *_ = packet()
    capture_waiting = asyncio.Event()
    capture_cancelled = asyncio.Event()
    first_frame = True

    async def get_frame():
        nonlocal first_frame
        if first_frame:
            first_frame = False
            return object()
        capture_waiting.set()
        try:
            await asyncio.Future()
        finally:
            capture_cancelled.set()

    stream = SimpleNamespace(
        config=SimpleNamespace(is_live=True, camera_external_id="CAM", stream_id="CAM"),
        start=AsyncMock(),
        stop=AsyncMock(),
        get_frame=get_frame,
    )
    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=SimpleNamespace(),
        pipeline=SimpleNamespace(),
        ppe_region_id="region",
        configurations=SimpleNamespace(get=AsyncMock(return_value=object())),
        outbox=SimpleNamespace(initialize=AsyncMock()),
        dispatcher=SimpleNamespace(drain_once=AsyncMock(return_value=0)),
        delivery_context_provider=AsyncMock(),
        tracked_capture=TrackedCapturePolicy(credential, SITE, "b" * 64, 30, 5, 2, 100_000),
    )

    async def stage():
        await capture_waiting.wait()
        raise TrackedScopeTransitionRequired("synthetic configuration changed")

    async def process(_frame):
        worker._last_frame_enqueued = False
        await worker._live_staging_queue.put(stage)

    worker._process_frame = process
    with pytest.raises(TrackedScopeTransitionRequired, match="configuration changed"):
        await asyncio.wait_for(worker.run(), timeout=1)
    assert capture_cancelled.is_set()
    stream.stop.assert_awaited_once()


@pytest.mark.anyio
async def test_final_frame_can_finish_after_predecessor_wait_within_its_pending_lease():
    import asyncio
    from unittest.mock import AsyncMock

    from smartsite_ai.ingestion.queue import QueueClosedError
    from smartsite_ai.ingestion.status import StreamState
    from smartsite_ai.integrations.outbox import OutboxCounts

    loop = asyncio.get_running_loop()
    staged_at = None

    async def pending(_credential):
        return staged_at is not None and loop.time() - staged_at < 0.05

    async def remaining(_credential):
        return 0.2 - (loop.time() - staged_at) if await pending(_credential) else None

    calls = 0

    async def get_frame():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise QueueClosedError("finite synthetic source complete")
        return SimpleNamespace()

    async def process(_frame):
        nonlocal staged_at
        staged_at = loop.time()
        worker._last_frame_enqueued = False

    _, _, credential, *_ = packet()
    stream = SimpleNamespace(
        config=SimpleNamespace(is_live=True, camera_external_id="CAM", stream_id="CAM"),
        start=AsyncMock(),
        stop=AsyncMock(),
        get_frame=get_frame,
        snapshot=lambda: SimpleNamespace(state=StreamState.STOPPED),
    )
    outbox = SimpleNamespace(
        initialize=AsyncMock(),
        has_pending_tracked_frame=pending,
        tracked_pending_remaining_seconds=remaining,
        counts=AsyncMock(return_value=OutboxCounts(0, 0, 0)),
    )
    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=SimpleNamespace(),
        pipeline=SimpleNamespace(),
        ppe_region_id="region",
        configurations=SimpleNamespace(get=AsyncMock(return_value=object())),
        outbox=outbox,
        dispatcher=SimpleNamespace(drain_once=AsyncMock(return_value=0)),
        delivery_interval_seconds=0.005,
        delivery_context_provider=AsyncMock(),
        tracked_capture=TrackedCapturePolicy(credential, SITE, "b" * 64, 0.2, 0.01, 2, 100_000),
    )
    worker._process_frame = process
    result = await worker.run()
    assert result.frames_processed == 1 and loop.time() - staged_at >= 0.05


@pytest.mark.anyio
async def test_changed_policy_is_reported_before_new_ack_inference_or_capture_submission(tmp_path):
    from unittest.mock import AsyncMock

    from test_policy_configuration_bundle import packet as bundle_packet

    from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle

    old = policy_bundle(outside=True)
    value = old.model_dump(mode="json", by_alias=True)
    value.pop("bundleSha256")
    value["configurationVersion"] += 1
    new = CameraPolicyConfigurationBundle.from_wire_bytes(bundle_packet(value))
    store = RegionConfigurationStore(allow_policy_v2=True)
    await store.apply_bundle(new)
    _, _, credential, *_ = packet()
    provider = AsyncMock(
        side_effect=AssertionError("new scope must not be acknowledged by this worker")
    )
    worker = HeadlessCameraProcessingWorker(
        stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),
        detector=SimpleNamespace(),
        pipeline=SimpleNamespace(),
        ppe_region_id=old.regions[0].region_id,
        configurations=store,
        outbox=SimpleNamespace(),
        dispatcher=SimpleNamespace(),
        delivery_context_provider=provider,
        tracked_capture=TrackedCapturePolicy(credential, SITE, "b" * 64, 30, 1, 2, 100000),
    )
    worker._active_snapshot = old
    with pytest.raises(RuntimeError, match="scope transition") as error:
        await worker._process_frame(SimpleNamespace(camera_external_id=old.camera_external_id))
    assert type(error.value).__name__ == "TrackedScopeTransitionRequired"
    provider.assert_not_awaited()


@pytest.mark.anyio
async def test_real_pipeline_worker_outbox_and_http_transport_keep_exact_frame_evidence(tmp_path):
    bundle = policy_bundle(outside=True)
    store = RegionConfigurationStore(allow_policy_v2=True)
    await store.apply_bundle(bundle)
    template = scene_batch()
    _, old_context, old_credential, *_ = packet()
    context = old_context.model_copy(
        update={
            "camera_id": bundle.camera_id,
            "stream_session_id": str(template.session_id),
            "configuration_version": bundle.configuration_version,
            "view_revision": bundle.view_revision,
            "bundle_sha256": bundle.bundle_sha256,
            "source_kind": "LIVE",
        }
    )
    credential = old_credential.model_copy(
        update={
            "camera_id": bundle.camera_id,
            "stream_session_id": str(template.session_id),
        }
    )
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite")
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    received = []
    seals = {}
    scope = compute_canonical_payload_hash(
        {
            **context.model_dump(by_alias=True, exclude={"schema_version", "receipt_command_id"}),
            "siteId": SITE,
            "profileSha256": "b" * 64,
        }
    )

    class Detector:
        async def detect(self, source):
            return template.model_copy(
                update={
                    "sequence_number": source.sequence_number,
                    "captured_at": source.captured_at,
                    "detections": (
                        template.detections[0],
                        detection("no-hardhat", (0.40, 0.22, 0.48, 0.30)),
                    ),
                }
            )

    async def issue(_bundle, _frame, kind):
        assert kind == "LIVE"
        return context

    def backend(request):
        if request.method == "GET":
            return httpx.Response(200, json=seals[request.url.path.rsplit("/", 2)[-2]])
        original = json.loads(request.content)["event"]
        received.append(original)
        partition = original["framePartition"]
        digest = compute_canonical_payload_hash(original)
        seals[partition["frameId"]] = {
            "kind": "ADMITTED_TRACKED_FRAME",
            "scopeSha256": scope,
            "sequenceNumber": partition["sequenceNumber"],
            "subjects": [
                {
                    "segmentId": item["segmentId"],
                    "eventId": original["eventId"],
                    "personObservationIndex": item["personObservationIndex"],
                }
                for item in original["continuity"]["subjects"]
            ],
            "retiredSegmentIds": [],
        }
        return httpx.Response(
            200,
            json={
                "kind": "RAW_STORED",
                "eventId": original["eventId"],
                "payloadHash": digest,
                "bindingHash": context.binding_hash(digest),
            },
        )

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox, client, generation_credential=credential, tracked_batch_limit=2
        )
        worker = HeadlessCameraProcessingWorker(
            stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),
            detector=Detector(),
            pipeline=pipeline(),
            ppe_region_id=bundle.regions[0].region_id,
            configurations=store,
            outbox=outbox,
            dispatcher=dispatcher,
            evidence_publisher=publisher,
            delivery_context_provider=issue,
            tracked_capture=TrackedCapturePolicy(credential, SITE, "b" * 64, 30, 1, 2, 100_000),
        )
        for sequence in (1, 2, 3):
            source = FrameEnvelope(
                stream_id=template.stream_id,
                session_id=template.session_id,
                camera_external_id=template.camera_external_id,
                captured_at=datetime(2026, 10, 27, 12, 0, sequence, tzinfo=UTC),
                sequence_number=sequence,
                width=640,
                height=480,
                payload=bytes([20, 40, 60]) * 640 * 480,
            )
            await worker._process_frame(source)
            await dispatcher.drain_once()
            await dispatcher.drain_once()
    assert len(received) == 3
    assert all(item["schemaVersion"] == "1.7.0" for item in received)
    assert [[item["type"] for item in event["observations"]] for event in received] == [
        ["PERSON"],
        ["PERSON"],
        ["PERSON", "PPE"],
    ]
    assert received[-1]["observations"][1]["scope"] == "CAMERA"
    assert received[-1]["observations"][1]["status"] == "MISSING"
    assert len(received[-1]["evidence"]) == 1
    assert len(list(tmp_path.glob("*.jpg"))) == 1
    assert not await outbox.has_pending_tracked_frame(credential)
    assert (await outbox.read_tracked_predecessor(scope, "3")).subjects[0].event_id == received[-1][
        "eventId"
    ]
