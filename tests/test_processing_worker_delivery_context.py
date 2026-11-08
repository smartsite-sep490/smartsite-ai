import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from test_processing_worker import REGION_ID, FakeStream, RecordingPipeline, _frame
from test_processing_worker_bundle_policy import Detector, bundle

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, TrackedCapturePolicy
from smartsite_ai.tracking import IoUPersonTracker


@pytest.mark.anyio
async def test_worker_freezes_confirmed_context_and_waits_when_ack_is_uncertain(tmp_path: Path):
    store = RegionConfigurationStore()
    active = bundle()
    await store.apply_bundle(active)
    calls = []

    async def snapshot(configuration, frame, source_kind):
        calls.append((configuration, frame, source_kind))
        if len(calls) == 1:
            return None
        return EventDeliveryContext.model_validate(
            {
                "schemaVersion": "1.0.0",
                "cameraId": active.camera_id,
                "workerGenerationId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "streamSessionId": str(frame.session_id),
                "configurationVersion": active.configuration_version,
                "bundleSha256": active.bundle_sha256,
                "viewRevision": active.view_revision,
                "receiptCommandId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "sourceKind": source_kind,
            }
        )

    outbox = SqliteEventOutbox(tmp_path / "worker.sqlite3")
    stream = FakeStream([])
    stream.config = stream.config.model_copy(update={"is_live": True})
    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=Detector(),
        pipeline=RecordingPipeline(),
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=AsyncMock(),
        delivery_context_provider=snapshot,
    )
    await worker._process_frame(_frame(0))
    assert (await outbox.counts()).pending == 0
    await worker._process_frame(_frame(1))
    (pending,) = await outbox.ready()
    assert pending.delivery_context is not None
    assert pending.delivery_context.bundle_sha256 == active.bundle_sha256
    assert pending.delivery_context.stream_session_id == pending.event.stream_session_id
    assert all(configuration is active for configuration, _, _ in calls)


@pytest.mark.anyio
async def test_replay_keeps_original_bound_policy_when_configuration_changes_during_inference(
    tmp_path,
):
    store = RegionConfigurationStore()
    original = bundle()
    await store.apply_bundle(original)
    context = EventDeliveryContext.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": original.camera_id,
            "workerGenerationId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "streamSessionId": str(_frame(0).session_id),
            "configurationVersion": 1,
            "bundleSha256": original.bundle_sha256,
            "viewRevision": 1,
            "receiptCommandId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "sourceKind": "REPLAY",
        }
    )

    async def snapshot(configuration, frame, kind):
        return context if await store.get_bundle("CAM-GATE-01") is configuration else None

    class UpdatingDetector(Detector):
        async def detect(self, frame):
            result = await super().detect(frame)
            await store.apply_bundle(bundle(2))
            return result

    outbox = SqliteEventOutbox(tmp_path / "replay.sqlite3")
    worker = HeadlessCameraProcessingWorker(
        stream=FakeStream([]),
        detector=UpdatingDetector(),
        pipeline=RecordingPipeline(),
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=AsyncMock(),
        delivery_context_provider=snapshot,
    )
    await worker._process_frame(_frame(0))
    (pending,) = await outbox.ready()
    assert pending.delivery_context == context


@pytest.mark.anyio
async def test_replay_without_definitive_ack_fails_instead_of_succeeding_with_no_events(tmp_path):
    store = RegionConfigurationStore()
    await store.apply_bundle(bundle())
    detector = AsyncMock()
    outbox = SqliteEventOutbox(tmp_path / "unconfirmed-replay.sqlite3")
    worker = HeadlessCameraProcessingWorker(
        stream=FakeStream([]),
        detector=detector,
        pipeline=RecordingPipeline(),
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=AsyncMock(),
        delivery_context_provider=AsyncMock(return_value=None),
    )
    with pytest.raises(RuntimeError, match="Replay frame has no confirmed"):
        await worker._process_frame(_frame(0))
    detector.detect.assert_not_awaited()
    assert (await outbox.counts()).pending == 0


@pytest.mark.anyio
@pytest.mark.parametrize("seal_available,frame_count", [(False, 2), (True, 2), (False, 1)])
async def test_tracked_worker_waits_for_admitted_predecessor(tmp_path, seal_available, frame_count):
    active = bundle()
    store = RegionConfigurationStore()
    await store.apply_bundle(active)
    stream = FakeStream([_frame(index) for index in range(frame_count)])
    stream.config = stream.config.model_copy(update={"is_live": True})
    site_id = "00000000-0000-4000-8000-000000000071"
    profile = "a" * 64
    context = EventDeliveryContext(
        schemaVersion="1.0.0",
        cameraId=active.camera_id,
        workerGenerationId="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        streamSessionId=str(_frame(0).session_id),
        configurationVersion=active.configuration_version,
        bundleSha256=active.bundle_sha256,
        viewRevision=active.view_revision,
        receiptCommandId="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        sourceKind="LIVE",
    )
    credential = WorkerGenerationCredential(
        schemaVersion="1.0.0",
        cameraId=context.camera_id,
        workerGenerationId=context.worker_generation_id,
        streamSessionId=context.stream_session_id,
        secret="b" * 43,
        expiresAt="2030-01-01T00:00:00.000Z",
    )
    scope = compute_canonical_payload_hash(
        {
            **context.model_dump(by_alias=True, exclude={"schema_version", "receipt_command_id"}),
            "siteId": site_id,
            "profileSha256": profile,
        }
    )
    posted = []

    def handle(request):
        if request.method == "POST":
            assert request.url.path.endswith("/tracked-parts")
            event = json.loads(request.content)["event"]
            posted.append(event)
            digest = compute_canonical_payload_hash(event)
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": event["eventId"],
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        if not seal_available:
            return httpx.Response(404)
        event = next(e for e in posted if e["framePartition"]["frameId"] in request.url.path)
        return httpx.Response(
            200,
            json={
                "kind": "ADMITTED_TRACKED_FRAME",
                "scopeSha256": scope,
                "sequenceNumber": event["framePartition"]["sequenceNumber"],
                "subjects": [
                    {
                        "segmentId": claim["segmentId"],
                        "eventId": event["eventId"],
                        "personObservationIndex": claim["personObservationIndex"],
                    }
                    for claim in event["continuity"]["subjects"]
                ],
                "retiredSegmentIds": [],
            },
        )

    outbox = SqliteEventOutbox(tmp_path / "tracked-worker.sqlite")
    async with BackendClient(
        "http://backend.test", "synthetic-token", transport=httpx.MockTransport(handle)
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,
            detector=Detector(),
            pipeline=Mf05Mf06Pipeline(
                tracker=IoUPersonTracker(),
                ppe=PpePipeline(),
                zones=RestrictedZonePipeline(),
                ppe_region_id=REGION_ID,
            ),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(
                outbox, backend, generation_credential=credential, tracked_batch_limit=1
            ),
            delivery_context_provider=AsyncMock(return_value=context),
            delivery_interval_seconds=0.01,
            tracked_capture=TrackedCapturePolicy(
                credential=credential,
                site_id=site_id,
                profile_sha256=profile,
                pending_seconds=10,
                predecessor_wait_seconds=0.15,
                max_pending=2,
                max_bytes=16000,
            ),
        )
        stream.config = stream.config.model_copy(update={"is_live": False})
        with pytest.raises(ValueError, match="LIVE"):
            HeadlessCameraProcessingWorker(
                stream=stream,
                detector=Detector(),
                pipeline=worker._pipeline,
                ppe_region_id=REGION_ID,
                configurations=store,
                outbox=outbox,
                dispatcher=worker._dispatcher,
                delivery_context_provider=AsyncMock(return_value=context),
                tracked_capture=worker._tracked_capture,
            )
        stream.config = stream.config.model_copy(update={"is_live": True})
        if seal_available:
            result = await worker.run()
            assert result.frames_processed == result.events_enqueued == 2
            assert len(posted) == 2
            assert await outbox.read_tracked_predecessor(scope, "1") is not None
            assert (
                posted[1]["continuity"]["subjects"][0]["link"]["previousEventId"]
                == posted[0]["eventId"]
            )
        else:
            with pytest.raises(TimeoutError, match="predecessor|final tracked"):
                await worker.run()
            assert len(posted) == 1
    assert stream.stopped
    assert not await outbox.ready()
