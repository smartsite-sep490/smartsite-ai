from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_processing_worker import REGION_ID, FakeStream, RecordingPipeline, _frame
from test_processing_worker_bundle_policy import Detector, bundle

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.integrations.outbox import SqliteEventOutbox
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker


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
