from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.processing_worker import (
    HeadlessCameraProcessingWorker,
    ProcessingWorkerSourceError,
)

SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
REGION_ID = "22222222-2222-4222-8222-222222222222"


def _configuration(version: int = 1) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": version,
            "cameraExternalId": "CAM-GATE-01",
            "regions": (
                {
                    "regionId": REGION_ID,
                    "geometryVersion": version,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))},
                },
            ),
        }
    )


def _frame(sequence: int) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="gate-stream",
        session_id=SESSION_ID,
        camera_external_id="CAM-GATE-01",
        captured_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC) + timedelta(seconds=sequence),
        width=2,
        height=2,
        sequence_number=sequence,
        payload=b"\x00" * 12,
    )


class FakeStream:
    def __init__(
        self,
        frames: list[FrameEnvelope],
        *,
        terminal_state: StreamState = StreamState.STOPPED,
    ) -> None:
        self.config = StreamConfig(
            stream_id="gate-stream",
            camera_external_id="CAM-GATE-01",
            source_url="test://gate",
            is_live=False,
        )
        self.frames = list(frames)
        self.started = False
        self.stopped = False
        self.terminal_state = terminal_state

    async def start(self) -> None:
        self.started = True

    async def get_frame(self) -> FrameEnvelope:
        await asyncio.sleep(0)
        if not self.frames:
            raise QueueClosedError("complete")
        return self.frames.pop(0)

    async def stop(self) -> None:
        self.stopped = True

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(state=self.terminal_state)


class FakeDetector:
    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="ppe-model",
            model_version="1",
            model_sha256="a" * 64,
            detections=(),
        )


class RecordingPipeline:
    def __init__(self) -> None:
        self.configuration_versions: list[int] = []
        self.event_ids: list[str] = []

    def process(
        self,
        batch: DetectionBatch,
        *,
        region_configuration: CameraRegionConfiguration,
        event_id: str,
    ) -> TechnicalObservationEvent:
        self.configuration_versions.append(region_configuration.configuration_version)
        self.event_ids.append(event_id)
        return TechnicalObservationEvent.create(
            event_id=event_id,
            camera_external_id=batch.camera_external_id,
            stream_session_id=str(batch.session_id),
            captured_at=batch.captured_at.isoformat(),
            frame_dimensions={"width": batch.frame_width, "height": batch.frame_height},
            observations=[
                {
                    "type": "ZONE_ENTRY",
                    "trackId": 7,
                    "regionId": REGION_ID,
                    "geometryVersion": region_configuration.configuration_version,
                    "confidence": 0.9,
                }
            ],
        )


@pytest.mark.anyio
@pytest.mark.parametrize("broken_preview", [False, True])
async def test_worker_runs_without_browser_and_delivers_deterministic_events(
    tmp_path: Path,
    broken_preview: bool,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        event_id = json.loads(request.content)["eventId"]
        return httpx.Response(
            202,
            json={"eventId": event_id, "status": "PROCESSED", "alertIds": []},
        )

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_frame(0), _frame(1)])
    pipeline = RecordingPipeline()
    analyzed = []

    def preview(item):
        analyzed.append(item)
        if broken_preview:
            raise RuntimeError("SYNTHETIC_PREVIEW_FAILURE")

    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FakeDetector(),
            pipeline=pipeline,
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
            delivery_interval_seconds=0.01,
            analysis_observer=preview,
        )
        result = await worker.run()

    assert stream.started is True
    assert stream.stopped is True
    assert result.frames_processed == 2
    assert result.events_enqueued == 2
    assert result.outbox.delivered == 2
    assert result.outbox.pending == 0
    assert len(requests) == 2
    assert len(set(pipeline.event_ids)) == 2
    assert len(analyzed) == (1 if broken_preview else 2)
    for index, item in enumerate(analyzed):
        assert item.frame.sequence_number == index
        assert item.observation.event_id == pipeline.event_ids[index]
        assert item.observation.captured_at == item.frame.captured_at_iso
        assert item.configuration.camera_external_id == item.frame.camera_external_id
        assert item.confirmed_ppe_items == frozenset()


@pytest.mark.anyio
async def test_worker_replays_crash_surviving_event_before_new_source_session(
    tmp_path: Path,
) -> None:
    submitted_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        event_id = json.loads(request.content)["eventId"]
        submitted_ids.append(event_id)
        return httpx.Response(
            202,
            json={"eventId": event_id, "status": "PROCESSED", "alertIds": []},
        )

    old_event = RecordingPipeline().process(
        DetectionBatch.from_frame(
            _frame(99),
            model_artifact_id="ppe-model",
            model_version="1",
            model_sha256="a" * 64,
            detections=(),
        ),
        region_configuration=_configuration(),
        event_id="99999999-9999-4999-8999-999999999999",
    )
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    await outbox.enqueue(old_event)
    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_frame(0)])
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FakeDetector(),
            pipeline=RecordingPipeline(),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
        )
        await worker.run()

    assert submitted_ids[0] == old_event.event_id
    assert len(submitted_ids) == 2


@pytest.mark.anyio
async def test_worker_reads_configuration_at_each_frame_boundary(tmp_path: Path) -> None:
    store = RegionConfigurationStore()
    await store.apply(_configuration(1))

    class UpdatingDetector(FakeDetector):
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            if frame.sequence_number == 0:
                await store.apply(_configuration(2))
            return await super().detect(frame)

    def handler(request: httpx.Request) -> httpx.Response:
        event_id = json.loads(request.content)["eventId"]
        return httpx.Response(
            202,
            json={"eventId": event_id, "status": "PROCESSED", "alertIds": []},
        )

    stream = FakeStream([_frame(0), _frame(1)])
    pipeline = RecordingPipeline()
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=UpdatingDetector(),
            pipeline=pipeline,
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
        )
        await worker.run()

    assert pipeline.configuration_versions == [1, 2]


@pytest.mark.anyio
async def test_worker_requires_configuration_before_opening_source(tmp_path: Path) -> None:
    stream = FakeStream([_frame(0)])
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FakeDetector(),
            pipeline=RecordingPipeline(),
            ppe_region_id=REGION_ID,
            configurations=RegionConfigurationStore(),
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
        )
        with pytest.raises(RuntimeError, match="loaded before worker start"):
            await worker.run()

    assert stream.started is False
    assert stream.stopped is False


@pytest.mark.anyio
async def test_worker_stops_source_when_processing_fails(tmp_path: Path) -> None:
    class FailingDetector(FakeDetector):
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            raise RuntimeError("inference failed")

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_frame(0)])
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        max_retries=0,
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FailingDetector(),
            pipeline=RecordingPipeline(),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
        )
        with pytest.raises(RuntimeError, match="inference failed"):
            await worker.run()

    assert stream.stopped is True


@pytest.mark.anyio
async def test_worker_reports_source_failure_instead_of_false_success(tmp_path: Path) -> None:
    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([], terminal_state=StreamState.ERROR)
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        max_retries=0,
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FakeDetector(),
            pipeline=RecordingPipeline(),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
        )
        with pytest.raises(ProcessingWorkerSourceError, match="error state"):
            await worker.run()

    assert stream.started is True
    assert stream.stopped is True


@pytest.mark.anyio
async def test_worker_stops_capture_when_delivery_loop_fails(tmp_path: Path) -> None:
    class BlockingStream(FakeStream):
        async def get_frame(self) -> FrameEnvelope:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class FailingDispatcher:
        def __init__(self) -> None:
            self.calls = 0

        async def drain_once(self) -> int:
            self.calls += 1
            if self.calls == 1:
                return 0
            raise RuntimeError("delivery storage failed")

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = BlockingStream([])
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    dispatcher = FailingDispatcher()
    worker = HeadlessCameraProcessingWorker(
        stream=stream,  # type: ignore[arg-type]
        detector=FakeDetector(),
        pipeline=RecordingPipeline(),
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=dispatcher,  # type: ignore[arg-type]
        delivery_interval_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="delivery storage failed"):
        await asyncio.wait_for(worker.run(), timeout=1)

    assert dispatcher.calls == 2
    assert stream.stopped is True


@pytest.mark.anyio
async def test_worker_cancels_blocked_delivery_on_abnormal_shutdown(tmp_path: Path) -> None:
    class BlockingStream(FakeStream):
        async def get_frame(self) -> FrameEnvelope:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class BlockingDispatcher:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def drain_once(self) -> int:
            self.calls += 1
            if self.calls == 1:
                return 0
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            raise AssertionError("unreachable")

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = BlockingStream([])
    dispatcher = BlockingDispatcher()
    worker = HeadlessCameraProcessingWorker(
        stream=stream,  # type: ignore[arg-type]
        detector=FakeDetector(),
        pipeline=RecordingPipeline(),
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve()),
        dispatcher=dispatcher,  # type: ignore[arg-type]
    )

    task = asyncio.create_task(worker.run())
    await asyncio.wait_for(dispatcher.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert dispatcher.cancelled.is_set()
    assert stream.stopped is True


@pytest.mark.anyio
async def test_ppe_confirmation_resets_when_region_geometry_changes(tmp_path: Path) -> None:
    class UpdatingDetector(FakeDetector):
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            if frame.sequence_number == 1:
                await store.apply(_configuration(2))
            return await super().detect(frame)

    class MissingPpePipeline:
        def process(
            self,
            batch: DetectionBatch,
            *,
            region_configuration: CameraRegionConfiguration,
            event_id: str,
        ) -> TechnicalObservationEvent:
            return TechnicalObservationEvent.create(
                event_id=event_id,
                camera_external_id=batch.camera_external_id,
                stream_session_id=str(batch.session_id),
                captured_at=batch.captured_at.isoformat(),
                frame_dimensions={"width": batch.frame_width, "height": batch.frame_height},
                observations=[
                    {
                        "type": "PERSON",
                        "trackId": 7,
                        "confidence": 0.9,
                        "boundingBox": {
                            "x1": 0.1,
                            "y1": 0.1,
                            "x2": 0.6,
                            "y2": 0.9,
                            "coordinateSpace": "NORMALIZED_0_1",
                        },
                    },
                    {
                        "type": "PPE",
                        "trackId": 7,
                        "ppeItem": "HARD_HAT",
                        "status": "MISSING",
                        "regionId": REGION_ID,
                        "geometryVersion": region_configuration.configuration_version,
                        "confidence": 0.9,
                    },
                ],
            )

    def handler(request: httpx.Request) -> httpx.Response:
        event_id = json.loads(request.content)["eventId"]
        return httpx.Response(
            202,
            json={"eventId": event_id, "status": "PROCESSED", "alertIds": []},
        )

    store = RegionConfigurationStore()
    await store.apply(_configuration(1))
    stream = FakeStream([_frame(index) for index in range(5)])
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=UpdatingDetector(),
            pipeline=MissingPpePipeline(),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
            delivery_interval_seconds=0.01,
        )
        result = await worker.run()

    assert result.frames_processed == 5
    assert result.events_enqueued == 1
    assert result.outbox.delivered == 1


@pytest.mark.anyio
async def test_analysis_preview_retains_confirmed_ppe_until_cleared_and_isolates_resets(
    tmp_path: Path,
) -> None:
    session_b = UUID("33333333-3333-4333-8333-333333333333")
    deliveries: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        deliveries.append(body)
        return httpx.Response(
            202,
            json={"eventId": body["eventId"], "status": "PROCESSED", "alertIds": []},
        )

    def frame_for(sequence: int, session_id: UUID = SESSION_ID) -> FrameEnvelope:
        return FrameEnvelope(
            stream_id="gate-stream",
            session_id=session_id,
            camera_external_id="CAM-GATE-01",
            captured_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC) + timedelta(seconds=sequence),
            width=2,
            height=2,
            sequence_number=sequence,
            payload=b"\x00" * 12,
        )

    class SequencePpePipeline:
        def __init__(self, statuses_by_seq: dict[int, str]) -> None:
            self._statuses = statuses_by_seq

        def process(
            self,
            batch: DetectionBatch,
            *,
            region_configuration: CameraRegionConfiguration,
            event_id: str,
        ) -> TechnicalObservationEvent:
            status = self._statuses[batch.sequence_number]
            return TechnicalObservationEvent.create(
                event_id=event_id,
                camera_external_id=batch.camera_external_id,
                stream_session_id=str(batch.session_id),
                captured_at=batch.captured_at.isoformat(),
                frame_dimensions={"width": batch.frame_width, "height": batch.frame_height},
                observations=[
                    {
                        "type": "PERSON",
                        "trackId": 7,
                        "confidence": 0.9,
                        "boundingBox": {
                            "x1": 0.1,
                            "y1": 0.1,
                            "x2": 0.6,
                            "y2": 0.9,
                            "coordinateSpace": "NORMALIZED_0_1",
                        },
                    },
                    {
                        "type": "PPE",
                        "trackId": 7,
                        "ppeItem": "HARD_HAT",
                        "status": status,
                        "regionId": REGION_ID,
                        "geometryVersion": region_configuration.configuration_version,
                        "confidence": 0.9,
                    },
                    {
                        "type": "PPE",
                        "trackId": 7,
                        "ppeItem": "SAFETY_VEST",
                        "status": status,
                        "regionId": REGION_ID,
                        "geometryVersion": region_configuration.configuration_version,
                        "confidence": 0.9,
                    },
                ],
            )

    expected = frozenset({(7, "HARD_HAT"), (7, "SAFETY_VEST")})

    # Frames 0..4: MISSING (confirm at 2; keep preview at 3..4, single delivery)
    # Frames 5..6: PRESENT x2 (still retained after frame 5, cleared after frame 6)
    # Frame 7..9: MISSING in new session_b (confirm at frame 9; frame 7 starts clean with no bleed)
    frames = [
        *(frame_for(i, SESSION_ID) for i in range(7)),
        *(frame_for(i, session_b) for i in range(7, 10)),
    ]
    statuses = {
        0: "MISSING",
        1: "MISSING",
        2: "MISSING",
        3: "MISSING",
        4: "MISSING",
        5: "PRESENT",
        6: "PRESENT",
        7: "MISSING",
        8: "MISSING",
        9: "MISSING",
    }
    analyzed: list[tuple[int, UUID, frozenset[tuple[int, str]]]] = []
    store = RegionConfigurationStore()
    await store.apply(_configuration(1))
    stream = FakeStream(frames)
    outbox = SqliteEventOutbox((tmp_path / "outbox-retained.sqlite3").resolve())

    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FakeDetector(),
            pipeline=SequencePpePipeline(statuses),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
            delivery_interval_seconds=0.01,
            analysis_observer=lambda item: analyzed.append(
                (item.frame.sequence_number, item.frame.session_id, item.confirmed_ppe_items)
            ),
        )
        result = await worker.run()

    assert result.frames_processed == 10
    # Exactly 1 confirmed delivery per session (frame 2 in SESSION_ID, frame 9 in session_b)
    assert result.events_enqueued == 2
    assert result.outbox.delivered == 2
    assert len(deliveries) == 2
    assert analyzed == [
        (0, SESSION_ID, frozenset()),
        (1, SESSION_ID, frozenset()),
        (2, SESSION_ID, expected),
        (3, SESSION_ID, expected),
        (4, SESSION_ID, expected),
        (5, SESSION_ID, expected),
        (6, SESSION_ID, frozenset()),
        (7, session_b, frozenset()),
        (8, session_b, frozenset()),
        (9, session_b, expected),
    ]

    # Geometry/configuration reset must also clear prior confirmed preview state (no bleed).
    reset_analyzed: list[tuple[int, frozenset[tuple[int, str]]]] = []
    reset_store = RegionConfigurationStore()
    await reset_store.apply(_configuration(1))

    class ResetDetector(FakeDetector):
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            if frame.sequence_number == 3:
                await reset_store.apply(_configuration(2))
            return await super().detect(frame)

    reset_deliveries: list[dict[str, object]] = []

    def reset_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        reset_deliveries.append(body)
        return httpx.Response(
            202,
            json={"eventId": body["eventId"], "status": "PROCESSED", "alertIds": []},
        )

    reset_outbox = SqliteEventOutbox((tmp_path / "outbox-reset.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(reset_handler),
    ) as backend:
        reset_worker = HeadlessCameraProcessingWorker(
            stream=FakeStream([frame_for(i, SESSION_ID) for i in range(7)]),  # type: ignore[arg-type]
            detector=ResetDetector(),
            pipeline=SequencePpePipeline({i: "MISSING" for i in range(7)}),
            ppe_region_id=REGION_ID,
            configurations=reset_store,
            outbox=reset_outbox,
            dispatcher=OutboxDispatcher(reset_outbox, backend),
            delivery_interval_seconds=0.01,
            analysis_observer=lambda item: reset_analyzed.append(
                (item.frame.sequence_number, item.confirmed_ppe_items)
            ),
        )
        reset_result = await reset_worker.run()

    assert reset_result.frames_processed == 7
    assert reset_result.events_enqueued == 2
    assert len(reset_deliveries) == 2
    assert reset_analyzed == [
        (0, frozenset()),
        (1, frozenset()),
        (2, expected),
        (3, expected),
        (4, frozenset()),
        (5, frozenset()),
        (6, expected),
    ]


@pytest.mark.anyio
async def test_v2_policy_bundle_preview_retains_confirmed_ppe_and_isolates_gap_reset(
    tmp_path: Path,
) -> None:
    from test_mf05_mf06_bundle_policy import detection, scene_batch
    from test_tracked_policy_pipeline import pipeline, policy_bundle
    from test_tracked_transport import packet

    from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
    from smartsite_ai.processing_worker import TrackedCapturePolicy

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
    site_id = "00000000-0000-4000-8000-000000000077"
    scope = compute_canonical_payload_hash(
        {
            **context.model_dump(by_alias=True, exclude={"schema_version", "receipt_command_id"}),
            "siteId": site_id,
            "profileSha256": "b" * 64,
        }
    )
    outbox = SqliteEventOutbox((tmp_path / "outbox-v2.sqlite").resolve())
    delivered_missing_ppe: list[list[tuple[str, str, str | None]]] = []
    seals: dict[str, dict[str, object]] = {}

    class SequenceV2Detector:
        async def detect(self, source: FrameEnvelope) -> DetectionBatch:
            head_cls = "helmet" if source.sequence_number in {6, 7} else "no-hardhat"
            return template.model_copy(
                update={
                    "sequence_number": source.sequence_number,
                    "captured_at": source.captured_at,
                    "detections": (
                        template.detections[0],
                        detection(head_cls, (0.40, 0.22, 0.48, 0.30)),
                        detection("vest", (0.38, 0.32, 0.52, 0.50)),
                    ),
                }
            )

    async def issue(_bundle, _frame, _kind):
        return context

    def backend(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=seals[request.url.path.rsplit("/", 2)[-2]])
        original = json.loads(request.content)["event"]
        delivered_missing_ppe.append(
            [
                (item["ppeItem"], item["status"], item.get("scope"))
                for item in original["observations"]
                if item["type"] == "PPE" and item["status"] == "MISSING"
            ]
        )
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
            "retiredSegmentIds": [
                item["segmentId"] for item in original["continuity"]["retiredSegments"]
            ],
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

    analyzed: list[tuple[int, frozenset[tuple[int, str]]]] = []
    async with BackendClient(
        "https://backend.example",
        "synthetic-service",
        transport=httpx.MockTransport(backend),
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox, client, generation_credential=credential, tracked_batch_limit=4
        )
        worker = HeadlessCameraProcessingWorker(
            stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),  # type: ignore[arg-type]
            detector=SequenceV2Detector(),
            pipeline=pipeline(),
            ppe_region_id=bundle.regions[0].region_id,
            configurations=store,
            outbox=outbox,
            dispatcher=dispatcher,
            delivery_context_provider=issue,
            tracked_capture=TrackedCapturePolicy(credential, site_id, "b" * 64, 30, 1, 8, 100_000),
            analysis_observer=lambda item: analyzed.append(
                (item.frame.sequence_number, item.confirmed_ppe_items)
            ),
        )
        for offset, sequence in (
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (16, 8),
            (17, 9),
            (18, 10),
            (20, 12),
        ):
            source = FrameEnvelope(
                stream_id=template.stream_id,
                session_id=template.session_id,
                camera_external_id=template.camera_external_id,
                captured_at=datetime(2026, 10, 27, 12, 0, offset, tzinfo=UTC),
                sequence_number=sequence,
                width=640,
                height=480,
                payload=bytes([20, 40, 60]) * 640 * 480,
            )
            await worker._process_frame(source)
            await dispatcher.drain_once()
            await dispatcher.drain_once()

    expected = frozenset({(1, "HARD_HAT")})
    assert analyzed == [
        (1, frozenset()),
        (2, frozenset()),
        (3, expected),
        (4, expected),
        (5, expected),
        (6, expected),
        (7, frozenset()),
        (8, frozenset()),
        (9, frozenset()),
        (10, expected),
        (12, frozenset()),
    ]
    assert delivered_missing_ppe == [
        [],
        [],
        [("HARD_HAT", "MISSING", "CAMERA")],
        [],
        [],
        [],
        [],
        [],
        [],
        [("HARD_HAT", "MISSING", "CAMERA")],
        [],
    ]
