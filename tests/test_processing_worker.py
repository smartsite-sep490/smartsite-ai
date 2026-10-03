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
from smartsite_ai.ingestion.queue import BoundedFrameQueue, QueueClosedError
from smartsite_ai.ingestion.status import StreamMetrics, StreamState
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
        self.frame_count = len(frames)

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
        return SimpleNamespace(
            state=self.terminal_state,
            metrics=StreamMetrics(
                frames_enqueued=self.frame_count,
                frames_dequeued=self.frame_count - len(self.frames),
            ),
        )


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
async def test_worker_runs_without_browser_and_delivers_deterministic_events(
    tmp_path: Path,
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


@pytest.mark.anyio
async def test_worker_reports_sampling_separately_from_actual_queue_drops(tmp_path: Path) -> None:
    class OverloadedStream(FakeStream):
        def __init__(self) -> None:
            super().__init__([])
            self.queue = BoundedFrameQueue(maxsize=2)

        async def start(self) -> None:
            self.started = True
            for sequence in range(5):
                self.queue.put(_frame(sequence))
            self.queue.close()

        async def get_frame(self) -> FrameEnvelope:
            return await self.queue.get()

        def snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                state=StreamState.STOPPED,
                metrics=StreamMetrics(
                    frames_enqueued=self.queue.enqueued_count,
                    frames_dequeued=self.queue.dequeued_count,
                    frames_dropped=self.queue.dropped_count,
                    sampled_out_frames=4,
                    last_error="rtsp://operator:secret@camera",
                ),
            )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            202,
            json={
                "eventId": json.loads(request.content)["eventId"],
                "status": "PROCESSED",
                "alertIds": [],
            },
        )

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = OverloadedStream()
    outbox = SqliteEventOutbox((tmp_path / "overload.sqlite3").resolve())
    async with BackendClient(
        "http://backend:3000", "token", transport=httpx.MockTransport(handler)
    ) as backend:
        result = await HeadlessCameraProcessingWorker(
            stream=stream,  # type: ignore[arg-type]
            detector=FakeDetector(),
            pipeline=RecordingPipeline(),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
        ).run()
    assert result.frames_processed == 2
    assert result.frame_flow == {
        "framesEnqueued": 5,
        "framesDequeued": 2,
        "framesDropped": 3,
        "sampledOutFrames": 4,
    }
    assert "secret" not in json.dumps(result.frame_flow)


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
