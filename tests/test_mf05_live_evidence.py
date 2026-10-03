from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from PIL import Image

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import EvidenceItem, TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.evidence.models import (
    EvidenceBindingError,
    EvidenceConflictError,
    EvidencePathContainmentError,
    EvidenceSizeLimitError,
    FrameBatchBindingError,
)
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamMetrics, StreamState
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.processing_worker import (
    HeadlessCameraProcessingWorker,
    validate_frame_batch_binding,
)

SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
REGION_ID = "22222222-2222-4222-8222-222222222222"
CAMERA_ID = "CAM-GATE-01"


def _make_frame(
    sequence: int,
    *,
    session_id: UUID = SESSION_ID,
    camera_id: str = CAMERA_ID,
    width: int = 10,
    height: int = 10,
    color_bgr: tuple[int, int, int] = (10, 20, 30),
) -> FrameEnvelope:
    b, g, r = color_bgr
    payload = bytes([b, g, r] * (width * height))
    return FrameEnvelope(
        stream_id="gate-stream",
        session_id=session_id,
        camera_external_id=camera_id,
        captured_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC) + timedelta(seconds=sequence),
        width=width,
        height=height,
        sequence_number=sequence,
        payload=payload,
    )


def _make_event(
    frame: FrameEnvelope,
    *,
    event_id: str = "33333333-3333-4333-8333-333333333333",
) -> TechnicalObservationEvent:
    return TechnicalObservationEvent.create(
        event_id=event_id,
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at.isoformat(),
        frame_dimensions={"width": frame.width, "height": frame.height},
        observations=[
            {
                "type": "ZONE_ENTRY",
                "trackId": 1,
                "regionId": REGION_ID,
                "geometryVersion": 1,
                "confidence": 0.95,
            }
        ],
        evidence=[],
    )


def _configuration(version: int = 1) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": version,
            "cameraExternalId": CAMERA_ID,
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


class FakeStream:
    def __init__(self, frames: list[FrameEnvelope]) -> None:
        self.config = StreamConfig(
            stream_id="gate-stream",
            camera_external_id=CAMERA_ID,
            source_url="test://gate",
            is_live=False,
        )
        self.frames = list(frames)
        self.frame_count = len(frames)

    async def start(self) -> None:
        pass

    async def get_frame(self) -> FrameEnvelope:
        await asyncio.sleep(0)
        if not self.frames:
            raise QueueClosedError("complete")
        return self.frames.pop(0)

    async def stop(self) -> None:
        pass

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(
            state=StreamState.STOPPED,
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


class FakePipeline:
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
                    "type": "ZONE_ENTRY",
                    "trackId": 7,
                    "regionId": REGION_ID,
                    "geometryVersion": region_configuration.configuration_version,
                    "confidence": 0.9,
                }
            ],
            evidence=[],
        )


# ==============================================================================
# 1. Exact-frame binding & collision tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_exact_frame_binding(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    frame = _make_frame(1, color_bgr=(255, 0, 0))  # pure blue in BGR
    event = _make_event(frame, event_id="44444444-4444-4444-8444-444444444444")

    published_event = await publisher.publish(frame=frame, event=event)

    # 1. Event evidence is populated
    assert len(published_event.evidence) == 1
    item = published_event.evidence[0]
    assert item.kind == "FRAME"

    # 2. URI is provider-neutral and embeds session, sequence, and eventId
    expected_uri = f"local://evidence/{SESSION_ID}/1/{event.event_id}.jpg"
    assert item.uri == expected_uri

    # 3. File exists on disk as flat file in root_dir
    saved_file = evidence_dir / f"{SESSION_ID}_1_{event.event_id}.jpg"
    assert saved_file.is_file()

    # 4. Manifest sidecar is NOT written
    manifest_file = evidence_dir / f"{SESSION_ID}_1_{event.event_id}.manifest.json"
    assert not manifest_file.exists()

    with Image.open(saved_file) as img:
        assert img.size == (frame.width, frame.height)
        # Check pixel color (PIL opens as RGB: (0, 0, 255) for BGR (255, 0, 0) with lossy tolerance)
        rgb = img.getpixel((0, 0))
        assert abs(rgb[0] - 0) <= 5
        assert abs(rgb[1] - 0) <= 5
        assert abs(rgb[2] - 255) <= 5


@pytest.mark.anyio
async def test_evidence_publisher_rejects_mismatched_frame_binding(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    frame_a = _make_frame(1)
    frame_b = _make_frame(2)
    event_b = _make_event(frame_b)

    # Providing frame_a with event_b must fail binding check
    with pytest.raises(EvidenceBindingError, match="captured_at|capturedAt"):
        await publisher.publish(frame=frame_a, event=event_b)


@pytest.mark.anyio
async def test_evidence_publisher_preserves_existing_evidence_and_deduplicates_on_retry(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    frame = _make_frame(1)
    base_event = _make_event(frame, event_id="44444444-4444-4444-8444-444444444444")

    existing_crop = EvidenceItem(
        kind="CROP",
        uri="local://evidence/crops/crop1.jpg",
    )
    event_with_existing = TechnicalObservationEvent.create(
        event_id=base_event.event_id,
        camera_external_id=base_event.camera_external_id,
        stream_session_id=base_event.stream_session_id,
        captured_at=base_event.captured_at,
        frame_dimensions=base_event.frame_dimensions,
        observations=base_event.observations,
        evidence=[existing_crop],
        schema_version=base_event.schema_version,
    )

    published_1 = await publisher.publish(frame=frame, event=event_with_existing)

    # 1. Existing evidence is preserved, FRAME is appended
    assert len(published_1.evidence) == 2
    assert published_1.evidence[0] == existing_crop
    assert published_1.evidence[1].kind == "FRAME"
    expected_uri = f"local://evidence/{SESSION_ID}/1/{base_event.event_id}.jpg"
    assert published_1.evidence[1].uri == expected_uri

    # 2. On retry (publishing same frame/event again), duplicate URI is not added
    published_2 = await publisher.publish(frame=frame, event=published_1)
    assert len(published_2.evidence) == 2
    assert published_2.evidence[0] == existing_crop
    assert published_2.evidence[1].kind == "FRAME"
    assert published_2.evidence[1].uri == expected_uri


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("attribute", "mismatched_value"),
    [
        ("stream_id", "wrong-stream"),
        ("camera_external_id", "CAM-DIFFERENT"),
        ("session_id", UUID("99999999-9999-4999-8999-999999999999")),
        ("sequence_number", 999),
        ("captured_at", datetime(2026, 9, 27, 12, 0, tzinfo=UTC)),
        ("frame_width", 640),
        ("frame_height", 480),
    ],
)
async def test_worker_rejects_detection_batch_mismatch(
    attribute: str,
    mismatched_value: object,
) -> None:
    frame = _make_frame(1, width=10, height=10)
    batch_kwargs = {
        "stream_id": frame.stream_id,
        "camera_external_id": frame.camera_external_id,
        "session_id": frame.session_id,
        "sequence_number": frame.sequence_number,
        "captured_at": frame.captured_at,
        "frame_width": frame.width,
        "frame_height": frame.height,
        "model_artifact_id": "ppe-model",
        "model_version": "1",
        "model_sha256": "a" * 64,
        "detections": (),
    }
    batch_kwargs[attribute] = mismatched_value
    bad_batch = DetectionBatch.model_validate(batch_kwargs)

    with pytest.raises(FrameBatchBindingError):
        validate_frame_batch_binding(frame, bad_batch)


@pytest.mark.anyio
async def test_evidence_publisher_handles_same_session_timestamp_collision(tmp_path: Path) -> None:
    """Publisher collision test with identical captured_at but distinct sequence and payload."""
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    collision_timestamp = datetime(2026, 9, 27, 10, 0, 0, tzinfo=UTC)
    frame_blue = FrameEnvelope(
        stream_id="gate-stream",
        session_id=SESSION_ID,
        camera_external_id=CAMERA_ID,
        captured_at=collision_timestamp,
        width=10,
        height=10,
        sequence_number=1,
        payload=bytes([255, 0, 0] * 100),  # blue
    )
    frame_green = FrameEnvelope(
        stream_id="gate-stream",
        session_id=SESSION_ID,
        camera_external_id=CAMERA_ID,
        captured_at=collision_timestamp,  # identical timestamp
        width=10,
        height=10,
        sequence_number=2,  # distinct sequence
        payload=bytes([0, 255, 0] * 100),  # green
    )

    event_blue = _make_event(frame_blue, event_id="11111111-aaaa-4111-8111-111111111111")
    event_green = _make_event(frame_green, event_id="22222222-bbbb-4222-8222-222222222222")

    published_blue = await publisher.publish(frame=frame_blue, event=event_blue)
    published_green = await publisher.publish(frame=frame_green, event=event_green)

    uri_blue = published_blue.evidence[0].uri
    uri_green = published_green.evidence[0].uri
    assert uri_blue != uri_green
    assert uri_blue == f"local://evidence/{SESSION_ID}/1/{event_blue.event_id}.jpg"
    assert uri_green == f"local://evidence/{SESSION_ID}/2/{event_green.event_id}.jpg"

    file_blue = evidence_dir / f"{SESSION_ID}_1_{event_blue.event_id}.jpg"
    file_green = evidence_dir / f"{SESSION_ID}_2_{event_green.event_id}.jpg"
    assert file_blue.is_file()
    assert file_green.is_file()

    with Image.open(file_blue) as img:
        rgb = img.getpixel((0, 0))
        assert abs(rgb[0] - 0) <= 5 and abs(rgb[2] - 255) <= 5  # blue
    with Image.open(file_green) as img:
        rgb = img.getpixel((0, 0))
        assert abs(rgb[1] - 255) <= 5 and abs(rgb[0] - 0) <= 5  # green


@pytest.mark.anyio
async def test_worker_rejects_detector_returning_mismatched_batch_before_pipeline(
    tmp_path: Path,
) -> None:
    """Reject a detector batch with a mismatched sequence before pipeline execution."""

    class BadDetector:
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            return DetectionBatch.from_frame(
                frame,
                model_artifact_id="ppe-model",
                model_version="1",
                model_sha256="a" * 64,
                detections=(),
            ).model_copy(update={"sequence_number": frame.sequence_number + 100})

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_make_frame(1)])
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    client = BackendClient(
        "http://backend:3000",
        "token",
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )
    dispatcher = OutboxDispatcher(outbox, client)

    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=BadDetector(),
        pipeline=FakePipeline(),
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=dispatcher,
        delivery_interval_seconds=0.01,
    )

    with pytest.raises(FrameBatchBindingError):
        await worker.run()

    # Outbox must have 0 enqueued events
    counts = await outbox.counts()
    assert counts.pending == 0
    assert counts.delivered == 0


# ==============================================================================
# 2. Atomic single JPEG artifact tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_atomic_single_jpeg_create(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    frame = _make_frame(10)
    event = _make_event(frame, event_id="55555555-5555-4555-8555-555555555555")

    await publisher.publish(frame=frame, event=event)

    image_path = evidence_dir / f"{SESSION_ID}_10_{event.event_id}.jpg"
    assert image_path.is_file()

    # Manifest sidecar does not exist (single artifact model)
    manifest_path = evidence_dir / f"{SESSION_ID}_10_{event.event_id}.manifest.json"
    assert not manifest_path.exists()

    # No leftover temp files
    temp_files = list(evidence_dir.glob(".*"))
    assert len(temp_files) == 0


@pytest.mark.anyio
async def test_evidence_publisher_is_idempotent_and_rejects_conflicting_pixels(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)
    event_id = "77777777-7777-4777-8777-777777777777"
    blue = _make_frame(11, color_bgr=(255, 0, 0))
    green = _make_frame(11, color_bgr=(0, 255, 0))
    event = _make_event(blue, event_id=event_id)

    await publisher.publish(frame=blue, event=event)
    image_path = evidence_dir / f"{SESSION_ID}_11_{event_id}.jpg"
    original = image_path.read_bytes()

    await publisher.publish(frame=blue, event=event)
    assert image_path.read_bytes() == original

    with pytest.raises(EvidenceConflictError):
        await publisher.publish(frame=green, event=event)
    assert image_path.read_bytes() == original


@pytest.mark.anyio
async def test_evidence_publisher_no_overwrite_conflict_and_idempotency(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    frame_blue = _make_frame(1, color_bgr=(255, 0, 0))
    event = _make_event(frame_blue, event_id="77777777-7777-4777-8777-777777777777")

    # 1. First publish succeeds
    published_1 = await publisher.publish(frame=frame_blue, event=event)
    file_path = evidence_dir / f"{SESSION_ID}_1_{event.event_id}.jpg"
    assert file_path.is_file()
    original_bytes = file_path.read_bytes()

    # 2. Idempotent publish with identical frame & event succeeds without error
    published_2 = await publisher.publish(frame=frame_blue, event=event)
    assert published_2.evidence[0].uri == published_1.evidence[0].uri
    assert file_path.read_bytes() == original_bytes

    # 3. Publish with same session, sequence, and eventId but DIFFERENT pixels must raise conflict
    frame_green = _make_frame(1, color_bgr=(0, 255, 0))
    with pytest.raises(EvidenceConflictError, match="already exists with conflicting content"):
        await publisher.publish(frame=frame_green, event=event)

    # 4. File on disk must remain unchanged (not overwritten)
    assert file_path.read_bytes() == original_bytes
    with Image.open(file_path) as img:
        rgb = img.getpixel((0, 0))
        assert abs(rgb[0] - 0) <= 5 and abs(rgb[2] - 255) <= 5


# ==============================================================================
# 3. Path containment, root validation & traversal tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_requires_existing_directory_root(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(EvidencePathContainmentError, match="must exist"):
        LocalEvidencePublisher(root_dir=non_existent)


@pytest.mark.anyio
async def test_evidence_publisher_rejects_file_as_root(tmp_path: Path) -> None:
    file_path = tmp_path / "some_file.txt"
    file_path.write_text("hello", encoding="utf-8")
    with pytest.raises(EvidencePathContainmentError, match="must be a directory"):
        LocalEvidencePublisher(root_dir=file_path)


@pytest.mark.anyio
async def test_evidence_publisher_rejects_link_like_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    import smartsite_ai.evidence.local_publisher as lp_mod

    monkeypatch.setattr(lp_mod, "is_link_like", lambda p: True)
    with pytest.raises(EvidencePathContainmentError, match="must not be a symlink"):
        LocalEvidencePublisher(root_dir=evidence_dir)


@pytest.mark.anyio
async def test_evidence_publisher_revalidates_root_before_write(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    # Simulate root directory being deleted after init
    evidence_dir.rmdir()
    frame = _make_frame(1)
    event = _make_event(frame)

    with pytest.raises(EvidencePathContainmentError, match="root directory is missing or invalid"):
        await publisher.publish(frame=frame, event=event)


@pytest.mark.anyio
async def test_evidence_publisher_rejects_path_traversal_event_id(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    with pytest.raises(EvidencePathContainmentError):
        publisher._resolve_evidence_path(
            session_id=str(SESSION_ID),
            sequence_number=1,
            event_id="../../escaped",
        )


@pytest.mark.anyio
async def test_evidence_publisher_rejects_path_traversal_session_id(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    with pytest.raises(EvidencePathContainmentError):
        publisher._resolve_evidence_path(
            session_id="../../escaped",
            sequence_number=1,
            event_id="55555555-5555-4555-8555-555555555555",
        )


# ==============================================================================
# 4. Immutable write tests (no count-based retention / prune)
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_immutable_write_does_not_prune(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    # Publish 5 frames sequentially
    for i in range(1, 6):
        frame = _make_frame(i)
        event = _make_event(frame, event_id=f"00000000-0000-0000-0000-00000000000{i}")
        await publisher.publish(frame=frame, event=event)

    # All 5 .jpg files must remain intact without pruning
    retained_images = sorted(evidence_dir.glob("*.jpg"))
    assert len(retained_images) == 5


# ==============================================================================
# 5. Max JPEG size tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_rejects_exceeded_jpeg_size(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    # Set limit tiny (50 bytes)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir, max_jpeg_bytes=50)

    frame = _make_frame(1, width=100, height=100)
    event = _make_event(frame)

    with pytest.raises(EvidenceSizeLimitError, match="exceeds maximum"):
        await publisher.publish(frame=frame, event=event)


# ==============================================================================
# 6. Worker integration & fail-open durable delivery tests
# ==============================================================================


@pytest.mark.anyio
async def test_worker_disabled_evidence_by_default(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        event_id = json.loads(request.content)["eventId"]
        return httpx.Response(
            202, json={"eventId": event_id, "status": "PROCESSED", "alertIds": []}
        )

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_make_frame(1)])
    pipeline = FakePipeline()
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    client = BackendClient("http://backend:3000", "token", transport=httpx.MockTransport(handler))
    dispatcher = OutboxDispatcher(outbox, client)

    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=FakeDetector(),
        pipeline=pipeline,
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=dispatcher,
        delivery_interval_seconds=0.01,
        # evidence_publisher is None by default
    )

    result = await worker.run()
    assert result.frames_processed == 1
    assert result.events_enqueued == 1
    assert result.outbox.delivered == 1

    # Check payload sent to backend
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["evidence"] == []


@pytest.mark.anyio
async def test_worker_publishes_evidence_when_configured(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        event_id = json.loads(request.content)["eventId"]
        return httpx.Response(
            202, json={"eventId": event_id, "status": "PROCESSED", "alertIds": []}
        )

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_make_frame(1)])
    pipeline = FakePipeline()
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    client = BackendClient("http://backend:3000", "token", transport=httpx.MockTransport(handler))
    dispatcher = OutboxDispatcher(outbox, client)

    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=FakeDetector(),
        pipeline=pipeline,
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=dispatcher,
        evidence_publisher=publisher,
        delivery_interval_seconds=0.01,
    )

    result = await worker.run()
    assert result.frames_processed == 1
    assert result.events_enqueued == 1
    assert result.outbox.delivered == 1

    # Check payload sent to backend
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["kind"] == "FRAME"
    assert payload["evidence"][0]["uri"].startswith(f"local://evidence/{SESSION_ID}/1/")

    # File exists on disk
    saved = list(evidence_dir.glob("*.jpg"))
    assert len(saved) == 1


@pytest.mark.anyio
async def test_worker_publisher_failure_fails_open_with_safe_diagnostic(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        event_id = json.loads(request.content)["eventId"]
        return httpx.Response(
            202, json={"eventId": event_id, "status": "PROCESSED", "alertIds": []}
        )

    class FailingPublisher:
        async def publish(
            self,
            *,
            frame: FrameEnvelope,
            event: TechnicalObservationEvent,
        ) -> TechnicalObservationEvent:
            raise OSError("failed writing C:/secrets/token.pem with secret_token_xyz")

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    stream = FakeStream([_make_frame(1)])
    pipeline = FakePipeline()
    outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
    client = BackendClient("http://backend:3000", "token", transport=httpx.MockTransport(handler))
    dispatcher = OutboxDispatcher(outbox, client)

    worker = HeadlessCameraProcessingWorker(
        stream=stream,
        detector=FakeDetector(),
        pipeline=pipeline,
        ppe_region_id=REGION_ID,
        configurations=store,
        outbox=outbox,
        dispatcher=dispatcher,
        evidence_publisher=FailingPublisher(),
        delivery_interval_seconds=0.01,
    )

    with caplog.at_level(logging.WARNING):
        result = await worker.run()

    # 1. Worker does not crash
    assert result.frames_processed == 1
    assert result.events_enqueued == 1
    assert result.outbox.delivered == 1

    # 2. Durable delivery succeeded (event delivered without evidence)
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["evidence"] == []

    # 3. Also verify raw outbox record in SQLite
    with sqlite3.connect(outbox.path) as conn:
        row = conn.execute("SELECT payload_json FROM observation_outbox").fetchone()
        stored_payload = json.loads(row[0])
        assert stored_payload["evidence"] == []

    # 4. Safe diagnostic logged: no sensitive strings or file paths in any log record
    assert any("evidence" in record.message.lower() for record in caplog.records)
    assert "secret_token_xyz" not in caplog.text
    assert "token.pem" not in caplog.text
    assert "C:/secrets" not in caplog.text

    warning_records = [r for r in caplog.records if "evidence publication failed" in r.message]
    assert len(warning_records) == 1
    record = warning_records[0]
    assert getattr(record, "exception_type", None) == "OSError"
    assert getattr(record, "stream_id", None) == "gate-stream"
    assert getattr(record, "sequence_number", None) == 1
    assert getattr(record, "event_id", None) is not None
    assert not hasattr(record, "error")


def test_build_parser_evidence_arguments() -> None:
    from smartsite_ai.tools.run_camera_worker import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--stream-id",
            "test",
            "--camera-id",
            "11111111-1111-4111-8111-111111111111",
            "--camera-external-id",
            "CAM-1",
            "--ppe-region-id",
            "22222222-2222-4222-8222-222222222222",
            "--model-spec",
            "spec.json",
            "--outbox",
            "outbox.sqlite3",
            "--evidence-dir",
            "C:/evidence",
            "--evidence-max-bytes",
            "2048",
        ]
    )
    assert args.evidence_dir == Path("C:/evidence")
    assert args.evidence_max_bytes == 2048
    # --evidence-retention-limit must not exist on parser
    assert not hasattr(args, "evidence_retention_limit")


def test_cli_requires_existing_directory_without_mkdir(tmp_path: Path) -> None:
    from smartsite_ai.tools.run_camera_worker import CameraWorkerRunError, _absolute_existing_dir

    # Relative path
    with pytest.raises(CameraWorkerRunError, match="must be absolute"):
        _absolute_existing_dir(Path("relative/path"), "test")

    # Non-existent directory
    with pytest.raises(
        CameraWorkerRunError, match="must identify an existing regular non-symlink directory"
    ):
        _absolute_existing_dir(tmp_path / "missing", "test")

    # Regular file
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a dir", encoding="utf-8")
    with pytest.raises(
        CameraWorkerRunError, match="must identify an existing regular non-symlink directory"
    ):
        _absolute_existing_dir(file_path, "test")

    # Valid existing directory
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    assert _absolute_existing_dir(valid_dir, "test") == valid_dir
