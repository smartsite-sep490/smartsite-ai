from __future__ import annotations

import asyncio
import hashlib
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
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.evidence.models import (
    EvidenceBindingError,
    EvidencePathContainmentError,
    EvidenceSizeLimitError,
)
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker

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
        return SimpleNamespace(state=StreamState.STOPPED)


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
# 1. Exact-frame binding tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_exact_frame_binding(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
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

    # 3. File exists on disk and matches frame content
    saved_file = evidence_dir / str(SESSION_ID) / f"1_{event.event_id}.jpg"
    assert saved_file.is_file()

    with Image.open(saved_file) as img:
        assert img.size == (frame.width, frame.height)
        # Check pixel color (PIL opens as RGB: (0, 0, 255) for BGR (255, 0, 0) with lossy tolerance)
        rgb = img.getpixel((0, 0))
        assert abs(rgb[0] - 0) <= 5
        assert abs(rgb[1] - 0) <= 5
        assert abs(rgb[2] - 255) <= 5


@pytest.mark.anyio
async def test_evidence_publisher_rejects_mismatched_frame_binding(tmp_path: Path) -> None:
    publisher = LocalEvidencePublisher(root_dir=tmp_path / "evidence")

    frame_a = _make_frame(1)
    frame_b = _make_frame(2)
    event_b = _make_event(frame_b)

    # Providing frame_a with event_b must fail binding check
    with pytest.raises(EvidenceBindingError, match="captured_at|capturedAt"):
        await publisher.publish(frame=frame_a, event=event_b)


# ==============================================================================
# 2. Atomic manifest and reference tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_atomic_manifest_and_reference(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    frame = _make_frame(10)
    event = _make_event(frame, event_id="55555555-5555-4555-8555-555555555555")

    await publisher.publish(frame=frame, event=event)

    image_path = evidence_dir / str(SESSION_ID) / f"10_{event.event_id}.jpg"
    manifest_path = evidence_dir / str(SESSION_ID) / f"10_{event.event_id}.manifest.json"

    assert image_path.is_file()
    assert manifest_path.is_file()

    # No leftover temp files
    temp_files = list((evidence_dir / str(SESSION_ID)).glob(".*"))
    assert len(temp_files) == 0

    # Verify manifest fields
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eventId"] == event.event_id
    assert manifest["streamSessionId"] == str(SESSION_ID)
    assert manifest["sequenceNumber"] == 10
    assert manifest["cameraExternalId"] == CAMERA_ID
    assert manifest["width"] == frame.width
    assert manifest["height"] == frame.height
    assert manifest["mediaType"] == "image/jpeg"
    assert manifest["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert manifest["sizeBytes"] == image_path.stat().st_size
    assert manifest["uri"] == f"local://evidence/{SESSION_ID}/10/{event.event_id}.jpg"


# ==============================================================================
# 3. Path containment (traversal) tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_rejects_path_traversal_event_id(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    with pytest.raises(EvidencePathContainmentError):
        publisher._resolve_evidence_paths(
            session_id=str(SESSION_ID),
            sequence_number=1,
            event_id="../../escaped",
        )


@pytest.mark.anyio
async def test_evidence_publisher_rejects_path_traversal_session_id(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    publisher = LocalEvidencePublisher(root_dir=evidence_dir)

    with pytest.raises(EvidencePathContainmentError):
        publisher._resolve_evidence_paths(
            session_id="../../escaped",
            sequence_number=1,
            event_id="55555555-5555-4555-8555-555555555555",
        )


# ==============================================================================
# 4. Retention bounded tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_bounds_retention(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    publisher = LocalEvidencePublisher(root_dir=evidence_dir, retention_limit=3)

    # Publish 5 frames sequentially
    for i in range(1, 6):
        frame = _make_frame(i)
        event = _make_event(frame, event_id=f"00000000-0000-0000-0000-00000000000{i}")
        await publisher.publish(frame=frame, event=event)

    # At most 3 .jpg and 3 .manifest.json files should remain
    retained_images = sorted(evidence_dir.glob("*/*.jpg"))
    retained_manifests = sorted(evidence_dir.glob("*/*.manifest.json"))

    assert len(retained_images) == 3
    assert len(retained_manifests) == 3

    # Retained should be the 3 newest (sequences 3, 4, 5)
    names = [p.name for p in retained_images]
    assert names == [
        "3_00000000-0000-0000-0000-000000000003.jpg",
        "4_00000000-0000-0000-0000-000000000004.jpg",
        "5_00000000-0000-0000-0000-000000000005.jpg",
    ]


# ==============================================================================
# 5. Max JPEG size tests
# ==============================================================================


@pytest.mark.anyio
async def test_evidence_publisher_rejects_exceeded_jpeg_size(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
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
    saved = list(evidence_dir.glob("*/*.jpg"))
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
            raise OSError("simulated disk full / I/O error")

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

    # 4. Safe diagnostic logged
    assert any("evidence" in record.message.lower() for record in caplog.records)
    # No raw sensitive tokens in log
    for record in caplog.records:
        assert "service_token" not in record.message


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
            "--evidence-retention-limit",
            "50",
        ]
    )
    assert args.evidence_dir == Path("C:/evidence")
    assert args.evidence_max_bytes == 2048
    assert args.evidence_retention_limit == 50
