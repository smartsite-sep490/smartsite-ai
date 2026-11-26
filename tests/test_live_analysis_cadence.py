"""Focused deterministic tests isolating the analyzed-preview cadence bottleneck."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from test_mf05_mf06_bundle_policy import PPE_REGION_ID, detection, scene_batch
from test_tracked_policy_pipeline import policy_bundle
from test_tracked_transport import packet

from smartsite_ai.camera_media_api import camera_media_router
from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.live_latest_source import LiveLatestFrameSource
from smartsite_ai.ingestion.queue import PreviewFrameQueue
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.frame_evidence_upload import TrackedFrameEvidenceUploader
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, TrackedCapturePolicy
from smartsite_ai.runtime.analysis_preview import AnalysisPreviewObserver
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.tracking.continuity import FrameContinuityPolicy

SITE = "00000000-0000-4000-8000-000000000077"
PREVIEW_SECRET = SecretStr("cadence-test-preview-token-0123456789abcdef")


def _parse_multipart(request: httpx.Request) -> dict[str, bytes]:
    raw_headers = f"Content-Type: {request.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n"
    message = BytesParser(policy=email_policy).parsebytes(
        raw_headers.encode("ascii") + request.content
    )
    parts: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if isinstance(name, str):
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            parts[name] = payload
    return parts


class _LivePacedSource:
    """Deterministic ~25 fps live FrameSource feeding current UTC wall-clock frames."""

    def __init__(self, *, stream_id: str, session_id: UUID, camera_external_id: str) -> None:
        self._stream_id = stream_id
        self._session_id = session_id
        self._camera_external_id = camera_external_id
        self._sequence = 0
        self._connected = False
        self._closed = False
        self._close_event = asyncio.Event()
        self._start_monotonic = 0.0
        self._payload = bytes([40, 80, 120]) * 64 * 64

    @property
    def source_id(self) -> str:
        return self._stream_id

    async def connect(self) -> None:
        self._connected = True
        self._closed = False
        self._sequence = 0
        self._close_event = asyncio.Event()
        self._start_monotonic = time.monotonic()

    async def read_frame(self) -> FrameEnvelope | None:
        if not self._connected or self._closed:
            return None
        target = self._start_monotonic + self._sequence * 0.04
        delay = target - time.monotonic()
        if delay > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._close_event.wait(), timeout=delay)
        if self._closed:
            return None
        frame = FrameEnvelope(
            stream_id=self._stream_id,
            session_id=self._session_id,
            camera_external_id=self._camera_external_id,
            captured_at=datetime.now(UTC),
            width=64,
            height=64,
            sequence_number=self._sequence,
            payload=self._payload,
        )
        self._sequence += 1
        return frame

    async def close(self) -> None:
        self._closed = True
        self._connected = False
        self._close_event.set()


class _TimedMissingPpeDetector:
    """Fast synthetic detector (5ms) recording per-frame inference timing."""

    def __init__(self, template, *, inference_delay_seconds: float) -> None:
        self._template = template
        self._delay = inference_delay_seconds
        self.detect_durations_ms: list[float] = []

    async def detect(self, source: FrameEnvelope):
        started = time.perf_counter()
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        batch = self._template.model_copy(
            update={
                "stream_id": source.stream_id,
                "session_id": source.session_id,
                "camera_external_id": source.camera_external_id,
                "sequence_number": source.sequence_number,
                "captured_at": source.captured_at,
                "frame_width": source.width,
                "frame_height": source.height,
                "detections": (
                    self._template.detections[0],
                    detection("no-hardhat", (0.40, 0.22, 0.48, 0.30)),
                ),
            }
        )
        self.detect_durations_ms.append((time.perf_counter() - started) * 1000.0)
        return batch


class _TrackedBackendStub:
    """Simulated Backend holding the 2nd tracked frame unsealed for a deterministic window."""

    def __init__(
        self,
        *,
        scope_sha256: str,
        expected_binding_hash: Callable[[str], str],
        slow_post_index: int | None,
        seal_delay_seconds: float,
    ) -> None:
        self._scope_sha256 = scope_sha256
        self._binding_hash = expected_binding_hash
        self._slow_post_index = slow_post_index
        self._seal_delay = seal_delay_seconds
        self._post_order: dict[str, int] = {}
        self._seals: dict[str, dict[str, object]] = {}
        self._ready_at: dict[str, float] = {}
        self.evidence_uploads = 0
        self.raw_posts = 0
        self.result_polls_404 = 0
        self.admitted_sequences: list[int] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            frame_id = request.url.path.rsplit("/", 2)[-2]
            ready_at = self._ready_at.get(frame_id)
            if ready_at is None or time.monotonic() < ready_at:
                self.result_polls_404 += 1
                return httpx.Response(404, json={"code": "NOT_READY"})
            seal = self._seals[frame_id]
            self.admitted_sequences.append(int(str(seal["sequenceNumber"])))
            return httpx.Response(200, json=seal)

        if request.url.path.endswith("/tracked-parts/evidence"):
            self.evidence_uploads += 1
            parts = _parse_multipart(request)
            original = json.loads(parts["event"].decode("utf-8"))
            digest = compute_canonical_payload_hash(original)
            sha = hashlib.sha256(parts["frame"]).hexdigest()
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "1.0.0",
                    "kind": "EVIDENCE_STORED",
                    "eventId": original["eventId"],
                    "payloadHash": digest,
                    "bindingHash": self._binding_hash(digest),
                    "sha256": sha,
                },
            )

        self.raw_posts += 1
        original = json.loads(request.content)["event"]
        partition = original["framePartition"]
        frame_id = partition["frameId"]
        digest = compute_canonical_payload_hash(original)
        if frame_id not in self._post_order:
            idx = len(self._post_order) + 1
            self._post_order[frame_id] = idx
            delay = self._seal_delay if idx == self._slow_post_index else 0.0
            self._ready_at[frame_id] = time.monotonic() + delay
            self._seals[frame_id] = {
                "kind": "ADMITTED_TRACKED_FRAME",
                "scopeSha256": self._scope_sha256,
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
                "bindingHash": self._binding_hash(digest),
            },
        )


def _tracked_setup(tmp_path: Path, *, evidence: bool):
    bundle = policy_bundle(outside=True)
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
    scope = compute_canonical_payload_hash(
        {
            **context.model_dump(by_alias=True, exclude={"schema_version", "receipt_command_id"}),
            "siteId": SITE,
            "profileSha256": "b" * 64,
        }
    )
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite")
    publisher = (
        LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True) if evidence else None
    )
    return bundle, template, context, credential, scope, outbox, publisher


@pytest.mark.anyio
async def test_analyzed_preview_cadence_survives_slow_backend_tracked_seal(tmp_path: Path) -> None:
    """Reproduce live 503 + >2s analysis capture gap while inference is 5ms and raw is 200.

    HeadlessCameraProcessingWorker awaits stage_tracked_capture() in _process_frame_once()
    before looping to stream.get_frame(). When Backend admission/seal for frame 2 (before
    frame 3's CONTINUE link) takes 3.3s, the next detect() + analysis_observer() call stalls
    past AnalysisPreviewObserver's <2s freshness window even with LIVE queue=1 and targetFps=1.
    """
    bundle, template, context, credential, scope, outbox, publisher = _tracked_setup(
        tmp_path, evidence=True
    )
    store = RegionConfigurationStore(allow_policy_v2=True)
    await store.apply_bundle(bundle)

    continuity = FrameContinuityPolicy(
        max_source_sequence_gap=64, max_capture_gap=timedelta(seconds=2)
    )
    raw_source = _LivePacedSource(
        stream_id=template.stream_id,
        session_id=template.session_id,
        camera_external_id=bundle.camera_external_id,
    )
    latest_source = LiveLatestFrameSource(raw_source)
    preview_queue = PreviewFrameQueue()
    stream_config = StreamConfig(
        stream_id=template.stream_id,
        camera_external_id=bundle.camera_external_id,
        source_url="0",
        target_fps=1.0,
        is_live=True,
        max_queue_size=1,
        max_consecutive_failures=1,
    )
    stream = StreamWorker(stream_config, source=latest_source, preview_frames=preview_queue)

    async def current_bundle(_: AnalysisFrame) -> bool:
        latest_bundle = await store.get_bundle(bundle.camera_external_id)
        return latest_bundle is bundle

    analysis_observer = AnalysisPreviewObserver(bundle.camera_external_id, current=current_bundle)
    raw_observer = CameraPreviewObserver(
        bundle.camera_external_id,
        preview_queue,
        source_kind="LIVE",
        is_streaming=lambda: stream.snapshot().state == StreamState.STREAMING,
        analysis=analysis_observer,
    )
    app = FastAPI()
    app.include_router(
        camera_media_router({bundle.camera_external_id: raw_observer}, PREVIEW_SECRET)
    )

    detector = _TimedMissingPpeDetector(template, inference_delay_seconds=0.005)
    tracked_pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(continuity_policy=continuity),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(continuity_policy=continuity),
        ppe_region_id=PPE_REGION_ID,
        tracked_v17=True,
    )
    backend_stub = _TrackedBackendStub(
        scope_sha256=scope,
        expected_binding_hash=context.binding_hash,
        slow_post_index=2,
        seal_delay_seconds=3.3,
    )

    async def issue_context(_bundle, _frame, kind):
        assert kind == "LIVE"
        return context

    tracked_policy = TrackedCapturePolicy(
        credential=credential,
        site_id=SITE,
        profile_sha256="b" * 64,
        pending_seconds=30.0,
        predecessor_wait_seconds=10.0,
        max_pending=8,
        max_bytes=100_000,
        continuity_policy=continuity,
    )
    raw_statuses: list[int] = []
    raw_sequences: list[int] = []
    analysis_samples: list[tuple[int, str | None, str | None]] = []

    async with (
        BackendClient(
            "https://backend.example",
            "synthetic-service",
            transport=httpx.MockTransport(backend_stub.handle),
        ) as backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://preview.test",
            headers={"Authorization": f"Bearer {PREVIEW_SECRET.get_secret_value()}"},
        ) as preview_client,
    ):
        dispatcher = OutboxDispatcher(
            outbox,
            backend_client,
            generation_credential=credential,
            tracked_batch_limit=tracked_policy.max_pending,
            tracked_evidence_upload=TrackedFrameEvidenceUploader(backend_client, tmp_path),
        )
        worker = HeadlessCameraProcessingWorker(
            stream=stream,
            detector=detector,
            pipeline=tracked_pipeline,
            ppe_region_id=bundle.regions[0].region_id,
            configurations=store,
            outbox=outbox,
            dispatcher=dispatcher,
            delivery_interval_seconds=0.25,
            evidence_publisher=publisher,
            delivery_context_provider=issue_context,
            tracked_capture=tracked_policy,
            analysis_observer=analysis_observer.offer,
        )

        await raw_observer.start()
        worker_task = asyncio.create_task(worker.run())
        try:
            # Wait for the initial analysis frame so warm-up completes cleanly.
            warmup_deadline = time.monotonic() + 2.0
            while time.monotonic() < warmup_deadline:
                resp = await preview_client.get(
                    f"/v1/realtime/preview/latest?cameraExternalId={bundle.camera_external_id}"
                )
                if resp.status_code == 200:
                    frame = resp.json()["frame"]
                    analysis_samples.append((200, frame["sequenceNumber"], frame["capturedAt"]))
                    break
                await asyncio.sleep(0.05)
            assert analysis_samples, "Initial analyzed frame did not appear within 2.0s"

            # Poll every 250ms across 5.5s, covering the 3.3s delayed seal of tracked frame #2.
            for _ in range(22):
                await asyncio.sleep(0.25)
                raw_resp, analysis_resp = await asyncio.gather(
                    preview_client.get(
                        "/v1/realtime/configuration-image"
                        f"?cameraExternalId={bundle.camera_external_id}"
                    ),
                    preview_client.get(
                        f"/v1/realtime/preview/latest?cameraExternalId={bundle.camera_external_id}"
                    ),
                )
                raw_statuses.append(raw_resp.status_code)
                if raw_resp.status_code == 200:
                    raw_sequences.append(int(raw_resp.json()["frame"]["sequenceNumber"]))
                if analysis_resp.status_code == 200:
                    frame = analysis_resp.json()["frame"]
                    analysis_samples.append((200, frame["sequenceNumber"], frame["capturedAt"]))
                else:
                    analysis_samples.append((analysis_resp.status_code, None, None))
        finally:
            await stream.stop()
            result = await asyncio.wait_for(worker_task, timeout=5.0)
            if not await raw_observer.stop():
                await raw_observer.wait_for_release()

    unique_analysis = list(
        {
            seq: datetime.fromisoformat(cap)
            for status, seq, cap in analysis_samples
            if status == 200 and seq is not None and cap is not None
        }.items()
    )
    capture_gaps_ms = [
        round((unique_analysis[i][1] - unique_analysis[i - 1][1]).total_seconds() * 1000.0)
        for i in range(1, len(unique_analysis))
    ]
    stale_503_count = sum(1 for status, _, _ in analysis_samples if status == 503)

    # Confirm fast inference + healthy live raw stream so failure cleanly isolates staging stall.
    assert max(detector.detect_durations_ms) < 50.0
    assert all(status == 200 for status in raw_statuses)
    assert len(set(raw_sequences)) >= 10
    assert backend_stub.result_polls_404 >= 1

    # A delayed seal must not stall preview during a bounded backlog.
    assert stale_503_count == 0 and max(capture_gaps_ms, default=0) <= 1600, (
        "Analyzed preview stalled during Backend tracked seal delay despite fast inference: "
        f"detect_max_ms={max(detector.detect_durations_ms):.1f}, "
        f"raw_200={len(raw_statuses)} (seq {min(raw_sequences)}..{max(raw_sequences)}), "
        f"analysis_503={stale_503_count}, "
        f"analysis_seqs={[s for s, _ in unique_analysis]}, "
        f"capture_gaps_ms={capture_gaps_ms}"
    )
    admitted = backend_stub.admitted_sequences
    assert admitted == sorted(set(admitted))
    assert len(admitted) == result.frames_processed
    assert {int(seq) for seq, _ in unique_analysis}.issubset(admitted)
    assert result.events_enqueued == backend_stub.raw_posts
    assert not await outbox.has_pending_tracked_frame(credential)
