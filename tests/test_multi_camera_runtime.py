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
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxCounts, OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, ProcessingWorkerResult
from smartsite_ai.runtime.assembly import assemble_sessions
from smartsite_ai.runtime.inference_lane import SerializingDetector, share_detector
from smartsite_ai.runtime.manifest import (
    CameraRuntimeError,
    bind_manifest,
    load_manifest_bytes,
    parse_manifest,
)
from smartsite_ai.runtime.supervisor import (
    CameraSession,
    RuntimePlan,
    execute_runtime,
    exit_code,
    report_payload,
    startup_outcomes,
    supervise_sessions,
)
from smartsite_ai.tools.run_camera_runtime import build_parser
from smartsite_ai.tools.run_camera_worker import build_parser as build_one_camera_parser

REGION_A = "22222222-2222-4222-8222-222222222222"
REGION_B = "44444444-4444-4444-8444-444444444444"
CAMERA_A = "11111111-1111-4111-8111-111111111111"
CAMERA_B = "33333333-3333-4333-8333-333333333333"
CAMERA_C = "55555555-5555-4555-8555-555555555555"


def _entry(
    stream_id: str,
    camera_id: str,
    external_id: str,
    *,
    source: dict[str, str],
    outbox: str,
    evidence_dir: str | None = None,
    ppe_region_id: str = REGION_A,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "streamId": stream_id,
        "cameraId": camera_id,
        "cameraExternalId": external_id,
        "ppeRegionId": ppe_region_id,
        "source": source,
        "live": False,
        "outbox": outbox,
    }
    if evidence_dir is not None:
        payload["evidenceDir"] = evidence_dir
    return payload


def _document(tmp_path: Path, cameras: list[dict[str, object]]) -> dict[str, object]:
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    return {
        "schemaVersion": "1",
        "modelSpec": str(model),
        "cameras": cameras,
    }


def _video(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"not-a-real-video")
    return str(path)


class _StopPoller:
    def __init__(self) -> None:
        self.stopped = False

    async def run(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        self.stopped = True


class _ResultWorker:
    def __init__(self, result: ProcessingWorkerResult) -> None:
        self.result = result
        self.cancelled = False

    async def run(self) -> ProcessingWorkerResult:
        try:
            await asyncio.sleep(0)
            return self.result
        finally:
            if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                self.cancelled = True


def _clean_result() -> ProcessingWorkerResult:
    return ProcessingWorkerResult(
        frames_processed=2,
        events_enqueued=1,
        outbox=OutboxCounts(pending=0, delivered=1, terminal=0),
    )


def test_manifest_rejects_bounds_duplicates_and_embedded_credentials(tmp_path: Path) -> None:
    video_a = _video(tmp_path, "a.mp4")
    outbox_a = str(tmp_path / "a.sqlite3")
    outbox_b = str(tmp_path / "b.sqlite3")
    evidence_a = tmp_path / "evidence-a"
    evidence_b = tmp_path / "evidence-b"
    evidence_a.mkdir()
    evidence_b.mkdir()
    first = _entry(
        "gate-01",
        CAMERA_A,
        "CAM-A",
        source={"kind": "path", "path": video_a},
        outbox=outbox_a,
        evidence_dir=str(evidence_a),
    )
    second = _entry(
        "yard-02",
        CAMERA_B,
        "CAM-B",
        source={"kind": "env", "name": "SMARTSITE_AI_CAMERA_YARD_02_SOURCE"},
        outbox=outbox_b,
        evidence_dir=str(evidence_b),
        ppe_region_id=REGION_B,
    )
    parsed = parse_manifest(_document(tmp_path, [first, second]))
    assert len(parsed.cameras) == 2

    with pytest.raises(CameraRuntimeError, match="1 to 3 cameras"):
        parse_manifest(_document(tmp_path, []))
    with pytest.raises(CameraRuntimeError, match="1 to 3 cameras"):
        parse_manifest(_document(tmp_path, [first, second, first, second]))
    with pytest.raises(CameraRuntimeError, match="duplicate stream id"):
        parse_manifest(_document(tmp_path, [first, {**second, "streamId": "GATE-01"}]))
    with pytest.raises(CameraRuntimeError, match="duplicate camera id"):
        parse_manifest(_document(tmp_path, [first, {**second, "cameraId": CAMERA_A}]))
    with pytest.raises(CameraRuntimeError, match="duplicate camera external id"):
        parse_manifest(_document(tmp_path, [first, {**second, "cameraExternalId": "cam-a"}]))
    with pytest.raises(CameraRuntimeError, match="duplicate outbox path"):
        parse_manifest(_document(tmp_path, [first, {**second, "outbox": outbox_a}]))
    with pytest.raises(CameraRuntimeError, match="duplicate evidence directory"):
        parse_manifest(_document(tmp_path, [first, {**second, "evidenceDir": str(evidence_a)}]))
    with pytest.raises(CameraRuntimeError, match="duplicate camera source"):
        parse_manifest(
            _document(tmp_path, [first, {**second, "source": {"kind": "path", "path": video_a}}])
        )
    with pytest.raises(CameraRuntimeError, match="environment reference"):
        parse_manifest(
            _document(
                tmp_path,
                [
                    {
                        **first,
                        "source": {
                            "kind": "path",
                            "path": "rtsp://operator:secret@10.0.0.8/live",
                        },
                    }
                ],
            )
        )
    with pytest.raises(CameraRuntimeError, match="environment name is invalid"):
        parse_manifest(
            _document(
                tmp_path,
                [first, {**second, "source": {"kind": "env", "name": "CAMERA_PASSWORD"}}],
            )
        )
    with pytest.raises(CameraRuntimeError, match="unsupported"):
        parse_manifest({**_document(tmp_path, [first]), "schemaVersion": "2"})
    with pytest.raises(CameraRuntimeError, match="manifest is invalid"):
        parse_manifest({"schemaVersion": "1", "cameras": [first]})

    secret = "rtsp://operator:secret-token@10.0.0.8/live"
    with pytest.raises(CameraRuntimeError, match="is unset") as missing:
        bind_manifest(parsed, {})
    assert secret not in str(missing.value)
    with pytest.raises(CameraRuntimeError, match="duplicate camera source") as duplicate:
        bind_manifest(
            parsed,
            {
                "SMARTSITE_AI_CAMERA_YARD_02_SOURCE": video_a,
            },
        )
    assert secret not in str(duplicate.value)
    with pytest.raises(CameraRuntimeError, match="source is invalid") as invalid:
        bind_manifest(
            parsed, {"SMARTSITE_AI_CAMERA_YARD_02_SOURCE": "rtsp://operator:secret-token@"}
        )
    assert "secret-token" not in str(invalid.value)


def test_manifest_binds_isolated_paths_without_printing_secrets(tmp_path: Path) -> None:
    video = _video(tmp_path, "gate.mp4")
    evidence = tmp_path / "evidence-gate"
    evidence.mkdir()
    secret = "rtsp://user:s3cret@192.0.2.10/stream"
    document = _document(
        tmp_path,
        [
            _entry(
                "gate-01",
                CAMERA_A,
                "CAM-A",
                source={"kind": "path", "path": video},
                outbox=str(tmp_path / "gate.sqlite3"),
                evidence_dir=str(evidence),
            ),
            _entry(
                "yard-02",
                CAMERA_B,
                "CAM-B",
                source={"kind": "env", "name": "SMARTSITE_AI_CAMERA_YARD_02_SOURCE"},
                outbox=str(tmp_path / "yard.sqlite3"),
                ppe_region_id=REGION_B,
            ),
        ],
    )
    bound = bind_manifest(
        parse_manifest(document),
        {"SMARTSITE_AI_CAMERA_YARD_02_SOURCE": secret},
    )
    assert bound.cameras[0].source == video
    assert bound.cameras[1].source == secret
    assert bound.cameras[0].evidence_dir == evidence
    assert bound.cameras[1].evidence_dir is None
    rendered = json.dumps(document)
    assert "s3cret" not in rendered


def test_manifest_binding_rejects_an_existing_link_like_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _video(tmp_path, "gate.mp4")
    outbox = tmp_path / "gate.sqlite3"
    outbox.write_bytes(b"not-a-database")
    document = _document(
        tmp_path,
        [
            _entry(
                "gate-01",
                CAMERA_A,
                "CAM-A",
                source={"kind": "path", "path": video},
                outbox=str(outbox),
            )
        ],
    )

    monkeypatch.setattr(
        "smartsite_ai.runtime.manifest.is_link_like",
        lambda path: path == outbox,
    )

    with pytest.raises(CameraRuntimeError, match="regular non-symlink file"):
        bind_manifest(parse_manifest(document), {})


def test_example_manifest_parses_without_reading_its_paths() -> None:
    example = (
        Path(__file__).parents[1] / "examples" / "camera-runtime-manifest.example.json"
    ).read_bytes()
    manifest = load_manifest_bytes(example)
    assert len(manifest.cameras) == 2
    assert manifest.cameras[1].source.kind == "env"


def test_manifest_parsing_accepts_posix_absolute_paths_on_a_windows_host(
    tmp_path: Path,
) -> None:
    document = _document(
        tmp_path,
        [
            _entry(
                "gate-01",
                CAMERA_A,
                "CAM-A",
                source={"kind": "path", "path": "/srv/smartsite/gate-01.mp4"},
                outbox="/var/lib/smartsite/gate-01.sqlite3",
                evidence_dir="/var/lib/smartsite/evidence-gate-01",
            )
        ],
    )
    document["modelSpec"] = "/opt/smartsite/yolo11s-ppe-artifact.json"

    manifest = parse_manifest(document)

    assert manifest.model_spec == "/opt/smartsite/yolo11s-ppe-artifact.json"


def test_ultralytics_runner_is_not_marked_thread_safe() -> None:
    assert UltralyticsYoloRunner.concurrent_inference_safe is False
    runner = UltralyticsYoloRunner(
        model_factory=lambda _path: object(),
        image_factory=lambda _frame: object(),
        provider_version_factory=lambda: "test",
    )
    assert isinstance(share_detector(_Detector(), runner), SerializingDetector)


def test_one_camera_cli_remains_a_separate_command() -> None:
    with pytest.raises(SystemExit):
        build_one_camera_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    one_camera = build_one_camera_parser().parse_args(
        [
            "--source",
            "0",
            "--stream-id",
            "gate",
            "--camera-id",
            CAMERA_A,
            "--camera-external-id",
            "CAM-A",
            "--ppe-region-id",
            REGION_A,
            "--model-spec",
            "C:\\SmartSiteData\\model.json",
            "--outbox",
            "C:\\SmartSiteData\\out.sqlite3",
        ]
    )
    assert one_camera.stream_id == "gate"
    assert not hasattr(one_camera, "manifest")


class _Detector:
    def __init__(self, *, safe: bool = False) -> None:
        self.concurrent_inference_safe = safe
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls += 1
        await asyncio.sleep(0)
        self.in_flight -= 1
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="ppe-model",
            model_version="1",
            model_sha256="a" * 64,
            detections=(),
        )


def _frame(camera_id: str, sequence: int) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id=camera_id,
        session_id=UUID(CAMERA_A),
        camera_external_id=camera_id,
        captured_at=datetime(2026, 9, 29, 8, 0, tzinfo=UTC) + timedelta(seconds=sequence),
        width=2,
        height=2,
        sequence_number=sequence,
        payload=b"\x00" * 12,
    )


@pytest.mark.anyio
async def test_shared_detector_serializes_until_runner_proves_safe() -> None:
    unsafe = _Detector()
    lane = share_detector(unsafe, unsafe)
    await asyncio.gather(lane.detect(_frame("CAM-A", 0)), lane.detect(_frame("CAM-B", 0)))
    assert unsafe.max_in_flight == 1
    assert unsafe.calls == 2

    safe = _Detector(safe=True)
    direct = share_detector(safe, safe)
    await asyncio.gather(direct.detect(_frame("CAM-A", 1)), direct.detect(_frame("CAM-B", 1)))
    assert safe.max_in_flight == 2


def test_startup_failure_marks_unchecked_cameras_not_started() -> None:
    outcomes = startup_outcomes(
        (("gate-01", "CAM-A"), ("yard-02", "CAM-B"), ("dock-03", "CAM-C")),
        (("yard-02", "BackendConfigurationNotFoundError"),),
    )
    assert [item.status for item in outcomes] == ["not_started", "failed", "not_started"]
    assert outcomes[1].error_class == "BackendConfigurationNotFoundError"


@pytest.mark.anyio
async def test_partial_startup_failure_does_not_load_the_model() -> None:
    opened = 0
    closed = 0

    async def preflight() -> tuple[object, ...]:
        return startup_outcomes(
            (("gate-01", "CAM-A"), ("yard-02", "CAM-B")), (("yard-02", "RuntimeError"),)
        )

    async def open_model() -> _Detector:
        nonlocal opened
        opened += 1
        return _Detector()

    async def close_model() -> None:
        nonlocal closed
        closed += 1

    def build_sessions(_detector: object) -> tuple[CameraSession, ...]:
        raise AssertionError("sessions must not be built after preflight failure")

    report = await execute_runtime(
        RuntimePlan(
            preflight=preflight,
            open_model=open_model,
            close_model=close_model,
            build_sessions=build_sessions,
        )
    )
    assert opened == 0
    assert closed == 0
    assert report.model_loaded is False
    assert [camera.status for camera in report.cameras] == ["not_started", "failed"]
    assert exit_code(report) == 1


@pytest.mark.anyio
async def test_runtime_failure_keeps_the_healthy_camera_and_closes_the_model_once() -> None:
    opened = 0
    closed = 0
    healthy_cancelled = False

    class Healthy:
        async def run(self) -> ProcessingWorkerResult:
            nonlocal healthy_cancelled
            try:
                await asyncio.sleep(0)
                return _clean_result()
            finally:
                task = asyncio.current_task()
                healthy_cancelled = task is not None and task.cancelling() > 0

    class Failing:
        async def run(self) -> ProcessingWorkerResult:
            raise RuntimeError("rtsp://user:secret@10.0.0.8/live exploded")

    async def preflight() -> None:
        return None

    async def open_model() -> _Detector:
        nonlocal opened
        opened += 1
        return _Detector()

    async def close_model() -> None:
        nonlocal closed
        closed += 1

    def build_sessions(_detector: object) -> tuple[CameraSession, ...]:
        return (
            CameraSession("gate-01", "CAM-A", Healthy(), _StopPoller()),
            CameraSession("yard-02", "CAM-B", Failing(), _StopPoller()),
        )

    report = await execute_runtime(
        RuntimePlan(
            preflight=preflight,
            open_model=open_model,
            close_model=close_model,
            build_sessions=build_sessions,
        )
    )
    assert opened == 1
    assert closed == 1
    assert report.model_loaded is True
    assert report.model_closed is True
    assert report.cameras[0].status == "completed"
    assert report.cameras[1].status == "failed"
    assert report.cameras[1].error_class == "RuntimeError"
    assert healthy_cancelled is False
    rendered = json.dumps(report_payload(report))
    assert "secret" not in rendered
    assert exit_code(report) == 1


@pytest.mark.anyio
async def test_cancellation_releases_every_session_and_closes_one_model() -> None:
    closed = 0
    started = 0
    ready = asyncio.Event()
    workers: list[_HangWorker] = []

    class _HangWorker:
        def __init__(self) -> None:
            self.cancelled = False

        async def run(self) -> ProcessingWorkerResult:
            nonlocal started
            started += 1
            if started == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                task = asyncio.current_task()
                self.cancelled = task is not None and task.cancelling() > 0
            raise AssertionError("hanging worker must be cancelled")

    async def preflight() -> None:
        return None

    async def open_model() -> _Detector:
        return _Detector()

    async def close_model() -> None:
        nonlocal closed
        closed += 1

    def build_sessions(_detector: object) -> tuple[CameraSession, ...]:
        workers.extend((_HangWorker(), _HangWorker()))
        return (
            CameraSession("gate-01", "CAM-A", workers[0], _StopPoller()),
            CameraSession("yard-02", "CAM-B", workers[1], _StopPoller()),
        )

    task = asyncio.create_task(
        execute_runtime(
            RuntimePlan(
                preflight=preflight,
                open_model=open_model,
                close_model=close_model,
                build_sessions=build_sessions,
            )
        )
    )
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == 1
    assert [worker.cancelled for worker in workers] == [True, True]


@pytest.mark.anyio
async def test_pending_outbox_uses_the_one_camera_review_exit_code() -> None:
    async def preflight() -> None:
        return None

    async def open_model() -> _Detector:
        return _Detector()

    async def close_model() -> None:
        return None

    pending = ProcessingWorkerResult(
        frames_processed=1,
        events_enqueued=1,
        outbox=OutboxCounts(pending=1, delivered=0, terminal=0),
    )

    def build_sessions(_detector: object) -> tuple[CameraSession, ...]:
        return (CameraSession("gate-01", "CAM-A", _ResultWorker(pending), _StopPoller()),)

    report = await execute_runtime(
        RuntimePlan(
            preflight=preflight,
            open_model=open_model,
            close_model=close_model,
            build_sessions=build_sessions,
        )
    )
    assert exit_code(report) == 2
    assert report.model_closed is True


def _configuration(external_id: str, region_id: str) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": external_id,
            "regions": (
                {
                    "regionId": region_id,
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))},
                },
            ),
        }
    )


class _Stream:
    def __init__(self, frame: FrameEnvelope) -> None:
        self.config = StreamConfig(
            stream_id=frame.stream_id,
            camera_external_id=frame.camera_external_id,
            source_url="test://camera",
            is_live=False,
        )
        self.frame: FrameEnvelope | None = frame
        self.stopped = False

    async def start(self) -> None:
        return None

    async def get_frame(self) -> FrameEnvelope:
        await asyncio.sleep(0)
        if self.frame is None:
            raise QueueClosedError("complete")
        frame = self.frame
        self.frame = None
        return frame

    async def stop(self) -> None:
        self.stopped = True

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(state=StreamState.STOPPED)


class _RecordingPipeline:
    def __init__(self, region_id: str) -> None:
        self.region_id = region_id
        self.cameras: list[str] = []

    def process(
        self,
        batch: DetectionBatch,
        *,
        region_configuration: CameraRegionConfiguration,
        event_id: str,
    ) -> TechnicalObservationEvent:
        self.cameras.append(batch.camera_external_id)
        return TechnicalObservationEvent.create(
            event_id=event_id,
            camera_external_id=batch.camera_external_id,
            stream_session_id=str(batch.session_id),
            captured_at=batch.captured_at.isoformat(),
            frame_dimensions={"width": batch.frame_width, "height": batch.frame_height},
            observations=[
                {
                    "type": "ZONE_ENTRY",
                    "trackId": 1,
                    "regionId": self.region_id,
                    "geometryVersion": 1,
                    "confidence": 0.9,
                }
            ],
        )


@pytest.mark.anyio
async def test_supervised_cameras_do_not_share_pipeline_or_outbox_state(tmp_path: Path) -> None:
    detector = share_detector(_Detector(), _Detector())
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body["cameraExternalId"])
        return httpx.Response(
            202,
            json={"eventId": body["eventId"], "status": "PROCESSED", "alertIds": []},
        )

    async def session(
        external_id: str, region_id: str, filename: str
    ) -> tuple[CameraSession, _RecordingPipeline]:
        store = RegionConfigurationStore()
        await store.apply(_configuration(external_id, region_id))
        pipeline = _RecordingPipeline(region_id)
        outbox = SqliteEventOutbox((tmp_path / filename).resolve())
        client = BackendClient(
            "http://backend.test",
            "token",
            transport=httpx.MockTransport(handler),
        )
        worker = HeadlessCameraProcessingWorker(
            stream=_Stream(_frame(external_id, 1)),
            detector=detector,
            pipeline=pipeline,
            ppe_region_id=region_id,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, client),
            delivery_interval_seconds=0.01,
        )
        return (
            CameraSession(external_id, external_id, worker, _StopPoller()),
            pipeline,
        )

    first, pipeline_a = await session("CAM-A", REGION_A, "a.sqlite3")
    second, pipeline_b = await session("CAM-B", REGION_B, "b.sqlite3")
    outcomes = await supervise_sessions((first, second))

    assert [item.status for item in outcomes] == ["completed", "completed"]
    assert pipeline_a.cameras == ["CAM-A"]
    assert pipeline_b.cameras == ["CAM-B"]
    assert sorted(requests) == ["CAM-A", "CAM-B"]
    assert first.worker is not second.worker
    assert isinstance(detector, SerializingDetector)


def test_assemble_sessions_keeps_one_detector_and_distinct_camera_state(tmp_path: Path) -> None:
    from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
    from smartsite_ai.runtime.assembly import CameraPlan

    evidence_a = tmp_path / "evidence-a"
    evidence_b = tmp_path / "evidence-b"
    evidence_a.mkdir()
    evidence_b.mkdir()
    video_a = _video(tmp_path, "a.mp4")
    video_b = _video(tmp_path, "b.mp4")
    document = _document(
        tmp_path,
        [
            _entry(
                "gate-01",
                CAMERA_A,
                "CAM-A",
                source={"kind": "path", "path": video_a},
                outbox=str(tmp_path / "a.sqlite3"),
                evidence_dir=str(evidence_a),
            ),
            _entry(
                "yard-02",
                CAMERA_B,
                "CAM-B",
                source={"kind": "path", "path": video_b},
                outbox=str(tmp_path / "b.sqlite3"),
                evidence_dir=str(evidence_b),
                ppe_region_id=REGION_B,
            ),
        ],
    )
    bound = bind_manifest(parse_manifest(document), {})
    detector = _Detector()

    class _Client:
        def __init__(self) -> None:
            self.closed = False

        async def fetch(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("assemble must not fetch configuration")

        async def aclose(self) -> None:
            self.closed = True

    plans = []
    for camera in bound.cameras:
        client = _Client()
        store = RegionConfigurationStore()
        poller = CameraConfigurationPoller(
            client=client,  # type: ignore[arg-type]
            store=store,
            camera_id=camera.entry.camera_id,
            camera_external_id=camera.camera_external_id,
        )
        plans.append(
            CameraPlan(bound=camera, store=store, poller=poller, configuration_client=client)
        )  # type: ignore[arg-type]

    def source_factory(config: StreamConfig) -> object:
        return SimpleNamespace(config=config)

    sessions = assemble_sessions(
        plans,
        detector,  # type: ignore[arg-type]
        backend_client_factory=lambda: SimpleNamespace(),  # type: ignore[arg-type,return-value]
        source_factory=source_factory,
        evidence_max_bytes=1024,
        delivery_interval_seconds=0.25,
    )
    workers = [session.worker for session in sessions]
    assert workers[0]._detector is detector  # type: ignore[attr-defined]
    assert workers[1]._detector is detector  # type: ignore[attr-defined]
    assert workers[0]._pipeline is not workers[1]._pipeline  # type: ignore[attr-defined]
    assert workers[0]._configurations is not workers[1]._configurations  # type: ignore[attr-defined]
    assert workers[0]._outbox is not workers[1]._outbox  # type: ignore[attr-defined]
    assert workers[0]._evidence_publisher is not workers[1]._evidence_publisher  # type: ignore[attr-defined]
    assert workers[0]._ppe_region_id != workers[1]._ppe_region_id  # type: ignore[attr-defined]
    assert workers[0]._temporal_gate is not workers[1]._temporal_gate  # type: ignore[attr-defined]


def test_third_camera_id_constant_is_distinct() -> None:
    assert len({CAMERA_A, CAMERA_B, CAMERA_C}) == 3
