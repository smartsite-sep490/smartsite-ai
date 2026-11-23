from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from uuid import UUID

import pytest
from test_generation_runtime_claim import CAMERA, REGION, ConfigurationClient
from test_source_control_child_profile import inputs

from smartsite_ai.config import Settings
from smartsite_ai.inference.artifacts import ArtifactValidationError, VerifiedModelArtifact
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.inference.yolo import (
    DetectorUnavailableError,
    RawYoloDetection,
    Yolo11Detector,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.outbox import OutboxCounts
from smartsite_ai.processing_worker import ProcessingWorkerResult
from smartsite_ai.runtime.generation_manifest import bind_generation_manifest
from smartsite_ai.runtime.inference_lane import SerializingDetector
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest
from smartsite_ai.runtime.source_control_child import bind_source_control_child_memory
from smartsite_ai.runtime.supervisor import CameraSession
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.tools import run_camera_runtime as runtime_cli

SESSION_ID = UUID("00000000-0000-4000-8000-000000000042")


class _FakeRunner(UltralyticsYoloRunner):
    def __init__(self, *, concurrent_inference_safe: bool = False) -> None:
        super().__init__()
        self.concurrent_inference_safe = concurrent_inference_safe
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeDetector:
    def __init__(self, detections: tuple[NormalizedDetection, ...], artifact_id: str) -> None:
        self._detections = detections
        self._artifact_id = artifact_id

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id=self._artifact_id,
            model_version="1",
            model_sha256="a" * 64,
            detections=self._detections,
        )


class _StopPoller:
    async def run(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()


def _frame() -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="synthetic",
        session_id=SESSION_ID,
        camera_external_id="CAM",
        captured_at=datetime(2026, 11, 22, 22, 0, 0, tzinfo=UTC),
        width=640,
        height=480,
        sequence_number=1,
        payload=b"\x00\x00\x00" * (640 * 480),
        pixel_format="BGR24",
    )


def _detection(
    class_id: int,
    class_name: str,
    confidence: float,
    bounds: tuple[float, float, float, float],
) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(
            x1=bounds[0],
            y1=bounds[1],
            x2=bounds[2],
            y2=bounds[3],
            coordinate_space="NORMALIZED_0_1",
        ),
    )


def _manifest_doc(
    tmp_path: Path,
    *,
    person_model_spec: object = ...,
) -> dict[str, object]:
    model_spec = tmp_path / "model.json"
    model_spec.write_text("{}", encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    document: dict[str, object] = {
        "schemaVersion": "1",
        "modelSpec": str(model_spec),
        "cameras": [
            {
                "streamId": "synthetic",
                "cameraId": str(CAMERA),
                "cameraExternalId": "CAM",
                "ppeRegionId": str(REGION),
                "source": {"kind": "path", "path": str(video)},
                "live": False,
                "outbox": str(tmp_path / "outbox.sqlite3"),
            }
        ],
    }
    if person_model_spec is not ...:
        document["personModelSpec"] = person_model_spec
    return document


def _write_manifest(
    tmp_path: Path,
    *,
    with_person_model: bool,
) -> tuple[Path, Path | None]:
    person_spec: Path | None = None
    person_arg: object = ...
    if with_person_model:
        person_spec = tmp_path / "person-model.json"
        person_spec.write_text("{}", encoding="utf-8")
        person_arg = str(person_spec)
    manifest = parse_manifest(_manifest_doc(tmp_path, person_model_spec=person_arg))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(by_alias=True), encoding="utf-8")
    return manifest_path, person_spec


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_runner: _FakeRunner,
    primary_detector: _FakeDetector,
    secondary_loader: Callable[..., object],
) -> list[tuple[Path, object]]:
    monkeypatch.setattr(
        runtime_cli.BackendConfigurationClient,
        "from_settings",
        classmethod(lambda _cls, *_a, **_kw: ConfigurationClient()),
    )
    primary_calls: list[tuple[Path, object]] = []

    def fake_primary(spec: Path, **kwargs: object):
        primary_calls.append((spec, kwargs.get("runner_factory")))
        return (
            primary_detector,
            primary_runner,
            None,
            MappingProxyType({0: "person", 1: "hardhat"}),
        )

    monkeypatch.setattr(runtime_cli, "load_yolo11_detector", fake_primary)
    monkeypatch.setattr(runtime_cli, "load_yolo11_person_detector", secondary_loader)
    return primary_calls


def test_manifest_absent_person_model_spec_preserves_legacy_defaults(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_doc(tmp_path))
    assert manifest.person_model_spec is None
    bound = bind_manifest(manifest, {})
    assert bound.person_model_spec is None


def test_manifest_accepts_person_model_spec_and_preserves_through_bindings(
    tmp_path: Path,
) -> None:
    value, bundle, credential_path, _, expected, env = inputs(tmp_path)
    person_spec = tmp_path / "person-model.json"
    person_spec.write_text("{}", encoding="utf-8")
    value["personModelSpec"] = str(person_spec)

    manifest = parse_manifest(value)
    assert manifest.person_model_spec == str(person_spec)

    gen_bound = bind_generation_manifest(
        manifest,
        credential_path,
        UUID(bundle.camera_id),
        Path(manifest.cameras[0].outbox),
    )
    assert gen_bound.person_model_spec == str(person_spec)

    op_bound, _ = bind_source_control_child_memory(
        gen_bound,
        policy=SourcePolicy({"usb0": 0}, {}, ()),
        scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
        camera_id=UUID(bundle.camera_id),
        environ=env,
        expected=expected,
    )
    assert op_bound.person_model_spec == str(person_spec)

    runtime_bound = bind_manifest(op_bound, env)
    assert runtime_bound.person_model_spec == person_spec


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "   ",
        "relative/person-model.json",
        123,
        True,
    ],
)
def test_manifest_person_model_spec_rejects_invalid_values(
    tmp_path: Path, bad_value: object
) -> None:
    with pytest.raises(CameraRuntimeError):
        parse_manifest(_manifest_doc(tmp_path, person_model_spec=bad_value))


@pytest.mark.parametrize("target", ["missing", "directory"])
def test_bind_manifest_person_model_spec_requires_existing_regular_file(
    tmp_path: Path, target: str
) -> None:
    candidate = tmp_path / "missing-person.json" if target == "missing" else tmp_path
    manifest = parse_manifest(_manifest_doc(tmp_path, person_model_spec=str(candidate)))
    with pytest.raises(CameraRuntimeError):
        bind_manifest(manifest, {})


def test_run_manifest_absent_person_model_spec_skips_secondary_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _write_manifest(tmp_path, with_person_model=False)
    primary_runner = _FakeRunner()
    primary_detector = _FakeDetector(
        (_detection(0, "person", 0.90, (0.1, 0.1, 0.5, 0.9)),),
        artifact_id="primary-ppe",
    )
    secondary_calls: list[Path] = []

    def fail_secondary(spec: Path, **_kwargs: object):
        secondary_calls.append(spec)
        raise AssertionError("secondary person loader must not be called when omitted")

    primary_calls = _patch_loaders(
        monkeypatch,
        primary_runner=primary_runner,
        primary_detector=primary_detector,
        secondary_loader=fail_secondary,
    )

    seen_batches: list[DetectionBatch] = []

    class _InspectWorker:
        def __init__(self, detector: object) -> None:
            self._detector = detector

        async def run(self) -> ProcessingWorkerResult:
            assert isinstance(self._detector, SerializingDetector)
            seen_batches.append(await self._detector.detect(_frame()))
            return ProcessingWorkerResult(1, 0, OutboxCounts(0, 0, 0))

    monkeypatch.setattr(
        runtime_cli,
        "assemble_sessions",
        lambda plans, detector, **_kwargs: (
            CameraSession(
                stream_id=plans[0].bound.stream_id,
                camera_external_id=plans[0].bound.camera_external_id,
                worker=_InspectWorker(detector),
                poller=_StopPoller(),
            ),
        ),
    )

    exit_status = asyncio.run(
        runtime_cli.run_manifest(manifest_path, settings=Settings(_env_file=None))
    )
    assert exit_status == 0
    assert primary_calls == [(tmp_path / "model.json", runtime_cli.UltralyticsYoloRunner)]
    assert secondary_calls == []
    assert seen_batches[0].model_artifact_id == "primary-ppe"
    assert primary_runner.close_calls == 1


@pytest.mark.parametrize("primary_concurrent_safe", [False, True])
def test_run_manifest_opt_in_person_model_composes_person_anchored_and_closes_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_concurrent_safe: bool,
) -> None:
    manifest_path, person_spec = _write_manifest(tmp_path, with_person_model=True)
    primary_runner = _FakeRunner(concurrent_inference_safe=primary_concurrent_safe)
    secondary_runner = _FakeRunner(concurrent_inference_safe=False)
    primary_detector = _FakeDetector(
        (
            _detection(0, "person", 0.91, (0.0, 0.0, 0.9, 1.0)),
            _detection(1, "hardhat", 0.88, (0.4, 0.1, 0.6, 0.25)),
        ),
        artifact_id="primary-ppe",
    )
    secondary_detector = _FakeDetector(
        (
            _detection(0, "person", 0.85, (0.45, 0.20, 0.65, 0.95)),
            _detection(2, "car", 0.75, (0.05, 0.40, 0.30, 0.80)),
        ),
        artifact_id="coco-person",
    )
    secondary_calls: list[tuple[Path, object, set[str]]] = []

    def fake_secondary(spec: Path, **kwargs: object):
        secondary_calls.append((spec, kwargs.get("runner_factory"), set(kwargs)))
        return (
            secondary_detector,
            secondary_runner,
            None,
            MappingProxyType({0: "person"}),
        )

    primary_calls = _patch_loaders(
        monkeypatch,
        primary_runner=primary_runner,
        primary_detector=primary_detector,
        secondary_loader=fake_secondary,
    )

    seen_batches: list[DetectionBatch] = []

    class _InspectWorker:
        def __init__(self, detector: object) -> None:
            self._detector = detector

        async def run(self) -> ProcessingWorkerResult:
            assert isinstance(self._detector, SerializingDetector)
            seen_batches.append(await self._detector.detect(_frame()))
            return ProcessingWorkerResult(1, 0, OutboxCounts(0, 0, 0))

    monkeypatch.setattr(
        runtime_cli,
        "assemble_sessions",
        lambda plans, detector, **_kwargs: (
            CameraSession(
                stream_id=plans[0].bound.stream_id,
                camera_external_id=plans[0].bound.camera_external_id,
                worker=_InspectWorker(detector),
                poller=_StopPoller(),
            ),
        ),
    )

    exit_status = asyncio.run(
        runtime_cli.run_manifest(manifest_path, settings=Settings(_env_file=None))
    )
    assert exit_status == 0
    assert primary_calls == [(tmp_path / "model.json", runtime_cli.UltralyticsYoloRunner)]
    assert secondary_calls == [(person_spec, runtime_cli.UltralyticsYoloRunner, {"runner_factory"})]
    assert len(seen_batches) == 1
    assert seen_batches[0].model_artifact_id == "person-anchored-ppe"
    assert seen_batches[0].model_version == "v1"
    assert [(d.class_id, d.class_name, d.bounding_box.x1) for d in seen_batches[0].detections] == [
        (1, "hardhat", 0.4),
        (0, "person", 0.45),
    ]
    assert primary_runner.close_calls == 1
    assert secondary_runner.close_calls == 1


def test_run_manifest_opt_in_person_model_closes_both_runners_on_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _write_manifest(tmp_path, with_person_model=True)
    primary_runner = _FakeRunner()
    secondary_runner = _FakeRunner()
    primary_detector = _FakeDetector((), artifact_id="primary-ppe")
    secondary_detector = _FakeDetector((), artifact_id="coco-person")

    _patch_loaders(
        monkeypatch,
        primary_runner=primary_runner,
        primary_detector=primary_detector,
        secondary_loader=lambda _spec, **_kw: (
            secondary_detector,
            secondary_runner,
            None,
            MappingProxyType({0: "person"}),
        ),
    )

    started = asyncio.Event()

    class _BlockingWorker:
        async def run(self) -> ProcessingWorkerResult:
            started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        runtime_cli,
        "assemble_sessions",
        lambda plans, _detector, **_kwargs: (
            CameraSession(
                stream_id=plans[0].bound.stream_id,
                camera_external_id=plans[0].bound.camera_external_id,
                worker=_BlockingWorker(),
                poller=_StopPoller(),
            ),
        ),
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            runtime_cli.run_manifest(manifest_path, settings=Settings(_env_file=None))
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert primary_runner.close_calls == 1
    assert secondary_runner.close_calls == 1


@pytest.mark.parametrize(
    "failure_mode",
    ["loader_error", "bad_runner_type", "cancelled_during_secondary"],
)
def test_run_manifest_closes_runners_on_secondary_failure_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    manifest_path, _ = _write_manifest(tmp_path, with_person_model=True)
    primary_runner = _FakeRunner()
    primary_detector = _FakeDetector((), artifact_id="primary-ppe")
    bad_runner_closed = 0

    def close_bad_runner() -> None:
        nonlocal bad_runner_closed
        bad_runner_closed += 1

    def failing_secondary(_spec: Path, **_kwargs: object):
        if failure_mode == "loader_error":
            raise ArtifactValidationError("invalid secondary person model")
        if failure_mode == "cancelled_during_secondary":
            raise asyncio.CancelledError()
        return (
            _FakeDetector((), artifact_id="coco-person"),
            SimpleNamespace(close=close_bad_runner),
            None,
            MappingProxyType({0: "person"}),
        )

    _patch_loaders(
        monkeypatch,
        primary_runner=primary_runner,
        primary_detector=primary_detector,
        secondary_loader=failing_secondary,
    )

    session_built = False

    def fail_assemble(*_args: object, **_kwargs: object):
        nonlocal session_built
        session_built = True
        raise AssertionError("runtime must not fall back when secondary person loader fails")

    monkeypatch.setattr(runtime_cli, "assemble_sessions", fail_assemble)

    expected_exc: type[BaseException]
    if failure_mode == "loader_error":
        expected_exc = ArtifactValidationError
    elif failure_mode == "cancelled_during_secondary":
        expected_exc = asyncio.CancelledError
    else:
        expected_exc = TypeError

    with pytest.raises(expected_exc):
        asyncio.run(runtime_cli.run_manifest(manifest_path, settings=Settings(_env_file=None)))

    assert not session_built
    assert primary_runner.close_calls == 1
    if failure_mode == "bad_runner_type":
        assert bad_runner_closed == 1


class _WarmupRecordingRunner(UltralyticsYoloRunner):
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        detections: tuple[RawYoloDetection, ...] = (),
        fail_predict: BaseException | None = None,
    ) -> None:
        super().__init__()
        self._label = label
        self._events = events
        self._detections = detections
        self._fail_predict = fail_predict
        self.frames: list[FrameEnvelope] = []
        self.close_calls = 0

    def predict(self, frame: FrameEnvelope) -> tuple[RawYoloDetection, ...]:
        self._events.append(f"predict:{self._label}")
        self.frames.append(frame)
        if self._fail_predict is not None:
            raise self._fail_predict
        return self._detections

    def close(self) -> None:
        self._events.append(f"close:{self._label}")
        self.close_calls += 1


def _verified_artifact(
    tmp_path: Path, name: str, class_map: tuple[tuple[int, str], ...]
) -> VerifiedModelArtifact:
    weights = tmp_path / f"{name}.pt"
    weights.write_bytes(name.encode("utf-8"))
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return VerifiedModelArtifact.model_validate(
        {
            "artifact_id": name,
            "version": "1",
            "model_family": "yolo11s",
            "artifact_path": weights,
            "sha256": digest,
            "source_url": f"https://models.example.test/{name}.pt",
            "license": "AGPL-3.0",
            "class_map": class_map,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "image_size": (640, 640),
            "device": "cpu",
            "resolved_path": weights.resolve(),
            "actual_sha256": digest,
        }
    )


@pytest.mark.parametrize("with_person_model", [False, True])
def test_run_manifest_warms_loaded_runners_before_opening_source_with_discarded_black_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_person_model: bool
) -> None:
    manifest_path, _ = _write_manifest(tmp_path, with_person_model=with_person_model)
    events: list[str] = []
    ppe_artifact = _verified_artifact(tmp_path, "ppe", ((0, "person"), (1, "hard_hat")))
    person_artifact = _verified_artifact(tmp_path, "persons", ((0, "person"),))

    primary_runner = _WarmupRecordingRunner(
        "ppe",
        events,
        detections=(
            RawYoloDetection(class_id=1, confidence=0.95, x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        ),
    )
    secondary_runner = _WarmupRecordingRunner(
        "persons",
        events,
        detections=(
            RawYoloDetection(class_id=0, confidence=0.95, x1=10.0, y1=10.0, x2=60.0, y2=90.0),
        ),
    )

    monkeypatch.setattr(
        runtime_cli.BackendConfigurationClient,
        "from_settings",
        classmethod(lambda _cls, *_a, **_kw: ConfigurationClient()),
    )
    monkeypatch.setattr(
        runtime_cli,
        "load_yolo11_detector",
        lambda *_a, **_kw: (
            Yolo11Detector(ppe_artifact, primary_runner),
            primary_runner,
            ppe_artifact,
            MappingProxyType(dict(ppe_artifact.class_map)),
        ),
    )
    monkeypatch.setattr(
        runtime_cli,
        "load_yolo11_person_detector",
        lambda *_a, **_kw: (
            Yolo11Detector(person_artifact, secondary_runner),
            secondary_runner,
            person_artifact,
            MappingProxyType(dict(person_artifact.class_map)),
        ),
    )

    class _RecordingSource:
        def __init__(self, config: object) -> None:
            events.append("source:construct")
            self.config = config

    monkeypatch.setattr(runtime_cli, "OpenCvFrameSource", _RecordingSource)
    observed_workers: list[object] = []

    class _AssertNoWarmupStateWorker:
        def __init__(self, detector: object, source: object) -> None:
            self._detector = detector
            self._source = source

        async def run(self) -> ProcessingWorkerResult:
            events.append("worker:run")
            # Warmup occurred before worker/source creation and left zero outbox state.
            assert not (tmp_path / "outbox.sqlite3").exists()
            return ProcessingWorkerResult(1, 0, OutboxCounts(0, 0, 0))

    def spy_assemble(plans, detector, *, source_factory, **_kwargs):
        events.append("assemble_sessions")
        assert len(primary_runner.frames) == 1
        if with_person_model:
            assert len(secondary_runner.frames) == 1
        src = source_factory(plans[0].bound.source_config)
        worker = _AssertNoWarmupStateWorker(detector, src)
        observed_workers.append(worker)
        return (
            CameraSession(
                stream_id=plans[0].bound.stream_id,
                camera_external_id=plans[0].bound.camera_external_id,
                worker=worker,
                poller=_StopPoller(),
            ),
        )

    monkeypatch.setattr(runtime_cli, "assemble_sessions", spy_assemble)

    status = asyncio.run(runtime_cli.run_manifest(manifest_path, settings=Settings(_env_file=None)))
    assert status == 0
    assert len(observed_workers) == 1

    expected_predicts = ["predict:ppe", "predict:persons"] if with_person_model else ["predict:ppe"]
    assemble_index = events.index("assemble_sessions")
    assert events[:assemble_index] == expected_predicts
    assert "source:construct" in events[assemble_index:]

    for runner in (primary_runner, secondary_runner) if with_person_model else (primary_runner,):
        assert len(runner.frames) == 1
        warmup_frame = runner.frames[0]
        assert warmup_frame.pixel_format == "BGR24"
        assert warmup_frame.width == 640 and warmup_frame.height == 480
        assert warmup_frame.payload == bytes(640 * 480 * 3)
        assert runner.close_calls == 1
    if not with_person_model:
        assert secondary_runner.frames == []


@pytest.mark.parametrize("fail_target", ["primary", "secondary", "secondary_unexpected_error"])
def test_run_manifest_closes_both_runners_on_warmup_failure_before_opening_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_target: str
) -> None:
    manifest_path, _ = _write_manifest(tmp_path, with_person_model=True)
    events: list[str] = []
    ppe_artifact = _verified_artifact(tmp_path, "ppe", ((0, "person"), (1, "hard_hat")))
    person_artifact = _verified_artifact(tmp_path, "persons", ((0, "person"),))

    primary_error = (
        DetectorUnavailableError("warmup failed on ppe") if fail_target == "primary" else None
    )
    if fail_target == "secondary":
        secondary_error: BaseException | None = DetectorUnavailableError("warmup failed on persons")
    elif fail_target == "secondary_unexpected_error":
        secondary_error = RuntimeError("unexpected runner abort")
    else:
        secondary_error = None

    primary_runner = _WarmupRecordingRunner("ppe", events, fail_predict=primary_error)
    secondary_runner = _WarmupRecordingRunner("persons", events, fail_predict=secondary_error)

    monkeypatch.setattr(
        runtime_cli.BackendConfigurationClient,
        "from_settings",
        classmethod(lambda _cls, *_a, **_kw: ConfigurationClient()),
    )
    monkeypatch.setattr(
        runtime_cli,
        "load_yolo11_detector",
        lambda *_a, **_kw: (
            Yolo11Detector(ppe_artifact, primary_runner),
            primary_runner,
            ppe_artifact,
            MappingProxyType(dict(ppe_artifact.class_map)),
        ),
    )
    monkeypatch.setattr(
        runtime_cli,
        "load_yolo11_person_detector",
        lambda *_a, **_kw: (
            Yolo11Detector(person_artifact, secondary_runner),
            secondary_runner,
            person_artifact,
            MappingProxyType(dict(person_artifact.class_map)),
        ),
    )

    source_opened = False

    def fail_assemble(*_args: object, **_kwargs: object):
        nonlocal source_opened
        source_opened = True
        raise AssertionError("No camera session/source may open if model warmup fails")

    def fail_source(*_args: object, **_kwargs: object):
        nonlocal source_opened
        source_opened = True
        raise AssertionError("OpenCvFrameSource must not be constructed if warmup fails")

    monkeypatch.setattr(runtime_cli, "assemble_sessions", fail_assemble)
    monkeypatch.setattr(runtime_cli, "OpenCvFrameSource", fail_source)

    with pytest.raises(DetectorUnavailableError):
        asyncio.run(runtime_cli.run_manifest(manifest_path, settings=Settings(_env_file=None)))

    assert not source_opened
    assert primary_runner.close_calls == 1
    assert secondary_runner.close_calls == 1
