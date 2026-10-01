from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import pytest

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.evaluation.dataset import LoadedEvaluationDataset
from smartsite_ai.evaluation.execution import (
    EvaluationExecutionError,
    EvaluationExecutionMetadata,
    EvaluationExecutionRequest,
    EvaluationExecutionServices,
    execute_evaluation,
    load_ground_truth_episodes,
)
from smartsite_ai.evaluation.models import EvaluationFrame, EvaluationManifest
from smartsite_ai.evaluation.provider_validation import (
    ProviderValidationArguments,
    ProviderValidationMetrics,
    ProviderValidationReport,
)
from smartsite_ai.evaluation.report import GitReportMetadata, RuntimeReportMetadata
from smartsite_ai.inference.artifacts import ModelArtifactSpec
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines.ppe_temporal import ConfirmedPpeCandidate

SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
REGION_ID = "22222222-2222-4222-8222-222222222222"


def _manifest() -> EvaluationManifest:
    return EvaluationManifest.model_validate(
        {
            "schemaVersion": "1.0.0",
            "datasetId": "ppe-eval",
            "datasetVersion": "1",
            "sourceUrl": "https://example.test/dataset",
            "license": "CC-BY-4.0",
            "aggregateSha256": "a" * 64,
            "classMap": {
                "Person": "Person",
                "Hardhat": "Hardhat",
                "NO-Hardhat": "NO-Hardhat",
                "Safety Vest": "Safety Vest",
                "NO-Safety Vest": "NO-Safety Vest",
            },
            "splits": {
                "train": "train.jsonl",
                "validation": "validation.jsonl",
                "test": "test.jsonl",
            },
        }
    )


def _frame(frame_id: str, index: int, seconds: float) -> EvaluationFrame:
    return EvaluationFrame.model_validate(
        {
            "frameId": frame_id,
            "mediaPath": "clip.mp4",
            "sha256": "b" * 64,
            "width": 2,
            "height": 2,
            "frameIndex": index,
            "videoTimeSeconds": seconds,
            "annotations": [
                {
                    "annotationId": f"person-{index}",
                    "className": "Person",
                    "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
                    "personInstanceId": "worker-1",
                    "observablePpeItems": ["HARD_HAT", "SAFETY_VEST"],
                },
                {
                    "annotationId": f"no-hat-{index}",
                    "className": "NO-Hardhat",
                    "boundingBox": {"x1": 0.2, "y1": 0.1, "x2": 0.5, "y2": 0.35},
                    "relatedPersonAnnotationId": f"person-{index}",
                },
            ],
        }
    )


def _artifact(tmp_path: Path, *, family: str = "yolo11s") -> ModelArtifactSpec:
    return ModelArtifactSpec(
        artifact_id="ppe-yolo11s",
        version="1",
        model_family=family,
        artifact_path=(tmp_path / "model.pt").resolve(),
        sha256="c" * 64,
        source_url="https://example.test/model",
        license="AGPL-3.0",
        class_map=(
            (0, "Person"),
            (1, "Hardhat"),
            (2, "NO-Hardhat"),
            (3, "Safety Vest"),
            (4, "NO-Safety Vest"),
        ),
        confidence_threshold=0.25,
        iou_threshold=0.45,
        image_size=(640, 640),
        device="cpu",
    )


def _region_configuration() -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "camera-1",
            "regions": (
                {
                    "regionId": REGION_ID,
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
                },
            ),
        }
    )


def _provider_report(tmp_path: Path) -> ProviderValidationReport:
    return ProviderValidationReport(
        arguments=ProviderValidationArguments(
            model_path=(tmp_path / "model.pt").resolve(),
            data_config_path=(tmp_path / "data.yaml").resolve(),
            image_size=(640, 640),
            confidence_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
            batch_size=1,
            workers=0,
            device="cpu",
            seed=0,
        ),
        metrics=ProviderValidationMetrics(
            provider_name="ultralytics",
            provider_version="8.4.155",
            class_map=(
                (0, "Person"),
                (1, "Hardhat"),
                (2, "NO-Hardhat"),
                (3, "Safety Vest"),
                (4, "NO-Safety Vest"),
            ),
            ap50=0.8,
            ap50_95=0.6,
        ),
    )


class _Reader:
    def read(self, root: Path, frame: EvaluationFrame) -> FrameEnvelope:
        del root
        return FrameEnvelope(
            stream_id=frame.media_path,
            session_id=SESSION_ID,
            camera_external_id="camera-1",
            captured_at=datetime(2026, 1, 1, tzinfo=UTC)
            + timedelta(seconds=frame.video_time_seconds or 0),
            width=2,
            height=2,
            sequence_number=frame.frame_index or 0,
            payload=b"\0" * 12,
        )


class _Detector:
    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        detections = (
            NormalizedDetection(
                class_id=0,
                class_name="Person",
                confidence=0.9,
                bounding_box=NormalizedBoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
            ),
            NormalizedDetection(
                class_id=2,
                class_name="NO-Hardhat",
                confidence=0.8,
                bounding_box=NormalizedBoundingBox(x1=0.2, y1=0.1, x2=0.5, y2=0.35),
            ),
        )
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="ppe-yolo11s",
            model_version="1",
            model_sha256="c" * 64,
            detections=detections,
        )


class _Pipeline:
    def process(
        self,
        batch: DetectionBatch,
        *,
        region_configuration: CameraRegionConfiguration,
        event_id: str,
    ) -> TechnicalObservationEvent:
        del region_configuration
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
                        "x2": 0.9,
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
                    "geometryVersion": 1,
                    "confidence": 0.8,
                    "boundingBox": {
                        "x1": 0.2,
                        "y1": 0.1,
                        "x2": 0.5,
                        "y2": 0.35,
                        "coordinateSpace": "NORMALIZED_0_1",
                    },
                },
            ],
        )


class _Gate:
    def update(
        self,
        *,
        stream_id: str,
        session_id: UUID,
        observed_at: datetime,
        active_track_ids: tuple[int, ...],
        observations: tuple[object, ...],
    ) -> tuple[ConfirmedPpeCandidate, ...]:
        assert active_track_ids == (7,)
        assert len(observations) == 1
        return (
            ConfirmedPpeCandidate(
                stream_id=stream_id,
                session_id=session_id,
                track_id=7,
                ppe_item="HARD_HAT",
                first_seen_at=observed_at,
                confirmed_at=observed_at,
            ),
        )


def _request(tmp_path: Path, artifact: ModelArtifactSpec) -> EvaluationExecutionRequest:
    return EvaluationExecutionRequest(
        dataset_manifest_path=(tmp_path / "manifest.json").resolve(),
        split="test",
        episodes_index_path=(tmp_path / "episodes.jsonl").resolve(),
        region_configuration_path=(tmp_path / "regions.json").resolve(),
        artifact=artifact,
        match_iou=0.5,
        onset_tolerance=timedelta(seconds=1),
        predictions_path=(tmp_path / "predictions.jsonl").resolve(),
        accuracy_report_path=(tmp_path / "accuracy.report.json").resolve(),
        candidate_report_path=(tmp_path / "candidates.json").resolve(),
        summary_path=(tmp_path / "summary.txt").resolve(),
    )


def test_load_ground_truth_episodes_is_strict_bounded_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "clipId": "z",
                        "personInstanceId": "p",
                        "ppeItem": "HARD_HAT",
                        "startTimeSeconds": 2.0,
                        "endTimeSeconds": 3.0,
                    }
                ),
                json.dumps(
                    {
                        "clipId": "a",
                        "personInstanceId": "p",
                        "ppeItem": "SAFETY_VEST",
                        "startTimeSeconds": 1.0,
                        "endTimeSeconds": 2.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    episodes = load_ground_truth_episodes(path)
    assert tuple(item.clip_id for item in episodes) == ("a", "z")

    path.write_text('{"clipId":"a","clipId":"b"}\n', encoding="utf-8")
    with pytest.raises(EvaluationExecutionError, match="duplicate JSON key"):
        load_ground_truth_episodes(path)


def test_execute_evaluation_writes_complete_outputs_from_real_metrics(tmp_path: Path) -> None:
    frames = (_frame("f-2", 2, 2.0), _frame("f-1", 1, 1.0))
    dataset = LoadedEvaluationDataset(
        manifest=_manifest(),
        splits=MappingProxyType({"test": frames}),
        dataset_root=tmp_path,
    )
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "clipId": "clip.mp4",
                "personInstanceId": "worker-1",
                "ppeItem": "HARD_HAT",
                "startTimeSeconds": 1.0,
                "endTimeSeconds": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = _artifact(tmp_path)
    metadata = EvaluationExecutionMetadata(
        git=GitReportMetadata(commit_sha="d" * 40, dirty_worktree=False),
        runtime=RuntimeReportMetadata(
            python_version="3.13", platform="test", device="cpu", package_versions={}, hardware={}
        ),
        command_arguments=(),
    )
    rendered: list[tuple[str, object]] = []
    services = EvaluationExecutionServices(
        dataset_loader=lambda path: dataset,
        region_loader=lambda path: _region_configuration(),
        frame_reader=_Reader(),
        detector=_Detector(),
        pipeline=_Pipeline(),
        temporal_gate=_Gate(),
        provider_validation=lambda: _provider_report(tmp_path),
        evidence_writer=lambda frame, source, plan: rendered.append((frame.frame_id, plan)),
    )

    result = asyncio.run(
        execute_evaluation(_request(tmp_path, artifact), metadata=metadata, services=services)
    )

    assert result.report.status == "COMPLETE"
    assert result.report.detection_metrics.micro.true_positives == 4
    assert result.report.episode_metrics.true_positives == 1
    assert [frame_id for frame_id, _plan in rendered] == ["f-1", "f-2"]
    no_hat_boxes = [
        box
        for _frame_id, plan in rendered
        for box in plan.boxes
        if box.source == "PREDICTION" and box.class_name == "NO-Hardhat"
    ]
    assert all(box.track_id == 7 for box in no_hat_boxes)
    assert all(box.related_person_object_id is not None for box in no_hat_boxes)
    assert all(box.confirmed_candidate is True for box in no_hat_boxes)
    prediction_lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["frameId"] for line in prediction_lines] == ["f-1", "f-2"]
    candidate_payload = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    assert candidate_payload["providerValidation"]["metrics"]["ap50"] == 0.8
    assert candidate_payload["episodeMetrics"]["true_positives"] == 1
    assert (tmp_path / "summary.txt").read_text(encoding="utf-8").strip()


def test_executor_rejects_non_yolo11s_before_publishing(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, family="yolov8n")
    request = _request(tmp_path, artifact)
    services = EvaluationExecutionServices(
        dataset_loader=lambda path: pytest.fail("must fail before loading"),
        region_loader=lambda path: pytest.fail("must fail before loading"),
        frame_reader=_Reader(),
        detector=_Detector(),
        pipeline=_Pipeline(),
        temporal_gate=_Gate(),
        provider_validation=lambda: pytest.fail("must fail before provider validation"),
    )
    metadata = EvaluationExecutionMetadata(
        git=GitReportMetadata(commit_sha="d" * 40, dirty_worktree=False),
        runtime=RuntimeReportMetadata(
            python_version="3.13", platform="test", device="cpu", package_versions={}, hardware={}
        ),
        command_arguments=(),
    )
    with pytest.raises(EvaluationExecutionError, match="model_family must equal yolo11s"):
        asyncio.run(execute_evaluation(request, metadata=metadata, services=services))
    assert not request.accuracy_report_path.exists()
    assert not request.predictions_path.exists()


def test_executor_rejects_batch_from_a_different_artifact_version(tmp_path: Path) -> None:
    frames = (_frame("f-1", 1, 1.0),)
    dataset = LoadedEvaluationDataset(
        manifest=_manifest(),
        splits=MappingProxyType({"test": frames}),
        dataset_root=tmp_path,
    )
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text("", encoding="utf-8")
    artifact = _artifact(tmp_path).model_copy(update={"version": "2"})
    metadata = EvaluationExecutionMetadata(
        git=GitReportMetadata(commit_sha="d" * 40, dirty_worktree=False),
        runtime=RuntimeReportMetadata(
            python_version="3.13",
            platform="test",
            device="cpu",
            package_versions={},
            hardware={},
        ),
        command_arguments=(),
    )
    services = EvaluationExecutionServices(
        dataset_loader=lambda path: dataset,
        region_loader=lambda path: _region_configuration(),
        frame_reader=_Reader(),
        detector=_Detector(),
        pipeline=_Pipeline(),
        temporal_gate=_Gate(),
        provider_validation=lambda: _provider_report(tmp_path),
    )

    with pytest.raises(EvaluationExecutionError, match="evaluated artifact"):
        asyncio.run(
            execute_evaluation(_request(tmp_path, artifact), metadata=metadata, services=services)
        )

    assert not (tmp_path / "accuracy.report.json").exists()
    assert not (tmp_path / "predictions.jsonl").exists()


def test_video_frame_without_event_still_advances_temporal_gate(tmp_path: Path) -> None:
    frame = _frame("f-1", 1, 1.0)
    dataset = LoadedEvaluationDataset(
        manifest=_manifest(),
        splits=MappingProxyType({"test": (frame,)}),
        dataset_root=tmp_path,
    )
    (tmp_path / "episodes.jsonl").write_text("", encoding="utf-8")
    calls: list[tuple[tuple[int, ...], tuple[object, ...]]] = []

    class NoEventPipeline:
        def process(self, *args: object, **kwargs: object) -> None:
            return None

    class RecordingGate:
        def update(self, **values: object) -> tuple[()]:
            calls.append((values["active_track_ids"], values["observations"]))  # type: ignore[arg-type]
            return ()

    artifact = _artifact(tmp_path)
    services = EvaluationExecutionServices(
        dataset_loader=lambda path: dataset,
        region_loader=lambda path: _region_configuration(),
        frame_reader=_Reader(),
        detector=_Detector(),
        pipeline=NoEventPipeline(),
        temporal_gate=RecordingGate(),
        provider_validation=lambda: _provider_report(tmp_path),
    )
    metadata = EvaluationExecutionMetadata(
        git=GitReportMetadata(commit_sha="d" * 40, dirty_worktree=False),
        runtime=RuntimeReportMetadata(
            python_version="3.13",
            platform="test",
            device="cpu",
            package_versions={},
            hardware={},
        ),
        command_arguments=(),
    )

    asyncio.run(
        execute_evaluation(_request(tmp_path, artifact), metadata=metadata, services=services)
    )

    assert calls == [((), ())]


def test_runtime_failure_never_publishes_complete_outputs(tmp_path: Path) -> None:
    frame = _frame("f-1", 1, 1.0)
    dataset = LoadedEvaluationDataset(
        manifest=_manifest(),
        splits=MappingProxyType({"test": (frame,)}),
        dataset_root=tmp_path,
    )
    (tmp_path / "episodes.jsonl").write_text("", encoding="utf-8")

    class FailingDetector:
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            del frame
            raise RuntimeError("inference failed")

    artifact = _artifact(tmp_path)
    request = _request(tmp_path, artifact)
    services = EvaluationExecutionServices(
        dataset_loader=lambda path: dataset,
        region_loader=lambda path: _region_configuration(),
        frame_reader=_Reader(),
        detector=FailingDetector(),
        pipeline=_Pipeline(),
        temporal_gate=_Gate(),
        provider_validation=lambda: _provider_report(tmp_path),
    )
    metadata = EvaluationExecutionMetadata(
        git=GitReportMetadata(commit_sha="d" * 40, dirty_worktree=False),
        runtime=RuntimeReportMetadata(
            python_version="3.13",
            platform="test",
            device="cpu",
            package_versions={},
            hardware={},
        ),
        command_arguments=(),
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        asyncio.run(execute_evaluation(request, metadata=metadata, services=services))

    assert not request.predictions_path.exists()
    assert not request.candidate_report_path.exists()
    assert not request.summary_path.exists()
    assert not request.accuracy_report_path.exists()
