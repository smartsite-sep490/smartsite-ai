"""Pure, provider-neutral orchestration for the locked YOLO11s evaluation gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict

from smartsite_ai.domain.observations import (
    PersonObservation,
    PpeObservation,
    TechnicalObservationEvent,
)
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.evaluation.alert_metrics import (
    EpisodeMetricsReport,
    PredictedPpeEpisode,
    match_candidate_episodes,
)
from smartsite_ai.evaluation.dataset import LoadedEvaluationDataset
from smartsite_ai.evaluation.detection_metrics import (
    DetectionEvaluationFrame,
    compute_detection_metrics,
)
from smartsite_ai.evaluation.matching import EvaluationPrediction, match_detections
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    EvaluationBoundingBox,
    EvaluationFrame,
    GroundTruthObject,
    GroundTruthPpeEpisode,
)
from smartsite_ai.evaluation.overlay import (
    OverlayPlan,
    PredictionOverlayContext,
    build_overlay_plan,
)
from smartsite_ai.evaluation.provider_validation import ProviderValidationReport
from smartsite_ai.evaluation.report import (
    CommandArgument,
    DatasetReportMetadata,
    EvaluationReport,
    GitReportMetadata,
    ImageSize,
    ModelReportMetadata,
    RuntimeReportMetadata,
    ThresholdReportMetadata,
    evaluate_accuracy_gate,
    write_evaluation_report,
)
from smartsite_ai.inference.artifacts import ModelArtifactSpec
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines.ppe_temporal import ConfirmedPpeCandidate

MAX_EPISODES_FILE_BYTES = 16 * 1024 * 1024
MAX_EPISODE_LINE_BYTES = 64 * 1024
MAX_EPISODES = 100_000


class EvaluationExecutionError(RuntimeError):
    """A safe failure before a complete evaluation report can be published."""


class FrameReader(Protocol):
    def read(self, dataset_root: Path, frame: EvaluationFrame) -> FrameEnvelope: ...


class AsyncDetector(Protocol):
    async def detect(self, frame: FrameEnvelope) -> DetectionBatch: ...


class EvaluationPipeline(Protocol):
    def process(
        self,
        batch: DetectionBatch,
        *,
        region_configuration: CameraRegionConfiguration,
        event_id: str,
    ) -> TechnicalObservationEvent | None: ...


class CandidateGate(Protocol):
    def update(
        self,
        *,
        stream_id: str,
        session_id: UUID,
        observed_at: datetime,
        active_track_ids: Sequence[int],
        observations: Sequence[PpeObservation],
    ) -> tuple[ConfirmedPpeCandidate, ...]: ...


class EvidenceWriter(Protocol):
    def __call__(
        self, frame: EvaluationFrame, source: FrameEnvelope, plan: OverlayPlan
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationExecutionRequest:
    dataset_manifest_path: Path
    split: Literal["train", "validation", "test"]
    episodes_index_path: Path
    region_configuration_path: Path
    artifact: ModelArtifactSpec
    match_iou: float
    onset_tolerance: timedelta
    predictions_path: Path
    accuracy_report_path: Path
    candidate_report_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationExecutionMetadata:
    git: GitReportMetadata
    runtime: RuntimeReportMetadata
    command_arguments: tuple[CommandArgument, ...]


@dataclass(frozen=True, slots=True)
class EvaluationExecutionServices:
    dataset_loader: Callable[[Path], LoadedEvaluationDataset]
    region_loader: Callable[[Path], CameraRegionConfiguration]
    frame_reader: FrameReader
    detector: AsyncDetector
    pipeline: EvaluationPipeline
    temporal_gate: CandidateGate
    provider_validation: Callable[[], ProviderValidationReport]
    evidence_writer: EvidenceWriter | None = None


@dataclass(frozen=True, slots=True)
class EvaluationExecutionResult:
    report: EvaluationReport
    provider_validation: ProviderValidationReport
    episode_metrics: EpisodeMetricsReport
    frame_count: int


def _to_camel(name: str) -> str:
    first, *remaining = name.split("_")
    return first + "".join(part.capitalize() for part in remaining)


class _CandidateReport(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        alias_generator=_to_camel,
        populate_by_name=True,
    )

    status: Literal["COMPLETE"] = "COMPLETE"
    provider_validation: ProviderValidationReport
    episode_metrics: EpisodeMetricsReport


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_ground_truth_episodes(path: Path) -> tuple[GroundTruthPpeEpisode, ...]:
    """Load strict bounded JSONL and return canonical stable episode order."""

    try:
        if not path.is_file():
            raise EvaluationExecutionError("ground-truth episode index is not a regular file")
        if path.stat().st_size > MAX_EPISODES_FILE_BYTES:
            raise EvaluationExecutionError("ground-truth episode index exceeds size limit")
        episodes: list[GroundTruthPpeEpisode] = []
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if len(raw_line) > MAX_EPISODE_LINE_BYTES:
                    raise EvaluationExecutionError(
                        f"ground-truth episode line {line_number} exceeds size limit"
                    )
                if not raw_line.strip():
                    raise EvaluationExecutionError(
                        f"ground-truth episode line {line_number} must not be blank"
                    )
                try:
                    value = json.loads(
                        raw_line.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
                    )
                    episodes.append(GroundTruthPpeEpisode.model_validate(value))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise EvaluationExecutionError(
                        f"invalid ground-truth episode at line {line_number}: {error}"
                    ) from error
                if len(episodes) > MAX_EPISODES:
                    raise EvaluationExecutionError("ground-truth episode count exceeds limit")
    except OSError as error:
        raise EvaluationExecutionError("could not read ground-truth episode index") from error

    return tuple(
        sorted(
            episodes,
            key=lambda item: (
                item.clip_id,
                type(item.person_instance_id).__name__,
                str(item.person_instance_id),
                item.ppe_item,
                item.start_time_seconds,
                item.end_time_seconds,
            ),
        )
    )


def _stable_frames(frames: Sequence[EvaluationFrame]) -> tuple[EvaluationFrame, ...]:
    return tuple(
        sorted(
            frames,
            key=lambda frame: (
                frame.media_path,
                frame.frame_index is not None,
                frame.frame_index if frame.frame_index is not None else -1,
                frame.frame_id,
            ),
        )
    )


def _prediction(frame_id: str, index: int, batch: DetectionBatch) -> EvaluationPrediction:
    item = batch.detections[index]
    return EvaluationPrediction(
        prediction_id=f"{frame_id}:{index:04d}",
        class_name=item.class_name,
        confidence=item.confidence,
        bounding_box=EvaluationBoundingBox(
            x1=item.bounding_box.x1,
            y1=item.bounding_box.y1,
            x2=item.bounding_box.x2,
            y2=item.bounding_box.y2,
        ),
    )


def _iou(left: EvaluationBoundingBox, right: EvaluationBoundingBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = (
        (left.x2 - left.x1) * (left.y2 - left.y1)
        + (right.x2 - right.x1) * (right.y2 - right.y1)
        - intersection
    )
    return intersection / union if union else 0.0


def _person_identity(
    person: PersonObservation,
    ground_truth: Sequence[GroundTruthObject],
    *,
    clip_id: str,
    session_id: UUID,
    threshold: float,
) -> int | str:
    if person.bounding_box is not None:
        person_box = EvaluationBoundingBox(
            x1=person.bounding_box.x1,
            y1=person.bounding_box.y1,
            x2=person.bounding_box.x2,
            y2=person.bounding_box.y2,
        )
        candidates = []
        for item in ground_truth:
            if item.class_name != "Person" or item.person_instance_id is None:
                continue
            overlap = _iou(person_box, item.bounding_box)
            if overlap >= threshold:
                candidates.append((-overlap, item.annotation_id, item.person_instance_id))
        if candidates:
            candidates.sort(key=lambda value: (value[0], value[1]))
            return candidates[0][2]
    return _synthetic_identity(clip_id, session_id, person.track_id)


def _synthetic_identity(clip_id: str, session_id: UUID, track_id: int) -> str:
    digest = hashlib.sha256(f"{clip_id}|{session_id}|{track_id}".encode()).hexdigest()[:24]
    return f"unmatched-{digest}"


def _prediction_overlay_context(
    predictions: Sequence[EvaluationPrediction],
    event: TechnicalObservationEvent | None,
    confirmed: Sequence[ConfirmedPpeCandidate],
) -> tuple[PredictionOverlayContext, ...]:
    """Map pipeline track/candidate facts back onto the exact detector predictions."""

    if event is None:
        return ()
    contexts: list[PredictionOverlayContext] = []
    used: set[str] = set()
    person_prediction_by_track: dict[int, str] = {}
    people = sorted(
        (item for item in event.observations if isinstance(item, PersonObservation)),
        key=lambda item: item.track_id,
    )
    for person in people:
        box = EvaluationBoundingBox(
            x1=person.bounding_box.x1,
            y1=person.bounding_box.y1,
            x2=person.bounding_box.x2,
            y2=person.bounding_box.y2,
        )
        candidates = sorted(
            (
                (-_iou(box, item.bounding_box), item.prediction_id, item)
                for item in predictions
                if item.class_name == "Person" and item.prediction_id not in used
            ),
            key=lambda value: (value[0], value[1]),
        )
        if not candidates or candidates[0][0] >= 0.0:
            continue
        prediction = candidates[0][2]
        used.add(prediction.prediction_id)
        person_prediction_by_track[person.track_id] = prediction.prediction_id
        contexts.append(
            PredictionOverlayContext(
                prediction_id=prediction.prediction_id,
                track_id=person.track_id,
            )
        )

    confirmed_keys = {(item.track_id, item.ppe_item) for item in confirmed}
    class_by_observation = {
        ("HARD_HAT", "PRESENT"): "Hardhat",
        ("HARD_HAT", "MISSING"): "NO-Hardhat",
        ("SAFETY_VEST", "PRESENT"): "Safety Vest",
        ("SAFETY_VEST", "MISSING"): "NO-Safety Vest",
    }
    observations = sorted(
        (
            item
            for item in event.observations
            if isinstance(item, PpeObservation) and item.bounding_box is not None
        ),
        key=lambda item: (item.track_id, item.ppe_item, item.status),
    )
    for observation in observations:
        related_person = person_prediction_by_track.get(observation.track_id)
        if related_person is None:
            continue
        box = EvaluationBoundingBox(
            x1=observation.bounding_box.x1,
            y1=observation.bounding_box.y1,
            x2=observation.bounding_box.x2,
            y2=observation.bounding_box.y2,
        )
        expected_class = class_by_observation[(observation.ppe_item, observation.status)]
        candidates = sorted(
            (
                (-_iou(box, item.bounding_box), item.prediction_id, item)
                for item in predictions
                if item.class_name == expected_class and item.prediction_id not in used
            ),
            key=lambda value: (value[0], value[1]),
        )
        if not candidates or candidates[0][0] >= 0.0:
            continue
        prediction = candidates[0][2]
        used.add(prediction.prediction_id)
        contexts.append(
            PredictionOverlayContext(
                prediction_id=prediction.prediction_id,
                related_person_prediction_id=related_person,
                track_id=observation.track_id,
                confirmed_candidate=(
                    observation.status == "MISSING"
                    and (observation.track_id, observation.ppe_item) in confirmed_keys
                ),
            )
        )
    return tuple(sorted(contexts, key=lambda item: item.prediction_id))


def _atomic_write(path: Path, content: str) -> None:
    if path.exists():
        raise EvaluationExecutionError(f"output already exists: {path.name}")
    if not path.parent.is_dir():
        raise EvaluationExecutionError(f"output directory does not exist: {path.parent}")
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as error:
            raise EvaluationExecutionError(f"output already exists: {path.name}") from error
    finally:
        temp.unlink(missing_ok=True)


def _evaluated_seconds(frames: Sequence[EvaluationFrame]) -> float:
    times: dict[str, list[float]] = {}
    for frame in frames:
        if frame.video_time_seconds is not None:
            times.setdefault(frame.media_path, []).append(frame.video_time_seconds)
    return sum(max(values) - min(values) for values in times.values() if values)


async def execute_evaluation(
    request: EvaluationExecutionRequest,
    *,
    metadata: EvaluationExecutionMetadata,
    services: EvaluationExecutionServices,
) -> EvaluationExecutionResult:
    """Execute the evaluation and publish COMPLETE only after every metric is real."""

    if request.artifact.model_family != "yolo11s":
        raise EvaluationExecutionError("artifact model_family must equal yolo11s")
    if not math.isfinite(request.match_iou) or not 0.0 < request.match_iou <= 1.0:
        raise EvaluationExecutionError("match_iou must be finite and in (0, 1]")

    started_at = datetime.now(UTC)
    dataset = services.dataset_loader(request.dataset_manifest_path)
    frames = _stable_frames(dataset.get_split(request.split))
    ground_truth_episodes = load_ground_truth_episodes(request.episodes_index_path)
    region_configuration = services.region_loader(request.region_configuration_path)
    provider_report = services.provider_validation()

    metric_frames: list[DetectionEvaluationFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    candidate_state: dict[tuple[str, UUID, int, str], dict[str, Any]] = {}

    for frame in frames:
        source = services.frame_reader.read(dataset.dataset_root, frame)
        batch = await services.detector.detect(source)
        if (
            batch.model_artifact_id != request.artifact.artifact_id
            or batch.model_version != request.artifact.version
            or batch.model_sha256 != request.artifact.sha256
        ):
            raise EvaluationExecutionError("detector batch does not match the evaluated artifact")
        if (
            batch.stream_id != source.stream_id
            or batch.session_id != source.session_id
            or batch.camera_external_id != source.camera_external_id
            or batch.captured_at != source.captured_at
            or batch.frame_width != source.width
            or batch.frame_height != source.height
            or batch.sequence_number != source.sequence_number
        ):
            raise EvaluationExecutionError("detector batch does not match its source frame")
        predictions = tuple(
            _prediction(frame.frame_id, index, batch) for index in range(len(batch.detections))
        )
        metric_frame = DetectionEvaluationFrame(
            frame_id=frame.frame_id,
            ground_truth=frame.annotations,
            predictions=predictions,
        )
        metric_frames.append(metric_frame)
        match_result = match_detections(
            frame.annotations, predictions, iou_threshold=request.match_iou
        )
        prediction_rows.append(
            {
                "frameId": frame.frame_id,
                "mediaPath": frame.media_path,
                "detections": [item.model_dump(mode="json", by_alias=True) for item in predictions],
            }
        )

        if frame.video_time_seconds is None:
            plan = build_overlay_plan(frame.annotations, predictions, match_result)
            if services.evidence_writer is not None:
                services.evidence_writer(frame, source, plan)
            continue
        event = services.pipeline.process(
            batch,
            region_configuration=region_configuration,
            event_id=str(uuid5(NAMESPACE_URL, f"smartsite-evaluation:{frame.frame_id}")),
        )
        persons = (
            ()
            if event is None
            else tuple(item for item in event.observations if isinstance(item, PersonObservation))
        )
        ppe_observations = (
            ()
            if event is None
            else tuple(item for item in event.observations if isinstance(item, PpeObservation))
        )
        identities = {
            person.track_id: _person_identity(
                person,
                frame.annotations,
                clip_id=frame.media_path,
                session_id=batch.session_id,
                threshold=request.match_iou,
            )
            for person in persons
        }
        confirmed = services.temporal_gate.update(
            stream_id=batch.stream_id,
            session_id=batch.session_id,
            observed_at=batch.captured_at,
            active_track_ids=tuple(sorted(identities)),
            observations=ppe_observations,
        )
        for candidate in confirmed:
            key = (frame.media_path, candidate.session_id, candidate.track_id, candidate.ppe_item)
            offset = (candidate.confirmed_at - candidate.first_seen_at).total_seconds()
            person_identity = identities.get(candidate.track_id)
            if person_identity is None:
                person_identity = _synthetic_identity(
                    frame.media_path, batch.session_id, candidate.track_id
                )
            candidate_state.setdefault(
                key,
                {
                    "person": person_identity,
                    "start": max(0.0, frame.video_time_seconds - offset),
                    "confirmed": frame.video_time_seconds,
                    "end": frame.video_time_seconds,
                },
            )
        for observation in ppe_observations:
            if observation.status != "MISSING":
                continue
            key = (frame.media_path, batch.session_id, observation.track_id, observation.ppe_item)
            if key in candidate_state:
                candidate_state[key]["end"] = frame.video_time_seconds

        plan = build_overlay_plan(
            frame.annotations,
            predictions,
            match_result,
            prediction_context=_prediction_overlay_context(predictions, event, confirmed),
        )
        if services.evidence_writer is not None:
            services.evidence_writer(frame, source, plan)

    predicted_episodes: list[PredictedPpeEpisode] = []
    for key in sorted(
        candidate_state, key=lambda value: (value[0], str(value[1]), value[2], value[3])
    ):
        clip_id, session_id, track_id, ppe_item = key
        state = candidate_state[key]
        candidate_id = str(
            uuid5(
                NAMESPACE_URL, f"{clip_id}|{session_id}|{track_id}|{ppe_item}|{state['confirmed']}"
            )
        )
        predicted_episodes.append(
            PredictedPpeEpisode(
                candidate_id=candidate_id,
                clip_id=clip_id,
                person_instance_id=state["person"],
                stream_session_id=session_id,
                ppe_item=ppe_item,
                start_time_seconds=state["start"],
                end_time_seconds=state["end"],
                confirmed_time_seconds=state["confirmed"],
                track_id=track_id,
            )
        )

    detection_metrics = compute_detection_metrics(
        metric_frames,
        class_names=CANONICAL_PPE_CLASSES,
        iou_threshold=request.match_iou,
    )
    episode_metrics = match_candidate_episodes(
        ground_truth_episodes,
        predicted_episodes,
        onset_tolerance=request.onset_tolerance,
        evaluated_seconds=_evaluated_seconds(frames),
    )
    completed_at = datetime.now(UTC)
    report = EvaluationReport(
        schema_version="1.0.0",
        run_id=uuid4(),
        status="COMPLETE",
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        git=metadata.git,
        model=ModelReportMetadata(
            artifact_id=request.artifact.artifact_id,
            version=request.artifact.version,
            family=request.artifact.model_family,
            sha256=request.artifact.sha256,
            source_url=request.artifact.source_url,
            license=request.artifact.license,
            class_map={str(class_id): name for class_id, name in request.artifact.class_map},
        ),
        dataset=DatasetReportMetadata(
            dataset_id=dataset.manifest.dataset_id,
            dataset_version=dataset.manifest.dataset_version,
            aggregate_sha256=dataset.manifest.aggregate_sha256,
            split=request.split,
            license=dataset.manifest.license,
        ),
        runtime=metadata.runtime,
        thresholds=ThresholdReportMetadata(
            confidence=request.artifact.confidence_threshold,
            nms_iou=request.artifact.iou_threshold,
            match_iou=request.match_iou,
        ),
        image_size=ImageSize(
            width=request.artifact.image_size[0], height=request.artifact.image_size[1]
        ),
        metric_definitions=(
            "Detection TP uses deterministic same-class IoU matching at matchIoU.",
            "Candidate TP uses person, PPE item, clip, and temporal overlap or onset tolerance.",
        ),
        command_arguments=metadata.command_arguments,
        detection_metrics=detection_metrics,
        episode_metrics=episode_metrics,
        accuracy_gate=evaluate_accuracy_gate(detection_metrics),
        failure=None,
    )
    candidate_report = _CandidateReport(
        provider_validation=provider_report,
        episode_metrics=episode_metrics,
    )
    predictions_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in prediction_rows
    )
    candidate_text = (
        json.dumps(
            candidate_report.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )
    summary = (
        f"SmartSite YOLO11s evaluation COMPLETE: {len(frames)} frames; "
        f"detection TP/FP/FN={detection_metrics.micro.true_positives}/"
        f"{detection_metrics.micro.false_positives}/{detection_metrics.micro.false_negatives}; "
        f"candidate TP/FP/FN={episode_metrics.true_positives}/"
        f"{episode_metrics.false_positives}/{episode_metrics.false_negatives}.\n"
    )

    published: list[Path] = []
    try:
        _atomic_write(request.predictions_path, predictions_text)
        published.append(request.predictions_path)
        _atomic_write(request.candidate_report_path, candidate_text)
        published.append(request.candidate_report_path)
        _atomic_write(request.summary_path, summary)
        published.append(request.summary_path)
        write_evaluation_report(request.accuracy_report_path, report)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise

    return EvaluationExecutionResult(
        report=report,
        provider_validation=provider_report,
        episode_metrics=episode_metrics,
        frame_count=len(frames),
    )


__all__ = [
    "EvaluationExecutionError",
    "EvaluationExecutionMetadata",
    "EvaluationExecutionRequest",
    "EvaluationExecutionResult",
    "EvaluationExecutionServices",
    "execute_evaluation",
    "load_ground_truth_episodes",
]
