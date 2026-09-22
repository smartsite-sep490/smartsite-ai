"""Run a reviewed detector over a local video and write annotated evidence.

This module deliberately owns only a synchronous, local-file validation command.  Production
camera ingestion remains under :mod:`smartsite_ai.ingestion`; all media/provider dependencies
used by this command are imported inside their explicit factories.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.artifacts import (
    ArtifactValidationError,
    ModelArtifactSpec,
    VerifiedModelArtifact,
    verify_model_artifact,
)
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.inference.yolo import Yolo11Detector
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.tracking import IoUPersonTracker

# OpenCV property identifiers are stable public constants.  Keeping these numeric values here
# lets the orchestration layer use deterministic fakes without importing cv2.
CAP_PROP_POS_MSEC = 0
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5

SUPPORTED_INPUT_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})
OUTPUT_CODEC = "mp4v"
_CLASS_ID_TEXT = re.compile(r"0|[1-9][0-9]*")


class VideoValidationError(RuntimeError):
    """The local validation run could not produce a complete result."""


@dataclass(slots=True)
class _UiTimelineCollector:
    """Collect local UI records from the real MF05/MF06 technical pipeline."""

    configuration: CameraRegionConfiguration
    ppe_region_id: str
    pipeline: Mf05Mf06Pipeline = field(init=False)
    entries: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pipeline = Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline(),
            zones=RestrictedZonePipeline(),
            ppe_region_id=self.ppe_region_id,
        )

    def observe(self, batch: DetectionBatch, video_time_seconds: float) -> None:
        event = self.pipeline.process(
            batch,
            region_configuration=self.configuration,
            event_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"smartsite-ui-test:{batch.stream_id}:{batch.session_id}:{batch.sequence_number}",
                )
            ),
        )
        if event is None:
            return

        event_payload = event.to_wire_dict()
        observations = event_payload["observations"]
        if not any(
            observation["type"] == "ZONE_ENTRY"
            or (observation["type"] == "PPE" and observation["status"] == "MISSING")
            for observation in observations
        ):
            return

        self.entries.append(
            {
                "videoTimeSeconds": round(video_time_seconds, 3),
                "event": event_payload,
            }
        )

    def write(self, path: Path) -> None:
        if path.suffix.lower() != ".json":
            raise VideoValidationError("UI timeline output extension must be .json")
        if path.exists():
            raise VideoValidationError("UI timeline output already exists")
        if not path.parent.is_dir():
            raise VideoValidationError("UI timeline output directory does not exist")
        payload = {
            "schemaVersion": "1.0.0",
            "cameraExternalId": self.configuration.camera_external_id,
            "entries": self.entries,
        }
        temp_path = _write_metadata_temp(path, payload)
        try:
            temp_path.replace(path)
        except OSError as error:
            _remove_if_exists(temp_path)
            raise VideoValidationError("could not finalize UI timeline output") from error


class CaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, prop: int) -> float: ...

    def read(self) -> tuple[bool, object | None]: ...

    def release(self) -> None: ...


class WriterProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: object) -> None: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[Path], CaptureProtocol]
WriterFactory = Callable[[Path, str, float, tuple[int, int]], WriterProtocol]
Renderer = Callable[[object, DetectionBatch], object]
BatchObserver = Callable[[DetectionBatch, float], None]
Clock = Callable[[], float]
UtcNowFactory = Callable[[], datetime]
SessionIdFactory = Callable[[], UUID]


def run_video_validation(
    *,
    input_path: Path,
    output_path: Path,
    metadata_output: Path,
    artifact: VerifiedModelArtifact,
    class_map: Mapping[int, str],
    detector: DetectorProtocol,
    capture_factory: CaptureFactory,
    writer_factory: WriterFactory,
    renderer: Renderer,
    monotonic_clock: Clock = monotonic,
    utc_now_factory: UtcNowFactory = lambda: datetime.now(UTC),
    session_id_factory: SessionIdFactory = uuid4,
    runtime_versions: Mapping[str, str] | None = None,
    configuration_paths: Sequence[Path] = (),
    camera_external_id: str | None = None,
    batch_observer: BatchObserver | None = None,
) -> dict[str, object]:
    """Validate one local video through an injected detector and media boundary.

    ``capture_factory``, ``writer_factory``, and ``renderer`` are explicit seams so the complete
    orchestration can be tested with no OpenCV/provider runtime.  The function owns the returned
    capture and writer for the duration of the run and always releases each object at most once.
    Metadata is written to a sibling temporary path and replaced only after the complete video
    has been processed and the writer has been released successfully.
    """

    input_path, output_path, metadata_output = _validate_paths(
        input_path=input_path,
        output_path=output_path,
        metadata_output=metadata_output,
        configuration_paths=configuration_paths,
    )
    if not class_map:
        raise VideoValidationError("class map must contain at least one class")
    _validate_class_map(class_map)
    if dict(artifact.class_map) != dict(class_map):
        raise VideoValidationError("class map does not match verified artifact")

    capture: CaptureProtocol | None = None
    writer: WriterProtocol | None = None
    event_loop: asyncio.AbstractEventLoop | None = None
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    metadata_temp: Path | None = None
    output_started = False
    capture_released = False
    writer_released = False
    completed = False

    session_id = session_id_factory()
    if not isinstance(session_id, UUID):
        raise VideoValidationError("session ID factory must return a UUID")
    capture_started_at = _ensure_utc(utc_now_factory(), "UTC clock")
    resolved_camera_external_id = camera_external_id or _camera_external_id(input_path)
    if not resolved_camera_external_id or len(resolved_camera_external_id) > 128:
        raise VideoValidationError("camera external ID must contain 1 to 128 characters")
    if "\x00" in resolved_camera_external_id:
        raise VideoValidationError("camera external ID contains a forbidden character")
    elapsed_started = monotonic_clock()
    frame_count = 0
    detection_count = 0
    detections_by_class: Counter[str] = Counter()
    first_capture_at: datetime | None = None
    last_capture_at: datetime | None = None

    try:
        try:
            capture = capture_factory(input_path)
        except Exception as error:
            raise VideoValidationError("could not open input video") from error
        if not _is_opened(capture):
            raise VideoValidationError("could not open input video")

        fps = _valid_positive_float(capture.get(CAP_PROP_FPS), "video FPS")
        width = _valid_positive_dimension(capture.get(CAP_PROP_FRAME_WIDTH), "video width")
        height = _valid_positive_dimension(capture.get(CAP_PROP_FRAME_HEIGHT), "video height")
        dimensions = (width, height)

        # The factory may create/truncate the destination before returning, so mark the path as
        # owned before calling it.  A factory failure must still clean up a partial output.
        output_started = True
        try:
            writer = writer_factory(output_path, OUTPUT_CODEC, fps, dimensions)
        except Exception as error:
            raise VideoValidationError("could not create output video writer") from error
        if not _is_opened(writer):
            raise VideoValidationError("could not open output video writer")

        # One loop and one inference executor are owned by this synchronous command.
        # Ctrl+C must drain that executor before the caller releases the model.
        event_loop = asyncio.new_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="smartsite-video-validation"
        )
        if setter := getattr(event_loop, "set_default_executor", None):
            setter(executor)
        while True:
            try:
                has_frame, source_frame = capture.read()
            except Exception as error:
                raise VideoValidationError("could not read input video") from error
            if not has_frame:
                break
            if source_frame is None:
                raise VideoValidationError("input video returned an empty frame")

            payload = _packed_bgr_payload(source_frame, width=width, height=height)
            timestamp_msec = _valid_non_negative_float(
                capture.get(CAP_PROP_POS_MSEC), "video presentation timestamp"
            )
            captured_at = capture_started_at + timedelta(milliseconds=timestamp_msec)
            frame = FrameEnvelope(
                stream_id="video-validation",
                session_id=session_id,
                camera_external_id=resolved_camera_external_id,
                captured_at=captured_at,
                width=width,
                height=height,
                sequence_number=frame_count,
                payload=payload,
            )
            try:
                batch = event_loop.run_until_complete(detector.detect(frame))
            except Exception as error:
                raise VideoValidationError("detector inference failed") from error
            if not isinstance(batch, DetectionBatch):
                raise VideoValidationError("detector returned an invalid detection batch")
            _validate_batch(batch, frame, artifact, class_map)
            if batch_observer is not None:
                try:
                    batch_observer(batch, timestamp_msec / 1_000.0)
                except VideoValidationError:
                    raise
                except Exception as error:
                    raise VideoValidationError("could not record pipeline observations") from error
            try:
                annotated_frame = renderer(source_frame, batch)
            except Exception as error:
                raise VideoValidationError("could not render detections") from error
            if annotated_frame is None:
                raise VideoValidationError("renderer returned an empty frame")
            try:
                writer.write(annotated_frame)
            except Exception as error:
                raise VideoValidationError("could not write output video") from error

            frame_count += 1
            detection_count += len(batch.detections)
            detections_by_class.update(detection.class_name for detection in batch.detections)
            first_capture_at = first_capture_at or captured_at
            last_capture_at = captured_at

        if frame_count == 0:
            raise VideoValidationError("input video contains no frames")

        writer_released = True
        _release_writer(writer)
        writer = None
        capture_released = True
        _release_capture(capture)
        capture = None

        elapsed_seconds = max(monotonic_clock() - elapsed_started, 0.0)
        metadata = _build_metadata(
            input_path=input_path,
            output_path=output_path,
            metadata_output=metadata_output,
            artifact=artifact,
            class_map=class_map,
            session_id=session_id,
            frame_count=frame_count,
            detection_count=detection_count,
            detections_by_class=detections_by_class,
            width=width,
            height=height,
            fps=fps,
            elapsed_seconds=elapsed_seconds,
            first_capture_at=first_capture_at,
            last_capture_at=last_capture_at,
            runtime_versions=runtime_versions,
        )
        metadata_temp = _write_metadata_temp(metadata_output, metadata)
        metadata_temp.replace(metadata_output)
        metadata_temp = None
        output_started = False
        completed = True
        return metadata
    except VideoValidationError:
        raise
    except Exception as error:
        raise VideoValidationError("video validation failed") from error
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        cleanup_errors.extend(_shutdown_inference(event_loop, executor))
        if metadata_temp is not None:
            try:
                _remove_if_exists(metadata_temp)
            except BaseException as error:
                cleanup_errors.append(error)
        if writer is not None and not writer_released:
            try:
                _release_writer(writer)
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                writer_released = True
        if capture is not None and not capture_released:
            try:
                _release_capture(capture)
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                capture_released = True
        if not completed and output_started:
            try:
                _remove_if_exists(output_path)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors and active_error is None:
            raise VideoValidationError("video resource cleanup failed") from cleanup_errors[0]


def validate_video(**kwargs: object) -> dict[str, object]:
    """Compatibility alias for callers that name the operation ``validate_video``."""

    return run_video_validation(**cast(dict[str, Any], kwargs))


def _shutdown_inference(
    event_loop: asyncio.AbstractEventLoop | None,
    executor: concurrent.futures.Executor | None,
) -> list[BaseException]:
    """Cancel pending inference work and join its thread before the loop closes."""

    errors: list[BaseException] = []
    if event_loop is not None:
        try:
            is_closed = getattr(event_loop, "is_closed", None)
            closed = bool(is_closed()) if callable(is_closed) else False
            if not closed:
                pending: list[asyncio.Task[object]] = []
                try:
                    pending = [task for task in asyncio.all_tasks(event_loop) if not task.done()]
                except (RuntimeError, TypeError):
                    pending = []
                for task in pending:
                    task.cancel()
                if pending:
                    event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                shutdown_executor = getattr(event_loop, "shutdown_default_executor", None)
                if shutdown_executor is not None:
                    event_loop.run_until_complete(shutdown_executor())
                shutdown_generators = getattr(event_loop, "shutdown_asyncgens", None)
                if shutdown_generators is not None:
                    event_loop.run_until_complete(shutdown_generators())
        except BaseException as error:
            errors.append(error)
        try:
            event_loop.close()
        except BaseException as error:
            errors.append(error)
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            errors.append(error)
    return errors


def _validate_paths(
    *,
    input_path: Path,
    output_path: Path,
    metadata_output: Path,
    configuration_paths: Sequence[Path] = (),
) -> tuple[Path, Path, Path]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    metadata_output = Path(metadata_output)
    if not input_path.exists():
        raise VideoValidationError("input video does not exist")
    if not input_path.is_file():
        raise VideoValidationError("input video must be a regular file")
    if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise VideoValidationError("unsupported input video extension")
    if output_path.suffix.lower() != ".mp4":
        raise VideoValidationError("output video extension must be .mp4")
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    metadata_resolved = metadata_output.resolve()
    if input_resolved == output_resolved or _same_existing_file(input_path, output_path):
        raise VideoValidationError("input and output paths must differ")
    if output_resolved == metadata_resolved or _same_existing_file(output_path, metadata_output):
        raise VideoValidationError("metadata and output paths must differ")
    if input_resolved == metadata_resolved or _same_existing_file(input_path, metadata_output):
        raise VideoValidationError("metadata and input paths must differ")
    _reject_configuration_overwrites(
        output_path=output_path,
        metadata_output=metadata_output,
        configuration_paths=configuration_paths,
    )
    if metadata_output.suffix.lower() != ".json":
        raise VideoValidationError("metadata output extension must be .json")
    if output_path.exists():
        raise VideoValidationError("output video already exists")
    if not output_path.parent.exists() or not output_path.parent.is_dir():
        raise VideoValidationError("output directory does not exist")
    if not metadata_output.parent.exists() or not metadata_output.parent.is_dir():
        raise VideoValidationError("metadata directory does not exist")
    return input_path, output_path, metadata_output


def _reject_configuration_overwrites(
    *,
    output_path: Path,
    metadata_output: Path,
    configuration_paths: Sequence[Path],
) -> None:
    """Reject a destination that would replace the class map, model, or input."""

    for configuration_path in configuration_paths:
        if _identifies_same_path(configuration_path, metadata_output) or _identifies_same_path(
            configuration_path, output_path
        ):
            raise VideoValidationError("output path must not replace an input configuration file")


def _absolute_cli_path(path: Path) -> Path:
    """Make a CLI path absolute without resolving a symlink away."""

    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _identifies_same_path(first: Path, second: Path) -> bool:
    """Return whether two paths name the same file, including unresolved aliases."""

    if _same_existing_file(first, second):
        return True
    try:
        return os.path.normcase(str(first.resolve())) == os.path.normcase(str(second.resolve()))
    except OSError:
        left = os.path.normcase(os.path.normpath(str(_absolute_cli_path(first))))
        right = os.path.normcase(os.path.normpath(str(_absolute_cli_path(second))))
        return left == right


def _same_existing_file(first: Path, second: Path) -> bool:
    if not first.exists() or not second.exists():
        return False
    try:
        return os.path.samefile(first, second)
    except (OSError, ValueError):
        return False


def _validate_class_map(class_map: Mapping[int, str]) -> None:
    for class_id, class_name in class_map.items():
        if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
            raise VideoValidationError("class map IDs must be non-negative integers")
        if not isinstance(class_name, str) or not class_name or "\x00" in class_name:
            raise VideoValidationError("class map names must be non-empty strings")


def _validate_batch(
    batch: DetectionBatch,
    frame: FrameEnvelope,
    artifact: VerifiedModelArtifact,
    class_map: Mapping[int, str],
) -> None:
    frame_identity = (
        batch.stream_id,
        batch.session_id,
        batch.camera_external_id,
        batch.captured_at,
        batch.frame_width,
        batch.frame_height,
        batch.sequence_number,
    )
    expected_identity = (
        frame.stream_id,
        frame.session_id,
        frame.camera_external_id,
        frame.captured_at,
        frame.width,
        frame.height,
        frame.sequence_number,
    )
    if frame_identity != expected_identity:
        raise VideoValidationError("detector batch identity does not match frame")
    if (
        batch.model_artifact_id != artifact.artifact_id
        or batch.model_version != artifact.version
        or batch.model_sha256 != artifact.actual_sha256
    ):
        raise VideoValidationError("detector batch artifact does not match verified artifact")
    for detection in batch.detections:
        expected_name = class_map.get(detection.class_id)
        if expected_name != detection.class_name:
            raise VideoValidationError("detector class name disagrees with class map")


def _is_opened(resource: object) -> bool:
    try:
        return bool(cast(Callable[[], bool], resource.isOpened)())
    except (AttributeError, TypeError, ValueError) as error:
        raise VideoValidationError("media resource has no valid open state") from error


def _release_capture(capture: CaptureProtocol) -> None:
    try:
        capture.release()
    except Exception as error:
        raise VideoValidationError("could not release input video") from error


def _release_writer(writer: WriterProtocol) -> None:
    try:
        writer.release()
    except Exception as error:
        raise VideoValidationError("could not release output video writer") from error


def _valid_positive_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VideoValidationError(f"{label} is invalid") from error
    if not math.isfinite(number) or number <= 0:
        raise VideoValidationError(f"{label} is invalid")
    return number


def _valid_non_negative_float(value: object, label: str) -> float:
    number = _valid_finite_float(value, label)
    if number < 0:
        raise VideoValidationError(f"{label} is invalid")
    return number


def _valid_finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VideoValidationError(f"{label} is invalid") from error
    if not math.isfinite(number):
        raise VideoValidationError(f"{label} is invalid")
    return number


def _valid_positive_dimension(value: object, label: str) -> int:
    number = _valid_positive_float(value, label)
    if not number.is_integer():
        raise VideoValidationError(f"{label} is invalid")
    return int(number)


def _packed_bgr_payload(source_frame: object, *, width: int, height: int) -> bytes:
    shape = getattr(source_frame, "shape", None)
    if not isinstance(shape, tuple) or shape != (height, width, 3):
        raise VideoValidationError("input frame shape does not match video dimensions")
    try:
        payload = bytes(cast(Callable[[], bytes], source_frame.tobytes)())
    except (AttributeError, TypeError, ValueError) as error:
        raise VideoValidationError("input frame could not be packed as BGR24") from error
    if len(payload) != width * height * 3:
        raise VideoValidationError("input frame payload size does not match dimensions")
    return payload


def _ensure_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise VideoValidationError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _camera_external_id(input_path: Path) -> str:
    value = input_path.stem or "local-video"
    return value[:128]


def _build_metadata(
    *,
    input_path: Path,
    output_path: Path,
    metadata_output: Path,
    artifact: VerifiedModelArtifact,
    class_map: Mapping[int, str],
    session_id: UUID,
    frame_count: int,
    detection_count: int,
    detections_by_class: Counter[str],
    width: int,
    height: int,
    fps: float,
    elapsed_seconds: float,
    first_capture_at: datetime | None,
    last_capture_at: datetime | None,
    runtime_versions: Mapping[str, str] | None,
) -> dict[str, object]:
    return {
        "status": "complete",
        "session_id": str(session_id),
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "metadata_output": str(metadata_output.resolve()),
        "frame_count": frame_count,
        "detection_count": detection_count,
        "detections_by_class": dict(sorted(detections_by_class.items())),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": OUTPUT_CODEC,
        },
        "capture_time": {
            "first": first_capture_at.isoformat() if first_capture_at else None,
            "last": last_capture_at.isoformat() if last_capture_at else None,
        },
        "processing_seconds": elapsed_seconds,
        "processing_fps": frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "version": artifact.version,
            "model_family": artifact.model_family,
            "sha256": artifact.actual_sha256,
            "license": artifact.license,
            "source_url": artifact.source_url,
            "class_map": {str(class_id): name for class_id, name in sorted(class_map.items())},
            "confidence_threshold": artifact.confidence_threshold,
            "iou_threshold": artifact.iou_threshold,
            "image_size": list(artifact.image_size),
            "device": artifact.device,
        },
        "runtime_versions": dict(runtime_versions or {}),
    }


def _write_metadata_temp(metadata_output: Path, metadata: Mapping[str, object]) -> Path:
    metadata_temp = metadata_output.with_name(f".{metadata_output.name}.{uuid4().hex}.tmp")
    try:
        metadata_temp.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        _remove_if_exists(metadata_temp)
        raise VideoValidationError("could not write run metadata") from error
    return metadata_temp


def _remove_if_exists(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


def _make_capture(input_path: Path) -> CaptureProtocol:
    import cv2

    return cast(CaptureProtocol, cv2.VideoCapture(str(input_path)))


def _make_writer(
    output_path: Path, codec: str, fps: float, dimensions: tuple[int, int]
) -> WriterProtocol:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cast(WriterProtocol, cv2.VideoWriter(str(output_path), fourcc, fps, dimensions))


def _make_renderer() -> Renderer:
    import cv2

    tracker = IoUPersonTracker()
    first_captured_at: datetime | None = None

    def render(source_frame: object, batch: DetectionBatch) -> object:
        nonlocal first_captured_at
        image = source_frame
        first_captured_at = first_captured_at or batch.captured_at
        elapsed_seconds = max((batch.captured_at - first_captured_at).total_seconds(), 0.0)
        tracked_frame = tracker.update(batch)
        track_ids = {
            person.detection: person.track_id
            for person in tracked_frame.persons
        }

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (min(image.shape[1], 330), 34), (17, 24, 39), -1)
        cv2.addWeighted(overlay, 0.88, image, 0.12, 0, image)
        elapsed_minutes, elapsed_remainder = divmod(round(elapsed_seconds), 60)
        cv2.putText(
            image,
            f"LOCAL YOLO  T+{elapsed_minutes:02d}:{elapsed_remainder:02d}",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for detection in batch.detections:
            box = detection.bounding_box
            height, width = image.shape[:2]
            x1 = max(0, min(width - 1, round(box.x1 * width)))
            y1 = max(0, min(height - 1, round(box.y1 * height)))
            x2 = max(0, min(width - 1, round(box.x2 * width)))
            y2 = max(0, min(height - 1, round(box.y2 * height)))
            track_id = track_ids.get(detection)
            is_person = track_id is not None
            color = (0, 220, 120) if is_person else (255, 190, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = (
                f"Track #{track_id} Person {detection.confidence:.2f}"
                if is_person
                else f"{detection.class_name} {detection.confidence:.2f}"
            )
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return image

    return render


class _JsonObject(list[tuple[object, object]]):
    """JSON object pairs, including duplicates that ``json.loads`` would otherwise drop."""


def _json_object_pairs(pairs: list[tuple[object, object]]) -> _JsonObject:
    keys = [key for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise VideoValidationError("class map IDs must be unique")
    return _JsonObject(pairs)


def _strict_class_id(raw_id: object) -> int:
    """Accept a JSON integer or a canonical decimal string, never a bool or fraction."""

    if isinstance(raw_id, bool | float):
        raise VideoValidationError("class map IDs must be integers")
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str) and _CLASS_ID_TEXT.fullmatch(raw_id):
        return int(raw_id)
    raise VideoValidationError("class map IDs must be integers")


def _parse_class_map(path: Path) -> dict[int, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object_pairs)
    except VideoValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VideoValidationError("class map JSON could not be read") from error
    if isinstance(value, _JsonObject):
        pairs = list(value)
    elif isinstance(value, list):
        pairs = []
        for item in value:
            if isinstance(item, _JsonObject):
                mapping = dict(item)
                if set(mapping) != {"id", "name"}:
                    raise VideoValidationError("class map JSON must contain id/name pairs")
                pairs.append((mapping["id"], mapping["name"]))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise VideoValidationError("class map JSON must contain id/name pairs")
    else:
        raise VideoValidationError("class map JSON must be an object or list")
    result: dict[int, str] = {}
    for raw_id, raw_name in pairs:
        class_id = _strict_class_id(raw_id)
        if class_id in result:
            raise VideoValidationError("class map IDs must be unique")
        if not isinstance(raw_name, str):
            raise VideoValidationError("class map names must be strings")
        result[class_id] = raw_name
    _validate_class_map(result)
    return result


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package_name, key in (("opencv-python", "opencv"), ("ultralytics", "ultralytics")):
        try:
            versions[key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartsite-ai-detect-video",
        description="Validate a local video with a verified model and write an annotated MP4.",
    )
    parser.add_argument("--input", type=Path, required=True, help="local input video")
    parser.add_argument("--model", type=Path, required=True, help="local model weight file")
    parser.add_argument("--model-artifact-id", required=True, help="stable model artifact ID")
    parser.add_argument("--model-version", required=True, help="model version")
    parser.add_argument(
        "--model-family",
        required=True,
        help="artifact family recorded in metadata, such as yolo11s or yolov8",
    )
    parser.add_argument("--model-sha256", required=True, help="expected lowercase SHA-256")
    parser.add_argument("--class-map", type=Path, required=True, help="JSON class map")
    parser.add_argument(
        "--camera-external-id",
        help="camera ID used for the technical pipeline; defaults to the input filename stem",
    )
    parser.add_argument("--output", type=Path, required=True, help="annotated .mp4 output")
    parser.add_argument(
        "--metadata-output", type=Path, required=True, help="atomic JSON run metadata output"
    )
    parser.add_argument(
        "--region-configuration",
        type=Path,
        help="camera-region configuration required together with --ui-timeline-output",
    )
    parser.add_argument(
        "--ppe-region-id",
        help="configured observation region UUID required together with --ui-timeline-output",
    )
    parser.add_argument(
        "--ui-timeline-output",
        type=Path,
        help="write MF05/MF06 UI timeline JSON from the real technical pipeline",
    )
    parser.add_argument(
        "--model-source-url",
        "--source-url",
        dest="model_source_url",
        required=True,
        help="HTTPS provenance URL recorded in metadata",
    )
    parser.add_argument(
        "--model-license",
        "--license",
        dest="model_license",
        required=True,
        help="declared model license recorded in metadata",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("WIDTH", "HEIGHT"),
        help="inference width then height; the provider receives height then width",
    )
    parser.add_argument("--device", default="cpu", help="inference device passed to Ultralytics")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``smartsite-ai-detect-video``."""

    parser = build_parser()
    args = parser.parse_args(argv)
    runner: UltralyticsYoloRunner | None = None
    try:
        _validate_cli_provenance(args.model_source_url, args.model_license)
        class_map = _parse_class_map(args.class_map)
        model_path = _absolute_cli_path(args.model)
        input_path = _absolute_cli_path(args.input)
        camera_external_id = args.camera_external_id or _camera_external_id(input_path)
        timeline_collector: _UiTimelineCollector | None = None
        timeline_output: Path | None = None
        configuration_paths: list[Path] = [args.class_map, model_path, args.input]
        timeline_options = (
            args.ui_timeline_output,
            args.region_configuration,
            args.ppe_region_id,
        )
        if any(option is not None for option in timeline_options):
            if not all(option is not None for option in timeline_options):
                raise VideoValidationError(
                    "--ui-timeline-output, --region-configuration and --ppe-region-id "
                    "must be used together"
                )
            region_configuration_path = _absolute_cli_path(args.region_configuration)
            configuration_paths.append(region_configuration_path)
            try:
                configuration = CameraRegionConfiguration.from_wire_bytes(
                    region_configuration_path.read_bytes()
                )
            except (OSError, ValueError) as error:
                raise VideoValidationError(
                    "camera region configuration could not be read"
                ) from error
            if configuration.camera_external_id != camera_external_id:
                raise VideoValidationError(
                    "camera external ID does not match the camera region configuration"
                )
            timeline_output = _absolute_cli_path(args.ui_timeline_output)
            protected_paths = (
                input_path,
                model_path,
                _absolute_cli_path(args.class_map),
                region_configuration_path,
                _absolute_cli_path(args.output),
                _absolute_cli_path(args.metadata_output),
            )
            if any(_identifies_same_path(timeline_output, path) for path in protected_paths):
                raise VideoValidationError(
                    "UI timeline output must not replace an input or run output"
                )
            timeline_collector = _UiTimelineCollector(
                configuration=configuration,
                ppe_region_id=args.ppe_region_id,
            )
        _reject_configuration_overwrites(
            output_path=args.output,
            metadata_output=args.metadata_output,
            configuration_paths=configuration_paths,
        )
        spec = ModelArtifactSpec(
            artifact_id=args.model_artifact_id,
            version=args.model_version,
            model_family=args.model_family,
            artifact_path=model_path,
            sha256=args.model_sha256,
            source_url=args.model_source_url,
            license=args.model_license,
            class_map=tuple(sorted(class_map.items())),
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            image_size=tuple(args.image_size),
            device=args.device,
        )
        artifact = verify_model_artifact(spec)
        runner = UltralyticsYoloRunner()
        runner.load(artifact)
        detector = Yolo11Detector(artifact, runner)
        run_video_validation(
            input_path=input_path,
            output_path=args.output,
            metadata_output=args.metadata_output,
            artifact=artifact,
            class_map=class_map,
            detector=detector,
            capture_factory=_make_capture,
            writer_factory=_make_writer,
            renderer=_make_renderer(),
            runtime_versions=_runtime_versions(),
            configuration_paths=configuration_paths,
            camera_external_id=camera_external_id,
            batch_observer=timeline_collector.observe if timeline_collector is not None else None,
        )
        if timeline_collector is not None and timeline_output is not None:
            timeline_collector.write(timeline_output)
        return 0
    except (ArtifactValidationError, VideoValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"error: video validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: video validation interrupted", file=sys.stderr)
        return 130
    finally:
        if runner is not None:
            runner.close()


def main() -> None:
    """Console-script entry point for local annotated-video validation."""

    raise SystemExit(run())


def _validate_cli_provenance(source_url: str, license_name: str) -> None:
    parsed_url = urlsplit(source_url)
    if parsed_url.hostname is not None and parsed_url.hostname.endswith(".invalid"):
        raise VideoValidationError("model provenance URL must identify a real source")
    if license_name.strip().upper() in {"N/A", "TBD", "UNKNOWN", "UNSPECIFIED"}:
        raise VideoValidationError("model license must be explicitly declared")


__all__ = [
    "CAP_PROP_FPS",
    "CAP_PROP_FRAME_HEIGHT",
    "CAP_PROP_FRAME_WIDTH",
    "CAP_PROP_POS_MSEC",
    "CaptureFactory",
    "Renderer",
    "SUPPORTED_INPUT_EXTENSIONS",
    "VideoValidationError",
    "WriterFactory",
    "build_parser",
    "main",
    "run",
    "run_video_validation",
    "validate_video",
]
