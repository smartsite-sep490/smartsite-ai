import json
from datetime import UTC, datetime
from math import inf, nan
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

SESSION = UUID("00000000-0000-4000-8000-000000000001")
MODEL_SHA256 = "a" * 64


def make_frame(**overrides: object) -> FrameEnvelope:
    data: dict[str, object] = {
        "stream_id": "stream-01",
        "session_id": SESSION,
        "camera_external_id": "cam-01",
        "captured_at": datetime(2026, 9, 20, 12, tzinfo=UTC),
        "width": 2,
        "height": 2,
        "sequence_number": 7,
        "payload": bytes(12),
        **overrides,
    }
    return FrameEnvelope.model_validate(data)


def make_box(**overrides: object) -> NormalizedBoundingBox:
    data: dict[str, object] = {
        "x1": 0.1,
        "y1": 0.2,
        "x2": 0.7,
        "y2": 0.9,
        "coordinate_space": "NORMALIZED_0_1",
        **overrides,
    }
    return NormalizedBoundingBox.model_validate(data)


def make_detection(**overrides: object) -> NormalizedDetection:
    data: dict[str, object] = {
        "class_id": 1,
        "class_name": "person",
        "confidence": 0.75,
        "bounding_box": make_box(),
        **overrides,
    }
    return NormalizedDetection.model_validate(data)


def batch_data(**overrides: object) -> dict[str, object]:
    return {
        "stream_id": "stream-01",
        "session_id": SESSION,
        "camera_external_id": "cam-01",
        "captured_at": datetime(2026, 9, 20, 12, tzinfo=UTC),
        "frame_width": 2,
        "frame_height": 2,
        "sequence_number": 7,
        "model_artifact_id": "detector-model",
        "model_version": "2026.09.20",
        "model_sha256": MODEL_SHA256,
        "detections": (make_detection(),),
        **overrides,
    }


def test_normalized_bounding_box_accepts_closed_unit_interval_and_is_frozen() -> None:
    bounding_box = make_box(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    assert bounding_box.coordinate_space == "NORMALIZED_0_1"
    with pytest.raises(ValidationError, match="frozen"):
        bounding_box.x1 = 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x1", -0.01),
        ("y1", -0.01),
        ("x2", 1.01),
        ("y2", 1.01),
        ("x1", nan),
        ("y1", inf),
        ("x2", -inf),
        ("y2", nan),
        ("coordinate_space", "PIXELS"),
        ("unexpected", True),
    ],
)
def test_normalized_bounding_box_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_box(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"x1": 0.7, "x2": 0.7},
        {"x1": 0.8, "x2": 0.7},
        {"y1": 0.9, "y2": 0.9},
        {"y1": 1.0, "y2": 0.9},
    ],
)
def test_normalized_bounding_box_requires_strictly_ordered_corners(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        make_box(**overrides)


def test_normalized_detection_accepts_inclusive_boundaries_and_is_frozen() -> None:
    detection = make_detection(class_id=2_147_483_647, class_name="x" * 128, confidence=1.0)
    assert detection.class_id == 2_147_483_647
    assert detection.confidence == 1.0
    with pytest.raises(ValidationError, match="frozen"):
        detection.confidence = 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_id", -1),
        ("class_id", 2_147_483_648),
        ("class_name", ""),
        ("class_name", "x" * 129),
        ("class_name", "hard\x00hat"),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("confidence", nan),
        ("confidence", inf),
        ("confidence", -inf),
        ("unexpected", True),
    ],
)
def test_normalized_detection_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_detection(**{field: value})


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (make_box, "x1", "0.1"),
        (make_detection, "class_id", "1"),
        (make_detection, "confidence", "0.75"),
    ],
)
def test_python_object_validation_rejects_numeric_string_coercion(
    factory: object, field: str, value: str
) -> None:
    with pytest.raises(ValidationError):
        factory(**{field: value})  # type: ignore[operator]


def test_detection_batch_from_frame_locks_identity_and_deterministically_sorts() -> None:
    frame = make_frame(
        stream_id="s" * 128,
        camera_external_id="c" * 128,
        width=3,
        height=1,
        sequence_number=2**63 - 1,
        payload=bytes(9),
    )
    first = make_detection(
        class_id=7,
        class_name="first",
        confidence=0.9,
        bounding_box=make_box(x1=0.4, y1=0.4, x2=0.8, y2=0.8),
    )
    second = make_detection(
        class_id=2,
        class_name="second",
        confidence=0.8,
        bounding_box=make_box(x1=0.2, y1=0.3, x2=0.7, y2=0.9),
    )
    third = make_detection(
        class_id=2,
        class_name="third",
        confidence=0.8,
        bounding_box=make_box(x1=0.3, y1=0.1, x2=0.6, y2=0.7),
    )

    batch = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=(item for item in (third, first, second)),
    )
    reversed_batch = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=reversed((third, first, second)),
    )

    assert batch.stream_id == frame.stream_id
    assert batch.session_id == frame.session_id
    assert batch.camera_external_id == frame.camera_external_id
    assert batch.captured_at == frame.captured_at
    assert batch.frame_width == frame.width
    assert batch.frame_height == frame.height
    assert batch.sequence_number == frame.sequence_number
    assert batch.detections == (first, second, third)
    assert reversed_batch.detections == batch.detections
    assert isinstance(batch.detections, tuple)


def test_detection_batch_sorts_class_id_before_box_and_name_tie_breakers() -> None:
    lower_class_id = make_detection(
        class_id=2,
        class_name="zulu",
        confidence=0.5,
        bounding_box=make_box(x1=0.4, y1=0.4, x2=0.9, y2=0.9),
    )
    higher_class_id = make_detection(
        class_id=7,
        class_name="alpha",
        confidence=0.5,
        bounding_box=make_box(x1=0.1, y1=0.1, x2=0.8, y2=0.8),
    )
    frame = make_frame()

    lower_class_id_first = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=(lower_class_id, higher_class_id),
    )
    higher_class_id_first = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=(higher_class_id, lower_class_id),
    )

    assert lower_class_id_first.detections == (lower_class_id, higher_class_id)
    assert higher_class_id_first.detections == (lower_class_id, higher_class_id)


@pytest.mark.parametrize(
    ("lower_box", "higher_box", "lower_name", "higher_name"),
    [
        (
            {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.9},
            {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
            "same-class",
            "same-class",
        ),
        (
            {"x1": 0.1, "y1": 0.2, "x2": 0.7, "y2": 0.9},
            {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
            "same-class",
            "same-class",
        ),
        (
            {"x1": 0.1, "y1": 0.2, "x2": 0.7, "y2": 0.8},
            {"x1": 0.1, "y1": 0.2, "x2": 0.7, "y2": 0.9},
            "same-class",
            "same-class",
        ),
        (
            {"x1": 0.1, "y1": 0.2, "x2": 0.7, "y2": 0.9},
            {"x1": 0.1, "y1": 0.2, "x2": 0.7, "y2": 0.9},
            "alpha",
            "bravo",
        ),
    ],
    ids=["y1", "x2", "y2", "class_name"],
)
def test_detection_batch_sorting_pins_each_remaining_tie_breaker(
    lower_box: dict[str, float],
    higher_box: dict[str, float],
    lower_name: str,
    higher_name: str,
) -> None:
    lower = make_detection(
        class_id=3,
        class_name=lower_name,
        confidence=0.5,
        bounding_box=make_box(**lower_box),
    )
    higher = make_detection(
        class_id=3,
        class_name=higher_name,
        confidence=0.5,
        bounding_box=make_box(**higher_box),
    )
    frame = make_frame()

    lower_first = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=(lower, higher),
    )
    higher_first = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=(higher, lower),
    )

    assert lower_first.detections == (lower, higher)
    assert higher_first.detections == (lower, higher)


def test_detection_batch_exposes_approved_frame_dimension_field_names() -> None:
    frame = make_frame(width=3, height=1, payload=bytes(9))

    batch = DetectionBatch.from_frame(
        frame,
        model_artifact_id="model-artifact",
        model_version="v1",
        model_sha256=MODEL_SHA256,
        detections=(),
    )

    assert batch.frame_width == 3
    assert batch.frame_height == 1
    assert set(batch.model_dump()) >= {"frame_width", "frame_height"}
    assert "width" not in batch.model_dump()
    assert "height" not in batch.model_dump()


def test_detection_batch_accepts_model_and_collection_boundaries() -> None:
    detections = tuple(
        make_detection(class_id=index, class_name=f"class-{index}") for index in range(1024)
    )
    batch = DetectionBatch.model_validate(
        batch_data(
            model_artifact_id="a" * 128,
            model_version="v" * 64,
            detections=detections,
        )
    )
    assert len(batch.detections) == 1024
    assert batch.detections[0].class_id == 0
    with pytest.raises(ValidationError, match="frozen"):
        batch.frame_width = 1


def test_detection_batch_bounds_iterable_consumption_before_rejecting_oversize_result() -> None:
    def oversize_detections():
        for index in range(1025):
            yield make_detection(class_id=index)
        raise AssertionError("from_frame consumed beyond the rejection boundary")

    with pytest.raises(ValidationError):
        DetectionBatch.from_frame(
            make_frame(),
            model_artifact_id="model-artifact",
            model_version="v1",
            model_sha256=MODEL_SHA256,
            detections=oversize_detections(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_id", ""),
        ("stream_id", "s" * 129),
        ("stream_id", "stream\x00id"),
        ("camera_external_id", ""),
        ("camera_external_id", "c" * 129),
        ("camera_external_id", "camera\x00id"),
        ("frame_width", 0),
        ("frame_width", 16_385),
        ("frame_height", 0),
        ("frame_height", 16_385),
        ("sequence_number", -1),
        ("sequence_number", 2**63),
        ("model_artifact_id", ""),
        ("model_artifact_id", "a" * 129),
        ("model_artifact_id", "model\x00artifact"),
        ("model_version", ""),
        ("model_version", "v" * 65),
        ("model_version", "version\x001"),
        ("model_sha256", "A" * 64),
        ("model_sha256", "a" * 63),
        ("model_sha256", "g" * 64),
        ("detections", tuple(make_detection(class_id=i) for i in range(1025))),
        ("unexpected", True),
    ],
)
def test_detection_batch_rejects_invalid_boundaries(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        DetectionBatch.model_validate(batch_data(**{field: value}))


@pytest.mark.parametrize("field", ["frame_width", "frame_height", "sequence_number"])
def test_detection_batch_rejects_python_integer_string_coercion(field: str) -> None:
    with pytest.raises(ValidationError):
        DetectionBatch.model_validate(batch_data(**{field: "2"}))


def test_detection_batch_rejects_python_uuid_and_datetime_strings() -> None:
    with pytest.raises(ValidationError):
        DetectionBatch.model_validate(batch_data(session_id=str(SESSION)))
    with pytest.raises(ValidationError):
        DetectionBatch.model_validate(batch_data(captured_at="2026-09-20T12:00:00Z"))


def test_detection_batch_json_deserializes_uuid_and_datetime_without_numeric_coercion() -> None:
    payload = json.loads(DetectionBatch.model_validate(batch_data()).model_dump_json())
    restored = DetectionBatch.model_validate_json(json.dumps(payload))
    assert restored.session_id == SESSION
    assert restored.captured_at == datetime(2026, 9, 20, 12, tzinfo=UTC)

    payload["frame_width"] = "2"
    with pytest.raises(ValidationError):
        DetectionBatch.model_validate_json(json.dumps(payload))


def test_all_detection_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_box(extra_field=True)
    with pytest.raises(ValidationError):
        make_detection(extra_field=True)
    with pytest.raises(ValidationError):
        DetectionBatch.model_validate(batch_data(extra_field=True))
