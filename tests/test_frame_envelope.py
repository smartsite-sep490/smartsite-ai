import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartsite_ai.ingestion.envelope import FrameEnvelope

SESSION = UUID("00000000-0000-4000-8000-000000000001")


def frame_data(**overrides: object) -> dict[str, object]:
    return {
        "stream_id": "stream-01",
        "session_id": SESSION,
        "camera_external_id": "cam-01",
        "captured_at": datetime(2026, 9, 19, 12, tzinfo=UTC),
        "width": 2,
        "height": 2,
        "sequence_number": 0,
        "payload": bytes(12),
        **overrides,
    }


def test_frame_envelope_valid_creation_and_readonly_buffer() -> None:
    envelope = FrameEnvelope.model_validate(frame_data())
    assert envelope.session_id == SESSION
    assert envelope.pixel_format == "BGR24"
    assert envelope.width == envelope.height == 2
    assert envelope.sequence_number == 0
    assert envelope.payload == bytes(12)
    assert envelope.buffer.tobytes() == bytes(12)
    assert envelope.buffer.readonly
    with pytest.raises(TypeError):
        envelope.buffer[0] = 1
    with pytest.raises(ValidationError, match="frozen"):
        envelope.width = 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extra_field", True),
        ("stream_id", ""),
        ("stream_id", "s" * 129),
        ("camera_external_id", ""),
        ("camera_external_id", "c" * 129),
        ("session_id", "invalid-uuid"),
        ("captured_at", datetime(2026, 9, 19, 12)),
        ("width", 0),
        ("width", 16385),
        ("height", 0),
        ("height", 16385),
        ("sequence_number", -1),
        ("sequence_number", 2**63),
        ("payload", "abcdefghijkl"),
        ("payload", bytes(11)),
        ("payload", bytes(13)),
        ("pixel_format", "RGB24"),
    ],
)
def test_frame_envelope_rejects_invalid_boundary(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FrameEnvelope.model_validate(frame_data(**{field: value}))
    # Check the intended boundary failed, not another invalid fixture field.
    assert any(
        error["loc"] == (field,) or (field == "payload" and error["loc"] == ())
        for error in exc_info.value.errors()
    )


@pytest.mark.parametrize("field", ["width", "height", "sequence_number"])
@pytest.mark.parametrize("value", ["2", 2.0, True])
def test_frame_envelope_rejects_python_numeric_coercion(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FrameEnvelope.model_validate(frame_data(**{field: value}))
    assert any(error["loc"] == (field,) for error in exc_info.value.errors())


def test_frame_envelope_rejects_uuid_string_in_python_input() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FrameEnvelope.model_validate(frame_data(session_id=str(SESSION)))
    assert any(error["loc"] == ("session_id",) for error in exc_info.value.errors())


@pytest.mark.parametrize("use_memoryview", [False, True])
def test_frame_envelope_defensively_copies_reused_buffer(use_memoryview: bool) -> None:
    reused = bytearray(range(12))
    payload = memoryview(reused) if use_memoryview else reused
    envelope = FrameEnvelope.model_validate(frame_data(payload=payload))
    reused[:] = bytes(12)
    assert envelope.payload == bytes(range(12))
    assert envelope.buffer.tobytes() == bytes(range(12))
    assert envelope.buffer.readonly


def test_frame_envelope_normalizes_aware_datetime_and_serializes_json() -> None:
    timestamp = datetime(2026, 9, 19, 19, tzinfo=timezone(timedelta(hours=7)))
    envelope = FrameEnvelope.model_validate(frame_data(captured_at=timestamp))
    assert envelope.captured_at.tzinfo is UTC
    assert envelope.captured_at_iso == "2026-09-19T12:00:00+00:00"
    serialized = json.loads(envelope.model_dump_json())
    assert serialized["session_id"] == "00000000-0000-4000-8000-000000000001"
    assert serialized["captured_at"] == "2026-09-19T12:00:00Z"
    assert serialized["pixel_format"] == "BGR24"


@pytest.mark.parametrize(("width", "height"), [(1, 1), (16384, 1), (1, 16384)])
def test_frame_envelope_accepts_inclusive_limits(width: int, height: int) -> None:
    envelope = FrameEnvelope.model_validate(
        frame_data(
            stream_id="s" * 128,
            camera_external_id="c" * 128,
            width=width,
            height=height,
            sequence_number=2**63 - 1,
            payload=bytes(width * height * 3),
        )
    )
    assert envelope.width == width
    assert envelope.height == height
    assert envelope.sequence_number == 9223372036854775807


def test_frame_envelope_rejects_string_payload_from_json() -> None:
    data = frame_data(payload="abcdefghijkl")
    data["session_id"] = str(SESSION)
    data["captured_at"] = "2026-09-19T12:00:00Z"
    with pytest.raises(ValidationError) as exc_info:
        FrameEnvelope.model_validate_json(json.dumps(data))
    assert any(error["loc"] == ("payload",) for error in exc_info.value.errors())
