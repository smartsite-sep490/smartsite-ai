from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from smartsite_ai.ingestion.envelope import FrameEnvelope


def test_frame_envelope_valid_creation() -> None:
    now = datetime.now(UTC)
    envelope = FrameEnvelope(
        stream_id="stream-01",
        session_id="session-abc-123",
        camera_external_id="cam-gate-north",
        captured_at=now,
        width=1920,
        height=1080,
        sequence_number=1,
        payload=b"raw-frame-bytes-payload",
    )
    assert envelope.stream_id == "stream-01"
    assert envelope.session_id == "session-abc-123"
    assert envelope.camera_external_id == "cam-gate-north"
    assert envelope.captured_at == now
    assert envelope.width == 1920
    assert envelope.height == 1080
    assert envelope.sequence_number == 1
    assert envelope.payload == b"raw-frame-bytes-payload"
    assert envelope.buffer.tobytes() == b"raw-frame-bytes-payload"


def test_frame_envelope_reused_mutable_buffer_immutability() -> None:
    """Ensure mutable buffer (e.g. bytearray) is defensively copied to prevent mutation."""
    reused_buffer = bytearray(b"original_frame_buffer_data")
    now = datetime.now(UTC)

    envelope = FrameEnvelope(
        stream_id="s1",
        session_id="sess1",
        camera_external_id="cam1",
        captured_at=now,
        width=640,
        height=480,
        sequence_number=1,
        payload=reused_buffer,
    )

    # In-place mutate the source buffer
    reused_buffer[0:8] = b"MUTATED!"
    assert reused_buffer.startswith(b"MUTATED!")

    # The envelope payload must retain the original data intact
    assert envelope.payload == b"original_frame_buffer_data"
    assert envelope.buffer.tobytes() == b"original_frame_buffer_data"


def test_frame_envelope_rejects_naive_datetime() -> None:
    naive_dt = datetime(2026, 9, 19, 12, 0, 0)
    with pytest.raises(ValidationError, match="captured_at must be timezone-aware"):
        FrameEnvelope(
            stream_id="stream-01",
            session_id="session-01",
            camera_external_id="cam-01",
            captured_at=naive_dt,
            width=640,
            height=480,
            sequence_number=0,
            payload=b"bytes",
        )


def test_frame_envelope_rejects_non_positive_dimensions() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        FrameEnvelope(
            stream_id="stream-01",
            session_id="session-01",
            camera_external_id="cam-01",
            captured_at=now,
            width=0,
            height=480,
            sequence_number=0,
            payload=b"bytes",
        )
    with pytest.raises(ValidationError):
        FrameEnvelope(
            stream_id="stream-01",
            session_id="session-01",
            camera_external_id="cam-01",
            captured_at=now,
            width=640,
            height=-10,
            sequence_number=0,
            payload=b"bytes",
        )


def test_frame_envelope_rejects_negative_sequence() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        FrameEnvelope(
            stream_id="stream-01",
            session_id="session-01",
            camera_external_id="cam-01",
            captured_at=now,
            width=640,
            height=480,
            sequence_number=-1,
            payload=b"bytes",
        )


def test_frame_envelope_iso_timestamp() -> None:
    now = datetime(2026, 9, 19, 14, 30, 0, tzinfo=UTC)
    envelope = FrameEnvelope(
        stream_id="stream-01",
        session_id="session-01",
        camera_external_id="cam-01",
        captured_at=now,
        width=1280,
        height=720,
        sequence_number=42,
        payload=b"frame-42",
    )
    assert envelope.captured_at_iso == "2026-09-19T14:30:00+00:00"
