from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from smartsite_ai.domain.observations import (
    BoundingBox,
    PersonObservation,
    PpeObservation,
    TechnicalObservationEvent,
    ZoneEntryObservation,
)
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.pipelines.ppe_temporal import (
    ConfirmedPpeCandidate,
    TemporalPpeCandidateGate,
    filter_event_for_delivery,
    should_post_event,
)
from smartsite_ai.realtime import parse_zone_polygon, resolve_realtime_device, safe_source_label


def test_zone_polygon_rejects_out_of_range_points() -> None:
    with pytest.raises(ValueError, match="within 0 and 1"):
        parse_zone_polygon("1.2,0.2;0.9,0.2;0.9,0.9")


def test_zone_polygon_rejects_fewer_than_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        parse_zone_polygon("0.1,0.1;0.2,0.2")


def test_zone_polygon_keeps_in_range_points() -> None:
    assert parse_zone_polygon("0,0;1,0;1,1") == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_source_errors_hide_credentials() -> None:
    assert safe_source_label("rtsp://user:secret@camera.local/live") == "rtsp://camera.local"
    assert safe_source_label(r"D:\videos\site.mp4") == "configured source"


@pytest.mark.parametrize(
    ("configured", "cuda_available", "cuda_device_count", "expected"),
    [
        ("auto", False, 0, "cpu"),
        ("auto", True, 1, "cuda:0"),
        ("cpu", True, 1, "cpu"),
        ("cuda", True, 1, "cuda:0"),
        ("cuda:1", True, 2, "cuda:1"),
    ],
)
def test_realtime_device_resolves_only_available_hardware(
    configured: str,
    cuda_available: bool,
    cuda_device_count: int,
    expected: str,
) -> None:
    assert (
        resolve_realtime_device(
            configured,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("configured", "cuda_available", "cuda_device_count"),
    [
        ("cuda", False, 0),
        ("cuda:0", False, 0),
        ("cuda:1", True, 1),
    ],
)
def test_realtime_device_rejects_unavailable_cuda(
    configured: str,
    cuda_available: bool,
    cuda_device_count: int,
) -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        resolve_realtime_device(
            configured,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
        )


def test_should_post_event_gating_behavior() -> None:
    def make_ev(*obs: object) -> TechnicalObservationEvent:
        return TechnicalObservationEvent.create(
            event_id="00000000-0000-4000-8000-000000000010",
            camera_external_id="camera-01",
            stream_session_id=str(UUID("00000000-0000-4000-8000-000000000001")),
            captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC).isoformat(),
            frame_dimensions={"width": 640, "height": 480},
            observations=obs,
        )

    person = PersonObservation.model_validate(
        {
            "type": "PERSON",
            "trackId": 1,
            "confidence": 0.9,
            "boundingBox": BoundingBox.model_validate(
                {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9, "coordinateSpace": "NORMALIZED_0_1"}
            ).to_wire_dict(),
        }
    )
    missing_ppe = PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": 1,
            "ppeItem": "HARD_HAT",
            "status": "MISSING",
            "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "geometryVersion": 1,
            "confidence": 0.95,
        }
    )
    zone_entry = ZoneEntryObservation.model_validate(
        {
            "type": "ZONE_ENTRY",
            "trackId": 1,
            "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "geometryVersion": 1,
            "confidence": 0.9,
        }
    )

    candidate = ConfirmedPpeCandidate(
        stream_id="stream-01",
        session_id=UUID("00000000-0000-4000-8000-000000000001"),
        track_id=1,
        ppe_item="HARD_HAT",
        first_seen_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        confirmed_at=datetime(2026, 9, 21, 12, 0, 1, tzinfo=UTC),
    )

    # Raw per-frame MISSING without confirmed candidates does not post
    assert not should_post_event(make_ev(person, missing_ppe), ())

    # Confirming frame posts once
    assert should_post_event(make_ev(person, missing_ppe), (candidate,))

    # Cooldown frames (no new candidate returned) do not post
    assert not should_post_event(make_ev(person, missing_ppe), ())

    # ZONE_ENTRY posts immediately even without confirmed PPE candidates
    assert should_post_event(make_ev(person, zone_entry), ())


def test_mixed_event_zone_entry_with_unconfirmed_missing_delivers_zone_only() -> None:
    session_id = UUID("00000000-0000-4000-8000-000000000001")
    person = PersonObservation.model_validate(
        {
            "type": "PERSON",
            "trackId": 1,
            "confidence": 0.9,
            "boundingBox": BoundingBox.model_validate(
                {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9, "coordinateSpace": "NORMALIZED_0_1"}
            ).to_wire_dict(),
        }
    )
    missing_ppe = PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": 1,
            "ppeItem": "HARD_HAT",
            "status": "MISSING",
            "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "geometryVersion": 1,
            "confidence": 0.95,
        }
    )
    zone_entry = ZoneEntryObservation.model_validate(
        {
            "type": "ZONE_ENTRY",
            "trackId": 1,
            "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "geometryVersion": 1,
            "confidence": 0.9,
        }
    )
    raw_mixed_event = TechnicalObservationEvent.create(
        event_id="00000000-0000-4000-8000-000000000010",
        camera_external_id="camera-01",
        stream_session_id=str(session_id),
        captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC).isoformat(),
        frame_dimensions={"width": 640, "height": 480},
        observations=[person, zone_entry, missing_ppe],
    )

    # When candidates are empty, delivery event preserves ZONE_ENTRY and PERSON,
    # but strips raw unconfirmed MISSING PPE completely.
    delivered = filter_event_for_delivery(raw_mixed_event, ())
    assert delivered is not None

    delivered_dict = delivered.to_wire_dict()
    obs_types = [o["type"] for o in delivered_dict["observations"]]
    assert "ZONE_ENTRY" in obs_types
    assert "PERSON" in obs_types
    assert not any(o["type"] == "PPE" for o in delivered_dict["observations"])


@pytest.mark.anyio
async def test_realtime_loop_gate_updates_every_batch_and_controls_post_count() -> None:
    """Verify temporal gate updates on every batch and delivery count is exact."""
    session_id = UUID("00000000-0000-4000-8000-000000000001")
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    def make_batch_at(dt: datetime, seq: int) -> DetectionBatch:
        return DetectionBatch(
            stream_id="stream-01",
            session_id=session_id,
            camera_external_id="camera-01",
            captured_at=dt,
            frame_width=640,
            frame_height=480,
            sequence_number=seq,
            model_artifact_id="fake-detector",
            model_version="test",
            model_sha256="a" * 64,
            detections=(),
        )

    person = PersonObservation.model_validate(
        {
            "type": "PERSON",
            "trackId": 1,
            "confidence": 0.9,
            "boundingBox": BoundingBox.model_validate(
                {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9, "coordinateSpace": "NORMALIZED_0_1"}
            ).to_wire_dict(),
        }
    )
    missing_ppe = PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": 1,
            "ppeItem": "HARD_HAT",
            "status": "MISSING",
            "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "geometryVersion": 1,
            "confidence": 0.95,
        }
    )
    zone_entry = ZoneEntryObservation.model_validate(
        {
            "type": "ZONE_ENTRY",
            "trackId": 1,
            "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "geometryVersion": 1,
            "confidence": 0.9,
        }
    )

    def make_event(dt: datetime, seq: int, *obs: object) -> TechnicalObservationEvent:
        return TechnicalObservationEvent.create(
            event_id=f"00000000-0000-4000-8000-{seq:012d}",
            camera_external_id="camera-01",
            stream_session_id=str(session_id),
            captured_at=dt.isoformat(),
            frame_dimensions={"width": 640, "height": 480},
            observations=obs,
        )

    # Simulated frames processed through realtime loop logic:
    # Frame 1 (t0): 1st missing PPE -> not posted
    b1 = make_batch_at(t0, 1)
    ev1 = make_event(t0, 1, person, missing_ppe)

    # Frame 2 (t0 + 200ms): empty frame -> pipeline returns None!
    b2 = make_batch_at(t0 + timedelta(milliseconds=200), 2)
    ev2 = None

    # Frame 3 (t0 + 400ms): ZONE_ENTRY + missing PPE -> zone posts, missing is suppressed!
    b3 = make_batch_at(t0 + timedelta(milliseconds=400), 3)
    ev3 = make_event(t0 + timedelta(milliseconds=400), 3, person, zone_entry, missing_ppe)

    # Frame 4 (t0 + 600ms): 3rd missing PPE -> confirmed candidate! Posts PPE/MISSING!
    b4 = make_batch_at(t0 + timedelta(milliseconds=600), 4)
    ev4 = make_event(t0 + timedelta(milliseconds=600), 4, person, missing_ppe)

    # Frame 5 (t0 + 800ms): 4th missing PPE (cooldown) -> not posted!
    b5 = make_batch_at(t0 + timedelta(milliseconds=800), 5)
    ev5 = make_event(t0 + timedelta(milliseconds=800), 5, person, missing_ppe)

    frames = [(b1, ev1), (b2, ev2), (b3, ev3), (b4, ev4), (b5, ev5)]

    backend = AsyncMock()
    temporal_gate = TemporalPpeCandidateGate()

    for batch, event in frames:
        active_track_ids: tuple[int, ...] = ()
        ppe_observations: tuple[object, ...] = ()
        if event is not None:
            active_track_ids = tuple(
                obs.track_id for obs in event.observations if obs.type == "PERSON"
            )
            ppe_observations = tuple(obs for obs in event.observations if obs.type == "PPE")
        confirmed_candidates = temporal_gate.update(
            stream_id=batch.stream_id,
            session_id=batch.session_id,
            observed_at=batch.captured_at,
            active_track_ids=active_track_ids,
            observations=ppe_observations,
        )

        if event is not None:
            delivery_event = filter_event_for_delivery(event, confirmed_candidates)
            if delivery_event is not None:
                await backend.post_event(delivery_event)

    # Total backend posts must be exactly 2:
    # 1. Frame 3: ZONE_ENTRY (without leaking missing PPE)
    # 2. Frame 4: Confirmed PPE MISSING
    assert backend.post_event.call_count == 2

    # Inspect 1st call payload (Frame 3)
    first_call_event = backend.post_event.call_args_list[0][0][0]
    first_types = [o["type"] for o in first_call_event.to_wire_dict()["observations"]]
    assert "ZONE_ENTRY" in first_types
    assert not any(o["type"] == "PPE" for o in first_call_event.to_wire_dict()["observations"])

    # Inspect 2nd call payload (Frame 4)
    second_call_event = backend.post_event.call_args_list[1][0][0]
    second_types = [o["type"] for o in second_call_event.to_wire_dict()["observations"]]
    assert "PPE" in second_types
    assert any(
        o["type"] == "PPE" and o["status"] == "MISSING"
        for o in second_call_event.to_wire_dict()["observations"]
    )
