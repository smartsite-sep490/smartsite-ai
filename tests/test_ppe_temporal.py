from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from smartsite_ai.domain.observations import (
    BoundingBox,
    PersonObservation,
    PpeObservation,
    TechnicalObservationEvent,
    ZoneEntryObservation,
)
from smartsite_ai.pipelines.ppe_temporal import (
    ConfirmedPpeCandidate,
    TemporalPpeCandidateGate,
    filter_event_for_delivery,
    should_post_event,
)

STREAM_ID = "stream-01"
SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
REGION_ID = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"


def ppe_obs(
    track_id: int,
    item: str = "HARD_HAT",
    status: str = "MISSING",
) -> PpeObservation:
    return PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": track_id,
            "ppeItem": item,
            "status": status,
            "regionId": REGION_ID,
            "geometryVersion": 1,
            "confidence": 0.95,
        }
    )


def test_confirmation_at_third_consecutive_missing_observation() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Frame 1: missing
    c1 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c1 == ()

    # Frame 2: missing
    c2 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c2 == ()

    # Frame 3: missing -> confirmed!
    t_confirm = t0 + timedelta(milliseconds=400)
    c3 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(c3) == 1
    assert c3[0] == ConfirmedPpeCandidate(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        track_id=1,
        ppe_item="HARD_HAT",
        first_seen_at=t0,
        confirmed_at=t_confirm,
    )


def test_cooldown_suppresses_repeat_candidates_for_ten_seconds() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    for i in range(3):
        res = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
    assert len(res) == 1

    # Frame 4..10 within 10s cooldown: still missing, but no new candidates
    for i in range(3, 10):
        c = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(seconds=i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
        assert c == ()

    # Clear state with 2 non-missing frames at t0 + 11s
    t_clear1 = t0 + timedelta(seconds=11)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_clear1,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    t_clear2 = t0 + timedelta(seconds=11, milliseconds=200)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_clear2,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # Now past cooldown (> 10s since confirmation at t0 + 0.4s).
    # 3 new consecutive missing frames trigger a new candidate:
    t_new = t0 + timedelta(seconds=12)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_new,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_new + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    c_new = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_new + timedelta(milliseconds=400),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(c_new) == 1
    assert c_new[0].first_seen_at == t_new


def test_exact_ten_second_cooldown_boundary() -> None:
    gate = TemporalPpeCandidateGate(
        confirmation_frames=3,
        clear_frames=2,
        cooldown=timedelta(seconds=10),
    )
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Confirm candidate at t0 + 400ms
    for i in range(3):
        res = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
    assert len(res) == 1
    t_confirm = res[0].confirmed_at
    assert t_confirm == t0 + timedelta(milliseconds=400)

    # Clear state at t_confirm + 5s
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=5),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=5, milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # Try 3 missing frames that end at t_confirm + 9.99s (< 10s cooldown)
    t_pre = t_confirm + timedelta(seconds=9, milliseconds=590)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_pre,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_pre + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    res_under = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=9, milliseconds=990),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert res_under == ()

    # Clear again
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=9, milliseconds=991),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=9, milliseconds=992),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # Frame 3 lands on exactly t_confirm + 10.0s (exact cooldown boundary)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=9, milliseconds=994),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=9, milliseconds=997),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    res_exact = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm + timedelta(seconds=10),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(res_exact) == 1
    assert res_exact[0].confirmed_at == t_confirm + timedelta(seconds=10)


def test_reset_after_two_consecutive_non_missing_frames() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # 2 missing frames
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )

    # 2 non-missing frames (reset!)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=400),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=600),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # 1 missing frame -> not confirmed yet (missing count restarted)
    c = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=800),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c == ()


def test_absent_item_for_active_track_counts_as_clear() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # 2 missing frames
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )

    # Track 1 is active, but HARD_HAT is omitted/absent -> counts as clear frames
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=400),
        active_track_ids=(1,),
        observations=(),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=600),
        active_track_ids=(1,),
        observations=(),
    )

    # Now 1 missing frame should not trigger confirmation
    c = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=800),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c == ()


def test_track_expiry_exact_and_over_one_second_boundary() -> None:
    # 1. Exact 1.0s boundary: not expired (<= 1s)
    gate_exact = TemporalPpeCandidateGate(track_expiry=timedelta(seconds=1))
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    gate_exact.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    # Track 1 seen at t0 + 200ms (count = 2)
    gate_exact.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )

    # Intermediate batch with empty track at t0 + 800ms
    gate_exact.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=800),
        active_track_ids=(),
        observations=(),
    )
    # Track 1 reappears at exactly t0 + 1200ms (delta from last seen t0+200ms is exactly 1.000s)
    # Since delta is <= 1.0s, state is retained, so 3rd consecutive missing confirms!
    c_exact = gate_exact.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(seconds=1, milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(c_exact) == 1
    assert c_exact[0].confirmed_at == t0 + timedelta(seconds=1, milliseconds=200)

    # 2. Over 1.0s boundary: expired (> 1s)
    gate_over = TemporalPpeCandidateGate(track_expiry=timedelta(seconds=1))
    gate_over.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate_over.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )

    # Intermediate batch with empty track at t0 + 800ms
    gate_over.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=800),
        active_track_ids=(),
        observations=(),
    )
    # Track 1 reappears at t0 + 1205ms (delta is 1.005s > 1.0s) -> expired!
    # Missing count restarts at 1, so no confirmation here.
    c_over = gate_over.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(seconds=1, milliseconds=205),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c_over == ()


def test_reappearance_expiry_without_intermediate_frames() -> None:
    gate = TemporalPpeCandidateGate(track_expiry=timedelta(seconds=1))
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Frame 1: missing at t0
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    # Frame 2: missing at t0 + 200ms (missing count = 2)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )

    # NO intermediate frame updates at all! Track disappears and reappears at
    # t0 + 2.0s (> 1s expiry).
    # Since observed_at - last_seen_at > track_expiry (1.8s > 1.0s), the old state must expire.
    # Therefore, this frame must NOT confirm candidate; it starts count at 1.
    c1 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(seconds=2),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c1 == ()

    # Frame at t0 + 2.2s: missing (count = 2)
    c2 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(seconds=2, milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c2 == ()

    # Frame at t0 + 2.4s: missing (count = 3 -> confirmed!)
    c3 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(seconds=2, milliseconds=400),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(c3) == 1
    assert c3[0].first_seen_at == t0 + timedelta(seconds=2)


def test_cooldown_survives_reappearance_expiry() -> None:
    gate = TemporalPpeCandidateGate(
        cooldown=timedelta(seconds=10),
        track_expiry=timedelta(seconds=1),
    )
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # 3 missing frames -> confirmed at t0 + 400ms
    for i in range(3):
        res = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
    assert len(res) == 1

    # NO intermediate frame updates. Track reappears at t0 + 3.0s (> 1s track expiry).
    # Cooldown of 10s must survive this reappearance expiry!
    for i in range(3):
        res = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(seconds=3, milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
    # 3 consecutive missing frames after reappearance, but still within 10s cooldown -> suppressed!
    assert res == ()


def test_stream_and_session_independence() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    session_a = UUID("00000000-0000-4000-8000-000000000001")
    session_b = UUID("00000000-0000-4000-8000-000000000002")

    # 2 missing frames on stream-01
    gate.update(
        stream_id="stream-01",
        session_id=session_a,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    gate.update(
        stream_id="stream-01",
        session_id=session_a,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )

    # 1 missing frame on stream-02 (same track 1)
    res_b = gate.update(
        stream_id="stream-02",
        session_id=session_a,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert res_b == ()

    # 1 missing frame on stream-01 with session_b (same track 1)
    res_c = gate.update(
        stream_id="stream-01",
        session_id=session_b,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert res_c == ()

    # 3rd missing frame on stream-01 session_a -> confirms stream-01 session_a only
    res_a = gate.update(
        stream_id="stream-01",
        session_id=session_a,
        observed_at=t0 + timedelta(milliseconds=400),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(res_a) == 1
    assert res_a[0].stream_id == "stream-01"
    assert res_a[0].session_id == session_a


def test_non_monotonic_timestamp_rejected() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0,  # duplicate timestamp
            active_track_ids=(1,),
            observations=(),
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 - timedelta(seconds=1),  # older timestamp
            active_track_ids=(1,),
            observations=(),
        )


def test_independent_items_and_tracks() -> None:
    gate = TemporalPpeCandidateGate()
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Track 1 missing Hardhat 3 times, Vest present
    # Track 2 missing Vest 3 times, Hardhat present
    for i in range(3):
        c = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * i),
            active_track_ids=(1, 2),
            observations=(
                ppe_obs(1, "HARD_HAT", "MISSING"),
                ppe_obs(1, "SAFETY_VEST", "PRESENT"),
                ppe_obs(2, "HARD_HAT", "PRESENT"),
                ppe_obs(2, "SAFETY_VEST", "MISSING"),
            ),
        )
    assert len(c) == 2
    assert {(item.track_id, item.ppe_item) for item in c} == {
        (1, "HARD_HAT"),
        (2, "SAFETY_VEST"),
    }


def _make_event(*observations: object) -> TechnicalObservationEvent:
    return TechnicalObservationEvent.create(
        event_id="00000000-0000-4000-8000-000000000010",
        camera_external_id="camera-01",
        stream_session_id=str(SESSION_ID),
        captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC).isoformat(),
        frame_dimensions={"width": 640, "height": 480},
        observations=observations,
    )


def test_filter_event_for_delivery_mixed_event_without_leaking_raw_missing() -> None:
    person1 = PersonObservation.model_validate(
        {
            "type": "PERSON",
            "trackId": 1,
            "confidence": 0.9,
            "boundingBox": BoundingBox.model_validate(
                {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.9, "coordinateSpace": "NORMALIZED_0_1"}
            ).to_wire_dict(),
        }
    )
    person2 = PersonObservation.model_validate(
        {
            "type": "PERSON",
            "trackId": 2,
            "confidence": 0.9,
            "boundingBox": BoundingBox.model_validate(
                {"x1": 0.5, "y1": 0.1, "x2": 0.8, "y2": 0.9, "coordinateSpace": "NORMALIZED_0_1"}
            ).to_wire_dict(),
        }
    )
    zone_entry = ZoneEntryObservation.model_validate(
        {
            "type": "ZONE_ENTRY",
            "trackId": 1,
            "regionId": REGION_ID,
            "geometryVersion": 1,
            "confidence": 0.9,
        }
    )
    raw_missing_ppe = ppe_obs(1, "HARD_HAT", "MISSING")
    present_ppe = ppe_obs(2, "SAFETY_VEST", "PRESENT")

    mixed_event = _make_event(person1, person2, zone_entry, raw_missing_ppe, present_ppe)

    # Case 1: No confirmed candidates.
    # Zone entry posts immediately, but raw unconfirmed missing PPE MUST be excluded.
    delivered = filter_event_for_delivery(mixed_event, ())
    assert delivered is not None
    assert should_post_event(mixed_event, ()) is True

    delivered_obs = delivered.to_wire_dict()["observations"]
    obs_types = [o["type"] for o in delivered_obs]
    assert "ZONE_ENTRY" in obs_types
    assert "PERSON" in obs_types
    # Present PPE is retained
    assert any(o["type"] == "PPE" and o["status"] == "PRESENT" for o in delivered_obs)
    # Raw missing PPE MUST NOT be present!
    assert not any(o["type"] == "PPE" and o["status"] == "MISSING" for o in delivered_obs)

    # Case 2: Track 1 HARD_HAT candidate is confirmed.
    candidate = ConfirmedPpeCandidate(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        track_id=1,
        ppe_item="HARD_HAT",
        first_seen_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        confirmed_at=datetime(2026, 9, 21, 12, 0, 1, tzinfo=UTC),
    )
    delivered_confirmed = filter_event_for_delivery(mixed_event, (candidate,))
    assert delivered_confirmed is not None
    assert should_post_event(mixed_event, (candidate,)) is True

    delivered_obs_confirmed = delivered_confirmed.to_wire_dict()["observations"]
    # Now confirmed missing PPE is included
    assert any(
        o["type"] == "PPE" and o["status"] == "MISSING" and o["trackId"] == 1
        for o in delivered_obs_confirmed
    )


def test_filter_event_for_delivery_returns_none_when_no_deliverable_trigger() -> None:
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
    unconfirmed_missing = ppe_obs(1, "HARD_HAT", "MISSING")
    present_ppe = ppe_obs(1, "HARD_HAT", "PRESENT")

    # Person only -> None
    assert filter_event_for_delivery(_make_event(person), ()) is None
    assert not should_post_event(_make_event(person), ())

    # Person + PRESENT only -> None (nothing to alert backend)
    assert filter_event_for_delivery(_make_event(person, present_ppe), ()) is None
    assert not should_post_event(_make_event(person, present_ppe), ())

    # Person + unconfirmed MISSING -> None (suppressed)
    assert filter_event_for_delivery(_make_event(person, unconfirmed_missing), ()) is None
    assert not should_post_event(_make_event(person, unconfirmed_missing), ())


def test_non_missing_frame_resets_pending_missing_streak() -> None:
    """A non-missing frame before confirmation resets consecutive missing streak immediately."""
    gate = TemporalPpeCandidateGate(confirmation_frames=3, clear_frames=2)
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Frame 1: MISSING (streak = 1)
    c1 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c1 == ()

    # Frame 2: MISSING (streak = 2)
    c2 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c2 == ()

    # Frame 3: PRESENT (breaks the pending missing streak immediately)
    c3 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=400),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    assert c3 == ()

    # Frame 4: MISSING (streak restarts at 1, must NOT emit candidate!)
    c4 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=600),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c4 == ()


def test_clear_hysteresis_after_confirmation() -> None:
    """After confirmation, 1 clear frame does not reset confirmed state, but 2 clear frames do."""
    gate = TemporalPpeCandidateGate(
        confirmation_frames=3,
        clear_frames=2,
        cooldown=timedelta(seconds=10),
    )
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Frames 1..3: MISSING -> confirmed!
    for i in range(3):
        res = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
    assert len(res) == 1
    t_confirm = res[0].confirmed_at

    # Frame 4: 1 PRESENT frame (hysteresis: does not yet end confirmed episode)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=800),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # Frame 5: 1 MISSING frame (remains in the same confirmed episode, no new candidate)
    res5 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=1000),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert res5 == ()

    # Clear episode with 2 consecutive PRESENT frames at t0 + 11s (> 10s cooldown)
    t_clear = t_confirm + timedelta(seconds=11)
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_clear,
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_clear + timedelta(milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # 3 new consecutive missing frames now start a new episode and emit candidate
    t_new = t_clear + timedelta(seconds=1)
    for i in range(2):
        c = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t_new + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
        assert c == ()

    c_new = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_new + timedelta(milliseconds=400),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(c_new) == 1


def test_duplicate_active_track_ids_normalized_to_single_count() -> None:
    """Duplicate track ids in active_track_ids advance the counter only once per frame."""
    gate = TemporalPpeCandidateGate(confirmation_frames=3)
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # Frame 1: active_track_ids=(1, 1) -> must only advance by 1
    c1 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1, 1),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c1 == ()

    # Frame 2: active_track_ids=(1, 1) -> must only advance to 2, NOT jump to 4 or confirm
    c2 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=200),
        active_track_ids=(1, 1),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert c2 == ()

    # Frame 3: active_track_ids=(1, 1) -> 3rd frame, confirms now!
    c3 = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=400),
        active_track_ids=(1, 1),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(c3) == 1


def test_delivery_filtering_rejects_candidate_from_different_session() -> None:
    """A candidate from a different session cannot authorize a raw MISSING observation."""
    session_a = UUID("00000000-0000-4000-8000-000000000001")
    session_b = UUID("00000000-0000-4000-8000-000000000002")

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
    missing_ppe = ppe_obs(1, "HARD_HAT", "MISSING")

    # Event belongs to session A
    event_session_a = TechnicalObservationEvent.create(
        event_id="00000000-0000-4000-8000-000000000010",
        camera_external_id="camera-01",
        stream_session_id=str(session_a),
        captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC).isoformat(),
        frame_dimensions={"width": 640, "height": 480},
        observations=[person, missing_ppe],
    )

    # Candidate was confirmed on session B
    candidate_session_b = ConfirmedPpeCandidate(
        stream_id=STREAM_ID,
        session_id=session_b,
        track_id=1,
        ppe_item="HARD_HAT",
        first_seen_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        confirmed_at=datetime(2026, 9, 21, 12, 0, 1, tzinfo=UTC),
    )

    # Must NOT deliver missing PPE authorized by foreign session
    assert filter_event_for_delivery(event_session_a, (candidate_session_b,)) is None
    assert not should_post_event(event_session_a, (candidate_session_b,))


def test_cooldown_blocked_new_episode_emits_at_cooldown_expiry_if_missing_persists() -> None:
    """When a new episode reaches 3 missing inside cooldown and persists, emit upon expiry."""
    gate = TemporalPpeCandidateGate(
        confirmation_frames=3,
        clear_frames=2,
        cooldown=timedelta(seconds=10),
    )
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    # 1. Episode 1: confirmed at t0 + 0.4s
    for i in range(3):
        res1 = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
    assert len(res1) == 1
    t_confirm1 = res1[0].confirmed_at
    assert t_confirm1 == t0 + timedelta(milliseconds=400)

    # 2. Episode 1 clears with 2 PRESENT frames
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=800),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )
    gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0 + timedelta(milliseconds=1000),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "PRESENT"),),
    )

    # 3. New episode starts at t0 + 5.0s, reaches 3 missing at t0 + 5.4s (inside 10s cooldown)
    t_ep2_start = t0 + timedelta(seconds=5)
    for i in range(3):
        res = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t_ep2_start + timedelta(milliseconds=200 * i),
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
        # Inside cooldown: must NOT emit candidate yet!
        assert res == ()

    # 4. Continuous MISSING frames up to t_confirm1 + 9.9s (still < 10.0s cooldown)
    # Frames arrive every 500ms so track does not expire
    current_t = t_ep2_start + timedelta(milliseconds=600)
    while current_t <= t_confirm1 + timedelta(seconds=9, milliseconds=900):
        res_pre = gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=current_t,
            active_track_ids=(1,),
            observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
        )
        assert res_pre == ()
        current_t += timedelta(milliseconds=500)

    # 5. Exactly at cooldown expiry: t_confirm1 + 10.0s (t0 + 10.4s), missing continues!
    # Latch was not set, so candidate is emitted NOW!
    res_expiry = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm1 + timedelta(seconds=10),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert len(res_expiry) == 1
    assert res_expiry[0].first_seen_at == t_ep2_start
    assert res_expiry[0].confirmed_at == t_confirm1 + timedelta(seconds=10)

    # 6. Subsequent missing frames in this confirmed episode do not emit again
    res_after = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t_confirm1 + timedelta(seconds=10, milliseconds=200),
        active_track_ids=(1,),
        observations=(ppe_obs(1, "HARD_HAT", "MISSING"),),
    )
    assert res_after == ()
