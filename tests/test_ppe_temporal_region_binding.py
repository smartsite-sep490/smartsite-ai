from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from smartsite_ai.domain.observations import PpeObservation, TechnicalObservationEvent
from smartsite_ai.pipelines.ppe_temporal import (
    ConfirmedPpeCandidate,
    TemporalPpeCandidateGate,
    filter_event_for_delivery,
    should_post_event,
)

STREAM_ID = "stream-01"
SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
OTHER_SESSION_ID = UUID("00000000-0000-4000-8000-000000000002")
REGION_A = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
REGION_B = "a12bc3d4-7dec-11d0-a765-00a0c91e6bf7"


def ppe_obs(
    track_id: int = 1,
    item: str = "HARD_HAT",
    status: str = "MISSING",
    *,
    region_id: str = REGION_A,
    geometry_version: int = 1,
) -> PpeObservation:
    return PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": track_id,
            "ppeItem": item,
            "status": status,
            "regionId": region_id,
            "geometryVersion": geometry_version,
            "confidence": 0.95,
        }
    )


def _make_event(
    *observations: object,
    session_id: UUID = SESSION_ID,
    captured_at: str | None = None,
) -> TechnicalObservationEvent:
    return TechnicalObservationEvent.create(
        event_id="00000000-0000-4000-8000-000000000010",
        camera_external_id="camera-01",
        stream_session_id=str(session_id),
        captured_at=captured_at or datetime(2026, 9, 21, 12, tzinfo=UTC).isoformat(),
        frame_dimensions={"width": 640, "height": 480},
        observations=observations,
    )


def test_missing_streak_in_region_a_does_not_confirm_first_missing_frame_in_region_b() -> None:
    gate = TemporalPpeCandidateGate(confirmation_frames=3, clear_frames=2)
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    def step(frame: int, *obs: PpeObservation) -> tuple[ConfirmedPpeCandidate, ...]:
        return gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * frame),
            active_track_ids=(1,),
            observations=obs,
        )

    assert step(0, ppe_obs(region_id=REGION_A, geometry_version=1)) == ()
    assert step(1, ppe_obs(region_id=REGION_A, geometry_version=1)) == ()
    # Frame 2 switches to REGION_B: must start REGION_B at 1, not confirm from REGION_A's 2 frames.
    assert step(2, ppe_obs(region_id=REGION_B, geometry_version=1)) == ()
    assert step(3, ppe_obs(region_id=REGION_B, geometry_version=1)) == ()

    confirmed = step(4, ppe_obs(region_id=REGION_B, geometry_version=1))
    assert len(confirmed) == 1
    assert confirmed[0] == ConfirmedPpeCandidate(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        track_id=1,
        ppe_item="HARD_HAT",
        region_id=REGION_B,
        geometry_version=1,
        first_seen_at=t0 + timedelta(milliseconds=400),
        confirmed_at=t0 + timedelta(milliseconds=800),
    )


def test_geometry_version_change_resets_region_bound_missing_streak() -> None:
    gate = TemporalPpeCandidateGate(confirmation_frames=3, clear_frames=2)
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    def step(frame: int, version: int) -> tuple[ConfirmedPpeCandidate, ...]:
        return gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * frame),
            active_track_ids=(1,),
            observations=(ppe_obs(region_id=REGION_A, geometry_version=version),),
        )

    assert step(0, 1) == ()
    assert step(1, 1) == ()
    # Same region_id but geometry_version changed to 2: must not confirm on 1st v2 frame.
    assert step(2, 2) == ()
    assert step(3, 2) == ()

    confirmed = step(4, 2)
    assert len(confirmed) == 1
    assert confirmed[0].region_id == REGION_A
    assert confirmed[0].geometry_version == 2
    assert confirmed[0].first_seen_at == t0 + timedelta(milliseconds=400)
    assert confirmed[0].confirmed_at == t0 + timedelta(milliseconds=800)


def test_overlapping_region_streaks_remain_independent_and_preserve_per_region_missing() -> None:
    gate = TemporalPpeCandidateGate(confirmation_frames=3, clear_frames=2)
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

    def step(frame: int, *obs: PpeObservation) -> tuple[ConfirmedPpeCandidate, ...]:
        return gate.update(
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            observed_at=t0 + timedelta(milliseconds=200 * frame),
            active_track_ids=(1,),
            observations=obs,
        )

    obs_a = ppe_obs(region_id=REGION_A, geometry_version=1)
    obs_b = ppe_obs(region_id=REGION_B, geometry_version=2)

    assert step(0, obs_a) == ()
    assert step(1, obs_a, obs_b) == ()

    # Frame 2: both missing -> REGION_A confirms (A=3), REGION_B stays pending (B=2)
    c_frame2 = step(2, obs_a, obs_b)
    assert len(c_frame2) == 1
    assert c_frame2[0].region_id == REGION_A
    assert c_frame2[0].geometry_version == 1
    assert c_frame2[0].first_seen_at == t0
    assert c_frame2[0].confirmed_at == t0 + timedelta(milliseconds=400)

    # Frame 3: REGION_A is PRESENT while REGION_B is MISSING (B=3).
    # REGION_A's PRESENT must not contradict or clear REGION_B's independent streak.
    c_frame3 = step(3, ppe_obs(status="PRESENT", region_id=REGION_A, geometry_version=1), obs_b)
    assert len(c_frame3) == 1
    assert c_frame3[0].region_id == REGION_B
    assert c_frame3[0].geometry_version == 2
    assert c_frame3[0].first_seen_at == t0 + timedelta(milliseconds=200)
    assert c_frame3[0].confirmed_at == t0 + timedelta(milliseconds=600)


def test_delivery_filter_binds_candidate_to_exact_region_geometry_session_and_capture() -> None:
    t0 = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
    gate = TemporalPpeCandidateGate(confirmation_frames=1)
    confirmed_a = gate.update(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        observed_at=t0,
        active_track_ids=(1,),
        observations=(ppe_obs(region_id=REGION_A, geometry_version=1),),
    )
    assert len(confirmed_a) == 1

    obs_a_v1 = ppe_obs(region_id=REGION_A, geometry_version=1)
    obs_b_v1 = ppe_obs(region_id=REGION_B, geometry_version=1)
    obs_a_v2 = ppe_obs(region_id=REGION_A, geometry_version=2)

    # Candidate from (REGION_A, v1) must NOT authorize REGION_B or changed geometryVersion.
    assert filter_event_for_delivery(_make_event(obs_b_v1), confirmed_a) is None
    assert not should_post_event(_make_event(obs_b_v1), confirmed_a)
    assert filter_event_for_delivery(_make_event(obs_a_v2), confirmed_a) is None
    assert not should_post_event(_make_event(obs_a_v2), confirmed_a)

    # Mixed event: only the exact (REGION_A, v1) observation may be delivered.
    delivered = filter_event_for_delivery(_make_event(obs_a_v1, obs_b_v1, obs_a_v2), confirmed_a)
    assert delivered is not None
    assert delivered.observations == [obs_a_v1]

    # Direct constructor with required region_id/geometry_version preserves session/capture binding.
    direct_candidate = ConfirmedPpeCandidate(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        track_id=1,
        ppe_item="HARD_HAT",
        region_id=REGION_A,
        geometry_version=1,
        first_seen_at=t0 - timedelta(milliseconds=200),
        confirmed_at=t0,
    )
    other_session_event = _make_event(obs_a_v1, session_id=OTHER_SESSION_ID)
    assert filter_event_for_delivery(other_session_event, (direct_candidate,)) is None
    stale_candidate = ConfirmedPpeCandidate(
        stream_id=STREAM_ID,
        session_id=SESSION_ID,
        track_id=1,
        ppe_item="HARD_HAT",
        region_id=REGION_A,
        geometry_version=1,
        first_seen_at=t0 - timedelta(milliseconds=400),
        confirmed_at=t0 - timedelta(microseconds=1),
    )
    assert filter_event_for_delivery(_make_event(obs_a_v1), (stale_candidate,)) is None
    with pytest.raises(TypeError):
        ConfirmedPpeCandidate(  # type: ignore[call-arg]
            stream_id=STREAM_ID,
            session_id=SESSION_ID,
            track_id=1,
            ppe_item="HARD_HAT",
            first_seen_at=t0,
            confirmed_at=t0,
        )
