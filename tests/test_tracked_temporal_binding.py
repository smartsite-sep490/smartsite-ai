from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from test_tracked_encoder import frame

from smartsite_ai.domain.observations import PpeObservation
from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate, tracked_delivery_frame

NOW = datetime(2026, 11, 17, tzinfo=UTC)
SESSION = UUID("00000000-0000-4000-8000-000000000002")


def regional(region=1, version=1):
    return PpeObservation(
        type="PPE",
        trackId=1,
        regionId=f"00000000-0000-4000-8000-{region:012d}",
        geometryVersion=version,
        ppeItem="HARD_HAT",
        status="MISSING",
    )


def step(gate, index, *observations, version=None):
    return gate.update(
        stream_id="camera-01",
        session_id=SESSION,
        observed_at=NOW + timedelta(milliseconds=index * 100),
        active_track_ids=(1,),
        observations=observations,
        camera_configuration_version=version,
    )


@pytest.mark.parametrize("change", ["region", "geometry"])
def test_region_or_geometry_change_never_borrows_previous_confirmation(change):
    gate = TemporalPpeCandidateGate(confirmation_frames=2)
    assert step(gate, 0, regional()) == ()
    observation = regional(region=2) if change == "region" else regional(version=2)
    assert step(gate, 1, observation) == ()
    confirmed = step(gate, 2, observation)
    assert len(confirmed) == 1
    assert confirmed[0].region_id == observation.region_id
    assert confirmed[0].geometry_version == observation.geometry_version


def test_camera_configuration_change_starts_new_scope_confirmation():
    ppe = frame().observations[1]
    gate = TemporalPpeCandidateGate(confirmation_frames=2)
    assert step(gate, 0, ppe, version=1) == ()
    assert step(gate, 1, ppe, version=2) == ()
    confirmed = step(gate, 2, ppe, version=2)
    assert len(confirmed) == 1 and confirmed[0].region_id is None
    assert confirmed[0].geometry_version == 2


@pytest.mark.parametrize(
    "statuses", [("MISSING", "PRESENT", "MISSING"), ("PRESENT", "MISSING", "PRESENT")]
)
def test_conflicting_same_frame_is_unknown_regardless_of_last_duplicate(statuses):
    template = frame().observations[1]
    observations = tuple(template.model_copy(update={"status": status}) for status in statuses)
    gate = TemporalPpeCandidateGate(confirmation_frames=1)
    assert step(gate, 0, *observations, version=1) == ()


@pytest.mark.parametrize("mismatch", ["time", "session", "configuration", "unbound", "nanosecond"])
def test_tracked_delivery_requires_exact_current_frame_candidate(mismatch):
    current = replace(frame(), captured_at=NOW.isoformat(), camera_configuration_version=1)
    gate = TemporalPpeCandidateGate(confirmation_frames=1)
    confirmed = step(gate, 0, current.observations[1], version=1)
    assert len(confirmed) == 1
    assert tracked_delivery_frame(current, confirmed).observations == current.observations
    if mismatch == "time":
        current = replace(current, captured_at=(NOW + timedelta(microseconds=1)).isoformat())
    elif mismatch == "session":
        current = replace(current, stream_session_id="00000000-0000-4000-8000-000000000077")
    elif mismatch == "configuration":
        current = replace(current, camera_configuration_version=2)
    elif mismatch == "unbound":
        confirmed = (replace(confirmed[0], geometry_version=None),)
    else:
        current = replace(current, captured_at="2026-11-17T00:00:00.000000001Z")
    selected = tracked_delivery_frame(current, confirmed)
    assert [item.type for item in selected.observations] == ["PERSON"]
    assert selected.subject_segments == current.subject_segments
