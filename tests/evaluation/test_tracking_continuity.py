"""Continuity failures counted against human-reviewed anonymous ground truth."""

import pytest

from smartsite_ai.evaluation.tracking_continuity import (
    TrackingMatch,
    evaluate_tracking_continuity,
)


def row(
    frame: int,
    person: str = "person-a",
    track: str | None = "1",
    *,
    camera: str = "camera-a",
    clip: str = "clip-a",
    session: str = "session-a",
) -> TrackingMatch:
    return TrackingMatch(camera, clip, session, frame, person, track)


def test_crossing_counts_both_id_switches_and_track_subject_changes() -> None:
    report = evaluate_tracking_continuity(
        [row(0, "a", "1"), row(0, "b", "2"), row(1, "a", "2"), row(1, "b", "1")]
    )
    assert report.identity_switches == 2
    assert report.track_subject_changes == 2
    assert report.distinct_ground_truth_subjects == 2
    assert report.distinct_predicted_tracks == 2


def test_unchanged_track_changes_physical_subject_without_an_id_switch() -> None:
    report = evaluate_tracking_continuity([row(0, "a"), row(1, "b")])
    assert report.track_subject_changes == 1
    assert report.identity_switches == 0


def test_explicit_miss_preserves_previous_match_and_counts_fragment_once() -> None:
    report = evaluate_tracking_continuity(
        [row(0), row(1, track=None), row(2, track=None), row(3, track="2")]
    )
    assert report.fragments_after_miss == 1
    assert report.identity_switches == 1
    assert report.matched_samples == report.unmatched_samples == 2


def test_leading_misses_and_unannotated_frame_gaps_are_not_fragments() -> None:
    report = evaluate_tracking_continuity([row(0, track=None), row(1, track=None), row(2), row(8)])
    assert report.fragments_after_miss == 0
    assert report.identity_switches == 0


def test_session_restart_is_reported_separately_from_switches_and_fragments() -> None:
    report = evaluate_tracking_continuity(
        [row(0, track="7"), row(1, track=None), row(2, track="1", session="session-b")]
    )
    assert report.identity_switches == 0
    assert report.fragments_after_miss == 0
    assert report.session_boundary_continuations == 1
    assert report.distinct_predicted_tracks == 2


def test_camera_and_clip_scope_prevent_accidental_cross_camera_correlations() -> None:
    report = evaluate_tracking_continuity(
        [row(0, "a", "1"), row(1, "b", "1", camera="camera-b"), row(2, clip="clip-b")]
    )
    assert report.identity_switches == report.track_subject_changes == 0
    assert report.session_boundary_continuations == 0
    assert report.distinct_ground_truth_subjects == 3
    assert report.distinct_predicted_tracks == 3


def test_shuffled_inputs_have_identical_results() -> None:
    rows = [row(0), row(1, track=None), row(2, track="2"), row(3, session="session-b")]
    assert evaluate_tracking_continuity(rows) == evaluate_tracking_continuity(rows[::-1])


@pytest.mark.parametrize(
    "rows",
    [
        [row(0), row(0)],
        [row(0), row(0, track="2")],
        [row(0, "a"), row(0, "b")],
        [row(0), row(0, session="session-b")],
        [row(0), row(1, session="session-b"), row(2)],
    ],
)
def test_conflicting_matches_or_overlapping_session_timelines_are_rejected(
    rows: list[TrackingMatch],
) -> None:
    with pytest.raises(ValueError):
        evaluate_tracking_continuity(rows)


@pytest.mark.parametrize("frame", [-1, True, 1.0, 2**53])
def test_frame_index_is_a_bounded_strict_integer(frame: object) -> None:
    with pytest.raises(ValueError):
        TrackingMatch("cam", "clip", "session", frame, "person", "track")  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["camera_id", "clip_id", "session_id", "person_key", "track_id"])
@pytest.mark.parametrize("value", ["", " padded", "padded ", "x" * 129, 7, True])
def test_identifiers_are_opaque_bounded_unpadded_strings(field: str, value: object) -> None:
    values = {
        "camera_id": "cam",
        "clip_id": "clip",
        "session_id": "session",
        "frame_index": 0,
        "person_key": "person",
        "track_id": "track",
    }
    values[field] = value
    with pytest.raises(ValueError):
        TrackingMatch(**values)  # type: ignore[arg-type]


def test_report_exposes_counts_with_an_explicit_diagnostic_scope() -> None:
    assert evaluate_tracking_continuity([]).to_dict() == {
        "scope": "MATCHED_LEDGER_DIAGNOSTICS",
        "samples": 0,
        "matchedSamples": 0,
        "unmatchedSamples": 0,
        "distinctGroundTruthSubjects": 0,
        "distinctPredictedTracks": 0,
        "identitySwitches": 0,
        "trackSubjectChanges": 0,
        "fragmentsAfterMiss": 0,
        "sessionBoundaryContinuations": 0,
    }


def test_sample_bound_is_enforced_before_processing() -> None:
    with pytest.raises(ValueError, match="sample limit"):
        evaluate_tracking_continuity([row(0)] * 20_001)


def test_error_does_not_echo_raw_identifiers() -> None:
    sensitive = "secret-like-test-value"
    with pytest.raises(ValueError) as error:
        evaluate_tracking_continuity([row(0, sensitive), row(0, sensitive)])
    assert sensitive not in str(error.value)
