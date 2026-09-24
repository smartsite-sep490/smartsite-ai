from datetime import timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.alert_metrics import (
    EpisodeMetricsReport,
    PredictedPpeEpisode,
    match_candidate_episodes,
)
from smartsite_ai.evaluation.models import GroundTruthPpeEpisode

SESSION_A = UUID("00000000-0000-0000-0000-000000000001")
SESSION_B = UUID("00000000-0000-0000-0000-000000000002")


def _ground_truth(
    clip_id: str,
    person_id: int | str,
    ppe_item: str,
    start: float,
    end: float,
) -> GroundTruthPpeEpisode:
    return GroundTruthPpeEpisode(
        clip_id=clip_id,
        person_instance_id=person_id,
        ppe_item=ppe_item,
        start_time_seconds=start,
        end_time_seconds=end,
    )


def _prediction(
    candidate_id: str,
    clip_id: str,
    person_id: int | str,
    ppe_item: str,
    start: float,
    end: float,
    *,
    track_id: int = 1,
    session_id: UUID = SESSION_A,
    confirmed: float | None = None,
) -> PredictedPpeEpisode:
    return PredictedPpeEpisode(
        candidate_id=candidate_id,
        clip_id=clip_id,
        person_instance_id=person_id,
        ppe_item=ppe_item,
        start_time_seconds=start,
        end_time_seconds=end,
        confirmed_time_seconds=start if confirmed is None else confirmed,
        stream_session_id=session_id,
        track_id=track_id,
    )


def test_matches_one_to_one_by_person_item_and_temporal_overlap() -> None:
    report = match_candidate_episodes(
        (
            _ground_truth("clip", 1, "HARD_HAT", 1.0, 3.0),
            _ground_truth("clip", 2, "SAFETY_VEST", 5.0, 6.0),
        ),
        (
            _prediction("p1", "clip", 1, "HARD_HAT", 1.5, 2.5),
            _prediction("wrong-person", "clip", 9, "SAFETY_VEST", 5.0, 6.0),
            _prediction("wrong-item", "clip", 2, "HARD_HAT", 5.0, 6.0),
        ),
        onset_tolerance=timedelta(milliseconds=500),
        evaluated_seconds=60.0,
    )

    assert [(item.candidate_id, item.overlap_seconds) for item in report.matches] == [("p1", 1.0)]
    assert report.true_positives == 1
    assert report.false_positives == 2
    assert report.false_negatives == 1
    assert report.precision.value == pytest.approx(1 / 3)
    assert report.recall.value == 0.5
    assert report.f1.value == 0.4
    assert [item.candidate_id for item in report.false_candidates] == [
        "wrong-item",
        "wrong-person",
    ]
    assert len(report.missed_episodes) == 1
    assert report.false_candidates_per_camera_hour.value == 120.0
    assert report.rate_gate_eligible is False


def test_onset_tolerance_boundary_matches_without_overlap_and_just_beyond_does_not() -> None:
    ground_truth = (_ground_truth("clip", 1, "HARD_HAT", 2.0, 3.0),)

    at_boundary = match_candidate_episodes(
        ground_truth,
        (_prediction("at", "clip", 1, "HARD_HAT", 1.5, 1.6),),
        onset_tolerance=timedelta(milliseconds=500),
    )
    beyond = match_candidate_episodes(
        ground_truth,
        (_prediction("beyond", "clip", 1, "HARD_HAT", 1.499, 1.6),),
        onset_tolerance=timedelta(milliseconds=500),
    )

    assert [item.candidate_id for item in at_boundary.matches] == ["at"]
    assert beyond.true_positives == 0
    assert beyond.false_positives == 1
    assert beyond.false_negatives == 1


def test_prefers_overlap_then_onset_difference_then_stable_ids() -> None:
    report = match_candidate_episodes(
        (
            _ground_truth("clip", 1, "HARD_HAT", 1.0, 4.0),
            _ground_truth("clip", 1, "HARD_HAT", 6.0, 8.0),
        ),
        (
            _prediction("lower-overlap", "clip", 1, "HARD_HAT", 1.0, 2.0),
            _prediction("higher-overlap", "clip", 1, "HARD_HAT", 1.2, 3.8),
            _prediction("tie-b", "clip", 1, "HARD_HAT", 6.0, 7.0, track_id=2),
            _prediction("tie-a", "clip", 1, "HARD_HAT", 6.0, 7.0, track_id=3),
        ),
        onset_tolerance=timedelta(milliseconds=500),
    )

    assert [item.candidate_id for item in report.matches] == ["higher-overlap", "tie-a"]


def test_duplicate_candidate_and_track_replacement_are_diagnostics_and_false_positives() -> None:
    report = match_candidate_episodes(
        (_ground_truth("clip", "worker-1", "HARD_HAT", 1.0, 5.0),),
        (
            _prediction("primary", "clip", "worker-1", "HARD_HAT", 1.0, 3.0, track_id=10),
            _prediction("same-track", "clip", "worker-1", "HARD_HAT", 3.2, 4.0, track_id=10),
            _prediction("new-track", "clip", "worker-1", "HARD_HAT", 4.1, 5.0, track_id=11),
        ),
        onset_tolerance=timedelta(milliseconds=500),
    )

    assert report.true_positives == 1
    assert report.false_positives == 2
    assert report.duplicate_count == 2
    assert report.duplicate_candidate_ids == ("new-track", "same-track")
    assert report.fragmentation_count == 1
    assert report.raw_false_candidate_count == report.false_positives
    assert [item.candidate_id for item in report.false_candidates] == [
        "new-track",
        "same-track",
    ]


def test_zero_duration_and_empty_inputs_emit_explicit_undefined_rates() -> None:
    report = match_candidate_episodes(
        (),
        (),
        onset_tolerance=timedelta(milliseconds=500),
        evaluated_seconds=0.0,
    )

    assert report.precision.value is None
    assert report.precision.reason == "no predicted candidate episodes"
    assert report.recall.value is None
    assert report.recall.reason == "no ground-truth candidate episodes"
    assert report.f1.value is None
    assert report.false_candidates_per_camera_hour.value is None
    assert report.false_candidates_per_camera_hour.reason == "evaluated duration is zero"
    assert report.rate_gate_eligible is False
    assert report.rate_gate_ineligibility_reason == (
        "requires at least 1800 evaluated camera-seconds"
    )


@pytest.mark.parametrize(
    ("evaluated_seconds", "eligible"),
    [(1799.999, False), (1800.0, True)],
)
def test_rate_gate_eligibility_boundary(evaluated_seconds: float, eligible: bool) -> None:
    report = match_candidate_episodes(
        (),
        (),
        onset_tolerance=timedelta(milliseconds=500),
        evaluated_seconds=evaluated_seconds,
    )

    assert report.rate_gate_eligible is eligible
    assert (report.rate_gate_ineligibility_reason is None) is eligible


def test_track_identity_includes_stream_session() -> None:
    report = match_candidate_episodes(
        (_ground_truth("clip", 1, "HARD_HAT", 1.0, 5.0),),
        (
            _prediction("session-a", "clip", 1, "HARD_HAT", 1.0, 3.0, track_id=7),
            _prediction(
                "session-b",
                "clip",
                1,
                "HARD_HAT",
                3.1,
                5.0,
                track_id=7,
                session_id=SESSION_B,
            ),
        ),
        onset_tolerance=timedelta(milliseconds=500),
    )

    assert report.fragmentation_count == 1
    assert len(report.fragmentation_diagnostics) == 1
    assert report.fragmentation_diagnostics[0].track_identities == (
        (str(SESSION_A), 7),
        (str(SESSION_B), 7),
    )


def test_rejects_invalid_inputs_and_keeps_report_immutable() -> None:
    prediction = _prediction("candidate", "clip", 1, "HARD_HAT", 1.0, 2.0)
    ground_truth = _ground_truth("clip", 1, "HARD_HAT", 1.0, 2.0)

    with pytest.raises(ValueError, match="onset_tolerance"):
        match_candidate_episodes((), (), onset_tolerance=timedelta(milliseconds=-1))
    with pytest.raises(ValueError, match="evaluated_seconds"):
        match_candidate_episodes(
            (), (), onset_tolerance=timedelta(0), evaluated_seconds=float("nan")
        )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        match_candidate_episodes(
            (), (prediction, prediction), onset_tolerance=timedelta(milliseconds=500)
        )
    with pytest.raises(ValueError, match="duplicate ground-truth episode"):
        match_candidate_episodes(
            (ground_truth, ground_truth), (), onset_tolerance=timedelta(milliseconds=500)
        )

    report = match_candidate_episodes((), (), onset_tolerance=timedelta(milliseconds=500))
    with pytest.raises(ValidationError):
        report.true_positives = 1


def test_report_rejects_derived_metrics_inconsistent_with_raw_counts_and_duration() -> None:
    valid = match_candidate_episodes(
        (), (), onset_tolerance=timedelta(milliseconds=500), evaluated_seconds=60.0
    )
    invalid_precision = valid.model_dump()
    invalid_precision["precision"] = {"value": 0.8, "reason": None}
    invalid_rate = valid.model_dump()
    invalid_rate["false_candidates_per_camera_hour"] = {"value": 999.0, "reason": None}

    with pytest.raises(ValidationError, match="precision does not match raw counts"):
        EpisodeMetricsReport.model_validate(invalid_precision)
    with pytest.raises(ValidationError, match="false_candidates_per_camera_hour"):
        EpisodeMetricsReport.model_validate(invalid_rate)
