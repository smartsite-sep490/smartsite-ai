import math

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.person_count_metrics import (
    PersonCountSample,
    compute_person_count_metrics,
)


def sample(frame: str, predicted: int, actual: int | None, status: str = "REVIEWED"):
    return PersonCountSample.model_validate(
        {
            "frameId": frame,
            "predictedCount": predicted,
            "groundTruthCount": actual,
            "reviewStatus": status,
        }
    )


def test_reviewed_frame_count_errors_do_not_hide_over_and_under_counts() -> None:
    report = compute_person_count_metrics(
        [
            sample("a", 4, 2),
            sample("b", 0, 2),
            sample("c", 3, 3),
        ]
    )
    assert report.evaluated_frames == 3
    assert report.mean_absolute_error == pytest.approx(4 / 3)
    assert report.root_mean_squared_error == pytest.approx(math.sqrt(8 / 3))
    assert report.mean_signed_error == 0.0
    assert report.exact_match_fraction == pytest.approx(1 / 3)
    assert report.overcount_frames == report.undercount_frames == 1
    assert report.ground_truth_people_total == report.predicted_people_total == 7
    assert report.is_model_acceptance is False


def test_unreviewed_and_excluded_frames_are_not_zero_person_ground_truth() -> None:
    report = compute_person_count_metrics(
        [
            sample("a", 2, 2),
            sample("b", 99, None, "UNREVIEWED"),
            sample("c", 50, None, "EXCLUDED"),
        ]
    )
    assert report.mean_absolute_error == 0.0
    assert report.evaluated_frames == 1
    assert report.unreviewed_frame_ids == ("b",)
    assert report.excluded_frame_ids == ("c",)
    assert report.predicted_people_total == 2


def test_no_reviewed_frames_yields_undefined_metrics_not_a_perfect_score() -> None:
    report = compute_person_count_metrics([sample("a", 0, None, "UNREVIEWED")])
    assert report.evaluated_frames == 0
    assert report.mean_absolute_error is None
    assert report.exact_match_fraction is None
    assert report.reason == "no reviewed frame counts"


def test_reviewed_empty_frame_is_valid_counting_ground_truth() -> None:
    report = compute_person_count_metrics([sample("a", 1, 0)])
    assert report.mean_absolute_error == 1.0
    assert report.overcount_frames == 1


def test_duplicate_frame_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate frameId"):
        compute_person_count_metrics([sample("a", 1, 1), sample("a", 1, 1)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("groundTruthCount", None),
        ("groundTruthCount", -1),
        ("predictedCount", True),
        ("predictedCount", 0.5),
        ("frameId", " a"),
        ("frameId", ""),
        ("reviewStatus", "AI_APPROVED"),
    ],
)
def test_invalid_reviewed_counts_are_rejected(field: str, value: object) -> None:
    row = {"frameId": "a", "predictedCount": 1, "groundTruthCount": 1, "reviewStatus": "REVIEWED"}
    row[field] = value
    with pytest.raises(ValidationError):
        PersonCountSample.model_validate(row)


def test_unreviewed_label_counts_cannot_be_scored_as_reviewed_truth() -> None:
    with pytest.raises(ValidationError, match="reviewed"):
        sample("a", 2, 2, "UNREVIEWED")


def test_frame_totals_are_not_unique_workers_and_do_not_measure_localization() -> None:
    report = compute_person_count_metrics([sample("a", 1, 1), sample("b", 1, 1)])
    assert report.predicted_people_total == 2
    assert report.scope == "REVIEWED_FRAME_COUNTS_ONLY"
    assert "not unique Workers" in report.warning
    assert "localization" in report.warning
