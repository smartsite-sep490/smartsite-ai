import json

import pytest
from pydantic import ValidationError

from smartsite_ai.domain.tracked_frame_result import TrackedFrameResult

SCOPE = "a" * 64
SEQUENCE = "18446744073709551615"
EVENT = "00000000-0000-4000-8000-000000000019"
SEGMENT = "00000000-0000-4000-8000-000000000020"
RETIRED = "00000000-0000-4000-8000-000000000021"
SUBJECT = {"segmentId": SEGMENT, "eventId": EVENT, "personObservationIndex": 1}
RESULT = {
    "kind": "ADMITTED_TRACKED_FRAME",
    "scopeSha256": SCOPE,
    "sequenceNumber": SEQUENCE,
    "subjects": [SUBJECT],
    "retiredSegmentIds": [RETIRED],
}


def test_admission_matches_exact_sent_frame_references():
    result = TrackedFrameResult.model_validate_json(json.dumps(RESULT))
    result.require_match(SCOPE, SEQUENCE, {SEGMENT: (EVENT, 1)}, {RETIRED})


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "RAW_STORED"},
        {"kind": "PENDING_PREDECESSOR"},
        {"sequenceNumber": "18446744073709551616"},
        {"sequenceNumber": 1},
        {"subjects": [{**SUBJECT, "personObservationIndex": True}]},
        {"subjects": [SUBJECT, SUBJECT]},
        {"retiredSegmentIds": [SEGMENT]},
        {"retiredSegmentIds": [RETIRED, RETIRED]},
        {"workerId": None},
    ],
)
def test_reject_non_admission_or_inconsistent_result(change):
    with pytest.raises(ValidationError):
        TrackedFrameResult.model_validate({**RESULT, **change})


@pytest.mark.parametrize(
    "expected",
    [
        ("b" * 64, SEQUENCE, {SEGMENT: (EVENT, 1)}, {RETIRED}),
        (SCOPE, "0", {SEGMENT: (EVENT, 1)}, {RETIRED}),
        (SCOPE, SEQUENCE, {SEGMENT: (EVENT, 0)}, {RETIRED}),
        (SCOPE, SEQUENCE, {}, {RETIRED}),
        (SCOPE, SEQUENCE, {SEGMENT: (EVENT, 1)}, set()),
    ],
)
def test_a_different_frame_cannot_advance_the_local_head(expected):
    with pytest.raises(ValueError, match="admitted frame binding mismatch"):
        TrackedFrameResult.model_validate(RESULT).require_match(*expected)
