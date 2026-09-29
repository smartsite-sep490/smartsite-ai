import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartsite_ai.inference.identity import (
    FaceRecognizerProtocol,
    FaceVerificationFrame,
    FaceVerificationResult,
    UnavailableFaceRecognizer,
)

VERIFICATION_ID = UUID("00000000-0000-4000-8000-000000000010")


def make_frame(**overrides: object) -> FaceVerificationFrame:
    return FaceVerificationFrame.model_validate(
        {
            "verification_id": VERIFICATION_ID,
            "captured_at": datetime(2026, 9, 29, 8, tzinfo=UTC),
            "mime_type": "image/jpeg",
            "content": b"synthetic-jpeg",
            **overrides,
        }
    )


def test_unavailable_recognizer_is_a_face_recognizer_and_fails_closed() -> None:
    recognizer = UnavailableFaceRecognizer()
    protocol: FaceRecognizerProtocol = recognizer

    result = asyncio.run(protocol.verify(make_frame()))

    assert isinstance(recognizer, FaceRecognizerProtocol)
    assert result.model_dump() == {
        "verification_id": VERIFICATION_ID,
        "status": "AI_UNAVAILABLE",
        "model_version": None,
        "candidate_profile_reference": None,
        "score_band": None,
        "reason_code": "FACE_MODEL_NOT_CONFIGURED",
    }


def test_frame_content_is_excluded_from_serialized_data() -> None:
    serialized = make_frame().model_dump()

    assert "content" not in serialized


@pytest.mark.parametrize(
    "data",
    [
        {
            "verification_id": VERIFICATION_ID,
            "status": "MATCHED",
            "reason_code": "MATCHED",
        },
        {
            "verification_id": VERIFICATION_ID,
            "status": "UNKNOWN",
            "candidate_profile_reference": "profile-1",
            "reason_code": "UNKNOWN_FACE",
        },
    ],
)
def test_result_rejects_ambiguous_identity_evidence(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FaceVerificationResult.model_validate(data)


def test_matched_result_never_contains_a_worker_or_access_decision() -> None:
    result = FaceVerificationResult.model_validate(
        {
            "verification_id": VERIFICATION_ID,
            "status": "MATCHED",
            "candidate_profile_reference": "profile-opaque-001",
            "score_band": "HIGH",
            "reason_code": "MATCHED",
        }
    )

    assert set(result.model_dump()) == {
        "verification_id",
        "status",
        "model_version",
        "candidate_profile_reference",
        "score_band",
        "reason_code",
    }
