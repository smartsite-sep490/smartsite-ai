"""Synthetic pixels and fake detector landmarks; no real person/model/GPU."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from smartsite_ai.inference.enrollment_quality import assess_enrollment_jpeg
from smartsite_ai.inference.identity import FaceVerificationFrame
from smartsite_ai.inference.insightface_recognizer import InsightFaceDemoRecognizer, TemplateCipher

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def image(brightness=128, texture=True):
    pixels = np.full((480, 640, 3), brightness, dtype=np.uint8)
    if texture:
        pixels[::2, ::2] = max(0, brightness - 50)
        pixels[1::2, 1::2] = min(255, brightness + 50)
    return cv2.imencode(".jpg", pixels)[1].tobytes()


def detector(turn=0, **changes):
    face = SimpleNamespace(
        bbox=np.array([180, 100, 460, 390]),
        det_score=0.95,
        kps=np.array(
            [[260, 200], [380, 200], [320 + turn * 120, 270], [275, 330], [365, 330]], dtype=float
        ),
    )
    for name, value in changes.items():
        setattr(face, name, value)
    return SimpleNamespace(get=lambda _image: [face])


@pytest.mark.parametrize(("turn", "target"), [(0, "front"), (0.25, "left"), (-0.25, "right")])
def test_accepts_clear_centered_image_in_requested_relative_pose(turn, target):
    assert assess_enrollment_jpeg(detector(turn), image(), target) == "FACE_QUALITY_ACCEPTED"


@pytest.mark.parametrize(
    ("turn", "target", "reason"),
    [
        (0, "left", "FACE_POSE_LEFT_REQUIRED"),
        (0, "right", "FACE_POSE_RIGHT_REQUIRED"),
        (0.25, "front", "FACE_POSE_FRONT_REQUIRED"),
        (-0.25, "left", "FACE_POSE_LEFT_REQUIRED"),
        (0.8, "left", "FACE_TURN_TOO_FAR"),
    ],
)
def test_cannot_reuse_front_pose_or_wrong_turn_for_an_angle(turn, target, reason):
    assert assess_enrollment_jpeg(detector(turn), image(), target) == reason


@pytest.mark.parametrize(
    ("pixels", "reason"),
    [
        (image(15, False), "FACE_TOO_DARK"),
        (image(245, False), "FACE_TOO_BRIGHT"),
        (image(128, False), "FACE_BLURRY"),
    ],
)
def test_image_quality_rejects_dark_overexposed_and_blurry_photos(pixels, reason):
    assert assess_enrollment_jpeg(detector(), pixels, "front") == reason


def test_face_count_size_clipping_and_missing_landmarks_fail_closed():
    assert (
        assess_enrollment_jpeg(SimpleNamespace(get=lambda _image: []), image(), "front")
        == "FACE_NOT_FOUND"
    )
    assert (
        assess_enrollment_jpeg(SimpleNamespace(get=lambda _image: [None, None]), image(), "front")
        == "FACE_MULTIPLE_FOUND"
    )
    assert (
        assess_enrollment_jpeg(detector(bbox=[280, 180, 360, 300]), image(), "front")
        == "FACE_TOO_SMALL"
    )
    assert (
        assess_enrollment_jpeg(detector(bbox=[0, 100, 460, 390]), image(), "front")
        == "FACE_CLIPPED"
    )
    assert (
        assess_enrollment_jpeg(detector(kps=[[float("nan"), 0]]), image(), "front")
        == "FACE_LANDMARKS_UNAVAILABLE"
    )
    assert assess_enrollment_jpeg(detector(), b"not-a-jpeg", "front") == "FACE_IMAGE_INVALID"


def test_quality_frame_never_performs_matching(tmp_path, monkeypatch):
    recognizer = InsightFaceDemoRecognizer(
        tmp_path, TemplateCipher(Fernet.generate_key().decode()), 0.45
    )

    async def quality(_jpeg, target):
        assert target == "left"
        return "FACE_POSE_LEFT_REQUIRED"

    monkeypatch.setattr(recognizer, "_quality", quality)
    result = asyncio.run(
        recognizer.verify(
            FaceVerificationFrame(
                verification_id=uuid4(),
                captured_at=datetime.now(UTC),
                mime_type="image/jpeg",
                content=b"synthetic",
                enrollment_target="left",
            )
        )
    )
    assert result.status == "QUALITY_FAILED"
    assert result.reason_code == "FACE_POSE_LEFT_REQUIRED"
    assert result.candidate_profile_reference is None
