import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from cryptography.fernet import Fernet

from smartsite_ai.inference.gate_continuity import GateContinuity
from smartsite_ai.inference.identity import FaceVerificationFrame
from smartsite_ai.inference.insightface_recognizer import InsightFaceDemoRecognizer, TemplateCipher


def test_same_face_is_suppressed_and_a_different_face_requires_two_observations():
    tracker = GateContinuity()
    assert tracker.observe("a", [1.0, 0.0], 0) == "FACE_PRESENCE_STABILIZING"
    assert tracker.observe("a", [1.0, 0.0], 1) == "FACE_PRESENCE_NEW"
    assert tracker.observe("a", [1.0, 0.0], 2) == "FACE_PRESENCE_SAME"
    assert tracker.observe("a", [0.0, 1.0], 3) == "FACE_PRESENCE_STABILIZING"
    assert tracker.observe("a", [0.0, 1.0], 4) == "FACE_PRESENCE_NEW"
    assert tracker.observe("a", [0.0, 1.0], 5) == "FACE_PRESENCE_SAME"


def test_reentry_requires_confirmed_absence_and_sessions_do_not_share_faces():
    tracker = GateContinuity()
    tracker.observe("a", [1.0, 0.0], 0)
    tracker.observe("a", [1.0, 0.0], 1)
    tracker.absent("a", 2)
    assert tracker.observe("a", [1.0, 0.0], 3) == "FACE_PRESENCE_SAME"
    tracker.absent("a", 4)
    tracker.absent("a", 6)
    assert tracker.observe("a", [1.0, 0.0], 7) == "FACE_PRESENCE_STABILIZING"
    assert tracker.observe("a", [1.0, 0.0], 8) == "FACE_PRESENCE_NEW"
    assert tracker.observe("b", [1.0, 0.0], 8) == "FACE_PRESENCE_STABILIZING"
    assert tracker.observe("a", [1.0, 0.0], 39) == "FACE_PRESENCE_STABILIZING"


def test_cache_is_bounded_and_uncertain_similarity_never_triggers_a_new_scan():
    tracker = GateContinuity()
    for index in range(200):
        tracker.observe(str(index), [1.0, 0.0], 0)
    assert len(tracker._entries) == 128
    tracker.observe("199", [1.0, 0.0], 1)
    assert tracker.observe("199", [0.5, 0.8660254], 2) == "FACE_MATCH_UNCERTAIN"


def test_presence_mode_returns_only_continuity_and_never_identity(tmp_path, monkeypatch):
    recognizer = InsightFaceDemoRecognizer(
        tmp_path, TemplateCipher(Fernet.generate_key().decode()), 0.45
    )

    async def quality(_jpeg, target):
        assert target is None
        return "FACE_QUALITY_ACCEPTED"

    async def embedding(_jpeg):
        return [1.0, 0.0]

    monkeypatch.setattr(recognizer, "_quality", quality)
    monkeypatch.setattr(recognizer, "_embedding", embedding)

    async def run():
        frame = FaceVerificationFrame(
            verification_id=uuid4(),
            captured_at=datetime.now(UTC),
            mime_type="image/jpeg",
            content=b"synthetic",
            gate_presence_session="a" * 64,
        )
        for expected in ("FACE_PRESENCE_STABILIZING", "FACE_PRESENCE_NEW", "FACE_PRESENCE_SAME"):
            result = await recognizer.verify(frame)
            assert result.status == "UNKNOWN"
            assert result.reason_code == expected
            assert result.candidate_profile_reference is None

    asyncio.run(run())
