import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from smartsite_ai.config import Settings
from smartsite_ai.inference.identity import UnavailableFaceRecognizer
from smartsite_ai.inference.insightface_recognizer import (
    InsightFaceDemoRecognizer,
    TemplateCipher,
    build_face_recognizer,
)


def test_database_template_encrypts_embeddings_without_writing_a_file(tmp_path: Path) -> None:
    from smartsite_ai.inference.identity import EncryptedFaceTemplate

    cipher = TemplateCipher(Fernet.generate_key().decode("ascii"))
    encrypted = cipher.encrypt("fp_demo", [0.6, 0.8])
    template = EncryptedFaceTemplate(
        profile_reference_hash=hashlib.sha256(b"fp_demo").hexdigest(),
        encrypted_template=encrypted,
    )
    assert cipher.decrypt(template) == ("fp_demo", [0.6, 0.8])
    assert "fp_demo" not in encrypted
    assert list(tmp_path.iterdir()) == []


def test_demo_recognizer_requires_explicit_private_configuration(tmp_path: Path) -> None:
    disabled = build_face_recognizer(Settings(_env_file=None))
    missing_key = build_face_recognizer(
        Settings(identity_demo_mode=True, identity_model_root=tmp_path, _env_file=None)
    )

    assert isinstance(disabled, UnavailableFaceRecognizer)
    assert isinstance(missing_key, UnavailableFaceRecognizer)


def test_demo_recognizer_is_configured_without_loading_or_downloading_model(tmp_path: Path) -> None:
    recognizer = build_face_recognizer(
        Settings(
            identity_demo_mode=True,
            identity_model_root=tmp_path,
            identity_template_encryption_key=Fernet.generate_key().decode("ascii"),
            _env_file=None,
        )
    )

    assert isinstance(recognizer, InsightFaceDemoRecognizer)


def test_enrollment_can_be_matched_after_ai_restart_using_database_ciphertext(
    tmp_path, monkeypatch
):
    from smartsite_ai.inference.identity import (
        EncryptedFaceTemplate,
        FaceEnrollmentRequest,
        FaceEnrollmentSample,
        FaceVerificationFrame,
    )

    key = Fernet.generate_key().decode("ascii")
    cipher = TemplateCipher(key)
    first = InsightFaceDemoRecognizer(tmp_path, cipher, 0.45)

    async def embedding(_self, _jpeg):
        return [0.6, 0.8]

    async def quality(_self, _jpeg, _target):
        return "FACE_QUALITY_ACCEPTED"

    monkeypatch.setattr(InsightFaceDemoRecognizer, "_embedding", embedding)
    monkeypatch.setattr(InsightFaceDemoRecognizer, "_quality", quality)
    enrollment_id = uuid4()
    samples = tuple(
        FaceEnrollmentSample(
            verification_id=enrollment_id,
            captured_at=datetime.now(UTC),
            mime_type="image/jpeg",
            content=b"synthetic-jpeg",
            sample_index=index,
        )
        for index in (1, 2, 3)
    )
    result = asyncio.run(
        first.enroll(FaceEnrollmentRequest(enrollment_id=enrollment_id, samples=samples))
    )
    assert result.status == "ENROLLED"
    stored = EncryptedFaceTemplate(
        profile_reference_hash=hashlib.sha256(result.profile_reference.encode()).hexdigest(),
        encrypted_template=result.encrypted_template,
    )
    restarted = InsightFaceDemoRecognizer(tmp_path, TemplateCipher(key), 0.45)
    frame = FaceVerificationFrame(
        verification_id=uuid4(),
        captured_at=datetime.now(UTC),
        mime_type="image/jpeg",
        content=b"synthetic-jpeg",
        templates=(stored,),
    )
    matched = asyncio.run(restarted.verify(frame))
    assert matched.status == "MATCHED"
    assert matched.candidate_profile_reference == result.profile_reference
    assert list(tmp_path.iterdir()) == []
    assert "encrypted_template" not in result.model_dump()
    assert "templates" not in frame.model_dump()
    without_database = asyncio.run(restarted.verify(frame.model_copy(update={"templates": ()})))
    assert without_database.status == "UNKNOWN"
    wrong_key = InsightFaceDemoRecognizer(
        tmp_path, TemplateCipher(Fernet.generate_key().decode()), 0.45
    )
    assert asyncio.run(wrong_key.verify(frame)).status == "AI_UNAVAILABLE"


@pytest.mark.parametrize("failure", ["pose", "identity"])
def test_enrollment_rechecks_pose_and_same_identity_before_creating_template(
    tmp_path, monkeypatch, failure
):
    from smartsite_ai.inference.identity import FaceEnrollmentRequest, FaceEnrollmentSample

    recognizer = InsightFaceDemoRecognizer(
        tmp_path, TemplateCipher(Fernet.generate_key().decode()), 0.45
    )
    checked = []

    async def quality(_jpeg, target):
        checked.append(target)
        return (
            "FACE_POSE_LEFT_REQUIRED"
            if failure == "pose" and target == "left"
            else "FACE_QUALITY_ACCEPTED"
        )

    async def embedding(jpeg):
        assert failure != "pose", "Rejected quality must not create an embedding"
        return [1.0, 0.0] if jpeg == b"1" else [0.0, 1.0]

    monkeypatch.setattr(recognizer, "_quality", quality)
    monkeypatch.setattr(recognizer, "_embedding", embedding)
    enrollment_id = uuid4()
    samples = tuple(
        FaceEnrollmentSample(
            verification_id=enrollment_id,
            captured_at=datetime.now(UTC),
            mime_type="image/jpeg",
            content=str(index).encode(),
            sample_index=index,
        )
        for index in (1, 2, 3)
    )
    result = asyncio.run(
        recognizer.enroll(FaceEnrollmentRequest(enrollment_id=enrollment_id, samples=samples))
    )
    assert checked == (["front", "left"] if failure == "pose" else ["front", "left", "right"])
    assert result.status == "QUALITY_FAILED"
    assert result.reason_code == (
        "FACE_POSE_LEFT_REQUIRED" if failure == "pose" else "FACE_SAMPLES_INCONSISTENT"
    )
    assert result.encrypted_template is None
    assert result.profile_reference is None


def test_cipher_rejects_template_swaps_corruption_and_incompatible_dimensions(
    tmp_path, monkeypatch
):
    from smartsite_ai.inference.identity import EncryptedFaceTemplate, FaceVerificationFrame

    cipher = TemplateCipher(Fernet.generate_key().decode())
    token = cipher.encrypt("profile1", [0.6, 0.8])
    with pytest.raises(RuntimeError):
        cipher.decrypt(
            EncryptedFaceTemplate(profile_reference_hash="a" * 64, encrypted_template=token)
        )
    with pytest.raises(RuntimeError):
        cipher.decrypt(
            EncryptedFaceTemplate(
                profile_reference_hash=hashlib.sha256(b"profile1").hexdigest(),
                encrypted_template="a" * 120,
            )
        )

    async def embedding(_self, _jpeg):
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(InsightFaceDemoRecognizer, "_embedding", embedding)
    template = EncryptedFaceTemplate(
        profile_reference_hash=hashlib.sha256(b"profile1").hexdigest(), encrypted_template=token
    )
    frame = FaceVerificationFrame(
        verification_id=uuid4(),
        captured_at=datetime.now(UTC),
        mime_type="image/jpeg",
        content=b"fake",
        templates=(template,),
    )
    assert (
        asyncio.run(InsightFaceDemoRecognizer(tmp_path, cipher, 0.45).verify(frame)).status
        == "AI_UNAVAILABLE"
    )
