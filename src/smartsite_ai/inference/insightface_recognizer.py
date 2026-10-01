"""Opt-in face inference; Backend PostgreSQL owns encrypted enrollment templates."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from smartsite_ai.config import Settings
from smartsite_ai.inference.enrollment_quality import (
    QUALITY_ACCEPTED,
    CaptureTarget,
    assess_enrollment_jpeg,
)
from smartsite_ai.inference.identity import (
    EncryptedFaceTemplate,
    FaceEnrollmentRequest,
    FaceEnrollmentResult,
    FaceRecognizerProtocol,
    FaceVerificationFrame,
    FaceVerificationResult,
    UnavailableFaceRecognizer,
)

_MODEL_NAME = "buffalo_l"
_MODEL_VERSION = "insightface-buffalo_l-public-demo"
_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")


class TemplateCipher:
    """Encrypt for PostgreSQL; never read or write local template files."""

    def __init__(self, key: str) -> None:
        self._fernet = Fernet(key.encode("ascii"))

    def encrypt(self, reference: str, embedding: list[float]) -> str:
        payload = {"reference": reference, "modelVersion": _MODEL_VERSION, "embedding": embedding}
        return self._fernet.encrypt(
            json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

    def decrypt(self, template: EncryptedFaceTemplate) -> tuple[str, list[float]]:
        try:
            data = json.loads(self._fernet.decrypt(template.encrypted_template.encode("ascii")))
        except (InvalidToken, UnicodeError, ValueError) as exc:
            raise RuntimeError("face template cannot be decrypted") from exc
        if not isinstance(data, dict):
            raise RuntimeError("face template is malformed")
        reference, embedding = data.get("reference"), data.get("embedding")
        if (
            not isinstance(reference, str)
            or hashlib.sha256(reference.encode("utf-8")).hexdigest()
            != template.profile_reference_hash
            or data.get("modelVersion") != _MODEL_VERSION
            or not isinstance(embedding, list)
            or not 1 <= len(embedding) <= 512
            or not all(type(item) in (float, int) and math.isfinite(item) for item in embedding)
            or _normalize(embedding) is None
        ):
            raise RuntimeError("face template is malformed or incompatible")
        return reference, embedding


class InsightFaceDemoRecognizer:
    capability_status = "configured_demo"
    capability_reason = (
        "Academic/non-commercial InsightFace demo with encrypted database templates. "
        "Model loads lazily; no liveness detection or production authorization claim."
    )

    def __init__(self, model_root: Path, cipher: TemplateCipher, threshold: float) -> None:
        self._model_root = model_root
        self._cipher = cipher
        self._threshold = threshold
        self._analysis: Any | None = None
        self._load_lock = asyncio.Lock()

    async def enroll(self, request: FaceEnrollmentRequest) -> FaceEnrollmentResult:
        try:
            for sample, target in zip(request.samples, ("front", "left", "right"), strict=True):
                reason = await self._quality(sample.content, target)
                if reason != QUALITY_ACCEPTED:
                    return FaceEnrollmentResult(
                        enrollment_id=request.enrollment_id,
                        status="QUALITY_FAILED",
                        reason_code=reason,
                    )
            embeddings = [await self._embedding(sample.content) for sample in request.samples]
        except (RuntimeError, AttributeError, ValueError):
            return FaceEnrollmentResult(
                enrollment_id=request.enrollment_id,
                status="AI_UNAVAILABLE",
                reason_code="FACE_MODEL_UNAVAILABLE",
            )
        if any(embedding is None for embedding in embeddings):
            return FaceEnrollmentResult(
                enrollment_id=request.enrollment_id,
                status="QUALITY_FAILED",
                reason_code="FACE_QUALITY_INSUFFICIENT",
            )
        if any(_cosine(embeddings[0], embedding) < self._threshold for embedding in embeddings[1:]):
            return FaceEnrollmentResult(
                enrollment_id=request.enrollment_id,
                status="QUALITY_FAILED",
                reason_code="FACE_SAMPLES_INCONSISTENT",
            )
        centroid = _normalize([sum(items) / len(items) for items in zip(*embeddings, strict=True)])
        if centroid is None:
            return FaceEnrollmentResult(
                enrollment_id=request.enrollment_id,
                status="QUALITY_FAILED",
                reason_code="FACE_QUALITY_INSUFFICIENT",
            )
        reference = f"fp_{request.enrollment_id.hex}"
        return FaceEnrollmentResult(
            enrollment_id=request.enrollment_id,
            status="ENROLLED",
            model_version=_MODEL_VERSION,
            profile_reference=reference,
            encrypted_template=self._cipher.encrypt(reference, centroid),
            reason_code="ENROLLED",
        )

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        if frame.enrollment_target is not None:
            try:
                reason = await self._quality(frame.content, frame.enrollment_target)
            except (RuntimeError, AttributeError, ValueError):
                return FaceVerificationResult(
                    verification_id=frame.verification_id,
                    status="AI_UNAVAILABLE",
                    reason_code="FACE_MODEL_UNAVAILABLE",
                )
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="UNKNOWN" if reason == QUALITY_ACCEPTED else "QUALITY_FAILED",
                reason_code=reason,
            )
        try:
            embedding = await self._embedding(frame.content)
            templates = [self._cipher.decrypt(template) for template in frame.templates]
        except RuntimeError:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="AI_UNAVAILABLE",
                reason_code="FACE_MODEL_OR_TEMPLATE_UNAVAILABLE",
            )
        if embedding is None:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="QUALITY_FAILED",
                reason_code="FACE_QUALITY_INSUFFICIENT",
            )
        if not templates:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="UNKNOWN",
                reason_code="NO_ENROLLMENTS",
            )
        try:
            scores = sorted(
                ((reference, _cosine(embedding, template)) for reference, template in templates),
                key=lambda entry: entry[1],
                reverse=True,
            )
        except RuntimeError:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="AI_UNAVAILABLE",
                reason_code="FACE_TEMPLATE_INCOMPATIBLE",
            )
        reference, score = scores[0]
        if score < self._threshold or (len(scores) > 1 and scores[1][1] >= self._threshold):
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="LOW_CONFIDENCE",
                reason_code="FACE_MATCH_UNCERTAIN",
            )
        return FaceVerificationResult(
            verification_id=frame.verification_id,
            status="MATCHED",
            model_version=_MODEL_VERSION,
            candidate_profile_reference=reference,
            score_band="HIGH" if score >= self._threshold + 0.12 else "MEDIUM",
            reason_code="FACE_MATCHED_DEMO",
        )

    async def _embedding(self, jpeg: bytes) -> list[float] | None:
        analysis = await self._analysis_instance()
        return await asyncio.to_thread(_embedding_from_jpeg, analysis, jpeg)

    async def _quality(self, jpeg: bytes, target: CaptureTarget) -> str:
        analysis = await self._analysis_instance()
        return await asyncio.to_thread(assess_enrollment_jpeg, analysis, jpeg, target)

    async def _analysis_instance(self) -> Any:
        if self._analysis is not None:
            return self._analysis
        async with self._load_lock:
            if self._analysis is not None:
                return self._analysis
            model_directory = self._model_root / "models" / _MODEL_NAME
            if not all((model_directory / item).is_file() for item in _MODEL_FILES):
                raise RuntimeError("required InsightFace model artifacts are not installed")
            self._analysis = await asyncio.to_thread(_load_analysis, self._model_root)
            return self._analysis


def _load_analysis(model_root: Path) -> Any:
    from insightface.app import FaceAnalysis

    analysis = FaceAnalysis(
        name=_MODEL_NAME,
        root=str(model_root),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    analysis.prepare(ctx_id=-1, det_size=(640, 640))
    return analysis


def _embedding_from_jpeg(analysis: Any, jpeg: bytes) -> list[float] | None:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    faces = analysis.get(image)
    if len(faces) != 1 or float(faces[0].det_score) < 0.70:
        return None
    return _normalize([float(value) for value in faces[0].normed_embedding])


def _normalize(values: list[float]) -> list[float] | None:
    length = math.sqrt(sum(value * value for value in values))
    return (
        None if not math.isfinite(length) or length == 0.0 else [value / length for value in values]
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("face template dimensions do not match configured model")
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_face_recognizer(settings: Settings) -> FaceRecognizerProtocol:
    if not settings.identity_demo_mode or settings.identity_model_root is None:
        return UnavailableFaceRecognizer()
    if (
        not settings.identity_model_root.is_absolute()
        or settings.identity_template_encryption_key is None
    ):
        return UnavailableFaceRecognizer()
    try:
        cipher = TemplateCipher(settings.identity_template_encryption_key.get_secret_value())
    except (ValueError, UnicodeEncodeError):
        return UnavailableFaceRecognizer()
    return InsightFaceDemoRecognizer(
        settings.identity_model_root, cipher, settings.identity_match_threshold
    )
