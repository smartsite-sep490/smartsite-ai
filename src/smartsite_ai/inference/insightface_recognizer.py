"""Local, opt-in InsightFace adapter for an academic demo.

No model import, download or image/template logging occurs at module import or
application startup.  The public model is only usable for non-commercial
research.  Production must replace this adapter with approved licensed models
and a calibrated threshold/evaluation record.
"""

import asyncio
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken

from smartsite_ai.config import Settings
from smartsite_ai.inference.identity import (
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


class EncryptedTemplateStore:
    """Small encrypted template store; it never accepts raw image bytes."""

    def __init__(self, path: Path, key: str) -> None:
        self._path = path
        self._fernet = Fernet(key.encode("ascii"))

    def upsert(self, profile_reference: str, embedding: list[float]) -> None:
        templates = self._load()
        templates[profile_reference] = embedding
        self._write(templates)

    def all(self) -> dict[str, list[float]]:
        return self._load()

    def _load(self) -> dict[str, list[float]]:
        if not self._path.exists():
            return {}
        try:
            raw = self._fernet.decrypt(self._path.read_bytes())
            data: Any = json.loads(raw.decode("utf-8"))
        except (InvalidToken, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("encrypted face template store cannot be read") from exc
        if not isinstance(data, dict) or not all(
            isinstance(key, str)
            and isinstance(value, list)
            and value
            and all(isinstance(item, (float, int)) and math.isfinite(item) for item in value)
            for key, value in data.items()
        ):
            raise RuntimeError("encrypted face template store is malformed")
        return {key: [float(item) for item in value] for key, value in data.items()}

    def _write(self, templates: dict[str, list[float]]) -> None:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        encrypted = self._fernet.encrypt(
            json.dumps(templates, allow_nan=False, separators=(",", ":")).encode("utf-8")
        )
        descriptor, temporary_name = tempfile.mkstemp(prefix=".templates-", dir=self._path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                os.chmod(temporary, 0o600)
                output.write(encrypted)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self._path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)


class InsightFaceDemoRecognizer:
    capability_status = "configured_demo"
    capability_reason = (
        "Academic/non-commercial local InsightFace demo configured. Model loads lazily; "
        "no liveness detection and no production authorization claim."
    )

    def __init__(self, model_root: Path, store: EncryptedTemplateStore, threshold: float) -> None:
        self._model_root = model_root
        self._store = store
        self._threshold = threshold
        self._analysis: Any | None = None
        self._load_lock = asyncio.Lock()

    async def enroll(self, request: FaceEnrollmentRequest) -> FaceEnrollmentResult:
        try:
            embeddings = [await self._embedding(sample.content) for sample in request.samples]
        except RuntimeError:
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
        centroid = _normalize([sum(items) / len(items) for items in zip(*embeddings, strict=True)])
        if centroid is None:
            return FaceEnrollmentResult(
                enrollment_id=request.enrollment_id,
                status="QUALITY_FAILED",
                reason_code="FACE_QUALITY_INSUFFICIENT",
            )
        profile_reference = f"fp_{request.enrollment_id.hex}"
        await asyncio.to_thread(self._store.upsert, profile_reference, centroid)
        return FaceEnrollmentResult(
            enrollment_id=request.enrollment_id,
            status="ENROLLED",
            model_version=_MODEL_VERSION,
            profile_reference=profile_reference,
            reason_code="ENROLLED",
        )

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        try:
            embedding = await self._embedding(frame.content)
        except RuntimeError:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="AI_UNAVAILABLE",
                reason_code="FACE_MODEL_UNAVAILABLE",
            )
        if embedding is None:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="QUALITY_FAILED",
                reason_code="FACE_QUALITY_INSUFFICIENT",
            )
        templates = await asyncio.to_thread(self._store.all)
        if not templates:
            return FaceVerificationResult(
                verification_id=frame.verification_id, status="UNKNOWN", reason_code="NO_ENROLLMENTS"
            )
        reference, score = max(
            ((reference, _cosine(embedding, template)) for reference, template in templates.items()),
            key=lambda entry: entry[1],
        )
        if score < self._threshold:
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="LOW_CONFIDENCE",
                reason_code="FACE_SCORE_BELOW_DEMO_THRESHOLD",
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
    # Imported only after checked local artifacts exist, preventing model-zoo downloads.
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
    return None if length == 0.0 else [value / length for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("face template dimensions do not match configured model")
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_face_recognizer(settings: Settings) -> FaceRecognizerProtocol:
    if not settings.identity_demo_mode:
        return UnavailableFaceRecognizer()
    if (
        settings.identity_model_root is None
        or settings.identity_template_store_path is None
        or settings.identity_template_encryption_key is None
    ):
        return UnavailableFaceRecognizer()
    if not settings.identity_model_root.is_absolute() or not settings.identity_template_store_path.is_absolute():
        return UnavailableFaceRecognizer()
    try:
        store = EncryptedTemplateStore(
            settings.identity_template_store_path,
            settings.identity_template_encryption_key.get_secret_value(),
        )
    except (ValueError, UnicodeEncodeError):
        return UnavailableFaceRecognizer()
    return InsightFaceDemoRecognizer(settings.identity_model_root, store, settings.identity_match_threshold)
