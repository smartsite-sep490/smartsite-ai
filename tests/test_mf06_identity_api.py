"""Focused protocol tests for the dedicated MF06 Safety face transport router."""

import base64
import hashlib
from io import BytesIO
from typing import Any
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from smartsite_ai.inference.identity import (
    FaceVerificationFrame,
    FaceVerificationResult,
    UnavailableFaceRecognizer,
)
from smartsite_ai.mf06_identity_api import router

ENDPOINT = "/v1/identity/safety-mf06/resolve"
TOKEN = "mf06-safety-identity-service-token-123456"
RAW_REFERENCE = "fp_synthetic"
REFERENCE_HASH = hashlib.sha256(RAW_REFERENCE.encode("utf-8")).hexdigest()
PROFILE_REVISION = "99999999-9999-4999-8999-999999999999"
INVOCATION_ID = "11111111-1111-4111-8111-111111111111"
OTHER_UUID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
MODEL_VERSION = "synthetic-face-v1"
FERNET_TEMPLATE = "gAAAAAB" + ("A" * 133)


def _make_jpeg_bytes(
    width: int = 8,
    height: int = 8,
    *,
    color: tuple[int, int, int] = (32, 96, 192),
    exif: bytes | None = None,
) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (width, height), color=color)
    if exif is not None:
        image.save(buffer, format="JPEG", exif=exif)
    else:
        image.save(buffer, format="JPEG")
    return buffer.getvalue()


CROP_JPEG = _make_jpeg_bytes(8, 8)
FRAME_JPEG = _make_jpeg_bytes(16, 16, color=(48, 128, 64))
CROP_SHA256 = hashlib.sha256(CROP_JPEG).hexdigest()
FRAME_SHA256 = hashlib.sha256(FRAME_JPEG).hexdigest()
ALLOWED_UNAVAILABLE_REASONS = {"NO_MATCH", "UNSUPPORTED", "SERVICE_UNAVAILABLE", "ABORTED"}


class _FakeFaceRecognizer(UnavailableFaceRecognizer):
    def __init__(self, result: FaceVerificationResult | None = None) -> None:
        self.frames: list[FaceVerificationFrame] = []
        self._result = result or FaceVerificationResult(
            verification_id=UUID(INVOCATION_ID),
            status="MATCHED",
            model_version=MODEL_VERSION,
            candidate_profile_reference=RAW_REFERENCE,
            score_band="HIGH",
            reason_code="FACE_MATCHED",
        )

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        self.frames.append(frame)
        return self._result


def _make_invocation(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocolVersion": "SAFETY_MF06_FACE_V1",
        "purpose": "SAFETY_MF06",
        "invocationId": INVOCATION_ID,
        "bindingDigest": "a" * 64,
        "requestId": "22222222-2222-4222-8222-222222222222",
        "eventId": "33333333-3333-4333-8333-333333333333",
        "payloadHash": "b" * 64,
        "siteId": "44444444-4444-4444-8444-444444444444",
        "cameraId": "55555555-5555-4555-8555-555555555555",
        "cameraExternalId": "synthetic-camera",
        "streamSessionId": "66666666-6666-4666-8666-666666666666",
        "capturedAt": "2026-10-11T00:00:00.000Z",
        "originalCapturedAt": "2026-10-11T00:00:00.000Z",
        "personObservationIndex": 2,
        "zoneObservationIndex": 3,
        "trackId": 17,
        "regionId": "77777777-7777-4777-8777-777777777777",
        "geometryVersion": 4,
        "zoneId": "88888888-8888-4888-8888-888888888888",
        "requestRevision": 1,
        "requestedPurpose": "INITIAL_OBSERVATION_ASSESSMENT",
        "frameSha256": FRAME_SHA256,
        "cropSha256": CROP_SHA256,
        "cropAlgorithmVersion": "PERSON_FLOOR_CEIL_JPEG95_444_V1",
        "modelVersion": MODEL_VERSION,
    }
    base.update(overrides)
    return base


def _make_candidate(
    profile_reference_hash: str = REFERENCE_HASH,
    profile_revision: str = PROFILE_REVISION,
    encrypted_template: str = FERNET_TEMPLATE,
) -> dict[str, Any]:
    return {
        "profileReferenceHash": profile_reference_hash,
        "profileRevision": profile_revision,
        "encryptedTemplate": encrypted_template,
    }


def _make_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "invocation": _make_invocation(),
        "cropJpegBase64": base64.b64encode(CROP_JPEG).decode("ascii"),
        "candidates": [_make_candidate()],
    }
    payload.update(overrides)
    return payload


def _make_client(
    recognizer: Any | None = None,
    token: str | None = TOKEN,
) -> tuple[TestClient, _FakeFaceRecognizer]:
    fake = recognizer if isinstance(recognizer, _FakeFaceRecognizer) else _FakeFaceRecognizer()
    app = FastAPI()
    app.state.identity_service_token = token
    app.state.face_recognizer = recognizer if recognizer is not None else fake
    app.include_router(router)
    return TestClient(app), fake


def _auth_headers() -> dict[str, str]:
    return {"authorization": f"Bearer {TOKEN}"}


def test_resolve_requires_single_valid_bearer_token() -> None:
    client, recognizer = _make_client()
    payload = _make_payload()

    assert client.post(ENDPOINT, json=payload).status_code == 401
    wrong = client.post(ENDPOINT, json=payload, headers={"authorization": "Bearer wrong-token"})
    assert wrong.status_code == 401
    basic = client.post(ENDPOINT, json=payload, headers={"authorization": f"Basic {TOKEN}"})
    assert basic.status_code == 401
    duplicate_auth = client.post(
        ENDPOINT,
        json=payload,
        headers=[("authorization", f"Bearer {TOKEN}"), ("authorization", f"Bearer {TOKEN}")],
    )
    assert duplicate_auth.status_code == 401

    unconfigured_client, _ = _make_client(token=None)
    unconfigured = unconfigured_client.post(ENDPOINT, json=payload, headers=_auth_headers())
    assert unconfigured.status_code == 503
    assert recognizer.frames == []


def test_resolve_matched_echoes_exact_invocation_and_maps_candidate_without_leaks() -> None:
    client, recognizer = _make_client()
    invocation = _make_invocation()
    response = client.post(
        ENDPOINT,
        json=_make_payload(invocation=invocation),
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        **invocation,
        "state": "CANDIDATE",
        "provenance": "TRUSTED_SERVICE",
        "profileReferenceHash": REFERENCE_HASH,
        "profileRevision": PROFILE_REVISION,
    }
    assert len(recognizer.frames) == 1
    frame = recognizer.frames[0]
    assert frame.verification_id == UUID(INVOCATION_ID)
    assert frame.content == CROP_JPEG
    assert frame.enrollment_target is None
    assert frame.gate_presence_session is None
    assert len(frame.templates) == 1
    assert frame.templates[0].profile_reference_hash == REFERENCE_HASH
    assert frame.templates[0].encrypted_template == FERNET_TEMPLATE

    assert RAW_REFERENCE not in response.text
    assert FERNET_TEMPLATE not in response.text
    for forbidden_key in (
        "candidateProfileReference",
        "encryptedTemplate",
        "scoreBand",
        "workerId",
    ):
        assert forbidden_key not in body


def test_resolve_rejects_unknown_fields_invalid_invocation_and_candidate_limits() -> None:
    client, recognizer = _make_client()
    secret_marker = "secret-untrusted-field-value"
    dup_hash = [
        _make_candidate(profile_revision=PROFILE_REVISION),
        _make_candidate(profile_reference_hash=REFERENCE_HASH, profile_revision=OTHER_UUID),
    ]
    dup_rev = [
        _make_candidate(profile_reference_hash=REFERENCE_HASH),
        _make_candidate(profile_reference_hash="c" * 64, profile_revision=PROFILE_REVISION),
    ]
    unknown_cases = [
        _make_payload(extraTopLevel=secret_marker),
        _make_payload(invocation=_make_invocation(workerId=secret_marker)),
        _make_payload(candidates=[{**_make_candidate(), "workerId": secret_marker}]),
        _make_payload(invocation=_make_invocation(protocolVersion="GATE_FACE_V1")),
        _make_payload(invocation=_make_invocation(purpose="GATE_ENTRY")),
        _make_payload(invocation=_make_invocation(frameSha256="c" * 64 + "\n")),
        _make_payload(
            invocation=_make_invocation(personObservationIndex=3, zoneObservationIndex=3)
        ),
        _make_payload(invocation=_make_invocation(originalCapturedAt="2026-10-11T00:00:01.000Z")),
        _make_payload(
            invocation=_make_invocation(
                capturedAt="0001-01-01T00:00:00.000Z",
                originalCapturedAt="0001-01-01T00:00:00+23:59",
            )
        ),
        _make_payload(candidates=[_make_candidate(profile_reference_hash="E" * 64)]),
        _make_payload(candidates=[_make_candidate(profile_revision="not-a-uuid")]),
        _make_payload(candidates=[_make_candidate(encrypted_template="short-template")]),
        _make_payload(candidates=dup_hash),
        _make_payload(candidates=dup_rev),
    ]
    for bad_payload in unknown_cases:
        res = client.post(ENDPOINT, json=bad_payload, headers=_auth_headers())
        assert res.status_code == 422
        assert secret_marker not in res.text
        assert FERNET_TEMPLATE not in res.text

    too_many = [
        _make_candidate(
            profile_reference_hash=f"{index:064x}",
            profile_revision=f"99999999-9999-4999-8999-{index:012x}",
        )
        for index in range(1, 34)
    ]
    excess = client.post(ENDPOINT, json=_make_payload(candidates=too_many), headers=_auth_headers())
    assert excess.status_code in (200, 422)
    if excess.status_code == 200:
        assert excess.json()["state"] == "UNAVAILABLE"
    assert recognizer.frames == []


def test_resolve_enforces_crop_jpeg_markers_digest_and_3mib_body_limit() -> None:
    client, recognizer = _make_client()

    wrong_type = client.post(
        ENDPOINT,
        content=CROP_JPEG,
        headers={**_auth_headers(), "content-type": "image/jpeg"},
    )
    assert wrong_type.status_code == 415

    oversized_body = b"{" + (b" " * (3 * 1024 * 1024 + 64)) + b"}"
    too_large = client.post(
        ENDPOINT,
        content=oversized_body,
        headers={**_auth_headers(), "content-type": "application/json"},
    )
    assert too_large.status_code == 413

    oversized_crop = b"\xff\xd8" + (b"\x00" * (1_048_576 + 1)) + b"\xff\xd9"
    oversized_crop_res = client.post(
        ENDPOINT,
        json=_make_payload(
            invocation=_make_invocation(cropSha256=hashlib.sha256(oversized_crop).hexdigest()),
            cropJpegBase64=base64.b64encode(oversized_crop).decode("ascii"),
        ),
        headers=_auth_headers(),
    )
    assert oversized_crop_res.status_code in (413, 422)

    non_jpeg = b"not-a-jpeg-payload"
    marker_only_jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    png_buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()
    assert b"\xff\xc0\x00\x11\x08\x00\x08\x00\x08" in CROP_JPEG
    overpixel_jpeg = CROP_JPEG.replace(
        b"\xff\xc0\x00\x11\x08\x00\x08\x00\x08",
        b"\xff\xc0\x00\x11\x08\x13\x88\x0f\xa0",
        1,
    )
    exif = Image.Exif()
    exif[0x0112] = 6
    rotated_jpeg = _make_jpeg_bytes(8, 8, exif=exif.tobytes())

    for bad_crop in (non_jpeg, marker_only_jpeg, png_bytes, overpixel_jpeg, rotated_jpeg):
        bad_crop_res = client.post(
            ENDPOINT,
            json=_make_payload(
                invocation=_make_invocation(cropSha256=hashlib.sha256(bad_crop).hexdigest()),
                cropJpegBase64=base64.b64encode(bad_crop).decode("ascii"),
            ),
            headers=_auth_headers(),
        )
        assert bad_crop_res.status_code == 422

    digest_mismatch = client.post(
        ENDPOINT,
        json=_make_payload(invocation=_make_invocation(cropSha256="f" * 64)),
        headers=_auth_headers(),
    )
    assert digest_mismatch.status_code == 422
    assert recognizer.frames == []


def test_resolve_maps_non_matching_or_unconfigured_recognizer_to_unavailable() -> None:
    invocation = _make_invocation()
    for status, expected_reason in (
        ("UNKNOWN", "NO_MATCH"),
        ("LOW_CONFIDENCE", "NO_MATCH"),
        ("QUALITY_FAILED", "NO_MATCH"),
        ("AI_UNAVAILABLE", "SERVICE_UNAVAILABLE"),
    ):
        result = FaceVerificationResult(
            verification_id=UUID(INVOCATION_ID),
            status=status,
            model_version=MODEL_VERSION,
            reason_code=f"TEST_{status}",
        )
        client, _ = _make_client(recognizer=_FakeFaceRecognizer(result))
        res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=invocation),
            headers=_auth_headers(),
        )
        assert res.status_code == 200
        assert res.json() == {
            **invocation,
            "state": "UNAVAILABLE",
            "provenance": "TRUSTED_SERVICE",
            "reason": expected_reason,
        }

    unavailable_client, _ = _make_client(recognizer=UnavailableFaceRecognizer())
    unconfigured_res = unavailable_client.post(
        ENDPOINT,
        json=_make_payload(invocation=invocation),
        headers=_auth_headers(),
    )
    assert unconfigured_res.status_code == 200
    assert unconfigured_res.json() == {
        **invocation,
        "state": "UNAVAILABLE",
        "provenance": "TRUSTED_SERVICE",
        "reason": "SERVICE_UNAVAILABLE",
    }


def test_resolve_fails_closed_on_mismatched_model_invocation_id_or_unlisted_reference() -> None:
    invocation = _make_invocation()
    for vid, model, ref in (
        (INVOCATION_ID, "other-face-model", RAW_REFERENCE),
        (OTHER_UUID, MODEL_VERSION, RAW_REFERENCE),
        (INVOCATION_ID, MODEL_VERSION, "fp_unlisted_candidate"),
    ):
        result = FaceVerificationResult(
            verification_id=UUID(vid),
            status="MATCHED",
            model_version=model,
            candidate_profile_reference=ref,
            score_band="HIGH",
            reason_code="FACE_MATCHED",
        )
        client, _ = _make_client(recognizer=_FakeFaceRecognizer(result))
        res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=invocation),
            headers=_auth_headers(),
        )
        assert res.status_code == 200
        body = res.json()
        assert body["state"] == "UNAVAILABLE"
        assert body["provenance"] == "TRUSTED_SERVICE"
        assert body["reason"] in ALLOWED_UNAVAILABLE_REASONS
        for key, value in invocation.items():
            assert body[key] == value
        assert "profileReferenceHash" not in body
        assert "profileRevision" not in body
        assert "fp_unlisted_candidate" not in res.text
        assert RAW_REFERENCE not in res.text
