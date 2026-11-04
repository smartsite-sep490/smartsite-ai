"""Bounded single-flight concurrency and timeout/cancellation tests for MF06 face resolve."""

import asyncio
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from test_mf06_identity_api import (
    ENDPOINT,
    FERNET_TEMPLATE,
    INVOCATION_ID,
    MODEL_VERSION,
    PROFILE_REVISION,
    RAW_REFERENCE,
    REFERENCE_HASH,
    TOKEN,
    _auth_headers,
    _make_invocation,
    _make_payload,
)

from smartsite_ai import mf06_identity_api
from smartsite_ai.app import create_app
from smartsite_ai.config import Settings
from smartsite_ai.inference.identity import (
    FaceVerificationFrame,
    FaceVerificationResult,
    UnavailableFaceRecognizer,
)

SECOND_INVOCATION_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
THIRD_INVOCATION_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
FORBIDDEN_RESPONSE_KEYS = (
    "workerId",
    "worker_id",
    "permission",
    "permissions",
    "authorized",
    "decision",
    "candidateProfileReference",
    "encryptedTemplate",
    "scoreBand",
)


class _BlockingFaceRecognizer(UnavailableFaceRecognizer):
    def __init__(self) -> None:
        self.frames: list[FaceVerificationFrame] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.in_flight = 0
        self.max_concurrent = 0

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        self.in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self.in_flight)
        call_index = len(self.frames)
        self.frames.append(frame)
        try:
            if call_index == 0:
                self.started.set()
                await self.release.wait()
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="MATCHED",
                model_version=MODEL_VERSION,
                candidate_profile_reference=RAW_REFERENCE,
                score_band="HIGH",
                reason_code="FACE_MATCHED",
            )
        finally:
            self.in_flight -= 1
            if call_index == 0:
                self.finished.set()

    async def release_and_wait_settled(self) -> None:
        self.release.set()
        await asyncio.wait_for(self.finished.wait(), timeout=1.0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)


def _make_app(recognizer: _BlockingFaceRecognizer) -> FastAPI:
    app = create_app(
        Settings(
            _env_file=None,
            identity_service_token=SecretStr(TOKEN),
        )
    )
    app.state.face_recognizer = recognizer
    return app


def _assert_no_store_and_no_authority_leaks(response: httpx.Response) -> dict[str, Any]:
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"
    assert RAW_REFERENCE not in response.text
    assert FERNET_TEMPLATE not in response.text
    body: dict[str, Any] = response.json()
    for forbidden_key in FORBIDDEN_RESPONSE_KEYS:
        assert forbidden_key not in body
    return body


def test_resolve_timeout_returns_aborted_and_holds_single_slot_until_task_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mf06_identity_api, "VERIFICATION_TIMEOUT_SECONDS", 0.03)
    recognizer = _BlockingFaceRecognizer()
    app = _make_app(recognizer)

    first_invocation = _make_invocation(invocationId=INVOCATION_ID)
    second_invocation = _make_invocation(invocationId=SECOND_INVOCATION_ID)
    third_invocation = _make_invocation(invocationId=THIRD_INVOCATION_ID)

    with TestClient(app) as client:
        first_res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=first_invocation),
            headers=_auth_headers(),
        )
        first_body = _assert_no_store_and_no_authority_leaks(first_res)
        assert first_body == {
            **first_invocation,
            "state": "UNAVAILABLE",
            "provenance": "TRUSTED_SERVICE",
            "reason": "ABORTED",
        }
        assert len(recognizer.frames) == 1
        assert recognizer.frames[0].verification_id == UUID(INVOCATION_ID)
        assert recognizer.in_flight == 1
        assert not recognizer.finished.is_set()

        second_res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=second_invocation),
            headers=_auth_headers(),
        )
        second_body = _assert_no_store_and_no_authority_leaks(second_res)
        assert second_body == {
            **second_invocation,
            "state": "UNAVAILABLE",
            "provenance": "TRUSTED_SERVICE",
            "reason": "SERVICE_UNAVAILABLE",
        }
        assert len(recognizer.frames) == 1
        assert recognizer.max_concurrent == 1

        assert client.portal is not None
        client.portal.call(recognizer.release_and_wait_settled)
        assert recognizer.in_flight == 0

        third_res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=third_invocation),
            headers=_auth_headers(),
        )
        third_body = _assert_no_store_and_no_authority_leaks(third_res)
        assert third_body == {
            **third_invocation,
            "state": "CANDIDATE",
            "provenance": "TRUSTED_SERVICE",
            "profileReferenceHash": REFERENCE_HASH,
            "profileRevision": PROFILE_REVISION,
        }
        assert len(recognizer.frames) == 2
        assert recognizer.frames[1].verification_id == UUID(THIRD_INVOCATION_ID)
        assert recognizer.max_concurrent == 1


def test_resolve_refuses_new_work_while_application_is_draining() -> None:
    recognizer = _BlockingFaceRecognizer()
    app = _make_app(recognizer)
    with TestClient(app) as client:
        app.state.mf06_face_draining = True
        response = client.post(ENDPOINT, json=_make_payload(), headers=_auth_headers())
        assert _assert_no_store_and_no_authority_leaks(response) == {
            **_make_invocation(),
            "state": "UNAVAILABLE",
            "provenance": "TRUSTED_SERVICE",
            "reason": "SERVICE_UNAVAILABLE",
        }
        assert recognizer.frames == []


def test_resolve_caller_cancellation_keeps_slot_busy_until_shielded_task_completes() -> None:
    recognizer = _BlockingFaceRecognizer()
    app = _make_app(recognizer)

    first_invocation = _make_invocation(invocationId=INVOCATION_ID)
    second_invocation = _make_invocation(invocationId=SECOND_INVOCATION_ID)
    third_invocation = _make_invocation(invocationId=THIRD_INVOCATION_ID)

    async def _start_and_cancel_first_caller() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as async_client:
            caller_task = asyncio.create_task(
                async_client.post(
                    ENDPOINT,
                    json=_make_payload(invocation=first_invocation),
                    headers=_auth_headers(),
                )
            )
            await asyncio.wait_for(recognizer.started.wait(), timeout=1.0)
            caller_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller_task
            await asyncio.sleep(0)

    with TestClient(app) as client:
        assert client.portal is not None
        client.portal.call(_start_and_cancel_first_caller)
        assert len(recognizer.frames) == 1
        assert recognizer.in_flight == 1
        assert not recognizer.finished.is_set()

        second_res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=second_invocation),
            headers=_auth_headers(),
        )
        second_body = _assert_no_store_and_no_authority_leaks(second_res)
        assert second_body == {
            **second_invocation,
            "state": "UNAVAILABLE",
            "provenance": "TRUSTED_SERVICE",
            "reason": "SERVICE_UNAVAILABLE",
        }
        assert len(recognizer.frames) == 1
        assert recognizer.max_concurrent == 1
        client.portal.call(recognizer.release_and_wait_settled)
        assert recognizer.in_flight == 0

        third_res = client.post(
            ENDPOINT,
            json=_make_payload(invocation=third_invocation),
            headers=_auth_headers(),
        )
        third_body = _assert_no_store_and_no_authority_leaks(third_res)
        assert third_body == {
            **third_invocation,
            "state": "CANDIDATE",
            "provenance": "TRUSTED_SERVICE",
            "profileReferenceHash": REFERENCE_HASH,
            "profileRevision": PROFILE_REVISION,
        }
        assert len(recognizer.frames) == 2
        assert recognizer.frames[1].verification_id == UUID(THIRD_INVOCATION_ID)
        assert recognizer.max_concurrent == 1
