"""A response lost through caller cancellation must not trigger repeat Face work."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from test_mf06_identity_api import ENDPOINT, _auth_headers, _make_invocation, _make_payload
from test_mf06_identity_concurrency import _BlockingFaceRecognizer, _make_app


def test_completed_work_after_caller_cancellation_is_replayed_without_reverification():
    recognizer = _BlockingFaceRecognizer()
    app = _make_app(recognizer)

    async def start_and_cancel():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            caller = asyncio.create_task(
                client.post(ENDPOINT, json=_make_payload(), headers=_auth_headers())
            )
            await asyncio.wait_for(recognizer.started.wait(), 1.0)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller

    with TestClient(app) as client:
        assert client.portal is not None
        client.portal.call(start_and_cancel)
        pending = client.post(ENDPOINT, json=_make_payload(), headers=_auth_headers())
        assert pending.json()["reason"] == "SERVICE_UNAVAILABLE"
        changed = client.post(
            ENDPOINT,
            json=_make_payload(invocation=_make_invocation(modelVersion="changed-model")),
            headers=_auth_headers(),
        )
        assert changed.status_code == 409
        assert changed.headers["cache-control"] == "no-store"
        client.portal.call(recognizer.release_and_wait_settled)
        replay = client.post(ENDPOINT, json=_make_payload(), headers=_auth_headers())
        assert replay.status_code == 200
        assert replay.headers["cache-control"] == "no-store"
        assert replay.json()["state"] == "CANDIDATE"
        assert len(recognizer.frames) == 1


def test_replay_still_requires_authentication_and_valid_crop():
    from test_mf06_identity_replay import _make_app

    app, recognizer = _make_app()
    with TestClient(app) as client:
        first = client.post(ENDPOINT, json=_make_payload(), headers=_auth_headers())
        assert first.status_code == 200
        assert client.post(ENDPOINT, json=_make_payload()).status_code == 401
        invalid = client.post(
            ENDPOINT, json=_make_payload(cropJpegBase64="AAAA"), headers=_auth_headers()
        )
        assert invalid.status_code == 422
        assert len(recognizer.frames) == 1
