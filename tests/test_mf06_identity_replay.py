"""Bounded HTTP replay and idempotency tests for the MF06 Safety face route."""

from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from test_mf06_identity_api import (
    ENDPOINT,
    FERNET_TEMPLATE,
    INVOCATION_ID,
    MODEL_VERSION,
    RAW_REFERENCE,
    REFERENCE_HASH,
    TOKEN,
    _auth_headers,
    _make_candidate,
    _make_invocation,
    _make_payload,
)

from smartsite_ai.app import create_app
from smartsite_ai.config import Settings
from smartsite_ai.inference.identity import (
    FaceVerificationFrame,
    FaceVerificationResult,
    FaceVerificationStatus,
    UnavailableFaceRecognizer,
)


class _DynamicFakeRecognizer(UnavailableFaceRecognizer):
    """Synthetic fake that echoes each frame's dynamic verification_id."""

    def __init__(
        self,
        statuses: list[FaceVerificationStatus] | None = None,
        *,
        candidate_reference: str = RAW_REFERENCE,
    ) -> None:
        self.frames: list[FaceVerificationFrame] = []
        self._statuses: list[FaceVerificationStatus] = list(statuses or ["MATCHED"])
        self._candidate_reference = candidate_reference

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        self.frames.append(frame)
        status: FaceVerificationStatus = (
            self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        )
        if status == "MATCHED":
            return FaceVerificationResult(
                verification_id=frame.verification_id,
                status="MATCHED",
                model_version=MODEL_VERSION,
                candidate_profile_reference=self._candidate_reference,
                score_band="HIGH",
                reason_code="FACE_MATCHED",
            )
        return FaceVerificationResult(
            verification_id=frame.verification_id,
            status=status,
            model_version=MODEL_VERSION,
            candidate_profile_reference=None,
            score_band=None,
            reason_code="NO_MATCH" if status != "AI_UNAVAILABLE" else "AI_UNAVAILABLE",
        )


def _make_app(
    recognizer: _DynamicFakeRecognizer | None = None,
) -> tuple[FastAPI, _DynamicFakeRecognizer]:
    fake = recognizer or _DynamicFakeRecognizer()
    app = create_app(Settings(_env_file=None, identity_service_token=SecretStr(TOKEN)))
    app.state.face_recognizer = fake
    return app, fake


def _uuid_for_index(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def test_identical_replay_returns_cached_candidate_and_no_match_with_single_verify() -> None:
    app, fake = _make_app(_DynamicFakeRecognizer(statuses=["MATCHED", "UNKNOWN"]))
    candidate_payload = _make_payload(invocation=_make_invocation(invocationId=_uuid_for_index(1)))
    no_match_payload = _make_payload(invocation=_make_invocation(invocationId=_uuid_for_index(2)))

    with TestClient(app) as client:
        first_cand = client.post(ENDPOINT, json=candidate_payload, headers=_auth_headers())
        replay_cand = client.post(ENDPOINT, json=candidate_payload, headers=_auth_headers())
        first_no_match = client.post(ENDPOINT, json=no_match_payload, headers=_auth_headers())
        replay_no_match = client.post(ENDPOINT, json=no_match_payload, headers=_auth_headers())

    assert first_cand.status_code == 200
    assert replay_cand.status_code == 200
    assert first_cand.headers.get("cache-control") == "no-store"
    assert replay_cand.headers.get("cache-control") == "no-store"
    assert first_cand.json()["state"] == "CANDIDATE"
    assert replay_cand.json() == first_cand.json()

    assert first_no_match.status_code == 200
    assert replay_no_match.status_code == 200
    assert first_no_match.json()["state"] == "UNAVAILABLE"
    assert first_no_match.json()["reason"] == "NO_MATCH"
    assert replay_no_match.json() == first_no_match.json()
    assert [frame.verification_id for frame in fake.frames] == [
        UUID(_uuid_for_index(1)),
        UUID(_uuid_for_index(2)),
    ]


def test_reused_id_with_changed_model_template_or_order_refuses_reverification() -> None:
    app, fake = _make_app()
    second_hash = "c" * 64
    second_template = "gAAAAAB" + ("B" * 133)
    base_candidates = [
        _make_candidate(profile_reference_hash=REFERENCE_HASH, encrypted_template=FERNET_TEMPLATE),
        _make_candidate(
            profile_reference_hash=second_hash,
            profile_revision="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            encrypted_template=second_template,
        ),
    ]
    initial_payload = _make_payload(
        invocation=_make_invocation(invocationId=INVOCATION_ID),
        candidates=base_candidates,
    )

    with TestClient(app) as client:
        first = client.post(ENDPOINT, json=initial_payload, headers=_auth_headers())
        assert first.status_code == 200
        assert len(fake.frames) == 1

        mutated_model = _make_payload(
            invocation=_make_invocation(
                invocationId=INVOCATION_ID,
                modelVersion="synthetic-face-v2",
            ),
            candidates=base_candidates,
        )
        res_model = client.post(ENDPOINT, json=mutated_model, headers=_auth_headers())
        assert res_model.status_code == 409
        assert res_model.headers.get("cache-control") == "no-store"

        mutated_template = _make_payload(
            invocation=_make_invocation(invocationId=INVOCATION_ID),
            candidates=[
                _make_candidate(
                    profile_reference_hash=REFERENCE_HASH,
                    encrypted_template="gAAAAAB" + ("C" * 133),
                ),
                base_candidates[1],
            ],
        )
        res_template = client.post(ENDPOINT, json=mutated_template, headers=_auth_headers())
        assert res_template.status_code == 409
        assert res_template.headers.get("cache-control") == "no-store"

        reordered_candidates = _make_payload(
            invocation=_make_invocation(invocationId=INVOCATION_ID),
            candidates=list(reversed(base_candidates)),
        )
        res_order = client.post(ENDPOINT, json=reordered_candidates, headers=_auth_headers())
        assert res_order.status_code == 409
        assert res_order.headers.get("cache-control") == "no-store"
        empty = client.post(ENDPOINT, json=_make_payload(candidates=[]), headers=_auth_headers())
        assert empty.status_code == 409
        assert empty.headers.get("cache-control") == "no-store"

    assert len(fake.frames) == 1


def test_transient_service_unavailable_is_not_cached() -> None:
    app, fake = _make_app(_DynamicFakeRecognizer(statuses=["AI_UNAVAILABLE", "MATCHED"]))
    payload = _make_payload()

    with TestClient(app) as client:
        first = client.post(ENDPOINT, json=payload, headers=_auth_headers())
        second = client.post(ENDPOINT, json=payload, headers=_auth_headers())

    assert first.status_code == 200
    assert first.json()["state"] == "UNAVAILABLE"
    assert first.json()["reason"] == "SERVICE_UNAVAILABLE"
    assert second.status_code == 200
    assert second.json()["state"] == "CANDIDATE"
    assert second.json()["profileReferenceHash"] == REFERENCE_HASH
    assert len(fake.frames) == 2


def test_replay_cache_caps_at_32_completed_entries_and_expires_after_60_monotonic_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1000.0
    import smartsite_ai.mf06_identity_cache as cache_module

    monkeypatch.setattr(cache_module, "monotonic", lambda: now)

    app, fake = _make_app()
    payloads = {
        idx: _make_payload(invocation=_make_invocation(invocationId=_uuid_for_index(idx)))
        for idx in range(1, 34)
    }

    with TestClient(app) as client:
        for idx in range(1, 33):
            res = client.post(ENDPOINT, json=payloads[idx], headers=_auth_headers())
            assert res.status_code == 200
        assert len(fake.frames) == 32

        cached_latest = client.post(ENDPOINT, json=payloads[32], headers=_auth_headers())
        assert cached_latest.status_code == 200
        assert len(fake.frames) == 32

        overflow = client.post(ENDPOINT, json=payloads[33], headers=_auth_headers())
        assert overflow.status_code == 200
        assert len(fake.frames) == 33

        evicted_first = client.post(ENDPOINT, json=payloads[1], headers=_auth_headers())
        assert evicted_first.status_code == 200
        assert len(fake.frames) == 34

        now += 60.1
        expired_latest = client.post(ENDPOINT, json=payloads[33], headers=_auth_headers())
        assert expired_latest.status_code == 200
        assert len(fake.frames) == 35


def test_recognizer_replacement_invalidates_replay_cache() -> None:
    app, fake_one = _make_app(_DynamicFakeRecognizer(statuses=["MATCHED"]))
    fake_two = _DynamicFakeRecognizer(statuses=["UNKNOWN"])
    payload = _make_payload()

    with TestClient(app) as client:
        first = client.post(ENDPOINT, json=payload, headers=_auth_headers())
        cached = client.post(ENDPOINT, json=payload, headers=_auth_headers())
        assert first.status_code == 200
        assert first.json()["state"] == "CANDIDATE"
        assert cached.json() == first.json()
        assert len(fake_one.frames) == 1

        app.state.face_recognizer = fake_two
        after_swap = client.post(ENDPOINT, json=payload, headers=_auth_headers())

    assert after_swap.status_code == 200
    assert after_swap.json()["state"] == "UNAVAILABLE"
    assert after_swap.json()["reason"] == "NO_MATCH"
    assert len(fake_one.frames) == 1
    assert len(fake_two.frames) == 1


def test_shutdown_and_new_lifespan_clears_replay_cache() -> None:
    app, fake = _make_app()
    payload = _make_payload()

    with TestClient(app) as client:
        first = client.post(ENDPOINT, json=payload, headers=_auth_headers())
        replay = client.post(ENDPOINT, json=payload, headers=_auth_headers())
        assert first.status_code == 200
        assert replay.json() == first.json()
        assert len(fake.frames) == 1

    with TestClient(app) as client:
        after_restart = client.post(ENDPOINT, json=payload, headers=_auth_headers())
        assert after_restart.status_code == 200
        assert len(fake.frames) == 2
