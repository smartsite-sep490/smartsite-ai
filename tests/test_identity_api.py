from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import SecretStr

TOKEN = "local-demo-identity-service-token-123456"
ENROLLMENT_ID = UUID("00000000-0000-4000-8000-000000000020")


def make_client() -> TestClient:
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    return TestClient(create_app(Settings(identity_service_token=SecretStr(TOKEN), _env_file=None)))


def sample_url(index: int) -> str:
    return f"/v1/identity/enrollments/{ENROLLMENT_ID}/samples/{index}"


def test_identity_enrollment_requires_service_authentication() -> None:
    with make_client() as client:
        response = client.post(
            sample_url(1), content=b"synthetic-jpeg", headers={"content-type": "image/jpeg"}
        )

    assert response.status_code == 401


def test_identity_enrollment_accepts_exactly_three_ordered_ephemeral_jpegs() -> None:
    headers = {"authorization": f"Bearer {TOKEN}", "content-type": "image/jpeg"}
    with make_client() as client:
        assert (
            client.post(sample_url(2), content=b"synthetic-jpeg", headers=headers).status_code
            == 409
        )
        for index in (1, 2, 3):
            response = client.post(sample_url(index), content=b"synthetic-jpeg", headers=headers)
            assert response.status_code == 200
            assert response.json() == {"acceptedSampleCount": index}
        completion = client.post(
            f"/v1/identity/enrollments/{ENROLLMENT_ID}/complete",
            headers={"authorization": f"Bearer {TOKEN}"},
        )

    assert completion.status_code == 200
    assert completion.json() == {
        "status": "AI_UNAVAILABLE",
        "modelVersion": None,
        "profileReference": None,
        "encryptedTemplate": None,
        "reasonCode": "FACE_MODEL_NOT_CONFIGURED",
    }


def test_database_verification_requires_auth_and_does_not_echo_sensitive_payload():
    url = f"/v1/identity/verifications/{ENROLLMENT_ID}"
    with make_client() as client:
        assert client.post(url, json={"jpegBase64": "YQ==", "templates": []}).status_code == 401
        response = client.post(
            url,
            json={"jpegBase64": "invalid-private-face", "templates": []},
            headers={"authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 422
        assert "invalid-private-face" not in response.text
        response = client.post(
            url,
            json={"jpegBase64": "YQ==", "templates": []},
            headers={"authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "AI_UNAVAILABLE"
        response = client.post(
            url,
            json={"jpegBase64": "YQ==", "templates": [], "enrollmentTarget": "left"},
            headers={"authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "AI_UNAVAILABLE"
        response = client.post(
            url,
            json={
                "jpegBase64": "YQ==",
                "templates": [],
                "enrollmentTarget": "secret-invalid-target",
            },
            headers={"authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 422
        assert "secret-invalid-target" not in response.text
