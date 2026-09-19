from fastapi.testclient import TestClient


def test_api_is_ready_without_models_cameras_or_openai_credentials():
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    with TestClient(create_app(Settings(_env_file=None))) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "ok", "service": "smartsite-ai"}
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "service": "smartsite-ai",
        "scope": "api",
        "inference_ready": False,
    }


def test_readiness_is_unavailable_before_startup_and_after_shutdown():
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    app = create_app(Settings(_env_file=None))
    client = TestClient(app)
    assert client.get("/health/ready").status_code == 503
    with client:
        assert client.get("/health/ready").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200


def test_capabilities_never_claim_inference_or_expose_external_credentials(monkeypatch):
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-secret")
    monkeypatch.setenv("CAMERA_URL", "rtsp://fake-user:fake-password@camera.invalid/live")
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_ready"] is False
    assert set(payload["capabilities"]) == {"camera", "detector", "zone", "identity", "openai"}
    assert payload["capabilities"]["detector"]["provider"] == "ultralytics-yolo11s + supervision"
    for capability in payload["capabilities"].values():
        assert capability["status"] == "not_configured"
        assert capability["reason"]
    assert "fake-test-secret" not in response.text
    assert "fake-password" not in response.text


def test_development_openapi_describes_real_endpoints():
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    with TestClient(create_app(Settings(environment="development", _env_file=None))) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert set(response.json()["paths"]) == {"/health/live", "/health/ready", "/v1/capabilities"}
    assert "503" in response.json()["paths"]["/health/ready"]["get"]["responses"]


def test_production_disables_documentation_routes():
    from smartsite_ai.app import create_app
    from smartsite_ai.config import Settings

    with TestClient(create_app(Settings(environment="production", _env_file=None))) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/health/live").status_code == 200
