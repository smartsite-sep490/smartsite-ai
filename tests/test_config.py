import pytest
from pydantic import ValidationError


def test_reads_namespaced_environment_and_defaults_to_loopback(monkeypatch):
    from smartsite_ai.config import Settings

    monkeypatch.setenv("SMARTSITE_AI_ENVIRONMENT", "test")
    monkeypatch.setenv("SMARTSITE_AI_PORT", "8123")
    monkeypatch.setenv("SMARTSITE_AI_LOG_LEVEL", "warning")
    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert str(settings.host) == "127.0.0.1"
    assert settings.port == 8123
    assert settings.log_level == "warning"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SMARTSITE_AI_PORT", "0"),
        ("SMARTSITE_AI_PORT", "65536"),
        ("SMARTSITE_AI_PORT", "not-a-port"),
        ("SMARTSITE_AI_HOST", "not-an-address"),
        ("SMARTSITE_AI_ENVIRONMENT", "unknown"),
        ("SMARTSITE_AI_LOG_LEVEL", "verbose"),
    ],
)
def test_rejects_invalid_environment_before_server_start(name, value, monkeypatch):
    from smartsite_ai.config import Settings

    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_dotenv_accepts_unrelated_values_without_leaking_them(tmp_path):
    from smartsite_ai.config import Settings

    env_file = tmp_path / ".env"
    env_file.write_text("SMARTSITE_AI_PORT=8124\nUNRELATED_SECRET=fake-only\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)

    assert settings.port == 8124
    assert "fake-only" not in settings.model_dump_json()


def test_backend_ingestion_settings_default_to_none():
    from smartsite_ai.config import Settings

    settings = Settings(_env_file=None)
    assert settings.backend_ingestion_url is None
    assert settings.backend_service_token is None


def test_realtime_device_defaults_to_auto():
    from smartsite_ai.config import Settings

    settings = Settings(_env_file=None)

    assert settings.realtime_device == "auto"


@pytest.mark.parametrize("value", ["gpu", "cuda:-1", "cuda:abc", "cuda:0:1"])
def test_rejects_invalid_realtime_device(value, monkeypatch):
    from smartsite_ai.config import Settings

    monkeypatch.setenv("SMARTSITE_AI_REALTIME_DEVICE", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_backend_ingestion_settings_read_from_env_and_mask_token(monkeypatch):
    from smartsite_ai.config import Settings

    raw_token = "secret-svc-token-12345"
    monkeypatch.setenv("SMARTSITE_AI_BACKEND_INGESTION_URL", "http://backend.internal:3000")
    monkeypatch.setenv("SMARTSITE_AI_BACKEND_SERVICE_TOKEN", raw_token)

    settings = Settings(_env_file=None)
    assert settings.backend_ingestion_url == "http://backend.internal:3000"
    assert settings.backend_service_token is not None
    assert settings.backend_service_token.get_secret_value() == raw_token

    # Ensure token is never printed in plain text
    assert raw_token not in repr(settings)
    assert raw_token not in str(settings)
    assert raw_token not in settings.model_dump_json()
    assert "**********" in repr(settings.backend_service_token)
