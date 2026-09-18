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
