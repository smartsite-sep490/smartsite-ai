"""Validated process configuration; no integration is enabled by environment alone."""

from pathlib import Path
from typing import Literal

from pydantic import Field, IPvAnyAddress, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SMARTSITE_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    host: IPvAnyAddress = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"

    backend_ingestion_url: str | None = None
    backend_service_token: SecretStr | None = None
    identity_service_token: SecretStr | None = None
    # Explicit opt-in only. The public InsightFace model pack is restricted to
    # non-commercial research; application startup never downloads a model.
    identity_demo_mode: bool = False
    identity_model_root: Path | None = None
    identity_template_store_path: Path | None = None
    identity_template_encryption_key: SecretStr | None = None
    identity_match_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
