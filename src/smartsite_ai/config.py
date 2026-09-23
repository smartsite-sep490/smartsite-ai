"""Validated process configuration; no integration is enabled by environment alone."""

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

    # Local realtime demo settings. Keep these opt-in so the API foundation
    # remains safe when no model/runtime is installed.
    realtime_model_path: str | None = None
    realtime_source: str | None = None
    realtime_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    realtime_zone_polygon: str = "0.63,0.2;0.98,0.2;0.98,0.9;0.63,0.9"
    realtime_camera_external_id: str = "ppe-demo"
    realtime_class_map_path: str | None = None
    realtime_region_configuration_path: str | None = None
    realtime_ppe_region_id: str = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
    realtime_model_artifact_id: str = "ansarimajid-construction-ppe-yolov8-reference"
    realtime_model_version: str = "8139436e91aecb109362e13cacfea44a16e08358"
    realtime_model_family: str = "yolov8"
    realtime_model_sha256: str = "5c981fd81432236cd6c88fa336697370f110383a62cc967f7759debf3c2b147e"
    realtime_model_source_url: str = (
        "https://raw.githubusercontent.com/Ansarimajid/Construction-PPE-Detection/"
        "8139436e91aecb109362e13cacfea44a16e08358/Model/ppe.pt"
    )
    realtime_model_license: str = "MIT (repository declaration; checkpoint terms unverified)"
