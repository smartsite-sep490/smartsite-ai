import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from smartsite_ai.domain.configuration_bundle import (
    _validate_lowercase_uuid,
    _validate_utc_millisecond_timestamp,
)
from smartsite_ai.domain.regions import MAX_SAFE_INTEGER, JsonInteger, _StrictFrozenModel

MAX_CONFIGURATION_ACK_BYTES = 4096


class CameraConfigurationAcknowledgement(_StrictFrozenModel):
    command_id: str = Field(alias="commandId")
    worker_generation_id: str = Field(alias="workerGenerationId")
    stream_session_id: str = Field(alias="streamSessionId")
    configuration_version: JsonInteger = Field(
        alias="configurationVersion", ge=1, le=MAX_SAFE_INTEGER
    )
    bundle_sha256: str = Field(alias="bundleSha256", pattern=r"^[0-9a-f]{64}$")
    view_revision: JsonInteger = Field(alias="viewRevision", ge=1, le=MAX_SAFE_INTEGER)
    outcome: Literal["LOADED", "REJECTED"]
    reason_code: str | None = Field(default=None, alias="reasonCode")

    @field_validator("command_id", "worker_generation_id", "stream_session_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="acknowledgement ID")

    @model_validator(mode="after")
    def validate_reason(self) -> "CameraConfigurationAcknowledgement":
        if "reason_code" in self.model_fields_set and (
            self.outcome != "REJECTED"
            or self.reason_code is None
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", self.reason_code) is None
        ):
            raise ValueError("reasonCode is permitted only as a bounded REJECTED reason")
        return self


class CameraConfigurationAcknowledgementReceipt(_StrictFrozenModel):
    command_id: str = Field(alias="commandId")
    configuration_version: JsonInteger = Field(
        alias="configurationVersion", ge=1, le=MAX_SAFE_INTEGER
    )
    outcome: Literal["LOADED", "REJECTED"]
    received_at: str = Field(alias="receivedAt")
    replayed: bool

    @field_validator("command_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="receipt commandId")

    @field_validator("received_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validate_utc_millisecond_timestamp(value)
