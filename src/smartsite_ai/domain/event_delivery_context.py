"""Original nonsecret execution claim retained separately from the raw event."""

from typing import Literal

from pydantic import Field, field_validator

from smartsite_ai.domain.configuration_bundle import _validate_lowercase_uuid
from smartsite_ai.domain.regions import MAX_SAFE_INTEGER, JsonInteger, _StrictFrozenModel


class EventDeliveryContext(_StrictFrozenModel):
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    camera_id: str = Field(alias="cameraId")
    worker_generation_id: str = Field(alias="workerGenerationId")
    stream_session_id: str = Field(alias="streamSessionId")
    configuration_version: JsonInteger = Field(
        alias="configurationVersion", ge=1, le=MAX_SAFE_INTEGER
    )
    bundle_sha256: str = Field(alias="bundleSha256", pattern=r"^[0-9a-f]{64}$")
    view_revision: JsonInteger = Field(alias="viewRevision", ge=1, le=MAX_SAFE_INTEGER)
    receipt_command_id: str = Field(alias="receiptCommandId")
    source_kind: Literal["LIVE", "REPLAY"] = Field(alias="sourceKind")

    @field_validator("camera_id", "worker_generation_id", "stream_session_id", "receipt_command_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="delivery context ID")

    def binding_hash(self, event_payload_hash: str) -> str:
        from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash

        return compute_canonical_payload_hash(
            {
                "eventPayloadHash": event_payload_hash,
                "context": self.model_dump(mode="json", by_alias=True),
            }
        )
