"""A private storage receipt is not a raw event receipt or frame admission."""

from typing import Literal

from pydantic import Field, field_validator

from smartsite_ai.domain.configuration_bundle import _validate_lowercase_uuid
from smartsite_ai.domain.regions import _StrictFrozenModel


class FrameEvidenceReceipt(_StrictFrozenModel):
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    kind: Literal["EVIDENCE_STORED"]
    event_id: str = Field(alias="eventId")
    payload_hash: str = Field(alias="payloadHash", pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(alias="bindingHash", pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="receipt eventId")

    def require_match(self, event_id: str, payload_hash: str, binding_hash: str, sha256: str) -> None:
        if (
            self.event_id != event_id
            or self.payload_hash != payload_hash
            or self.binding_hash != binding_hash
            or self.sha256 != sha256
        ):
            raise ValueError("Frame evidence receipt binding mismatch")
