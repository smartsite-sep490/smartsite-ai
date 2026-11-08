"""Candidate C04 raw-part storage receipt, separate from frame admission."""

from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from smartsite_ai.domain.configuration_bundle import _validate_lowercase_uuid
from smartsite_ai.domain.regions import _StrictFrozenModel


class TrackedRawReceipt(_StrictFrozenModel):
    kind: Literal["RAW_STORED"]
    event_id: str = Field(alias="eventId")
    payload_hash: str = Field(alias="payloadHash", pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(alias="bindingHash", pattern=r"^[0-9a-f]{64}$")

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="receipt eventId")

    def require_match(self, event_id: str, payload_hash: str, binding_hash: str) -> None:
        """Confirm this part was stored, with no implication for predecessor selection."""
        if (
            self.event_id != str(UUID(event_id))
            or self.payload_hash != payload_hash
            or self.binding_hash != binding_hash
        ):
            raise ValueError("Tracked raw receipt binding mismatch")
