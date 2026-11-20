"""Draft control types; canonical Backend schemas are not vendored here yet."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _FrozenBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ResumeConfiguration(_FrozenBoundary):
    configurationVersion: int = Field(ge=1, le=2**53 - 1)
    viewRevision: int = Field(ge=1, le=2**53 - 1)
    bundleSha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ResumeBinding(_FrozenBoundary):
    cameraId: str
    workerGenerationId: str
    streamSessionId: str

    @field_validator("cameraId", "workerGenerationId", "streamSessionId")
    @classmethod
    def canonical_uuid(cls, value: str) -> str:
        return str(UUID(value))


class ResumePrepare(_FrozenBoundary):
    commandId: str
    expectedRevision: int = Field(ge=0, le=2**53 - 1)
    generationBinding: ResumeBinding
    configuration: ResumeConfiguration

    @field_validator("commandId")
    @classmethod
    def canonical_uuid(cls, value: str) -> str:
        return str(UUID(value))


class ResumeManifest(_FrozenBoundary):
    generationBinding: ResumeBinding
    configuration: ResumeConfiguration

    def require_configuration(self, configuration: ResumeConfiguration) -> None:
        """Call after authenticated bundle loading and before model/capture startup."""
        if configuration != self.configuration:
            raise ValueError("Expected camera configuration changed")
