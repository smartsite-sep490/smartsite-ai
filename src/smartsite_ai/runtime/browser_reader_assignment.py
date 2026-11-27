"""Private supervisor handoff for the scoped, read-only browser frame grant."""

import re
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from smartsite_ai.integrations.worker_generation import read_private_runtime_json
from smartsite_ai.runtime.manifest import CameraRuntimeError


class BrowserReaderAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    site_id: UUID = Field(alias="siteId")
    camera_id: UUID = Field(alias="cameraId")
    lease_id: UUID = Field(alias="leaseId")
    reader_secret: SecretStr = Field(alias="readerSecret")

    @field_validator("reader_secret")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        if re.fullmatch(r"[A-Za-z0-9_-]{43}", value.get_secret_value()) is None:
            raise ValueError("Invalid browser reader credential")
        return value


def load_browser_reader_assignment(path: Path) -> BrowserReaderAssignment:
    try:
        return BrowserReaderAssignment.model_validate(read_private_runtime_json(path))
    except (OSError, ValueError, RuntimeError):
        raise CameraRuntimeError("Browser reader assignment is unavailable or invalid") from None
