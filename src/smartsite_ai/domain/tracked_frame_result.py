"""Candidate Backend seal result. A raw-storage receipt cannot select a predecessor."""

from typing import Annotated, Literal

from pydantic import AfterValidator, Field, model_validator

from smartsite_ai.domain.configuration_bundle import _validate_lowercase_uuid
from smartsite_ai.domain.regions import _StrictFrozenModel


def _canonical_uuid(value: str) -> str:
    return _validate_lowercase_uuid(value, field_name="tracked result ID")


CanonicalUuid = Annotated[str, AfterValidator(_canonical_uuid)]


class AdmittedSubject(_StrictFrozenModel):
    segment_id: CanonicalUuid = Field(alias="segmentId")
    event_id: CanonicalUuid = Field(alias="eventId")
    person_index: int = Field(alias="personObservationIndex", ge=0, le=255)


class TrackedFrameReferences(_StrictFrozenModel):
    scope_sha256: str = Field(alias="scopeSha256", pattern=r"^[0-9a-f]{64}$")
    sequence_number: str = Field(alias="sequenceNumber", pattern=r"^(0|[1-9][0-9]{0,19})$")
    subjects: Annotated[tuple[AdmittedSubject, ...], Field(strict=False)]
    retired_segment_ids: Annotated[tuple[CanonicalUuid, ...], Field(strict=False)] = Field(
        alias="retiredSegmentIds"
    )

    @model_validator(mode="after")
    def validate_frame(self) -> "TrackedFrameReferences":
        subjects = self.subject_references()
        retired = set(self.retired_segment_ids)
        if (
            int(self.sequence_number) > 18_446_744_073_709_551_615
            or len(subjects) != len(self.subjects)
            or len(set(subjects.values())) != len(subjects)
            or len(retired) != len(self.retired_segment_ids)
            or retired.intersection(subjects)
        ):
            raise ValueError("Invalid admitted tracked frame")
        return self

    def subject_references(self) -> dict[str, tuple[str, int]]:
        return {
            subject.segment_id: (subject.event_id, subject.person_index)
            for subject in self.subjects
        }

    def require_match(
        self,
        scope_sha256: str,
        sequence_number: str,
        subjects: dict[str, tuple[str, int]],
        retired_segment_ids: set[str],
    ) -> None:
        """Bind a result from trusted Backend transport to exact sent-frame references."""
        if (
            self.scope_sha256 != scope_sha256
            or self.sequence_number != sequence_number
            or self.subject_references() != subjects
            or set(self.retired_segment_ids) != retired_segment_ids
        ):
            raise ValueError("Tracked admitted frame binding mismatch")


class TrackedFrameResult(TrackedFrameReferences):
    kind: Literal["ADMITTED_TRACKED_FRAME"]
