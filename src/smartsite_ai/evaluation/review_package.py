"""Strict DRAFT-only contracts for local temporal PPE pre-label review packages."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from smartsite_ai.evaluation.models import EvaluationBoundingBox

MAX_REVIEW_FLAGS = 16
ReviewFlags = Annotated[tuple[str, ...], BeforeValidator(lambda value: tuple(value))]
_PPE_CLASS_SEMANTICS = {
    "hardhat": ("HARD_HAT", "PRESENT"),
    "no-hardhat": ("HARD_HAT", "MISSING"),
    "safety vest": ("SAFETY_VEST", "PRESENT"),
    "no-safety vest": ("SAFETY_VEST", "MISSING"),
}


class _StrictDraftModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        populate_by_name=True,
    )


class ProposalDetection(_StrictDraftModel):
    """One detector proposal tied to a sampled frame."""

    proposal_id: str = Field(alias="proposalId", min_length=1, max_length=256)
    class_id: int = Field(alias="classId", ge=0, le=2_147_483_647)
    class_name: str = Field(alias="className", min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: EvaluationBoundingBox = Field(alias="boundingBox")


class ProposalPerson(_StrictDraftModel):
    """A provisional clip-local tracker result; it is never a worker identity."""

    provisional_track_id: int = Field(alias="provisionalTrackId", ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: EvaluationBoundingBox = Field(alias="boundingBox")


class ProposalPpeAssociation(_StrictDraftModel):
    """A technical PPE proposal associated to one provisional person track."""

    provisional_track_id: int = Field(alias="provisionalTrackId", ge=1)
    source_detection_proposal_id: str = Field(
        alias="sourceDetectionProposalId", min_length=1, max_length=256
    )
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"] = Field(alias="ppeItem")
    proposed_status: Literal["PRESENT", "MISSING"] = Field(alias="proposedStatus")
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: EvaluationBoundingBox = Field(alias="boundingBox")


class DraftFrameProposal(_StrictDraftModel):
    """One machine-generated frame proposal that requires human review."""

    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    status: Literal["DRAFT"]
    frame_proposal_id: str = Field(alias="frameProposalId", min_length=1, max_length=192)
    clip_id: str = Field(alias="clipId", min_length=1, max_length=128)
    frame_index: int = Field(alias="frameIndex", ge=0, le=10_000_000)
    video_time_seconds: float = Field(alias="videoTimeSeconds", ge=0.0, le=604_800.0)
    width: int = Field(gt=0, le=16_384)
    height: int = Field(gt=0, le=16_384)
    image_path: str = Field(alias="imagePath", min_length=1, max_length=512)
    overlay_path: str = Field(alias="overlayPath", min_length=1, max_length=512)
    detections: tuple[ProposalDetection, ...] = Field(max_length=1_024)
    persons: tuple[ProposalPerson, ...] = Field(max_length=512)
    ppe_associations: tuple[ProposalPpeAssociation, ...] = Field(
        alias="ppeAssociations", max_length=1_024
    )
    review_priority: Literal["STANDARD", "MEDIUM", "HIGH"] = Field(alias="reviewPriority")
    review_flags: ReviewFlags = Field(
        alias="reviewFlags", min_length=1, max_length=MAX_REVIEW_FLAGS
    )

    @model_validator(mode="after")
    def validate_draft_semantics(self) -> DraftFrameProposal:
        if len(set(self.review_flags)) != len(self.review_flags):
            raise ValueError("reviewFlags must be unique")
        if "PROVISIONAL_TRACK_IDS" not in self.review_flags:
            raise ValueError("reviewFlags must disclose provisional track IDs")
        detections = {detection.proposal_id: detection for detection in self.detections}
        if len(detections) != len(self.detections):
            raise ValueError("detection proposalId values must be unique")
        track_ids = {person.provisional_track_id for person in self.persons}
        if len(track_ids) != len(self.persons):
            raise ValueError("person provisionalTrackId values must be unique")
        association_keys: set[tuple[int, str]] = set()
        for association in self.ppe_associations:
            if association.provisional_track_id not in track_ids:
                raise ValueError("PPE association references an unknown provisional track")
            source = detections.get(association.source_detection_proposal_id)
            if source is None:
                raise ValueError("PPE association references an unknown source detection")
            expected = _PPE_CLASS_SEMANTICS.get(source.class_name.casefold())
            if expected != (association.ppe_item, association.proposed_status):
                raise ValueError("PPE association contradicts source detection class semantics")
            if source.confidence != association.confidence:
                raise ValueError("PPE association confidence differs from its source detection")
            if source.bounding_box != association.bounding_box:
                raise ValueError("PPE association box differs from its source detection")
            key = (association.provisional_track_id, association.ppe_item)
            if key in association_keys:
                raise ValueError("PPE associations must be unique per track and PPE item")
            association_keys.add(key)
        return self


class DraftPackageConfiguration(_StrictDraftModel):
    cadence_seconds: float = Field(alias="cadenceSeconds", ge=0.1, le=1.0)
    maximum_allowed_gap_seconds: Literal[1.0] = Field(alias="maximumAllowedGapSeconds")
    tracker_scope: Literal["reset-at-each-clip-boundary"] = Field(alias="trackerScope")
    track_id_semantics: Literal["provisional-clip-local-not-worker-identity"] = Field(
        alias="trackIdSemantics"
    )
    local_only: Literal[True] = Field(alias="localOnly")


class DraftArtifactProvenance(_StrictDraftModel):
    artifact_spec_path: str = Field(alias="artifactSpecPath", min_length=1, max_length=4096)
    artifact_spec_sha256_before: str = Field(
        alias="artifactSpecSha256Before", pattern=r"^[0-9a-f]{64}$"
    )
    artifact_spec_sha256_after: str = Field(
        alias="artifactSpecSha256After", pattern=r"^[0-9a-f]{64}$"
    )
    checkpoint_path: str = Field(alias="checkpointPath", min_length=1, max_length=4096)
    checkpoint_sha256_before: str = Field(alias="checkpointSha256Before", pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256_after: str = Field(alias="checkpointSha256After", pattern=r"^[0-9a-f]{64}$")
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    model_family: Literal["yolo11s"] = Field(alias="modelFamily")
    class_map: dict[str, str] = Field(alias="classMap", min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_stable_hashes(self) -> DraftArtifactProvenance:
        if self.artifact_spec_sha256_before != self.artifact_spec_sha256_after:
            raise ValueError("artifact spec hash changed during package generation")
        if self.checkpoint_sha256_before != self.checkpoint_sha256_after:
            raise ValueError("checkpoint hash changed during package generation")
        return self


class DraftClipProvenance(_StrictDraftModel):
    clip_id: str = Field(alias="clipId", min_length=1, max_length=128)
    media_path: str = Field(alias="mediaPath", min_length=1, max_length=4096)
    sha256_before: str = Field(alias="sha256Before", pattern=r"^[0-9a-f]{64}$")
    sha256_after: str = Field(alias="sha256After", pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(alias="sizeBytes", gt=0, le=64 * 1024 * 1024 * 1024)
    width: int = Field(gt=0, le=16_384)
    height: int = Field(gt=0, le=16_384)
    fps: float = Field(gt=0.0, le=1_000.0)
    frame_count: int = Field(alias="frameCount", gt=0, le=10_000_000)
    duration_seconds: float = Field(alias="durationSeconds", gt=0.0, le=604_800.0)
    sample_count: int = Field(alias="sampleCount", gt=0, le=20_000)

    @model_validator(mode="after")
    def validate_stable_media(self) -> DraftClipProvenance:
        if self.sha256_before != self.sha256_after:
            raise ValueError("source media hash changed during package generation")
        return self


class DraftGitProvenance(_StrictDraftModel):
    available: Literal[True]
    commit_sha: str = Field(alias="commitSha", pattern=r"^[0-9a-f]{40}$")
    dirty: Literal[False]


class DraftReviewPackageManifest(_StrictDraftModel):
    """The only accepted manifest shape for unreviewed proposal output."""

    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    status: Literal["DRAFT"]
    package_type: Literal["PPE_TEMPORAL_PRELABEL_REVIEW_PACKAGE"] = Field(alias="packageType")
    ground_truth: Literal[False] = Field(alias="groundTruth")
    human_review_required: Literal[True] = Field(alias="humanReviewRequired")
    warning: Literal["Machine proposals only. This package is not reviewed ground truth."]
    generated_at_utc: datetime = Field(alias="generatedAtUtc")
    configuration: DraftPackageConfiguration
    artifact: DraftArtifactProvenance
    clips: tuple[DraftClipProvenance, ...] = Field(min_length=1, max_length=32)
    proposal_count: int = Field(alias="proposalCount", gt=0, le=20_000)
    proposals_path: Literal["proposals.jsonl"] = Field(alias="proposalsPath")
    git: DraftGitProvenance

    @model_validator(mode="after")
    def validate_utc_and_unique_clips(self) -> DraftReviewPackageManifest:
        if self.generated_at_utc.tzinfo is None or self.generated_at_utc.utcoffset() is None:
            raise ValueError("generatedAtUtc must be timezone-aware")
        if self.generated_at_utc.utcoffset().total_seconds() != 0:
            raise ValueError("generatedAtUtc must use UTC")
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clipId values must be unique")
        if self.proposal_count != sum(clip.sample_count for clip in self.clips):
            raise ValueError("proposalCount must equal the clip sample total")
        return self


__all__ = [
    "DraftFrameProposal",
    "DraftReviewPackageManifest",
    "ProposalDetection",
    "ProposalPerson",
    "ProposalPpeAssociation",
]
