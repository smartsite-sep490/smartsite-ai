"""Audit declared source groups; candidate similarity never proves independence."""

import hashlib
import json
from collections import defaultdict
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _unambiguous_text(value: str) -> str:
    if not value.strip() or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(
            "identifiers/reasons must be nonblank, unpadded and free of control characters"
        )
    return value


AuditText = Annotated[str, AfterValidator(_unambiguous_text)]


class SplitSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sample_id: AuditText = Field(alias="sampleId", min_length=1, max_length=512)
    split: Literal["train", "val", "test"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_group_id: AuditText | None = Field(
        alias="sourceGroupId", default=None, min_length=1, max_length=256
    )


class CandidateLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    left_sample_id: AuditText = Field(alias="leftSampleId", min_length=1, max_length=512)
    right_sample_id: AuditText = Field(alias="rightSampleId", min_length=1, max_length=512)
    reason: AuditText = Field(min_length=1, max_length=256)


class SplitAuditManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    samples: list[SplitSample] = Field(min_length=1, max_length=100_000)
    links: list[CandidateLink] = Field(default_factory=list, max_length=250_000)

    @model_validator(mode="after")
    def validate_references(self) -> "SplitAuditManifest":
        ids = {sample.sample_id for sample in self.samples}
        if len(ids) != len(self.samples):
            raise ValueError("sampleId must be unique")
        for link in self.links:
            if link.left_sample_id not in ids or link.right_sample_id not in ids:
                raise ValueError("candidate link references an unknown sample")
            if link.left_sample_id == link.right_sample_id:
                raise ValueError("candidate link must connect distinct samples")
        return self


class SplitGroup(TypedDict):
    groupId: str
    sampleIds: list[str]
    splits: list[str]


class SplitAuditReport(TypedDict):
    manifestSha256: NotRequired[str]
    schemaVersion: str
    status: str
    scope: str
    datasetAccepted: bool
    fileBytesVerified: bool
    datasetModified: bool
    sampleCount: int
    declaredSourceGroupCount: int
    samplesWithoutSourceGroup: list[str]
    missingSplits: list[str]
    confirmedCrossSplitGroups: list[SplitGroup]
    candidateCrossSplitGroups: list[SplitGroup]
    warning: str


class _Groups:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {key: key for key in ids}
        self.size = dict.fromkeys(ids, 1)

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            previous = self.parent[key]
            self.parent[key] = root
            key = previous
        return root

    def union(self, a: str, b: str) -> None:
        left, right = self.find(a), self.find(b)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def _cross_split_groups(groups: _Groups, samples: list[SplitSample]) -> list[SplitGroup]:
    members: dict[str, list[SplitSample]] = defaultdict(list)
    for sample in samples:
        members[groups.find(sample.sample_id)].append(sample)
    result: list[SplitGroup] = []
    for rows in members.values():
        splits = sorted({row.split for row in rows})
        if len(splits) < 2:
            continue
        ids = sorted(row.sample_id for row in rows)
        digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()
        result.append({"groupId": digest, "sampleIds": ids, "splits": splits})
    return sorted(result, key=lambda group: group["sampleIds"])


def audit_split_groups(manifest: SplitAuditManifest) -> SplitAuditReport:
    """Report overlaps from metadata, without reading files or moving samples."""
    ids = [sample.sample_id for sample in manifest.samples]
    confirmed = _Groups(ids)
    hashes: dict[str, str] = {}
    source_groups: dict[str, str] = {}
    for sample in manifest.samples:
        previous = hashes.setdefault(sample.sha256, sample.sample_id)
        confirmed.union(previous, sample.sample_id)
        if sample.source_group_id is not None:
            previous = source_groups.setdefault(sample.source_group_id, sample.sample_id)
            confirmed.union(previous, sample.sample_id)

    candidates = _Groups(ids)
    for sample_id in ids:
        candidates.union(sample_id, confirmed.find(sample_id))
    for link in manifest.links:
        candidates.union(link.left_sample_id, link.right_sample_id)
    review_roots = {
        candidates.find(link.left_sample_id)
        for link in manifest.links
        if confirmed.find(link.left_sample_id) != confirmed.find(link.right_sample_id)
    }
    known = _cross_split_groups(confirmed, manifest.samples)
    pending = [
        group
        for group in _cross_split_groups(candidates, manifest.samples)
        if candidates.find(group["sampleIds"][0]) in review_roots
    ]
    missing = sorted(
        sample.sample_id for sample in manifest.samples if sample.source_group_id is None
    )
    if known:
        status = "BLOCKED_KNOWN_OVERLAP"
    elif pending:
        status = "REVIEW_REQUIRED"
    elif missing:
        status = "INCOMPLETE_SOURCE_GROUPS"
    else:
        status = "NO_KNOWN_OVERLAP"
    return {
        "schemaVersion": "1.0.0",
        "status": status,
        "scope": "DECLARED_SPLIT_METADATA_ONLY",
        "datasetAccepted": False,
        "fileBytesVerified": False,
        "datasetModified": False,
        "sampleCount": len(ids),
        "declaredSourceGroupCount": len(source_groups),
        "samplesWithoutSourceGroup": missing,
        "missingSplits": [
            split
            for split in ("train", "val", "test")
            if split not in {sample.split for sample in manifest.samples}
        ],
        "confirmedCrossSplitGroups": known,
        "candidateCrossSplitGroups": pending,
        "warning": (
            "Declared hashes/groups require source verification. Similarity links are only "
            "candidates. No known overlap is not annotation, license, independent-holdout "
            "or model acceptance."
        ),
    }
