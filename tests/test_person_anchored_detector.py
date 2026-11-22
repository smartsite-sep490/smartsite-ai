import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.inference.person_anchored import PersonAnchoredPpeDetector
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope


def frame():
    return FrameEnvelope(
        stream_id="test",
        session_id=uuid4(),
        camera_external_id="camera",
        captured_at=datetime.now(UTC),
        width=4,
        height=4,
        sequence_number=1,
        payload=bytes(48),
    )


def detection(name, left=0.1):
    return NormalizedDetection(
        class_id=0,
        class_name=name,
        confidence=0.8,
        bounding_box=NormalizedBoundingBox(x1=left, y1=0.1, x2=left + 0.2, y2=0.8),
    )


class FakeDetector:
    def __init__(self, objects, sha="a" * 64, mismatch=None):
        self.objects, self.sha, self.mismatch = objects, sha, mismatch
        self.seen = []

    async def detect(self, source):
        self.seen.append(source)
        batch = DetectionBatch.from_frame(
            source,
            model_artifact_id="fake",
            model_version="test",
            model_sha256=self.sha,
            detections=self.objects,
        )
        return batch.model_copy(update=self.mismatch) if self.mismatch else batch


@pytest.mark.parametrize(
    "people", [(), (detection("person"),), (detection("person"), detection("person", 0.6))]
)
def test_reference_people_replace_duplicate_ppe_people_without_removing_ppe(people):
    source = frame()
    hat = detection("Hardhat")
    ppe = FakeDetector((detection("Person"), detection("Person", 0.2), hat))
    persons = FakeDetector((*people, detection("chair")), sha="b" * 64)
    result = asyncio.run(PersonAnchoredPpeDetector(ppe=ppe, persons=persons).detect(source))
    assert result.detections == tuple(
        sorted(
            (*people, hat),
            key=lambda d: (
                -d.confidence,
                d.class_id,
                d.bounding_box.x1,
                d.bounding_box.y1,
                d.bounding_box.x2,
                d.bounding_box.y2,
                d.class_name,
            ),
        )
    )
    assert ppe.seen[0] is source and persons.seen[0] is source
    assert result.model_artifact_id == "person-anchored-ppe"
    assert result.model_sha256 not in ("a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_external_id", "other"),
        ("stream_id", "other"),
        ("session_id", uuid4()),
        ("sequence_number", 3),
        ("frame_width", 6),
        ("frame_height", 6),
        ("captured_at", datetime(2026, 1, 1, tzinfo=UTC)),
    ],
)
@pytest.mark.parametrize("side", ["ppe", "persons"])
def test_mismatched_source_batch_is_not_relabelled_as_the_current_frame(field, value, side):
    detectors = {
        key: FakeDetector((), mismatch={field: value} if key == side else None)
        for key in ("ppe", "persons")
    }
    with pytest.raises(ValueError, match="frame"):
        asyncio.run(PersonAnchoredPpeDetector(**detectors).detect(frame()))


def test_composite_identity_binds_both_artifact_versions():
    source = frame()

    def run(ppe_sha, person_sha):
        return asyncio.run(
            PersonAnchoredPpeDetector(
                ppe=FakeDetector((), ppe_sha), persons=FakeDetector((), person_sha)
            ).detect(source)
        ).model_sha256

    baseline = run("a" * 64, "b" * 64)
    assert baseline == run("a" * 64, "b" * 64)
    assert len({baseline, run("c" * 64, "b" * 64), run("a" * 64, "c" * 64)}) == 3


@pytest.mark.parametrize(
    "error", [RuntimeError("person provider unavailable"), asyncio.CancelledError()]
)
def test_person_provider_failure_never_falls_back_to_duplicate_ppe_people(error):
    class FailedPersons:
        async def detect(self, source):
            raise error

    with pytest.raises(type(error)):
        asyncio.run(
            PersonAnchoredPpeDetector(
                ppe=FakeDetector((detection("Person"), detection("Person", 0.2))),
                persons=FailedPersons(),
            ).detect(frame())
        )


def test_composite_fingerprint_changes_when_artifact_role_order_id_or_version_changes():
    source = frame()

    class NamedDetector:
        def __init__(self, artifact_id: str, version: str, sha: str) -> None:
            self._artifact_id = artifact_id
            self._version = version
            self._sha = sha

        async def detect(self, current: FrameEnvelope) -> DetectionBatch:
            return DetectionBatch.from_frame(
                current,
                model_artifact_id=self._artifact_id,
                model_version=self._version,
                model_sha256=self._sha,
                detections=(),
            )

    first = NamedDetector("ppe-artifact", "v1", "a" * 64)
    second = NamedDetector("coco-person", "v2", "b" * 64)
    adapter = PersonAnchoredPpeDetector(ppe=first, persons=second)
    assert isinstance(adapter, DetectorProtocol)

    normal = asyncio.run(adapter.detect(source)).model_sha256
    swapped = asyncio.run(
        PersonAnchoredPpeDetector(ppe=second, persons=first).detect(source)
    ).model_sha256
    renamed = asyncio.run(
        PersonAnchoredPpeDetector(
            ppe=NamedDetector("ppe-artifact-other", "v1", "a" * 64),
            persons=second,
        ).detect(source)
    ).model_sha256
    reversioned = asyncio.run(
        PersonAnchoredPpeDetector(
            ppe=first,
            persons=NamedDetector("coco-person", "v3", "b" * 64),
        ).detect(source)
    ).model_sha256

    assert len({normal, swapped, renamed, reversioned}) == 4


def test_primary_non_person_ppe_detections_are_preserved_verbatim_and_primary_failure_propagates():
    source = frame()
    hardhat = NormalizedDetection(
        class_id=1,
        class_name="Hardhat",
        confidence=0.91,
        bounding_box=NormalizedBoundingBox(x1=0.12, y1=0.05, x2=0.25, y2=0.18),
    )
    no_vest = NormalizedDetection(
        class_id=4,
        class_name="NO-Safety Vest",
        confidence=0.67,
        bounding_box=NormalizedBoundingBox(x1=0.15, y1=0.35, x2=0.45, y2=0.72),
    )
    coco_person = NormalizedDetection(
        class_id=0,
        class_name="person",
        confidence=0.93,
        bounding_box=NormalizedBoundingBox(x1=0.1, y1=0.05, x2=0.5, y2=0.95),
    )

    result = asyncio.run(
        PersonAnchoredPpeDetector(
            ppe=FakeDetector(
                (detection("Person", 0.0), detection("person", 0.5), hardhat, no_vest)
            ),
            persons=FakeDetector((coco_person, detection("couch", 0.0)), sha="b" * 64),
        ).detect(source)
    )

    assert result.detections == (coco_person, hardhat, no_vest)

    class FailedPpe:
        async def detect(self, current: FrameEnvelope) -> DetectionBatch:
            raise RuntimeError("primary ppe failed")

    with pytest.raises(RuntimeError, match="primary ppe failed"):
        asyncio.run(
            PersonAnchoredPpeDetector(
                ppe=FailedPpe(),
                persons=FakeDetector((coco_person,), sha="b" * 64),
            ).detect(source)
        )
