"""A supplemental detector cannot replace the existing Person/hat/vest evidence."""

import asyncio
from datetime import timedelta
from uuid import UUID

import pytest
from test_person_anchored_detector import FakeDetector, frame
from test_processing_worker import _configuration

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.inference.models import DetectionBatch, NormalizedDetection
from smartsite_ai.inference.person_anchored import (
    PersonAnchoredPpeDetector,
    SupplementalPpeDetector,
)
from smartsite_ai.inference.ppe_profiles import UNIFIED_PPE_15_PROFILE
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.runtime.analysis_preview import encode_analysis_frame


def objects():
    return tuple(
        NormalizedDetection(
            class_id=i,
            class_name=name,
            confidence=0.8,
            bounding_box={"x1": 0.1, "y1": 0.1, "x2": 0.6, "y2": 0.8},
        )
        for i, name in UNIFIED_PPE_15_PROFILE.class_map
    )


def test_core_objects_survive_unchanged_and_supplement_cannot_create_people_or_override_core():
    source = frame()
    original = objects()[:5]
    extra = tuple(d.model_copy(update={"confidence": 0.99}) for d in objects())
    core, supplement = FakeDetector(original), FakeDetector(extra, sha="b" * 64)
    result = asyncio.run(SupplementalPpeDetector(core=core, supplement=supplement).detect(source))
    assert len(result.detections) == 15
    assert tuple(d for d in result.detections if d.class_id < 5) == original
    assert all(
        next(d for d in result.detections if d.class_id == i) is original[i] for i in range(5)
    )
    assert tuple(d for d in result.detections if d.class_id >= 5) == extra[5:]
    assert len([d for d in result.detections if d.class_name == "Person"]) == 1
    assert core.seen == [source] and supplement.seen == [source]
    assert result.model_artifact_id == "supplemental-ppe"
    assert result.model_sha256 not in ("a" * 64, "b" * 64)


@pytest.mark.parametrize("owner", ["core", "supplement"])
@pytest.mark.parametrize(
    "field",
    [
        "camera_external_id",
        "stream_id",
        "session_id",
        "captured_at",
        "sequence_number",
        "frame_width",
        "frame_height",
    ],
)
def test_neither_detector_may_return_another_frame(owner, field):
    source = frame()
    values = {
        "camera_external_id": "other",
        "stream_id": "other",
        "session_id": UUID("99999999-9999-4999-8999-999999999999"),
        "captured_at": source.captured_at + timedelta(seconds=1),
        "sequence_number": source.sequence_number + 1,
        "frame_width": source.width + 1,
        "frame_height": source.height + 1,
    }
    core = FakeDetector(objects()[:5], mismatch={field: values[field]} if owner == "core" else None)
    extra = FakeDetector(
        objects(), mismatch={field: values[field]} if owner == "supplement" else None
    )
    with pytest.raises(ValueError, match="frame"):
        asyncio.run(SupplementalPpeDetector(core=core, supplement=extra).detect(source))


@pytest.mark.parametrize("owner", ["core", "supplement"])
def test_taxonomy_drift_is_rejected_instead_of_silently_reinterpreted(owner):
    wrong = objects()[1].model_copy(update={"class_name": "Gloves"})
    core = FakeDetector((wrong,) if owner == "core" else objects()[:5])
    extra = FakeDetector((wrong,) if owner == "supplement" else objects())
    with pytest.raises(ValueError, match="class"):
        asyncio.run(SupplementalPpeDetector(core=core, supplement=extra).detect(frame()))


def test_core_cannot_claim_supplemental_classes_and_empty_supplement_does_not_infer_absence():
    source = frame()
    with pytest.raises(ValueError, match="class"):
        asyncio.run(
            SupplementalPpeDetector(
                core=FakeDetector(objects()),
                supplement=FakeDetector(()),
            ).detect(source)
        )
    result = asyncio.run(
        SupplementalPpeDetector(
            core=FakeDetector(objects()[:5]),
            supplement=FakeDetector(()),
        ).detect(source)
    )
    assert result.detections == objects()[:5]


def test_composition_provenance_is_deterministic_and_binds_each_source_version_and_sha():
    source = frame()

    def compose(core_sha="a" * 64, extra_sha="b" * 64):
        return asyncio.run(
            SupplementalPpeDetector(
                core=FakeDetector(objects()[:5], sha=core_sha),
                supplement=FakeDetector(objects(), sha=extra_sha),
            ).detect(source)
        )

    assert compose().model_sha256 == compose().model_sha256
    assert (
        len(
            {
                compose().model_sha256,
                compose(core_sha="c" * 64).model_sha256,
                compose(extra_sha="c" * 64).model_sha256,
            }
        )
        == 3
    )


def test_outer_person_adapter_remains_the_only_person_owner():
    source = frame()
    core = FakeDetector(objects()[:5])
    extra = FakeDetector(objects(), sha="b" * 64)
    coco_person = objects()[0].model_copy(update={"class_name": "person", "confidence": 0.95})
    result = asyncio.run(
        PersonAnchoredPpeDetector(
            ppe=SupplementalPpeDetector(core=core, supplement=extra),
            persons=FakeDetector((coco_person,), sha="c" * 64),
        ).detect(source)
    )
    assert tuple(d for d in result.detections if d.class_id == 0) == (coco_person,)
    assert len(result.detections) == 15


@pytest.mark.parametrize("owner", ["core", "supplement"])
@pytest.mark.parametrize("exc_type", [RuntimeError, asyncio.CancelledError])
def test_errors_and_cancellation_propagate_from_both_detectors(owner, exc_type):
    source = frame()

    class _FailingDetector:
        async def detect(self, src):
            raise exc_type("boom")

    valid_core = FakeDetector(objects()[:5])
    valid_extra = FakeDetector(objects(), sha="b" * 64)
    core = _FailingDetector() if owner == "core" else valid_core
    extra = _FailingDetector() if owner == "supplement" else valid_extra
    with pytest.raises(exc_type):
        asyncio.run(SupplementalPpeDetector(core=core, supplement=extra).detect(source))


def test_more_than_1024_merged_detections_is_rejected():
    source = frame()
    core_items = tuple(
        NormalizedDetection(
            class_id=1,
            class_name="Hardhat",
            confidence=0.8,
            bounding_box={"x1": 0.1 + (i % 20) * 0.01, "y1": 0.1, "x2": 0.5, "y2": 0.8},
        )
        for i in range(500)
    )
    extra_items = tuple(
        NormalizedDetection(
            class_id=5,
            class_name="Gloves",
            confidence=0.8,
            bounding_box={"x1": 0.1 + (i % 20) * 0.01, "y1": 0.1, "x2": 0.5, "y2": 0.8},
        )
        for i in range(525)
    )
    with pytest.raises(ValueError):
        asyncio.run(
            SupplementalPpeDetector(
                core=FakeDetector(core_items),
                supplement=FakeDetector(extra_items, sha="b" * 64),
            ).detect(source)
        )


def test_composition_provenance_binds_model_artifact_id_and_model_version():
    source = frame()

    class _MetaDetector:
        def __init__(self, items, artifact_id: str, version: str, sha: str) -> None:
            self._items = items
            self._artifact_id = artifact_id
            self._version = version
            self._sha = sha

        async def detect(self, src):
            return DetectionBatch.from_frame(
                src,
                model_artifact_id=self._artifact_id,
                model_version=self._version,
                model_sha256=self._sha,
                detections=self._items,
            )

    def digest(
        *,
        core_id: str = "core-a",
        core_ver: str = "v1",
        extra_id: str = "extra-a",
        extra_ver: str = "v1",
    ) -> str:
        return asyncio.run(
            SupplementalPpeDetector(
                core=_MetaDetector(objects()[:5], core_id, core_ver, "a" * 64),
                supplement=_MetaDetector(objects(), extra_id, extra_ver, "b" * 64),
            ).detect(source)
        ).model_sha256

    digests = {
        digest(),
        digest(core_id="core-b"),
        digest(core_ver="v2"),
        digest(extra_id="extra-b"),
        digest(extra_ver="v2"),
    }
    assert len(digests) == 5


def test_outer_person_anchored_supplemental_preview_preserves_taxonomy_and_coexistence():
    configuration = _configuration()
    region_id = configuration.regions[0].region_id
    source = frame().model_copy(update={"camera_external_id": configuration.camera_external_id})
    coco_person = objects()[0].model_copy(update={"class_name": "person"})
    core = FakeDetector(objects()[:5])
    extra = FakeDetector(objects(), sha="b" * 64)
    composite = asyncio.run(
        PersonAnchoredPpeDetector(
            ppe=SupplementalPpeDetector(core=core, supplement=extra),
            persons=FakeDetector((coco_person,), sha="c" * 64),
        ).detect(source)
    )
    event = TechnicalObservationEvent.create(
        event_id="11111111-1111-4111-8111-111111111111",
        camera_external_id=source.camera_external_id,
        stream_session_id=str(source.session_id),
        captured_at=source.captured_at_iso,
        frame_dimensions={"width": source.width, "height": source.height},
        observations=[
            {
                "type": "PERSON",
                "trackId": 7,
                "boundingBox": {
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.6,
                    "y2": 0.8,
                    "coordinateSpace": "NORMALIZED_0_1",
                },
            },
            {
                "type": "PPE",
                "trackId": 7,
                "regionId": region_id,
                "geometryVersion": 1,
                "ppeItem": "HARD_HAT",
                "status": "MISSING",
            },
        ],
    )
    payload = encode_analysis_frame(
        AnalysisFrame(
            frame=source,
            observation=event,
            configuration=configuration,
            ppe_region_id=region_id,
            confirmed_ppe_items=frozenset({(7, "HARD_HAT")}),
            raw_detections=composite,
        )
    )
    preview_objects = payload["modelObjects"]
    assert preview_objects["artifactId"] == "person-anchored-ppe"
    assert preview_objects["artifactSha256"] == composite.model_sha256
    assert [(d["classId"], d["className"]) for d in preview_objects["detections"]] == list(
        UNIFIED_PPE_15_PROFILE.class_map
    )
    assert composite.detections[0].class_name == "person"
    assert payload["detections"][0]["ppeStatus"]["HARD_HAT"] == "MISSING"
    assert payload["detections"][0]["confirmedMissingItems"] == ["HARD_HAT"]
