from datetime import UTC, datetime
from uuid import UUID

import pytest
from test_ppe_model_profiles import document

from smartsite_ai.experimental_object_preview import unified_object_preview
from smartsite_ai.inference.loading import load_artifact_spec
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.inference.ppe_profiles import UNIFIED_PPE_15_PROFILE, experimental_ppe_profile
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.realtime import build_realtime_preview


def fixture():
    frame = FrameEnvelope(
        stream_id="test",
        session_id=UUID(int=1),
        camera_external_id="CAM-A",
        captured_at=datetime(2026, 10, 8, tzinfo=UTC),
        sequence_number=1,
        width=4,
        height=2,
        payload=bytes(24),
    )
    box = NormalizedBoundingBox(x1=0.1, y1=0.2, x2=0.7, y2=0.8)
    batch = DetectionBatch.from_frame(
        frame,
        model_artifact_id="experiment",
        model_version="v2",
        model_sha256="a" * 64,
        detections=[
            NormalizedDetection(class_id=i, class_name=name, confidence=0.8, bounding_box=box)
            for i, name in UNIFIED_PPE_15_PROFILE.class_map
        ],
    )
    return frame, batch


def test_all_fifteen_classes_are_same_frame_objects_without_wearer_states():
    frame, batch = fixture()
    objects = unified_object_preview(batch, frame)
    assert len(objects["detections"]) == 15
    assert objects["experimental"] is True
    assert objects["scope"] == "OBJECT_DETECTION_ONLY"
    assert not any("ppeStatus" in d or "trackId" in d for d in objects["detections"])
    result = build_realtime_preview(None, frame, model_objects=objects)
    assert result["modelObjects"] == objects
    assert result["detections"] == []
    assert result["sequenceNumber"] == "1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_external_id", "OTHER"),
        ("stream_id", "other"),
        ("sequence_number", 2),
        ("session_id", UUID(int=2)),
        ("frame_width", 5),
        ("captured_at", datetime(2026, 10, 9, tzinfo=UTC)),
    ],
)
def test_raw_objects_refuse_any_frame_lineage_mismatch(field, value):
    frame, batch = fixture()
    with pytest.raises(ValueError, match="lineage"):
        unified_object_preview(batch.model_copy(update={field: value}), frame)


def test_wrong_class_identity_is_not_projected():
    frame, batch = fixture()
    altered = batch.detections[0].model_copy(update={"class_name": "Not-A-Target"})
    with pytest.raises(ValueError, match="taxonomy"):
        unified_object_preview(batch.model_copy(update={"detections": (altered,)}), frame)


def test_empty_objects_do_not_create_missing_or_compliance():
    frame, batch = fixture()
    result = unified_object_preview(batch.model_copy(update={"detections": ()}), frame)
    assert result["detections"] == []
    assert "status" not in result
    assert "modelObjects" not in build_realtime_preview(None, frame)


def test_explicit_unified_profile_does_not_expand_immutable_wearer_contract(tmp_path):
    classes = {str(k): v for k, v in UNIFIED_PPE_15_PROFILE.class_map}
    artifact = document(tmp_path, classes)
    with pytest.raises(ValueError):
        load_artifact_spec(artifact)
    spec = load_artifact_spec(artifact, experimental_profile="unified-ppe-15")
    assert spec.class_map == UNIFIED_PPE_15_PROFILE.class_map
    assert experimental_ppe_profile("unified-ppe-15") == UNIFIED_PPE_15_PROFILE
    pipeline = PpePipeline.for_model_profile(UNIFIED_PPE_15_PROFILE)
    assert pipeline.minimum_schema_version == "1.1.0"
    assert UNIFIED_PPE_15_PROFILE.present_items == (
        "HARD_HAT",
        "SAFETY_VEST",
        "GLOVES",
        "BOOTS",
        "GOGGLES",
    )
    assert "MASK" not in UNIFIED_PPE_15_PROFILE.present_items


def test_unified_preview_profile_needs_explicit_consumer_version():
    from pydantic import ValidationError

    from smartsite_ai.config import Settings

    with pytest.raises(ValidationError, match="consumer"):
        Settings(_env_file=None, realtime_experimental_model_profile="unified-ppe-15")
    configured = Settings(
        _env_file=None,
        realtime_experimental_model_profile="unified-ppe-15",
        realtime_observation_schema_version="1.1.0",
    )
    assert configured.realtime_experimental_model_profile == "unified-ppe-15"
    assert Settings(_env_file=None).realtime_experimental_model_profile is None
