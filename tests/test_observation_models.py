import json
from pathlib import Path
from uuid import uuid4

import jsonschema
import pytest
from pydantic import ValidationError

from smartsite_ai.domain.observations import (
    BoundingBox,
    EvidenceItem,
    FrameDimensions,
    IdentityCandidateObservation,
    PersonObservation,
    PpeObservation,
    TechnicalObservationEvent,
    ZoneEntryObservation,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts/schemas/v1/technical-observation-event.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def create_sample_event_dict(**kwargs) -> dict:
    base = {
        "eventId": str(uuid4()),
        "schemaVersion": "1.0.0",
        "cameraExternalId": "CAM-01",
        "streamSessionId": str(uuid4()),
        "capturedAt": "2026-09-19T12:00:00Z",
        "frameDimensions": {"width": 1920, "height": 1080},
        "observations": [
            {
                "type": "PERSON",
                "trackId": 1,
                "confidence": 0.95,
                "boundingBox": {
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.5,
                    "y2": 0.8,
                    "coordinateSpace": "NORMALIZED_0_1",
                },
            }
        ],
        "evidence": [],
    }
    base.update(kwargs)
    return base


def test_valid_minimal_event_passes_model_and_json_schema():
    data = create_sample_event_dict()
    event = TechnicalObservationEvent.model_validate(data)
    assert str(event.event_id) == data["eventId"]
    assert event.camera_external_id == "CAM-01"

    wire = event.to_wire_dict()
    assert wire["eventId"] == data["eventId"]
    assert "event_id" not in wire

    # Cross-verify against vendored canonical JSON Schema
    errors = list(VALIDATOR.iter_errors(wire))
    assert not errors


def test_nested_extra_fields_rejected_at_all_levels():
    # 1. Root level extra
    with pytest.raises(ValidationError, match="extra"):
        TechnicalObservationEvent.model_validate(create_sample_event_dict(unexpectedRoot="bad"))

    # 2. FrameDimensions extra
    with pytest.raises(ValidationError, match="extra"):
        FrameDimensions.model_validate({"width": 1920, "height": 1080, "fps": 30})

    # 3. BoundingBox extra
    with pytest.raises(ValidationError, match="extra"):
        BoundingBox.model_validate(
            {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.5,
                "y2": 0.8,
                "coordinateSpace": "NORMALIZED_0_1",
                "extra": 1,
            }
        )

    # 4. PersonObservation extra
    with pytest.raises(ValidationError, match="extra"):
        PersonObservation.model_validate({"type": "PERSON", "trackId": 1, "custom": "field"})

    # 5. EvidenceItem extra
    with pytest.raises(ValidationError, match="extra"):
        EvidenceItem.model_validate(
            {"kind": "FRAME", "uri": "https://example.com/f.jpg", "extra": "data"}
        )


def test_bounding_box_geometry_invariants():
    # x1 >= x2 must fail
    with pytest.raises(ValidationError, match="strictly less than x2"):
        BoundingBox.model_validate(
            {"x1": 0.5, "y1": 0.1, "x2": 0.2, "y2": 0.8, "coordinateSpace": "NORMALIZED_0_1"}
        )
    with pytest.raises(ValidationError, match="strictly less than x2"):
        BoundingBox.model_validate(
            {"x1": 0.5, "y1": 0.1, "x2": 0.5, "y2": 0.8, "coordinateSpace": "NORMALIZED_0_1"}
        )

    # y1 >= y2 must fail
    with pytest.raises(ValidationError, match="strictly less than y2"):
        BoundingBox.model_validate(
            {"x1": 0.1, "y1": 0.8, "x2": 0.5, "y2": 0.3, "coordinateSpace": "NORMALIZED_0_1"}
        )
    with pytest.raises(ValidationError, match="strictly less than y2"):
        BoundingBox.model_validate(
            {"x1": 0.1, "y1": 0.5, "x2": 0.5, "y2": 0.5, "coordinateSpace": "NORMALIZED_0_1"}
        )

    # Out of bounds coordinates (< 0 or > 1)
    with pytest.raises(ValidationError):
        BoundingBox.model_validate(
            {"x1": -0.1, "y1": 0.1, "x2": 0.5, "y2": 0.8, "coordinateSpace": "NORMALIZED_0_1"}
        )
    with pytest.raises(ValidationError):
        BoundingBox.model_validate(
            {"x1": 0.1, "y1": 0.1, "x2": 1.2, "y2": 0.8, "coordinateSpace": "NORMALIZED_0_1"}
        )

    # Wrong coordinateSpace
    with pytest.raises(ValidationError):
        BoundingBox.model_validate(
            {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.8, "coordinateSpace": "PIXEL"}
        )


def test_identity_candidate_conditional_rules():
    # CANDIDATE status requires candidateWorkerId and similarityScore
    valid_cand = IdentityCandidateObservation.model_validate(
        {
            "type": "IDENTITY_CANDIDATE",
            "trackId": 10,
            "status": "CANDIDATE",
            "candidateWorkerId": "EMP-100",
            "similarityScore": 0.92,
            "qualityScore": 0.85,
        }
    )
    assert valid_cand.candidate_worker_id == "EMP-100"
    assert valid_cand.similarity_score == 0.92

    with pytest.raises(ValidationError, match="candidateWorkerId is required"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 10,
                "status": "CANDIDATE",
                "similarityScore": 0.92,
            }
        )

    with pytest.raises(ValidationError, match="similarityScore is required"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 10,
                "status": "CANDIDATE",
                "candidateWorkerId": "EMP-100",
            }
        )

    # UNKNOWN/UNAVAILABLE rejects candidateWorkerId and similarityScore
    valid_unknown = IdentityCandidateObservation.model_validate(
        {"type": "IDENTITY_CANDIDATE", "trackId": 10, "status": "UNKNOWN", "qualityScore": 0.5}
    )
    assert valid_unknown.candidate_worker_id is None
    assert valid_unknown.similarity_score is None

    with pytest.raises(ValidationError, match="candidateWorkerId is not allowed"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 10,
                "status": "UNKNOWN",
                "candidateWorkerId": "EMP-100",
            }
        )

    with pytest.raises(ValidationError, match="similarityScore is not allowed"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 10,
                "status": "UNAVAILABLE",
                "similarityScore": 0.8,
            }
        )


def test_ppe_and_zone_entry_observation_rules():
    region = str(uuid4())

    # Valid PPE
    ppe = PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": 5,
            "ppeItem": "HARD_HAT",
            "status": "MISSING",
            "regionId": region,
            "geometryVersion": 1,
            "confidence": 0.99,
        }
    )
    assert ppe.ppe_item == "HARD_HAT"
    assert ppe.status == "MISSING"

    # Missing regionId or geometryVersion
    with pytest.raises(ValidationError):
        PpeObservation.model_validate(
            {"type": "PPE", "trackId": 5, "ppeItem": "HARD_HAT", "status": "MISSING"}
        )

    # Invalid geometryVersion (< 1)
    with pytest.raises(ValidationError):
        PpeObservation.model_validate(
            {
                "type": "PPE",
                "trackId": 5,
                "ppeItem": "HARD_HAT",
                "status": "MISSING",
                "regionId": region,
                "geometryVersion": 0,
            }
        )

    # Valid Zone Entry
    zone = ZoneEntryObservation.model_validate(
        {"type": "ZONE_ENTRY", "trackId": 7, "regionId": region, "geometryVersion": 2}
    )
    assert zone.geometry_version == 2

    # Invalid Zone Entry (missing regionId)
    with pytest.raises(ValidationError):
        ZoneEntryObservation.model_validate(
            {"type": "ZONE_ENTRY", "trackId": 7, "geometryVersion": 2}
        )


def test_event_requires_at_least_one_observation():
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(create_sample_event_dict(observations=[]))


def test_serialization_uses_camel_case_and_excludes_none():
    event = TechnicalObservationEvent.model_validate(
        create_sample_event_dict(
            observations=[
                {
                    "type": "PPE",
                    "trackId": 2,
                    "ppeItem": "SAFETY_VEST",
                    "status": "PRESENT",
                    "regionId": str(uuid4()),
                    "geometryVersion": 1,
                    # confidence and boundingBox left None
                },
                {
                    "type": "IDENTITY_CANDIDATE",
                    "trackId": 2,
                    "status": "UNKNOWN",
                    # candidateWorkerId and similarityScore are None
                },
            ]
        )
    )

    wire = event.to_wire_dict()

    # None fields must be excluded
    ppe_obs = wire["observations"][0]
    assert "confidence" not in ppe_obs
    assert "boundingBox" not in ppe_obs

    ident_obs = wire["observations"][1]
    assert "candidateWorkerId" not in ident_obs
    assert "similarityScore" not in ident_obs

    # Must pass schema validation with strict additionalProperties=false
    errors = list(VALIDATOR.iter_errors(wire))
    assert not errors
