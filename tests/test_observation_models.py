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
    validate_rfc3339_captured_at,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts/schemas/v1/technical-observation-event.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

# Configure FormatChecker with robust RFC 3339 date-time and UUID checkers
FORMAT_CHECKER = jsonschema.FormatChecker()


@FORMAT_CHECKER.checks("date-time")
def check_rfc3339_datetime(val: object) -> bool:
    if not isinstance(val, str):
        return False
    try:
        validate_rfc3339_captured_at(val)
        return True
    except ValueError:
        return False


VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, format_checker=FORMAT_CHECKER)


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


# ============================================================================
# 1. Anti-drift test with functioning FormatChecker
# ============================================================================


def test_anti_drift_format_checker_detects_invalid_formats():
    """Verifies that the test harness FormatChecker actively rejects invalid date-time and UUID."""
    assert not FORMAT_CHECKER.conforms("not-a-date", "date-time")
    assert not FORMAT_CHECKER.conforms("not-a-uuid", "uuid")
    assert FORMAT_CHECKER.conforms("2026-09-18T10:00:00Z", "date-time")
    assert FORMAT_CHECKER.conforms("2026-12-31T23:59:60Z", "date-time")
    assert FORMAT_CHECKER.conforms(str(uuid4()), "uuid")

    bad_date_payload = create_sample_event_dict(capturedAt="not-a-date")
    errors = list(VALIDATOR.iter_errors(bad_date_payload))
    assert any("date-time" in err.message for err in errors)

    bad_uuid_payload = create_sample_event_dict(eventId="not-a-uuid")
    errors = list(VALIDATOR.iter_errors(bad_uuid_payload))
    assert any("uuid" in err.message for err in errors)


# ============================================================================
# 2. Positive tests: schema conformance & wire roundtrip
# ============================================================================


def test_valid_minimal_event_passes_model_and_json_schema():
    data = create_sample_event_dict()
    event = TechnicalObservationEvent.model_validate(data)
    assert event.event_id == data["eventId"]
    assert event.camera_external_id == "CAM-01"

    wire = event.to_wire_dict()
    assert wire == data

    errors = list(VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema validation errors: {errors}"


@pytest.mark.parametrize(
    "captured_at",
    [
        "2026-09-18T10:00:00Z",
        "2026-12-31T23:59:60Z",
        "2026-12-31T18:59:60-05:00",
        "2026-12-31T23:59:60+00",
        "2026-12-31T23:59:59+00",
        "2026-12-31T23:59:60+00:00",
        "2027-01-01T06:59:60+07:00",
        "2026-12-31T23:59:60+0000",
        "2026-12-31T12:00:00+23:59",
    ],
)
def test_valid_captured_at_formats_accepted(captured_at: str):
    data = create_sample_event_dict(capturedAt=captured_at)
    event = TechnicalObservationEvent.model_validate(data)
    assert event.captured_at == captured_at

    wire = event.to_wire_dict()
    assert wire["capturedAt"] == captured_at
    errors = list(VALIDATOR.iter_errors(wire))
    assert not errors


def test_all_observation_types_schema_conformance():
    region_id = str(uuid4())
    obs_person = {
        "type": "PERSON",
        "trackId": 1,
        "confidence": 0.8,
        "boundingBox": {
            "x1": 0.2,
            "y1": 0.2,
            "x2": 0.6,
            "y2": 0.7,
            "coordinateSpace": "NORMALIZED_0_1",
        },
    }
    obs_ppe = {
        "type": "PPE",
        "trackId": 1,
        "ppeItem": "HARD_HAT",
        "status": "PRESENT",
        "regionId": region_id,
        "geometryVersion": 2,
        "confidence": 0.9,
    }
    obs_zone = {
        "type": "ZONE_ENTRY",
        "trackId": 1,
        "regionId": region_id,
        "geometryVersion": 1,
    }
    obs_id_candidate = {
        "type": "IDENTITY_CANDIDATE",
        "trackId": 1,
        "status": "CANDIDATE",
        "candidateWorkerId": "worker-42",
        "similarityScore": 0.88,
        "qualityScore": 0.95,
    }
    obs_id_unknown = {
        "type": "IDENTITY_CANDIDATE",
        "trackId": 2,
        "status": "UNKNOWN",
    }

    evidence = [
        {"kind": "FRAME", "uri": "https://storage.local/ev1.jpg"},
        {"kind": "CROP", "uri": "https://storage.local/ev2.jpg", "trackId": 1},
        {"kind": "SNAPSHOT", "uri": "https://storage.local/ev3.jpg"},
    ]

    data = create_sample_event_dict(
        observations=[obs_person, obs_ppe, obs_zone, obs_id_candidate, obs_id_unknown],
        evidence=evidence,
    )

    event = TechnicalObservationEvent.model_validate(data)
    assert len(event.observations) == 5
    assert len(event.evidence) == 3

    wire = event.to_wire_dict()
    errors = list(VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema errors: {errors}"


def test_convenience_constructor_create():
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-02",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-19T12:00:00Z",
        frame_dimensions=FrameDimensions(width=1280, height=720),
        observations=[
            PersonObservation(
                type="PERSON",
                trackId=1,
                boundingBox=BoundingBox(
                    x1=0.1, y1=0.1, x2=0.4, y2=0.6, coordinateSpace="NORMALIZED_0_1"
                ),
            )
        ],
    )
    assert event.camera_external_id == "CAM-02"
    assert event.schema_version == "1.0.0"
    assert event.evidence == []
    wire = event.to_wire_dict()
    assert VALIDATOR.is_valid(wire)


# ============================================================================
# 3. Negative tests: invalid capturedAt
# ============================================================================


@pytest.mark.parametrize(
    "bad_date",
    [
        "not-a-date",
        "2026-02-31T10:00:00Z",
        "2026-13-01T10:00:00Z",
        "2026-09-18T25:00:00Z",
        "2026-12-31T22:59:60Z",
        "2026-12-31T23:59:60+07:00",
        "2026-12-31T23:59:61Z",
        "2026-02-29T10:00:00Z",  # 2026 is not a leap year
        "2026/09/18 10:00:00",
        "2026-12-31T12:00:00+99:00",
        "2026-12-31T12:00:00+24:00",
        "2026-12-31T12:00:00+2400",
        "2026-12-31T12:00:00+24",
        "2026-12-31T12:00:00+00:60",
        "٢٠٢٦-٠٩-١٩T١٢:٠٠:٠٠Z",  # Python \d matches Unicode digits; Ajv accepts ASCII only.
        "2026-09-19\u008512:00:00Z",  # Python \s accepts U+0085, but Ajv does not.
    ],
)
def test_invalid_captured_at_rejected(bad_date: str):
    data = create_sample_event_dict(capturedAt=bad_date)
    with pytest.raises(ValidationError, match="capturedAt"):
        TechnicalObservationEvent.model_validate(data)


# ============================================================================
# 4. Negative tests: boolean and string coercion for integers rejected
# ============================================================================


def test_track_id_boolean_or_numeric_string_coercion_rejected():
    # bool True -> 1 must be rejected
    with pytest.raises(ValidationError):
        PersonObservation.model_validate({"type": "PERSON", "trackId": True})

    # bool False -> 0 must be rejected
    with pytest.raises(ValidationError):
        PersonObservation.model_validate({"type": "PERSON", "trackId": False})

    # string "1" must be rejected
    with pytest.raises(ValidationError):
        PersonObservation.model_validate({"type": "PERSON", "trackId": "1"})


def test_width_height_geometry_version_coercion_rejected():
    with pytest.raises(ValidationError):
        FrameDimensions.model_validate({"width": True, "height": 1080})

    with pytest.raises(ValidationError):
        FrameDimensions.model_validate({"width": 1920, "height": "1080"})

    with pytest.raises(ValidationError):
        ZoneEntryObservation.model_validate(
            {"type": "ZONE_ENTRY", "trackId": 1, "regionId": str(uuid4()), "geometryVersion": True}
        )

    with pytest.raises(ValidationError):
        ZoneEntryObservation.model_validate(
            {"type": "ZONE_ENTRY", "trackId": 1, "regionId": str(uuid4()), "geometryVersion": "1"}
        )


# ============================================================================
# 5. Negative tests: explicit null rejection
# ============================================================================


def test_explicit_null_rejected_across_all_models():
    # 1. confidence=None in PersonObservation
    with pytest.raises(ValidationError, match="Explicit null"):
        PersonObservation.model_validate({"type": "PERSON", "trackId": 1, "confidence": None})

    # 2. boundingBox=None in PersonObservation
    with pytest.raises(ValidationError, match="Explicit null"):
        PersonObservation.model_validate({"type": "PERSON", "trackId": 1, "boundingBox": None})

    # 3. confidence=None in PpeObservation
    with pytest.raises(ValidationError, match="Explicit null"):
        PpeObservation.model_validate(
            {
                "type": "PPE",
                "trackId": 1,
                "ppeItem": "HARD_HAT",
                "status": "PRESENT",
                "regionId": str(uuid4()),
                "geometryVersion": 1,
                "confidence": None,
            }
        )

    # 4. qualityScore=None in IdentityCandidateObservation
    with pytest.raises(ValidationError, match="Explicit null"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 1,
                "status": "UNKNOWN",
                "qualityScore": None,
            }
        )

    # 5. trackId=None in EvidenceItem
    with pytest.raises(ValidationError, match="Explicit null"):
        EvidenceItem.model_validate(
            {"kind": "FRAME", "uri": "https://example.com/f.jpg", "trackId": None}
        )

    # 6. evidence=None in TechnicalObservationEvent
    data = create_sample_event_dict()
    data["evidence"] = None
    with pytest.raises(ValidationError, match="Explicit null"):
        TechnicalObservationEvent.model_validate(data)


# ============================================================================
# 6. Negative tests: missing required fields (no defaulting)
# ============================================================================


def test_missing_required_fields_not_defaulted():
    # Missing schemaVersion must raise
    data = create_sample_event_dict()
    del data["schemaVersion"]
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(data)

    # Missing evidence must raise
    data = create_sample_event_dict()
    del data["evidence"]
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(data)

    # Missing coordinateSpace in BoundingBox must raise
    with pytest.raises(ValidationError):
        BoundingBox.model_validate({"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.8})

    # Missing type in observation must raise
    with pytest.raises(ValidationError):
        PersonObservation.model_validate({"trackId": 1})

    # Missing regionId or geometryVersion in PpeObservation
    with pytest.raises(ValidationError):
        PpeObservation.model_validate(
            {"type": "PPE", "trackId": 1, "ppeItem": "HARD_HAT", "status": "PRESENT"}
        )


# ============================================================================
# 7. Negative tests: snake_case inputs rejected (only camelCase aliases accepted)
# ============================================================================


def test_snake_case_inputs_rejected():
    data = create_sample_event_dict()
    data["camera_external_id"] = data.pop("cameraExternalId")
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(data)

    data = create_sample_event_dict()
    data["stream_session_id"] = data.pop("streamSessionId")
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(data)

    data = create_sample_event_dict()
    data["schema_version"] = data.pop("schemaVersion")
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(data)

    with pytest.raises(ValidationError):
        PersonObservation.model_validate({"type": "PERSON", "track_id": 1})

    with pytest.raises(ValidationError):
        BoundingBox.model_validate(
            {
                "x1": 0.1,
                "y1": 0.1,
                "x2": 0.5,
                "y2": 0.8,
                "coordinate_space": "NORMALIZED_0_1",
            }
        )


# ============================================================================
# 8. Negative tests: extra properties forbidden
# ============================================================================


def test_extra_properties_rejected_at_all_levels():
    # Extra field on event
    data = create_sample_event_dict(unexpected="value")
    with pytest.raises(ValidationError, match="extra"):
        TechnicalObservationEvent.model_validate(data)

    # Extra field on FrameDimensions (e.g. fps is not in schema)
    with pytest.raises(ValidationError, match="extra"):
        FrameDimensions.model_validate({"width": 1920, "height": 1080, "fps": 30})

    # Extra field on BoundingBox
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

    # Extra field on EvidenceItem
    with pytest.raises(ValidationError, match="extra"):
        EvidenceItem.model_validate(
            {"kind": "FRAME", "uri": "https://example.com/f.jpg", "extra": "data"}
        )


# ============================================================================
# 9. Domain invariant tests: geometry, identity, min_length
# ============================================================================


def test_bounding_box_geometry_invariants():
    with pytest.raises(ValidationError, match="strictly less than x2"):
        BoundingBox.model_validate(
            {"x1": 0.5, "y1": 0.1, "x2": 0.2, "y2": 0.8, "coordinateSpace": "NORMALIZED_0_1"}
        )
    with pytest.raises(ValidationError, match="strictly less than x2"):
        BoundingBox.model_validate(
            {"x1": 0.5, "y1": 0.1, "x2": 0.5, "y2": 0.8, "coordinateSpace": "NORMALIZED_0_1"}
        )
    with pytest.raises(ValidationError, match="strictly less than y2"):
        BoundingBox.model_validate(
            {"x1": 0.1, "y1": 0.8, "x2": 0.5, "y2": 0.2, "coordinateSpace": "NORMALIZED_0_1"}
        )


def test_identity_candidate_conditional_rules():
    # CANDIDATE requires candidateWorkerId and similarityScore
    with pytest.raises(ValidationError, match="candidateWorkerId is required"):
        IdentityCandidateObservation.model_validate(
            {"type": "IDENTITY_CANDIDATE", "trackId": 1, "status": "CANDIDATE"}
        )
    with pytest.raises(ValidationError, match="similarityScore is required"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 1,
                "status": "CANDIDATE",
                "candidateWorkerId": "worker-1",
            }
        )

    # UNKNOWN/UNAVAILABLE forbid candidateWorkerId and similarityScore
    with pytest.raises(ValidationError, match="not allowed when status is UNKNOWN"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 1,
                "status": "UNKNOWN",
                "candidateWorkerId": "worker-1",
            }
        )
    with pytest.raises(ValidationError, match="not allowed when status is UNAVAILABLE"):
        IdentityCandidateObservation.model_validate(
            {
                "type": "IDENTITY_CANDIDATE",
                "trackId": 1,
                "status": "UNAVAILABLE",
                "similarityScore": 0.8,
            }
        )


def test_empty_observations_list_rejected():
    data = create_sample_event_dict(observations=[])
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(data)
