import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import jsonschema
import pytest

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
METADATA_PATH = ROOT / "contracts/metadata.json"

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

# Independent FormatChecker from installed format-nongpl extra
# NOTE: Not monkeypatched or overridden by Python validator under test
INDEPENDENT_FORMAT_CHECKER = jsonschema.FormatChecker()
INDEPENDENT_VALIDATOR = jsonschema.Draft202012Validator(
    SCHEMA, format_checker=INDEPENDENT_FORMAT_CHECKER
)


# ============================================================================
# 1. Provenance, Metadata, and Source Commit Schema Blob Integrity
# ============================================================================


def test_contracts_provenance_and_schema_hash_integrity():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    schema_bytes = SCHEMA_PATH.read_bytes()
    golden_bytes = (ROOT / "contracts/golden-vectors.json").read_bytes()

    assert metadata["sourceRepository"] == "smartsite-sep490/smartsite"
    assert re.fullmatch(r"[0-9a-f]{40}", metadata["sourceCommitSha"])
    assert metadata["schemaVersion"] == "1.0.0"

    computed_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    assert metadata["schemaSha256"] == computed_sha256
    assert metadata["goldenVectorsSha256"] == hashlib.sha256(golden_bytes).hexdigest()

    # CI can check out the AI repository alone. When the source checkout is
    # present, a missing commit or blob must fail rather than skip provenance.
    sibling_repo = ROOT.parent / "smartsite"
    if (sibling_repo / ".git").exists():
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(sibling_repo),
                "show",
                f"{metadata['sourceCommitSha']}:contracts/schemas/v1/technical-observation-event.json",
            ],
            capture_output=True,
            text=False,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
        assert proc.stdout == schema_bytes, "Vendored schema does not match source commit blob"

        golden_proc = subprocess.run(
            [
                "git",
                "-C",
                str(sibling_repo),
                "show",
                f"{metadata['sourceCommitSha']}:contracts/test/golden-vectors.json",
            ],
            capture_output=True,
            text=False,
            check=False,
        )
        assert golden_proc.returncode == 0, golden_proc.stderr.decode("utf-8", errors="replace")
        assert golden_proc.stdout == golden_bytes, (
            "Vendored vectors do not match source commit blob"
        )


# ============================================================================
# 2. Independent FormatChecker actively rejects invalid date-time & UUID
# ============================================================================


def test_independent_format_checker_actively_detects_malformed_formats():
    assert not INDEPENDENT_FORMAT_CHECKER.conforms("not-a-date", "date-time")
    assert not INDEPENDENT_FORMAT_CHECKER.conforms("not-a-uuid", "uuid")
    assert INDEPENDENT_FORMAT_CHECKER.conforms("2026-09-18T10:00:00Z", "date-time")
    assert INDEPENDENT_FORMAT_CHECKER.conforms(str(uuid4()), "uuid")

    base_payload: dict[str, Any] = {
        "eventId": str(uuid4()),
        "schemaVersion": "1.0.0",
        "cameraExternalId": "CAM-01",
        "streamSessionId": str(uuid4()),
        "capturedAt": "2026-09-18T10:00:00Z",
        "frameDimensions": {"width": 1920, "height": 1080},
        "observations": [{"type": "PERSON", "trackId": 1}],
        "evidence": [],
    }

    # Invalid date-time must be caught by independent validator
    bad_date = {**base_payload, "capturedAt": "not-a-date"}
    date_errors = list(INDEPENDENT_VALIDATOR.iter_errors(bad_date))
    assert any("date-time" in err.message for err in date_errors)

    # Invalid UUID must be caught by independent validator
    bad_uuid = {**base_payload, "eventId": "not-a-uuid"}
    uuid_errors = list(INDEPENDENT_VALIDATOR.iter_errors(bad_uuid))
    assert any("uuid" in err.message for err in uuid_errors)


# ============================================================================
# 3. Pydantic wire serialization & independent JSON Schema validation
#    covers all 4 observation variants + nested data
# ============================================================================


def test_variant_person_observation_wire_conformance():
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-PERSON-01",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-18T10:00:00Z",
        frame_dimensions=FrameDimensions(width=1920, height=1080),
        observations=[
            PersonObservation(
                type="PERSON",
                trackId=42,
                confidence=0.94,
                boundingBox=BoundingBox(
                    x1=0.15,
                    y1=0.20,
                    x2=0.55,
                    y2=0.85,
                    coordinateSpace="NORMALIZED_0_1",
                ),
            )
        ],
        evidence=[],
    )

    wire = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert wire == event.to_wire_dict()

    # Structural checks on wire representation
    assert wire["schemaVersion"] == "1.0.0"
    obs = wire["observations"][0]
    assert obs["type"] == "PERSON"
    assert obs["trackId"] == 42
    assert obs["confidence"] == 0.94
    assert obs["boundingBox"]["coordinateSpace"] == "NORMALIZED_0_1"
    assert "track_id" not in obs
    assert "coordinate_space" not in obs["boundingBox"]

    # Independent JSON Schema verification
    errors = list(INDEPENDENT_VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema errors: {errors}"


def test_variant_ppe_observation_wire_conformance():
    region_id = str(uuid4())
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-PPE-01",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-18T10:00:00Z",
        frame_dimensions=FrameDimensions(width=1280, height=720),
        observations=[
            PpeObservation(
                type="PPE",
                trackId=1,
                ppeItem="HARD_HAT",
                status="PRESENT",
                regionId=region_id,
                geometryVersion=1,
                confidence=0.98,
                boundingBox=BoundingBox(
                    x1=0.2,
                    y1=0.2,
                    x2=0.4,
                    y2=0.4,
                    coordinateSpace="NORMALIZED_0_1",
                ),
            ),
            PpeObservation(
                type="PPE",
                trackId=1,
                ppeItem="SAFETY_VEST",
                status="MISSING",
                regionId=region_id,
                geometryVersion=2,
            ),
        ],
        evidence=[],
    )

    wire = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert len(wire["observations"]) == 2

    # Wire verification for PPE fields
    hat_obs = wire["observations"][0]
    assert hat_obs["ppeItem"] == "HARD_HAT"
    assert hat_obs["status"] == "PRESENT"
    assert hat_obs["regionId"] == region_id
    assert hat_obs["geometryVersion"] == 1
    assert "boundingBox" in hat_obs

    vest_obs = wire["observations"][1]
    assert vest_obs["ppeItem"] == "SAFETY_VEST"
    assert vest_obs["status"] == "MISSING"
    assert "boundingBox" not in vest_obs  # None must be omitted from wire

    errors = list(INDEPENDENT_VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema errors: {errors}"


def test_variant_zone_entry_observation_wire_conformance():
    region_id = str(uuid4())
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-ZONE-01",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-18T10:00:00Z",
        frame_dimensions=FrameDimensions(width=3840, height=2160),
        observations=[
            ZoneEntryObservation(
                type="ZONE_ENTRY",
                trackId=99,
                regionId=region_id,
                geometryVersion=5,
                confidence=0.89,
            )
        ],
        evidence=[],
    )

    wire = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    obs = wire["observations"][0]
    assert obs["type"] == "ZONE_ENTRY"
    assert obs["trackId"] == 99
    assert obs["regionId"] == region_id
    assert obs["geometryVersion"] == 5
    assert obs["confidence"] == 0.89

    errors = list(INDEPENDENT_VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema errors: {errors}"


def test_variant_identity_candidate_observation_wire_conformance():
    # Covers all 3 identity statuses: CANDIDATE, UNKNOWN, UNAVAILABLE
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-ID-01",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-18T10:00:00Z",
        frame_dimensions=FrameDimensions(width=1920, height=1080),
        observations=[
            IdentityCandidateObservation(
                type="IDENTITY_CANDIDATE",
                trackId=10,
                status="CANDIDATE",
                candidateWorkerId="worker-uuid-1234",
                similarityScore=0.91,
                qualityScore=0.88,
            ),
            IdentityCandidateObservation(
                type="IDENTITY_CANDIDATE",
                trackId=11,
                status="UNKNOWN",
                qualityScore=0.72,
            ),
            IdentityCandidateObservation(
                type="IDENTITY_CANDIDATE",
                trackId=12,
                status="UNAVAILABLE",
            ),
        ],
        evidence=[],
    )

    wire = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert len(wire["observations"]) == 3

    cand_obs = wire["observations"][0]
    assert cand_obs["status"] == "CANDIDATE"
    assert cand_obs["candidateWorkerId"] == "worker-uuid-1234"
    assert cand_obs["similarityScore"] == 0.91
    assert cand_obs["qualityScore"] == 0.88

    unk_obs = wire["observations"][1]
    assert unk_obs["status"] == "UNKNOWN"
    assert "candidateWorkerId" not in unk_obs
    assert "similarityScore" not in unk_obs
    assert unk_obs["qualityScore"] == 0.72

    unav_obs = wire["observations"][2]
    assert unav_obs["status"] == "UNAVAILABLE"
    assert "candidateWorkerId" not in unav_obs
    assert "similarityScore" not in unav_obs
    assert "qualityScore" not in unav_obs

    errors = list(INDEPENDENT_VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema errors: {errors}"


def test_nested_evidence_items_wire_conformance():
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-EV-01",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-18T10:00:00Z",
        frame_dimensions=FrameDimensions(width=1920, height=1080),
        observations=[
            PersonObservation(
                type="PERSON",
                trackId=1,
            )
        ],
        evidence=[
            EvidenceItem(kind="FRAME", uri="https://storage.example.com/frames/101.jpg"),
            EvidenceItem(
                kind="CROP",
                uri="https://storage.example.com/crops/101.jpg",
                trackId=1,
                boundingBox=BoundingBox(
                    x1=0.1,
                    y1=0.2,
                    x2=0.5,
                    y2=0.7,
                    coordinateSpace="NORMALIZED_0_1",
                ),
            ),
            EvidenceItem(kind="SNAPSHOT", uri="https://storage.example.com/snapshots/101.jpg"),
        ],
    )

    wire = event.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert len(wire["evidence"]) == 3

    assert wire["evidence"][0]["kind"] == "FRAME"
    assert "trackId" not in wire["evidence"][0]

    assert wire["evidence"][1]["kind"] == "CROP"
    assert wire["evidence"][1]["trackId"] == 1
    assert wire["evidence"][1]["boundingBox"]["coordinateSpace"] == "NORMALIZED_0_1"

    assert wire["evidence"][2]["kind"] == "SNAPSHOT"

    errors = list(INDEPENDENT_VALIDATOR.iter_errors(wire))
    assert not errors, f"JSON Schema errors: {errors}"


def test_full_composite_event_wire_and_canonical_hash():
    region_id = str(uuid4())
    event = TechnicalObservationEvent.create(
        event_id=str(uuid4()),
        camera_external_id="CAM-FULL-01",
        stream_session_id=str(uuid4()),
        captured_at="2026-09-18T10:00:00Z",
        frame_dimensions=FrameDimensions(width=1920, height=1080),
        observations=[
            PersonObservation(
                type="PERSON",
                trackId=1,
                confidence=0.95,
                boundingBox=BoundingBox(
                    x1=0.1, y1=0.1, x2=0.5, y2=0.8, coordinateSpace="NORMALIZED_0_1"
                ),
            ),
            PpeObservation(
                type="PPE",
                trackId=1,
                ppeItem="HARD_HAT",
                status="PRESENT",
                regionId=region_id,
                geometryVersion=1,
                confidence=0.9,
            ),
            ZoneEntryObservation(
                type="ZONE_ENTRY",
                trackId=1,
                regionId=region_id,
                geometryVersion=1,
                confidence=0.85,
            ),
            IdentityCandidateObservation(
                type="IDENTITY_CANDIDATE",
                trackId=1,
                status="CANDIDATE",
                candidateWorkerId="worker-99",
                similarityScore=0.93,
            ),
        ],
        evidence=[
            EvidenceItem(kind="FRAME", uri="https://storage.local/f.jpg"),
        ],
    )

    wire = event.to_wire_dict()
    assert INDEPENDENT_VALIDATOR.is_valid(wire)

    canonical_hash = event.compute_payload_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", canonical_hash)


# ============================================================================
# 4. Anti-drift negative checks against JSON Schema
# ============================================================================


@pytest.mark.parametrize(
    ("mutator", "expected_err"),
    [
        (lambda d: d.update({"unexpected": "extra"}), "unexpected"),
        (lambda d: d["frameDimensions"].update({"fps": 30}), "fps"),
        (lambda d: d["observations"][0].update({"confidence": None}), "None"),
        (lambda d: d["observations"][0].update({"trackId": "1"}), "1"),
        (lambda d: d["frameDimensions"].update({"width": 1920.5}), "1920.5"),
        (lambda d: d.update({"observations": []}), "minItems"),
        (lambda d: d.pop("schemaVersion"), "schemaVersion"),
        (lambda d: d.pop("evidence"), "evidence"),
    ],
)
def test_schema_actively_rejects_malformed_wire_payloads(mutator: Any, expected_err: str):
    valid_payload: dict[str, Any] = {
        "eventId": str(uuid4()),
        "schemaVersion": "1.0.0",
        "cameraExternalId": "CAM-01",
        "streamSessionId": str(uuid4()),
        "capturedAt": "2026-09-18T10:00:00Z",
        "frameDimensions": {"width": 1920, "height": 1080},
        "observations": [
            {
                "type": "PERSON",
                "trackId": 1,
                "confidence": 0.9,
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

    payload = json.loads(json.dumps(valid_payload))
    mutator(payload)
    errors = list(INDEPENDENT_VALIDATOR.iter_errors(payload))
    assert len(errors) > 0, f"Schema failed to reject invalid payload: {payload}"
    assert any(
        err.validator == expected_err
        or expected_err in err.message
        or expected_err in str(err.path)
        for err in errors
    ), f"Expected '{expected_err}' in errors: {[(e.validator, e.message) for e in errors]}"
