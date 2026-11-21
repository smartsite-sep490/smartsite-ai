import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from smartsite_ai.domain.frame_evidence_receipt import FrameEvidenceReceipt

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_BACKEND_PR294_COMMIT = "b0917fd31a88bb840d0591ea97b2f82fdf4aacd5"
SCHEMA_REL = "contracts/schemas/v1/ai-frame-evidence-receipt.json"
EXAMPLE_REL = "contracts/examples/v1/ai-frame-evidence-receipt.json"


def _load_schema() -> dict:
    return json.loads((ROOT / SCHEMA_REL).read_text(encoding="utf-8"))


def _load_example() -> dict:
    return json.loads((ROOT / EXAMPLE_REL).read_text(encoding="utf-8"))


def test_frame_evidence_receipt_vendored_provenance_matches_canonical_bytes():
    metadata = json.loads((ROOT / "contracts/metadata.json").read_text(encoding="utf-8"))
    entry = metadata["aiFrameEvidenceReceipt"]

    assert entry["schemaVersion"] == "1.0.0"
    assert entry["sourceRepository"] == "smartsite-sep490/smartsite"
    assert entry["sourceCommitSha"] == CANONICAL_BACKEND_PR294_COMMIT
    assert entry["path"] == SCHEMA_REL
    assert entry["sourcePath"] == SCHEMA_REL
    assert entry["examplePath"] == EXAMPLE_REL
    assert entry["sourceExamplePath"] == EXAMPLE_REL

    schema_bytes = (ROOT / SCHEMA_REL).read_bytes()
    example_bytes = (ROOT / EXAMPLE_REL).read_bytes()
    assert re.fullmatch(r"[0-9a-f]{64}", entry["schemaSha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", entry["exampleSha256"])
    assert entry["schemaSha256"] == hashlib.sha256(schema_bytes).hexdigest()
    assert entry["exampleSha256"] == hashlib.sha256(example_bytes).hexdigest()

    # Keep provenance independent across observation profiles and v1/v1.7 tracks.
    assert metadata["sourceCommitSha"] != CANONICAL_BACKEND_PR294_COMMIT
    assert metadata["technicalObservationVersions"][-1]["schemaVersion"] == "1.4.0"
    assert metadata["trackedTechnicalObservationVersions"][-1]["schemaVersion"] == "1.7.0"


def test_canonical_receipt_example_satisfies_schema_and_strict_model():
    schema = _load_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    example = _load_example()
    validator.validate(example)

    parsed = FrameEvidenceReceipt.model_validate(example)
    assert parsed.schema_version == "1.0.0"
    assert parsed.kind == "EVIDENCE_STORED"
    parsed.require_match(
        example["eventId"],
        example["payloadHash"],
        example["bindingHash"],
        example["sha256"],
    )


@pytest.mark.parametrize(
    ("mutator", "pydantic_accepts"),
    [
        (lambda d: d.__setitem__("kind", "RAW_STORED"), False),
        (lambda d: d.__setitem__("kind", "FRAME_ADMITTED"), False),
        (lambda d: d.__setitem__("schemaVersion", "1.1.0"), False),
        (lambda d: d.__setitem__("allowed", True), False),
        (lambda d: d.__setitem__("publicUrl", "https://example.invalid/frame.jpg"), False),
        (lambda d: d.pop("bindingHash"), False),
        (lambda d: d.pop("payloadHash"), False),
        (lambda d: d.pop("sha256"), False),
        (lambda d: d.pop("eventId"), False),
        (lambda d: d.__setitem__("sha256", "C" * 64), False),
        (lambda d: d.__setitem__("payloadHash", "A" * 64), False),
        (lambda d: d.__setitem__("bindingHash", "short"), False),
        (lambda d: d.__setitem__("eventId", "11111111-1111-4111-8111-11111111111G"), False),
    ],
)
def test_receipt_schema_and_model_reject_tampered_or_permission_fields(mutator, pydantic_accepts):
    validator = Draft202012Validator(_load_schema())
    candidate = copy.deepcopy(_load_example())
    mutator(candidate)

    with pytest.raises(JsonSchemaValidationError):
        validator.validate(candidate)

    if not pydantic_accepts:
        with pytest.raises(PydanticValidationError):
            FrameEvidenceReceipt.model_validate(candidate)


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("event_id", "99999999-9999-4999-8999-999999999999"),
        ("payload_hash", "0" * 64),
        ("binding_hash", "0" * 64),
        ("sha256", "0" * 64),
    ],
)
def test_receipt_require_match_rejects_any_binding_mismatch(field, override):
    example = _load_example()
    parsed = FrameEvidenceReceipt.model_validate(example)
    kwargs = {
        "event_id": example["eventId"],
        "payload_hash": example["payloadHash"],
        "binding_hash": example["bindingHash"],
        "sha256": example["sha256"],
    }
    kwargs[field] = override
    with pytest.raises(ValueError, match="mismatch"):
        parsed.require_match(**kwargs)
