"""Provenance and compatibility checks for vendored Mask v1.8 / v2.1 schemas."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import ValidationError as JsonSchemaError
from pydantic import ValidationError as PydanticValidationError
from referencing import Registry, Resource

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import PPE_ITEMS
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = ROOT / "contracts"
METADATA_PATH = CONTRACTS_DIR / "metadata.json"

PLATFORM_MASK_COMMIT = "fb9b53adc3c67f7b4e1d28251108daeef2986d81"
SIX_PPE_ITEMS = ("HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES", "MASK")

EXPECTED_DIGESTS = {
    "contracts/mask-versioned-adoption.md": (
        "d66e5e450e8559242f7451197f2dae5920e644002847254c77b8094a6fb933e4"
    ),
    "contracts/schemas/v1.8/technical-observation-event.json": (
        "25f48f2db28bb93068fb94b3d43f6b61b84d4b52cce0a81d3c727cec241fd84e"
    ),
    "contracts/schemas/v1.8/examples/mask-camera-observation.json": (
        "cf089bf2e5029840032400be985da2d7b6e44071f4cf73759440460390a595f1"
    ),
    "contracts/schemas/v1.8/examples/mask-contract-hashes.json": (
        "19d56479fc59471e069c1ce81d8edc99aae12c61c4f187d7343b877077989352"
    ),
    "contracts/schemas/v2.1/camera-policy-configuration-bundle.json": (
        "696b89cad9fed7341d03bbe2a8ee234860abc092e15c29cec0ddcc885656e3b7"
    ),
    "contracts/schemas/v2.1/examples/mask-policy-configuration-bundle.json": (
        "6231ffc24d17382cb001e7ca59c26c9d856ccf2dbc03ac261490667c50ecaa6c"
    ),
}


def _schema_registry() -> tuple[dict[str, dict[str, Any]], Registry[Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for path in (CONTRACTS_DIR / "schemas").rglob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and "$id" in doc:
            resources[str(doc["$id"])] = doc
    registry = Registry().with_resources(
        (schema_id, Resource.from_contents(schema)) for schema_id, schema in resources.items()
    )
    return resources, registry


def _ecma_pattern_validator(
    validator: Draft202012Validator, patrn: str, instance: Any, schema: Any
) -> Iterator[ValidationError]:
    del schema
    if not validator.is_type(instance, "string"):
        return
    if patrn == r"^(?=.*\S)[^\p{Cc}\p{Cs}]+$":
        valid = (
            bool(instance)
            and any(not c.isspace() for c in instance)
            and all(unicodedata.category(c) not in {"Cc", "Cs"} for c in instance)
        )
        if not valid:
            yield ValidationError(f"{instance!r} does not match ECMA pattern {patrn!r}")
        return
    if not re.search(patrn, instance):
        yield ValidationError(f"{instance!r} does not match {patrn!r}")


_ECMA_DRAFT_2020_12 = jsonschema.validators.extend(
    Draft202012Validator, validators={"pattern": _ecma_pattern_validator}
)


def _v1_8_validator() -> Draft202012Validator:
    resources, registry = _schema_registry()
    schema = resources["https://smartsite.local/contracts/v1.8/technical-observation-event.json"]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    )


def _v2_1_validator() -> Draft202012Validator:
    resources, registry = _schema_registry()
    schema = resources[
        "https://smartsite.local/contracts/v2.1/camera-policy-configuration-bundle.json"
    ]
    meta_checker = copy.deepcopy(Draft202012Validator.FORMAT_CHECKER)

    @meta_checker.checks("regex", raises=re.error)
    def _check_ecma_regex(value: object) -> bool:
        if not isinstance(value, str):
            return True
        if value == r"^(?=.*\S)[^\p{Cc}\p{Cs}]+$":
            return True
        re.compile(value)
        return True

    Draft202012Validator.check_schema(schema, format_checker=meta_checker)
    return _ECMA_DRAFT_2020_12(schema, registry=registry, format_checker=jsonschema.FormatChecker())


def test_mask_contracts_provenance_lf_and_source_blob_parity() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    entry = metadata["maskVersionedContracts"]
    assert entry["sourceRepository"] == "smartsite-sep490/smartsite"
    assert entry["sourceCommitSha"] == PLATFORM_MASK_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", entry["sourceCommitSha"])
    assert entry["sourceCommitSha"] not in {
        metadata["sourceCommitSha"],
        metadata["cameraPolicyConfigurationBundle"]["sourceCommitSha"],
        metadata["trackedTechnicalObservationVersions"][-1]["sourceCommitSha"],
        metadata["aiFrameEvidenceReceipt"]["sourceCommitSha"],
        metadata["cameraSourceActivationState"]["sourceCommitSha"],
    }

    assert entry["adoptionDocPath"] == "contracts/mask-versioned-adoption.md"
    assert entry["sourceAdoptionDocPath"] == entry["adoptionDocPath"]
    assert entry["technicalObservation"]["schemaVersion"] == "1.8.0"
    assert entry["cameraPolicyConfigurationBundle"]["schemaVersion"] == "2.1.0"

    pinned_pairs = [
        (entry["adoptionDocPath"], entry["adoptionDocSha256"]),
        (entry["technicalObservation"]["path"], entry["technicalObservation"]["schemaSha256"]),
        (
            entry["technicalObservation"]["examplePath"],
            entry["technicalObservation"]["exampleSha256"],
        ),
        (
            entry["technicalObservation"]["hashesPath"],
            entry["technicalObservation"]["hashesSha256"],
        ),
        (
            entry["cameraPolicyConfigurationBundle"]["path"],
            entry["cameraPolicyConfigurationBundle"]["schemaSha256"],
        ),
        (
            entry["cameraPolicyConfigurationBundle"]["examplePath"],
            entry["cameraPolicyConfigurationBundle"]["exampleSha256"],
        ),
    ]
    assert {rel for rel, _digest in pinned_pairs} == set(EXPECTED_DIGESTS)
    for rel_path, expected_digest in pinned_pairs:
        assert expected_digest == EXPECTED_DIGESTS[rel_path]
        raw = (ROOT / rel_path).read_bytes()
        assert b"\r\n" not in raw, f"CRLF detected in {rel_path}"
        assert hashlib.sha256(raw).hexdigest() == expected_digest

    ci_text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for rel_path in EXPECTED_DIGESTS:
        assert rel_path in ci_text
        assert f"smartsite_ai/{rel_path}" in ci_text

    source_candidates = (ROOT.parent / "smartsite", ROOT.parents[1] / "repos/smartsite")
    for candidate in source_candidates:
        if not (candidate / ".git").exists():
            continue
        exists = subprocess.run(
            ["git", "-C", str(candidate), "cat-file", "-t", PLATFORM_MASK_COMMIT],
            capture_output=True,
            text=True,
            check=False,
        )
        if exists.returncode != 0 or exists.stdout.strip() != "commit":
            continue
        for rel_path in EXPECTED_DIGESTS:
            blob = subprocess.run(
                ["git", "-C", str(candidate), "show", f"{PLATFORM_MASK_COMMIT}:{rel_path}"],
                capture_output=True,
                check=True,
            ).stdout
            assert (ROOT / rel_path).read_bytes() == blob


def test_legacy_released_schemas_and_constants_remain_unchanged() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["technicalObservationVersions"][-1]["schemaVersion"] == "1.4.0"
    assert [e["schemaVersion"] for e in metadata["trackedTechnicalObservationVersions"]] == [
        "1.5.0",
        "1.6.0",
        "1.7.0",
    ]
    assert metadata["cameraPolicyConfigurationBundle"]["schemaVersion"] == "2.0.0"
    assert metadata["cameraSourceActivationState"]["schemaVersion"] == "1.1.0"
    assert (
        hashlib.sha256(
            (CONTRACTS_DIR / "schemas/v1.7/technical-observation-event.json").read_bytes()
        ).hexdigest()
        == "e5d5813c94b480e6e06075be9744393e5311f0188fad9445e91c23803d6b996f"
    )
    assert (
        hashlib.sha256(
            (CONTRACTS_DIR / "schemas/v2/camera-policy-configuration-bundle.json").read_bytes()
        ).hexdigest()
        == "dce75c6f92bfb4c7705dab57c2415a1819b39c8d43696989c5f286368b54df5a"
    )
    assert PPE_ITEMS == ("HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES")


def test_v1_8_observation_schema_accepts_mask_and_preserves_harness_hook_present_only() -> None:
    validator = _v1_8_validator()
    example = json.loads(
        (CONTRACTS_DIR / "schemas/v1.8/examples/mask-camera-observation.json").read_text(
            encoding="utf-8"
        )
    )
    before = copy.deepcopy(example)
    validator.validate(example)
    assert example == before

    for item in SIX_PPE_ITEMS:
        for status in ("PRESENT", "MISSING"):
            candidate = copy.deepcopy(example)
            candidate["observations"][1].update({"ppeItem": item, "status": status})
            validator.validate(candidate)

    for status in ("PRESENT", "MISSING"):
        region_event = copy.deepcopy(example)
        region_event["observations"][1] = {
            "type": "PPE",
            "trackId": 0,
            "regionId": "00000000-0000-4000-8000-000000000010",
            "geometryVersion": 1,
            "ppeItem": "MASK",
            "status": status,
        }
        validator.validate(region_event)

    # Harness & Hook stay PRESENT_ONLY in REGION scope and disallowed in CAMERA scope.
    for present_only_item in ("FULL_BODY_HARNESS", "LANYARD_HOOK"):
        camera_scoped = copy.deepcopy(example)
        camera_scoped["observations"][1]["ppeItem"] = present_only_item
        with pytest.raises(JsonSchemaError):
            validator.validate(camera_scoped)

        region_present = copy.deepcopy(example)
        region_present["observations"][1] = {
            "type": "PPE",
            "trackId": 0,
            "regionId": "00000000-0000-4000-8000-000000000010",
            "geometryVersion": 1,
            "ppeItem": present_only_item,
            "status": "PRESENT",
        }
        validator.validate(region_present)

        region_missing = copy.deepcopy(region_present)
        region_missing["observations"][1]["status"] = "MISSING"
        with pytest.raises(JsonSchemaError):
            validator.validate(region_missing)

    for patch in (
        {"ppeItem": "UNKNOWN_PPE"},
        {"status": "UNKNOWN"},
        {"status": "COMPLIANT"},
        {"allowed": True},
        {"workerId": "invented"},
        {"zoneId": "00000000-0000-4000-8000-000000000010"},
        {"scope": "SITE"},
    ):
        tampered = copy.deepcopy(example)
        tampered["observations"][1].update(patch)
        with pytest.raises(JsonSchemaError):
            validator.validate(tampered)

    # Legacy Python DTOs still reject unrolled 1.8.0 payloads.
    with pytest.raises(PydanticValidationError):
        TrackedObservationEvent.model_validate(example)
    with pytest.raises(PydanticValidationError):
        TechnicalObservationEvent.model_validate(example)


def test_v2_1_policy_bundle_schema_validates_six_items_and_rejects_harness_hook() -> None:
    validator = _v2_1_validator()
    raw_example = (
        CONTRACTS_DIR / "schemas/v2.1/examples/mask-policy-configuration-bundle.json"
    ).read_bytes()
    example = json.loads(raw_example.decode("utf-8"))
    validator.validate(example)

    all_six = copy.deepcopy(example)
    all_six["cameraPpePolicy"] = {"mode": "EXPLICIT", "requiredPpe": list(SIX_PPE_ITEMS)}
    all_six["regions"][0]["ppePolicy"] = {"mode": "EXPLICIT", "requiredPpe": list(SIX_PPE_ITEMS)}
    all_six["regions"][1]["ppePolicy"] = {"mode": "INHERIT"}
    validator.validate(all_six)

    empty_regions = copy.deepcopy(example)
    empty_regions["regions"] = []
    validator.validate(empty_regions)

    for invalid_policy in (
        {"mode": "EXPLICIT", "requiredPpe": ["MASK", "MASK"]},
        {"mode": "EXPLICIT", "requiredPpe": [*SIX_PPE_ITEMS, "MASK"]},
        {"mode": "EXPLICIT", "requiredPpe": []},
        {"mode": "EXPLICIT", "requiredPpe": ["FULL_BODY_HARNESS"]},
        {"mode": "EXPLICIT", "requiredPpe": ["LANYARD_HOOK"]},
        {"mode": "EXPLICIT", "requiredPpe": ["UNKNOWN_PPE"]},
        {"mode": "EXPLICIT", "requiredPpe": ["MASK"], "workerId": "invented"},
        {"mode": "DISABLED", "reason": "   "},
        {"mode": "DISABLED", "reason": "bad\ncontrol"},
    ):
        bad_camera = copy.deepcopy(example)
        bad_camera["cameraPpePolicy"] = invalid_policy
        with pytest.raises(JsonSchemaError):
            validator.validate(bad_camera)

        bad_region = copy.deepcopy(example)
        bad_region["regions"][0]["ppePolicy"] = invalid_policy
        with pytest.raises(JsonSchemaError):
            validator.validate(bad_region)

    wrong_version = copy.deepcopy(example)
    wrong_version["schemaVersion"] = "2.0.0"
    with pytest.raises(JsonSchemaError):
        validator.validate(wrong_version)

    # Legacy v2.0.0 wire parser must still reject 2.1.0 and 2.0.0 with MASK.
    with pytest.raises(ValueError):
        CameraPolicyConfigurationBundle.from_wire_bytes(raw_example)
    with pytest.raises(ValueError):
        CameraPolicyConfigurationBundle.from_wire_bytes(
            json.dumps(wrong_version, separators=(",", ":")).encode("utf-8")
        )


def test_canonical_mask_examples_preserve_reproducible_jcs_hashes() -> None:
    hashes = json.loads(
        (CONTRACTS_DIR / "schemas/v1.8/examples/mask-contract-hashes.json").read_text(
            encoding="utf-8"
        )
    )
    event = json.loads(
        (CONTRACTS_DIR / "schemas/v1.8/examples/mask-camera-observation.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = json.loads(
        (CONTRACTS_DIR / "schemas/v2.1/examples/mask-policy-configuration-bundle.json").read_text(
            encoding="utf-8"
        )
    )
    bundle_body = {key: value for key, value in bundle.items() if key != "bundleSha256"}

    assert compute_canonical_payload_hash(event) == hashes["eventSha256"]
    assert compute_canonical_payload_hash(bundle_body) == hashes["bundleSha256"]
    assert bundle["bundleSha256"] == hashes["bundleSha256"]
    assert (
        compute_canonical_payload_hash({**event, "schemaVersion": "1.7.0"})
        != (hashes["eventSha256"])
    )
    assert (
        compute_canonical_payload_hash({**bundle_body, "schemaVersion": "2.0.0"})
        != (hashes["bundleSha256"])
    )


def test_adoption_doc_documents_non_operational_boundary() -> None:
    doc = (CONTRACTS_DIR / "mask-versioned-adoption.md").read_text(encoding="utf-8")
    assert "not operational" in doc
    assert "v1.8" in doc
    assert "v2.1" in doc
    assert "PRESENT_ONLY" in doc
