from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import (
    MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES,
    MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS,
    PPE_ITEMS,
    CameraConfigurationBundle,
    CameraConfigurationBundleRegion,
)
from smartsite_ai.domain.regions import MAX_SAFE_INTEGER

CAMERA_ID = "11111111-aaaa-4111-8111-11111111111a"
REGION_ID_1 = "22222222-bbbb-4222-8222-22222222222b"
REGION_ID_2 = "33333333-cccc-4333-8333-33333333333c"
ZONE_ID_1 = "44444444-dddd-4444-8444-44444444444d"
ZONE_ID_2 = "55555555-eeee-4555-8555-55555555555e"


def _make_region(
    region_id: str = REGION_ID_1,
    *,
    geometry_version: int = 2,
    zone_id: str = ZONE_ID_1,
    policy_revision: int = 3,
    required_ppe: list[str] | None = None,
    apex_x: float = 0.5,
) -> dict[str, Any]:
    return {
        "regionId": region_id,
        "geometryVersion": geometry_version,
        "coordinateSpace": "NORMALIZED_0_1",
        "polygon": {"coordinates": [[0.0, 0.0], [1.0, 0.0], [apex_x, 1.0]]},
        "zoneId": zone_id,
        "policyRevision": policy_revision,
        "requiredPpe": (
            ["SAFETY_VEST", "HARD_HAT", "GOGGLES"] if required_ppe is None else required_ppe
        ),
    }


def _make_bundle_dict(
    *,
    camera_id: str = CAMERA_ID,
    camera_external_id: str = "CAM-GATE-01",
    configuration_version: int = 7,
    view_revision: int = 4,
    issued_at: str = "2026-10-27T08:00:00.123Z",
    regions: list[dict[str, Any]] | None = None,
    bundle_sha256: str | None = None,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "cameraId": camera_id,
        "cameraExternalId": camera_external_id,
        "configurationVersion": configuration_version,
        "viewRevision": view_revision,
        "issuedAt": issued_at,
        "regions": [_make_region()] if regions is None else regions,
    }
    computed_hash = (
        compute_canonical_payload_hash(content) if bundle_sha256 is None else bundle_sha256
    )
    return {
        **content,
        "bundleSha256": computed_hash,
    }


def _to_wire_bytes(payload: dict[str, Any], *, indent: int | None = None) -> bytes:
    if indent is None:
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return json.dumps(payload, indent=indent).encode("utf-8")


def test_parses_valid_bundle_preserving_whole_bundle_and_region_ppe_policy():
    payload = _make_bundle_dict(
        regions=[
            _make_region(
                REGION_ID_1,
                geometry_version=2,
                zone_id=ZONE_ID_1,
                policy_revision=5,
                required_ppe=["GOGGLES", "HARD_HAT"],
            ),
            _make_region(
                REGION_ID_2,
                geometry_version=1,
                zone_id=ZONE_ID_2,
                policy_revision=1,
                required_ppe=[],
                apex_x=0.25,
            ),
        ]
    )
    bundle = CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(payload, indent=2))

    assert bundle.schema_version == "1.0.0"
    assert bundle.camera_id == CAMERA_ID
    assert bundle.camera_external_id == "CAM-GATE-01"
    assert bundle.configuration_version == 7
    assert bundle.view_revision == 4
    assert bundle.issued_at == "2026-10-27T08:00:00.123Z"
    assert bundle.bundle_sha256 == payload["bundleSha256"]
    assert len(bundle.regions) == 2

    first, second = bundle.regions
    assert isinstance(first, CameraConfigurationBundleRegion)
    assert first.region_id == REGION_ID_1
    assert first.geometry_version == 2
    assert first.coordinate_space == "NORMALIZED_0_1"
    assert first.polygon.coordinates == ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))
    assert first.zone_id == ZONE_ID_1
    assert first.policy_revision == 5
    assert first.required_ppe == ("GOGGLES", "HARD_HAT")

    assert second.region_id == REGION_ID_2
    assert second.zone_id == ZONE_ID_2
    assert second.policy_revision == 1
    assert second.required_ppe == ()

    dumped = bundle.model_dump(mode="json", by_alias=True)
    assert dumped == payload

    with pytest.raises(ValidationError):
        bundle.view_revision = 99  # type: ignore[misc]
    with pytest.raises(ValidationError):
        first.policy_revision = 99  # type: ignore[misc]


def test_bundle_hash_is_sensitive_to_content_and_array_order():
    r1 = _make_region(
        REGION_ID_1,
        zone_id=ZONE_ID_1,
        required_ppe=["HARD_HAT", "SAFETY_VEST"],
    )
    r2 = _make_region(
        REGION_ID_2,
        zone_id=ZONE_ID_2,
        required_ppe=["GOGGLES", "BOOTS", "GLOVES"],
        apex_x=0.75,
    )
    valid = _make_bundle_dict(regions=[r1, r2])
    original_hash = valid["bundleSha256"]

    reordered_ppe = copy.deepcopy(valid)
    reordered_ppe["regions"][0]["requiredPpe"] = ["SAFETY_VEST", "HARD_HAT"]
    with pytest.raises(ValueError, match="bundleSha256"):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(reordered_ppe))

    reordered_regions = copy.deepcopy(valid)
    reordered_regions["regions"] = [r2, r1]
    reordered_regions["bundleSha256"] = original_hash
    with pytest.raises(ValueError, match="bundleSha256"):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(reordered_regions))

    reordered_vertices = copy.deepcopy(valid)
    reordered_vertices["regions"][0]["polygon"]["coordinates"] = [
        [1.0, 0.0],
        [0.5, 1.0],
        [0.0, 0.0],
    ]
    with pytest.raises(ValueError, match="bundleSha256"):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(reordered_vertices))

    tampered_policy = copy.deepcopy(valid)
    tampered_policy["regions"][0]["policyRevision"] = 99
    with pytest.raises(ValueError, match="bundleSha256"):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(tampered_policy))


def test_enforces_exact_wire_keys_no_duplicate_json_keys_and_byte_limit():
    assert MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES == 262_144
    assert MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS == 64
    assert PPE_ITEMS == ("HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES")

    with pytest.raises(TypeError, match="payload must be bytes"):
        CameraConfigurationBundle.from_wire_bytes("{}")  # type: ignore[arg-type]

    oversized = b" " * (MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES + 1)
    with pytest.raises(ValueError, match="262144"):
        CameraConfigurationBundle.from_wire_bytes(oversized)

    extra_bundle_key = _make_bundle_dict()
    extra_bundle_key["unexpected"] = True
    extra_bundle_key["bundleSha256"] = compute_canonical_payload_hash(
        {k: v for k, v in extra_bundle_key.items() if k != "bundleSha256"}
    )
    with pytest.raises(ValidationError):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(extra_bundle_key))

    extra_region_key = _make_bundle_dict()
    extra_region_key["regions"][0]["unexpected"] = 1
    extra_region_key["bundleSha256"] = compute_canonical_payload_hash(
        {k: v for k, v in extra_region_key.items() if k != "bundleSha256"}
    )
    with pytest.raises(ValidationError):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(extra_region_key))

    snake_case_payload = {
        "schema_version": "1.0.0",
        "camera_id": CAMERA_ID,
        "camera_external_id": "CAM-GATE-01",
        "configuration_version": 1,
        "view_revision": 1,
        "issued_at": "2026-10-27T08:00:00.000Z",
        "regions": [],
    }
    snake_case_payload["bundle_sha256"] = compute_canonical_payload_hash(snake_case_payload)
    with pytest.raises(ValidationError):
        CameraConfigurationBundle.model_validate(snake_case_payload)

    valid = _make_bundle_dict(regions=[])
    raw_text = json.dumps(valid, separators=(",", ":"))
    duplicated_key_bytes = (raw_text[:-1] + f',"viewRevision":{valid["viewRevision"]}}}').encode(
        "utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        CameraConfigurationBundle.from_wire_bytes(duplicated_key_bytes)


def test_enforces_region_count_unique_ids_and_lowercase_uuids():
    max_regions = [
        _make_region(
            f"{index:08x}-aaaa-4111-8111-11111111111a",
            zone_id=ZONE_ID_1,
            required_ppe=list(PPE_ITEMS),
        )
        for index in range(MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS)
    ]
    bundle_64 = CameraConfigurationBundle.from_wire_bytes(
        _to_wire_bytes(_make_bundle_dict(regions=max_regions))
    )
    assert len(bundle_64.regions) == 64

    too_many_regions = [
        *max_regions,
        _make_region("ffffffff-aaaa-4111-8111-11111111111a"),
    ]
    with pytest.raises(ValidationError):
        CameraConfigurationBundle.from_wire_bytes(
            _to_wire_bytes(_make_bundle_dict(regions=too_many_regions))
        )

    duplicate_regions = [
        _make_region(REGION_ID_1),
        _make_region(REGION_ID_1, zone_id=ZONE_ID_2),
    ]
    with pytest.raises(ValidationError, match="Region IDs must be unique"):
        CameraConfigurationBundle.from_wire_bytes(
            _to_wire_bytes(_make_bundle_dict(regions=duplicate_regions))
        )

    for bad_camera_id in (CAMERA_ID.upper(), "not-a-uuid"):
        with pytest.raises(ValidationError):
            CameraConfigurationBundle.from_wire_bytes(
                _to_wire_bytes(_make_bundle_dict(camera_id=bad_camera_id))
            )

    for bad_region in (
        _make_region(REGION_ID_1.upper()),
        _make_region(REGION_ID_1, zone_id=ZONE_ID_1.upper()),
        _make_region(REGION_ID_1, zone_id="not-a-uuid"),
    ):
        with pytest.raises(ValidationError):
            CameraConfigurationBundle.from_wire_bytes(
                _to_wire_bytes(_make_bundle_dict(regions=[bad_region]))
            )


def test_enforces_safe_integers_utc_milliseconds_ppe_items_and_geometry():
    max_int_payload = _make_bundle_dict(
        configuration_version=MAX_SAFE_INTEGER,
        view_revision=MAX_SAFE_INTEGER,
        regions=[
            _make_region(
                geometry_version=MAX_SAFE_INTEGER,
                policy_revision=MAX_SAFE_INTEGER,
            )
        ],
    )
    parsed_max = CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(max_int_payload))
    assert parsed_max.configuration_version == MAX_SAFE_INTEGER
    assert parsed_max.view_revision == MAX_SAFE_INTEGER
    assert parsed_max.regions[0].geometry_version == MAX_SAFE_INTEGER
    assert parsed_max.regions[0].policy_revision == MAX_SAFE_INTEGER

    for bad_int in (0, -1, True):
        for field_name in ("configurationVersion", "viewRevision"):
            candidate = _make_bundle_dict(regions=[])
            candidate[field_name] = bad_int
            candidate["bundleSha256"] = "a" * 64
            with pytest.raises(ValidationError):
                CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(candidate))

    unsafe_int = MAX_SAFE_INTEGER + 1
    unsafe_bundle = _make_bundle_dict(regions=[])
    unsafe_bundle["configurationVersion"] = unsafe_int
    with pytest.raises((ValidationError, ValueError)):
        CameraConfigurationBundle.from_wire_bytes(_to_wire_bytes(unsafe_bundle))

    for bad_issued_at in (
        "2026-10-27T08:00:00Z",
        "2026-10-27T08:00:00.123456Z",
        "2026-10-27T08:00:00.123+00:00",
        "2026-02-30T08:00:00.000Z",
        "2026-10-27T24:00:00.000Z",
        "2026-10-27T08:00:60.000Z",
    ):
        with pytest.raises(ValidationError):
            CameraConfigurationBundle.from_wire_bytes(
                _to_wire_bytes(_make_bundle_dict(issued_at=bad_issued_at, regions=[]))
            )

    for bad_ppe in (
        ["HARD_HAT", "HARD_HAT"],
        ["UNKNOWN_PPE"],
        ["hard_hat"],
    ):
        with pytest.raises(ValidationError):
            CameraConfigurationBundle.from_wire_bytes(
                _to_wire_bytes(_make_bundle_dict(regions=[_make_region(required_ppe=bad_ppe)]))
            )

    collinear_region = _make_region()
    collinear_region["polygon"] = {"coordinates": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]}
    with pytest.raises(ValidationError):
        CameraConfigurationBundle.from_wire_bytes(
            _to_wire_bytes(_make_bundle_dict(regions=[collinear_region]))
        )
