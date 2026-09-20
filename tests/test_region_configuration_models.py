import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.regions import CameraRegionConfiguration

ROOT = Path(__file__).resolve().parents[1]
VECTORS = json.loads(
    (ROOT / "contracts" / "camera-region-configuration-vectors.json").read_text(encoding="utf-8")
)
MAX_WIRE_BYTES = 262_144
REGION_ID = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"


def _wire_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _python_payload(
    coordinates: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "configurationVersion": 42,
        "cameraExternalId": "CAM-GATE-01",
        "regions": (
            {
                "regionId": REGION_ID,
                "geometryVersion": 3,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": coordinates},
            },
        ),
    }


@pytest.mark.parametrize(
    "vector",
    [vector for vector in VECTORS if vector["valid"]],
    ids=lambda vector: vector["description"],
)
def test_every_valid_golden_vector_round_trips_with_its_canonical_hash(
    vector: dict[str, Any],
):
    snapshot = CameraRegionConfiguration.from_wire_bytes(_wire_bytes(vector["payload"]))
    wire = snapshot.model_dump(mode="json", by_alias=True)

    assert wire == vector["payload"]
    assert compute_canonical_payload_hash(wire) == vector["expectedSha256"]
    assert isinstance(snapshot.regions, tuple)
    for region in snapshot.regions:
        assert isinstance(region.polygon.coordinates, tuple)
        assert all(isinstance(coordinate, tuple) for coordinate in region.polygon.coordinates)


@pytest.mark.parametrize(
    "vector",
    [vector for vector in VECTORS if not vector["valid"]],
    ids=lambda vector: vector["description"],
)
def test_every_invalid_golden_vector_is_rejected_by_the_runtime_model(
    vector: dict[str, Any],
):
    with pytest.raises(ValidationError):
        CameraRegionConfiguration.from_wire_bytes(_wire_bytes(vector["payload"]))


def test_wire_shape_uses_only_the_canonical_camel_case_fields():
    snapshot = CameraRegionConfiguration.model_validate(_python_payload())

    assert snapshot.model_dump(mode="json", by_alias=True) == {
        "schemaVersion": "1.0.0",
        "configurationVersion": 42,
        "cameraExternalId": "CAM-GATE-01",
        "regions": [
            {
                "regionId": REGION_ID,
                "geometryVersion": 3,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]},
            }
        ],
    }

    snake_case = _python_payload()
    snake_case["configuration_version"] = snake_case.pop("configurationVersion")
    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(snake_case)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("root", "unexpected", True),
        ("region", "zoneId", "zone-1"),
        ("polygon", "closed", True),
    ],
)
def test_unknown_fields_are_forbidden_at_every_object_level(scope: str, field: str, value: object):
    payload = _python_payload()
    if scope == "root":
        payload[field] = value
    elif scope == "region":
        payload["regions"][0][field] = value
    else:
        payload["regions"][0]["polygon"][field] = value

    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(payload)


def test_snapshot_and_nested_geometry_are_immutable():
    snapshot = CameraRegionConfiguration.model_validate(_python_payload())

    with pytest.raises(ValidationError):
        snapshot.configuration_version = 43
    with pytest.raises(ValidationError):
        snapshot.regions[0].geometry_version = 4
    with pytest.raises(TypeError):
        snapshot.regions[0].polygon.coordinates[0] = (0.2, 0.2)


@pytest.mark.parametrize("field", ["configurationVersion", "geometryVersion"])
@pytest.mark.parametrize("invalid", ["42", 42.0, True])
def test_python_object_versions_do_not_coerce_strings_floats_or_booleans(
    field: str, invalid: object
):
    payload = _python_payload()
    if field == "configurationVersion":
        payload[field] = invalid
    else:
        payload["regions"][0][field] = invalid

    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(payload)


@pytest.mark.parametrize("invalid", ["0.25", True, None])
def test_python_object_coordinates_do_not_coerce_non_numeric_values(invalid: object):
    payload = _python_payload(((invalid, 0.0), (1.0, 0.0), (0.0, 1.0)))

    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(payload)


def test_wire_json_accepts_json_number_forms_and_uuid_strings_without_changing_spelling():
    payload = (
        b'{"schemaVersion":"1.0.0","configurationVersion":42.0,'
        b'"cameraExternalId":"CAM-GATE-01","regions":[{'
        b'"regionId":"F81D4FAE-7DEC-11D0-A765-00A0C91E6BF6",'
        b'"geometryVersion":3.0,"coordinateSpace":"NORMALIZED_0_1",'
        b'"polygon":{"coordinates":[[0,0],[1.0,0],[0,1.0]]}}]}'
    )

    snapshot = CameraRegionConfiguration.from_wire_bytes(payload)

    assert snapshot.configuration_version == 42
    assert snapshot.regions[0].geometry_version == 3
    assert snapshot.regions[0].region_id == "F81D4FAE-7DEC-11D0-A765-00A0C91E6BF6"


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -0.1, 1.1])
def test_coordinates_must_be_finite_and_normalized(coordinate: float):
    payload = _python_payload(((coordinate, 0.0), (1.0, 0.0), (0.0, 1.0)))

    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(payload)


def test_polygon_rejects_more_than_64_vertices():
    coordinates = tuple((index / 65, 0.5) for index in range(65))

    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(_python_payload(coordinates))


INVALID_POLYGONS = [
    (
        "exactly collinear despite floating-point cancellation",
        (
            (450000004 / 2**30, 123000000 / 2**30),
            (450010005 / 2**30, 123020002 / 2**30),
            (450020006 / 2**30, 123040004 / 2**30),
        ),
    ),
    ("repeated closing vertex", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    (
        "repeated interior vertex",
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
    ),
    ("collinear zero area", ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))),
    ("bow-tie self-intersection", ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))),
    (
        "nonzero-area self-intersection",
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.0)),
    ),
    (
        "non-adjacent endpoint touch",
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (0.0, 1.0)),
    ),
    (
        "collinear overlap",
        ((0.0, 0.0), (0.75, 0.0), (0.75, 1.0), (0.25, 0.0), (1.0, 0.0), (0.0, 1.0)),
    ),
    (
        "nonzero-area crossing below floating-point product range",
        ((0.0, 1e-200), (2e-200, 1e-200), (1e-200, 0.0), (1e-200, 3e-200)),
    ),
]


@pytest.mark.parametrize(("name", "coordinates"), INVALID_POLYGONS, ids=lambda value: value)
def test_polygon_semantics_match_canonical_typescript_rejections(
    name: str, coordinates: tuple[tuple[float, float], ...]
):
    del name
    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(_python_payload(coordinates))


VALID_POLYGONS = [
    ("tiny triangle", ((0.5, 0.5), (0.500000001, 0.5), (0.5, 0.500000001))),
    (
        "subnormal triangle",
        (
            (0.0, 0.0),
            (float.fromhex("0x0.0000000000001p-1022"), 0.0),
            (0.0, float.fromhex("0x0.0000000000001p-1022")),
        ),
    ),
    ("counter-clockwise", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
    ("clockwise", ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))),
    (
        "concave",
        ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0)),
    ),
    (
        "collinear adjacent edges",
        ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    ),
    (
        "parallelogram below floating-point product range",
        ((0.0, 0.0), (3e-200, 2e-200), (3e-200, 3e-200), (0.0, 1e-200)),
    ),
]


@pytest.mark.parametrize(("name", "coordinates"), VALID_POLYGONS, ids=lambda value: value)
def test_polygon_semantics_match_canonical_typescript_acceptance(
    name: str, coordinates: tuple[tuple[float, float], ...]
):
    del name
    snapshot = CameraRegionConfiguration.model_validate(_python_payload(coordinates))
    assert snapshot.regions[0].polygon.coordinates == coordinates


def test_mixed_case_region_ids_are_duplicate_but_accepted_spelling_is_preserved():
    payload = _python_payload()
    duplicate = copy.deepcopy(payload["regions"][0])
    duplicate["regionId"] = REGION_ID.upper()
    payload["regions"] = (*payload["regions"], duplicate)

    with pytest.raises(ValidationError):
        CameraRegionConfiguration.model_validate(payload)

    uppercase = _python_payload()
    uppercase["regions"][0]["regionId"] = REGION_ID.upper()
    snapshot = CameraRegionConfiguration.model_validate(uppercase)
    assert snapshot.regions[0].region_id == REGION_ID.upper()


def test_from_wire_bytes_accepts_valid_utf8_and_rejects_invalid_utf8_or_json():
    valid = _wire_bytes(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "CAM-CỔNG-01",
            "regions": [],
        }
    )
    assert CameraRegionConfiguration.from_wire_bytes(valid).camera_external_id == "CAM-CỔNG-01"

    for invalid in (b"\xff", b'{"schemaVersion":', b"{} {}"):
        with pytest.raises(ValidationError):
            CameraRegionConfiguration.from_wire_bytes(invalid)


def test_from_wire_bytes_accepts_exact_limit_and_checks_oversize_before_parsing():
    valid = _wire_bytes(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "CAM-LIMIT",
            "regions": [],
        }
    )
    exact_limit = valid + b" " * (MAX_WIRE_BYTES - len(valid))
    assert len(exact_limit) == MAX_WIRE_BYTES
    assert CameraRegionConfiguration.from_wire_bytes(exact_limit).camera_external_id == "CAM-LIMIT"

    with pytest.raises(ValueError, match="262144") as exc_info:
        CameraRegionConfiguration.from_wire_bytes(b"\xff" * (MAX_WIRE_BYTES + 1))
    assert "exceeds" in str(exc_info.value)
