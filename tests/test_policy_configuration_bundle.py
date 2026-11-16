from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle


def content() -> dict[str, Any]:
    policies = [
        {"mode": "INHERIT"},
        {"mode": "EXPLICIT", "requiredPpe": ["GLOVES"]},
        {"mode": "DISABLED", "reason": "Temporary PPE policy exception"},
    ]
    return {
        "schemaVersion": "2.0.0",
        "cameraId": "11111111-2222-4333-8444-555555555555",
        "cameraExternalId": "camera-01",
        "configurationVersion": 4,
        "viewRevision": 2,
        "issuedAt": "2026-11-13T00:00:00.000Z",
        "cameraPpePolicy": {"mode": "EXPLICIT", "requiredPpe": ["HARD_HAT", "SAFETY_VEST"]},
        "regions": [
            {
                "regionId": f"aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeee{index + 1}",
                "geometryVersion": 1,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0.2, 0.2], [0.7, 0.2], [0.5, 0.7]]},
                "zoneId": "99999999-8888-4777-8666-555555555551",
                "policyRevision": 1,
                "ppePolicy": policy,
            }
            for index, policy in enumerate(policies)
        ],
    }


def packet(value: dict[str, Any]) -> bytes:
    return json.dumps(
        {**value, "bundleSha256": compute_canonical_payload_hash(value)},
        separators=(",", ":"),
    ).encode()


@pytest.fixture
def model():
    name = "smartsite_ai.domain.policy_configuration_bundle"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail("The v2 policy bundle decoder is not implemented")
    return module.CameraPolicyConfigurationBundle


def test_matches_backend_golden_hash_and_preserves_wire_policy(model):
    original = content()
    bundle = model.from_wire_bytes(packet(original))
    assert (
        bundle.bundle_sha256 == "5e1277da37e6c51644689c6a4c0481e8940dcfc5b2e86f66a7f0fe127297c5ca"
    )
    assert bundle.model_dump(mode="json", by_alias=True, exclude={"bundle_sha256"}) == original
    assert bundle.regions[0].ppe_policy.mode == "INHERIT"
    assert bundle.regions[1].ppe_policy.required_ppe == ("GLOVES",)


def test_accepts_camera_scope_without_inventing_regions(model):
    assert model.from_wire_bytes(packet({**content(), "regions": []})).regions == ()


def test_policy_mutations_cannot_reuse_the_original_hash(model):
    original = json.loads(packet(content()))
    for change in (
        {"cameraPpePolicy": {"mode": "DISABLED", "reason": "Changed default"}},
        {
            "regions": [
                {
                    **original["regions"][0],
                    "ppePolicy": {"mode": "EXPLICIT", "requiredPpe": ["BOOTS"]},
                }
            ]
        },
    ):
        with pytest.raises(ValueError):
            model.from_wire_bytes(json.dumps({**original, **change}).encode())


def test_rejects_invalid_policies_and_caller_invented_scope(model):
    for policy in (
        {"mode": "INHERIT"},
        {"mode": "EXPLICIT", "requiredPpe": []},
        {"mode": "EXPLICIT", "requiredPpe": ["MASK"]},
        {"mode": "EXPLICIT", "requiredPpe": ["GLOVES", "GLOVES"]},
        {"mode": "DISABLED", "reason": " "},
        {"mode": "DISABLED", "reason": "hidden" + chr(10) + "control"},
    ):
        with pytest.raises(ValueError):
            model.from_wire_bytes(packet({**content(), "cameraPpePolicy": policy}))
    for extra in ({"actorId": "invented"}, {"requiredPpe": []}):
        with pytest.raises(ValueError):
            model.from_wire_bytes(packet({**content(), **extra}))
        value = content()
        value["regions"][0].update(extra)
        with pytest.raises(ValueError):
            model.from_wire_bytes(packet(value))


def test_disabled_reason_uses_javascript_utf16_limit_without_mutating_wire(model):
    for reason in (chr(0xFEFF), "😀" * 251):
        with pytest.raises(ValueError):
            model.from_wire_bytes(
                packet({**content(), "cameraPpePolicy": {"mode": "DISABLED", "reason": reason}})
            )
    for reason in ("😀" * 250, "  Explicit exception " + chr(0xFEFF)):
        bundle = model.from_wire_bytes(
            packet({**content(), "cameraPpePolicy": {"mode": "DISABLED", "reason": reason}})
        )
        assert bundle.camera_ppe_policy.reason == reason


def test_preserves_geometry_revision_and_version_checks(model):
    for field, invalid in (
        ("configurationVersion", True),
        ("configurationVersion", 0),
        ("viewRevision", 1.5),
        ("cameraId", "invented"),
        ("issuedAt", "2026-02-30T00:00:00.000Z"),
    ):
        with pytest.raises(ValueError):
            model.from_wire_bytes(packet({**content(), field: invalid}))
    value = content()
    value["regions"][1]["regionId"] = value["regions"][0]["regionId"]
    with pytest.raises(ValueError):
        model.from_wire_bytes(packet(value))
    value = content()
    value["regions"][0]["polygon"]["coordinates"] = [[0, 0], [1, 1], [0, 1], [1, 0]]
    with pytest.raises(ValueError):
        model.from_wire_bytes(packet(value))


def test_v1_and_v2_require_explicit_decoders(model):
    with pytest.raises(ValueError):
        CameraConfigurationBundle.from_wire_bytes(packet(content()))
    legacy = copy.deepcopy(content())
    legacy["schemaVersion"] = "1.0.0"
    legacy.pop("cameraPpePolicy")
    for region in legacy["regions"]:
        region.pop("ppePolicy")
        region["requiredPpe"] = []
    assert CameraConfigurationBundle.from_wire_bytes(packet(legacy))
    with pytest.raises(ValueError):
        model.from_wire_bytes(packet(legacy))


def test_bounds_wire_and_rejects_duplicate_keys_without_vision_imports(model):
    duplicate = packet(content()).replace(
        b'"configurationVersion":4', b'"configurationVersion":4,"configurationVersion":4'
    )
    for wire in (bytes([255]), b"{", b" " * 262145, duplicate):
        with pytest.raises(ValueError):
            model.from_wire_bytes(wire)
    with pytest.raises(TypeError):
        model.from_wire_bytes("{}")
    # Collection of unrelated API tests may already import FastAPI. Check this
    # decoder's import boundary in a fresh process, not global pytest state.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd() / 'src')); "
            "import smartsite_ai.domain.policy_configuration_bundle as module; "
            "assert Path(module.__file__).is_relative_to(Path.cwd() / 'src'); "
            "assert not {'torch', 'cv2', 'ultralytics', 'fastapi'}.intersection(sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
        timeout=10,
    )
