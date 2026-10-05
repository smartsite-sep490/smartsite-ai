import json
from pathlib import Path

import pytest

from smartsite_ai.inference.artifacts import ArtifactValidationError
from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP, load_artifact_spec

NATIVE_MAP = {
    "0": "Person",
    "1": "Hardhat",
    "2": "NO-Hardhat",
    "3": "Safety Vest",
    "4": "Gloves",
    "5": "NO-Gloves",
    "6": "Boots",
    "7": "NO-Boots",
    "8": "Goggles",
    "9": "NO-Goggles",
}


def document(tmp_path: Path, classes: dict[str, str]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "artifactId": "test-profile",
                "version": "1",
                "modelFamily": "yolo11s",
                "artifactPath": str((tmp_path / "model.pt").resolve()),
                "sha256": "a" * 64,
                "sourceUrl": "https://models.example.test/model.pt",
                "license": "AGPL-3.0",
                "classMap": classes,
                "confidenceThreshold": 0.25,
                "iouThreshold": 0.7,
                "imageSize": [832, 832],
                "device": "cuda:0",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_default_profile_stays_strict_five_class(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="five canonical"):
        load_artifact_spec(document(tmp_path, NATIVE_MAP))
    spec = load_artifact_spec(
        document(tmp_path, dict((str(k), v) for k, v in CANONICAL_PPE_CLASS_MAP))
    )
    assert spec.class_map[4] == (4, "NO-Safety Vest")


def test_explicit_experimental_profile_preserves_native_class_ids(tmp_path: Path) -> None:
    spec = load_artifact_spec(document(tmp_path, NATIVE_MAP), experimental_profile="native-ppe-10")
    assert spec.class_map == tuple((int(key), name) for key, name in NATIVE_MAP.items())
    assert spec.class_map[4] == (4, "Gloves")


@pytest.mark.parametrize("mutation", ["swap", "extra", "missing"])
def test_experimental_profile_rejects_wrong_taxonomy(tmp_path: Path, mutation: str) -> None:
    classes = dict(NATIVE_MAP)
    if mutation == "swap":
        classes["4"] = "NO-Safety Vest"
    elif mutation == "extra":
        classes["10"] = "Harness"
    else:
        del classes["9"]
    with pytest.raises(ArtifactValidationError, match="native-ppe-10"):
        load_artifact_spec(document(tmp_path, classes), experimental_profile="native-ppe-10")


def test_invalid_profile_fails_before_reading_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="unsupported experimental"):
        load_artifact_spec(tmp_path / "missing.json", experimental_profile="anything")


def test_native_capability_does_not_invent_missing_vest_or_acceptance() -> None:
    from smartsite_ai.inference.ppe_profiles import NATIVE_PPE_10_PROFILE

    profile = NATIVE_PPE_10_PROFILE
    assert profile.experimental is True
    assert profile.present_items == ("HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES")
    assert profile.explicit_missing_items == ("HARD_HAT", "GLOVES", "BOOTS", "GOGGLES")
    assert "SAFETY_VEST" not in profile.explicit_missing_items
    assert "HARNESS" not in profile.present_items


def test_legacy_profile_keeps_missing_vest() -> None:
    from smartsite_ai.inference.ppe_profiles import LEGACY_PPE_5_PROFILE

    assert LEGACY_PPE_5_PROFILE.explicit_missing_items == ("HARD_HAT", "SAFETY_VEST")


def test_profile_import_has_no_inference_side_effects() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import smartsite_ai.inference.ppe_profiles; "
            "assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules; "
            "assert 'ultralytics' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
