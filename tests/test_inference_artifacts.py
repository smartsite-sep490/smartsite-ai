import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartsite_ai.inference.artifacts import (
    ArtifactChecksumMismatchError,
    ArtifactNotFoundError,
    ArtifactValidationError,
    ModelArtifactSpec,
    verify_model_artifact,
)


def artifact_data(artifact_path: Path, **overrides: object) -> dict[str, object]:
    sha256 = (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if artifact_path.is_file()
        else "a" * 64
    )
    return {
        "artifact_id": "yolo11s-ppe",
        "version": "2026.09.21",
        "artifact_path": artifact_path,
        "sha256": sha256,
        "source_url": "https://models.example.test/yolo11s-ppe.pt",
        "license": "AGPL-3.0",
        "class_map": ((0, "person"), (1, "hard_hat")),
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "image_size": (640, 640),
        "device": "cuda:0",
        **overrides,
    }


def make_spec(tmp_path: Path, **overrides: object) -> ModelArtifactSpec:
    artifact_path = tmp_path / "model.pt"
    artifact_path.write_bytes(b"\x00\x01")
    return ModelArtifactSpec.model_validate(artifact_data(artifact_path, **overrides))


def test_verifies_two_byte_local_artifact_and_preserves_immutable_metadata(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)

    verified = verify_model_artifact(spec)

    assert verified.artifact_id == "yolo11s-ppe"
    assert verified.version == "2026.09.21"
    assert verified.resolved_path == (tmp_path / "model.pt").resolve()
    assert verified.actual_sha256 == hashlib.sha256(b"\x00\x01").hexdigest()
    assert verified.class_map == ((0, "person"), (1, "hard_hat"))
    with pytest.raises(ValidationError, match="frozen"):
        verified.device = "cpu"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", ""),
        ("artifact_id", "artifact\x00id"),
        ("version", ""),
        ("version", "version\x00one"),
        ("sha256", "A" * 64),
        ("sha256", "a" * 63),
        ("sha256", "g" * 64),
        ("source_url", "http://models.example.test/model.pt"),
        ("source_url", "not-a-url"),
        ("license", ""),
        ("license", "AGPL\x00-3.0"),
        ("class_map", ()),
        ("class_map", ((0, "person"), (0, "hard_hat"))),
        ("class_map", ((-1, "person"),)),
        ("class_map", ((0, ""),)),
        ("class_map", ((0, "hard\x00hat"),)),
        ("confidence_threshold", -0.01),
        ("confidence_threshold", 1.01),
        ("iou_threshold", -0.01),
        ("iou_threshold", 1.01),
        ("image_size", (0, 640)),
        ("image_size", (640, 0)),
        ("device", ""),
        ("device", "cuda\x00:0"),
        ("unexpected", True),
    ],
)
def test_artifact_spec_rejects_invalid_declared_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    artifact_path = tmp_path / "model.pt"
    artifact_path.write_bytes(b"\x00\x01")

    with pytest.raises(ValidationError) as exc_info:
        ModelArtifactSpec.model_validate(artifact_data(artifact_path, **{field: value}))

    assert any(error["loc"][0] == field for error in exc_info.value.errors())


def test_artifact_spec_requires_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_path = Path("model.pt")
    relative_path.write_bytes(b"\x00\x01")

    with pytest.raises(ValidationError) as exc_info:
        ModelArtifactSpec.model_validate(artifact_data(relative_path))

    assert any(error["loc"] == ("artifact_path",) for error in exc_info.value.errors())


def test_verification_rejects_missing_and_directory_artifacts(tmp_path: Path) -> None:
    missing = ModelArtifactSpec.model_validate(
        artifact_data(tmp_path / "missing.pt", sha256="a" * 64)
    )
    directory = ModelArtifactSpec.model_validate(artifact_data(tmp_path, sha256="a" * 64))

    with pytest.raises(ArtifactNotFoundError):
        verify_model_artifact(missing)
    with pytest.raises(ArtifactValidationError):
        verify_model_artifact(directory)


def test_verification_rejects_checksum_mismatch_without_exposing_file_content(
    tmp_path: Path,
) -> None:
    spec = make_spec(tmp_path, sha256="a" * 64)

    with pytest.raises(ArtifactChecksumMismatchError) as exc_info:
        verify_model_artifact(spec)

    assert "\x00\x01" not in str(exc_info.value)


def test_verification_rejects_symlink_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"\x00\x01")
    symlink = tmp_path / "model-link.pt"
    try:
        symlink.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable on this platform: {error}")
    spec = ModelArtifactSpec.model_validate(artifact_data(symlink))

    with pytest.raises(ArtifactValidationError):
        verify_model_artifact(spec)
