"""Build a verified local YOLO11s artifact spec from one COMPLETE training manifest."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from smartsite_ai.inference.loading import (
    CANONICAL_PPE_CLASS_MAP,
    load_artifact_spec,
)
from smartsite_ai.training.dataset_integrity import (
    DatasetIntegrityError,
    is_link_like,
    safe_contained_path,
    sha256_file,
    verify_prepared_dataset,
)

_MAX_MANIFEST_BYTES = 1024 * 1024
_CLASS_MAP = {str(class_id): name for class_id, name in CANONICAL_PPE_CLASS_MAP}
_UNREVIEWED_LICENSE_MARKERS = ("replace", "todo", "tbd", "unknown", "unverified")


class ArtifactSpecBuildError(ValueError):
    """Raised when training evidence cannot produce a safe artifact specification."""


def _finite_probability(label: str):
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{label} must be a number") from error
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise argparse.ArgumentTypeError(f"{label} must be finite and between 0 and 1")
        return number

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartsite-ai-build-artifact-spec",
        description="Create a local verified artifact spec from one COMPLETE training manifest.",
    )
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-id", default="smartsite-yolo11s-ppe")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True, dest="artifact_license")
    parser.add_argument(
        "--license-reviewed",
        required=True,
        action="store_true",
        help="confirm the supplied artifact license was reviewed for the intended use",
    )
    parser.add_argument(
        "--confidence-threshold", type=_finite_probability("confidence threshold"), default=0.25
    )
    parser.add_argument("--iou-threshold", type=_finite_probability("IoU threshold"), default=0.45)
    parser.add_argument(
        "--device",
        default="trained",
        help="runtime device or 'trained' to reuse the manifest's resolved training device",
    )
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactSpecBuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> tuple[Path, Mapping[str, object]]:
    if not path.is_absolute():
        raise ArtifactSpecBuildError("training manifest must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactSpecBuildError("training manifest must exist") from error
    if is_link_like(path) or not resolved.is_file():
        raise ArtifactSpecBuildError("training manifest must be a non-link regular file")
    if resolved.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ArtifactSpecBuildError("training manifest exceeds 1 MiB")
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactSpecBuildError("training manifest must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ArtifactSpecBuildError("training manifest root must be an object")
    return resolved, value


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactSpecBuildError(f"training manifest {label} must be an object")
    return value


def _required_string(value: object, *, label: str, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArtifactSpecBuildError(f"training manifest {label} must be a non-blank string")
    if len(value) > max_length or "\x00" in value:
        raise ArtifactSpecBuildError(f"training manifest {label} exceeds its boundary")
    return value


def _public_source_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname.casefold() == "localhost"
        or parsed.hostname.casefold().endswith((".localhost", ".invalid", ".local"))
    ):
        raise ArtifactSpecBuildError(
            "source URL must be a public HTTPS URL without credentials, query, or fragment"
        )
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return value
    if not address.is_global:
        raise ArtifactSpecBuildError("source URL must not use a private or reserved IP address")
    return value


def _reviewed_license(value: str, confirmed: bool) -> str:
    if not confirmed:
        raise ArtifactSpecBuildError("artifact license review must be explicitly confirmed")
    if not value.strip() or value != value.strip() or len(value) > 128 or "\x00" in value:
        raise ArtifactSpecBuildError("artifact license must be a bounded non-blank value")
    lowered = value.casefold()
    if any(marker in lowered for marker in _UNREVIEWED_LICENSE_MARKERS):
        raise ArtifactSpecBuildError("artifact license still contains an unreviewed placeholder")
    return value


def _require_ignored_repository_file(path: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if result.returncode != 0:
        return
    repository_root = Path(result.stdout.strip()).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "-C", str(repository_root), "check-ignore", "--quiet", "--no-index", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if ignored.returncode != 0:
        raise ArtifactSpecBuildError("in-repository artifact spec output must be ignored by Git")


def _new_output_file(path: Path) -> Path:
    if not path.is_absolute():
        raise ArtifactSpecBuildError("artifact spec output must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ArtifactSpecBuildError("artifact spec output must not already exist")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactSpecBuildError("artifact spec output parent must exist") from error
    if is_link_like(parent) or not parent.is_dir():
        raise ArtifactSpecBuildError("artifact spec output parent must be a non-link directory")
    output = parent / path.name
    _require_ignored_repository_file(output)
    return output


def _verified_checkpoint(
    manifest_path: Path, manifest: Mapping[str, object]
) -> tuple[Path, str, str, int, str]:
    if manifest.get("schemaVersion") != "1.0.0" or manifest.get("status") != "COMPLETE":
        raise ArtifactSpecBuildError("training manifest must be COMPLETE schema 1.0.0")
    configuration = _required_mapping(manifest.get("configuration"), label="configuration")
    checkpoint = _required_mapping(manifest.get("checkpoint"), label="checkpoint")
    run_name = _required_string(
        configuration.get("name"), label="configuration.name", max_length=64
    )
    image_size = configuration.get("imageSize")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or not 1 <= image_size <= 16384
    ):
        raise ArtifactSpecBuildError("training manifest configuration.imageSize is invalid")
    raw_checkpoint = _required_string(checkpoint.get("path"), label="checkpoint.path")
    checkpoint_path = Path(raw_checkpoint)
    if not checkpoint_path.is_absolute():
        raise ArtifactSpecBuildError("training checkpoint path must be absolute")
    try:
        resolved_checkpoint = checkpoint_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactSpecBuildError("training checkpoint does not exist") from error
    expected_checkpoint = (manifest_path.parent / "weights" / "best.pt").resolve(strict=False)
    try:
        safe_contained_path(checkpoint_path, manifest_path.parent, kind="training checkpoint")
    except DatasetIntegrityError as error:
        raise ArtifactSpecBuildError(str(error)) from error
    if (
        is_link_like(checkpoint_path)
        or not resolved_checkpoint.is_file()
        or resolved_checkpoint != expected_checkpoint
    ):
        raise ArtifactSpecBuildError(
            "training checkpoint must be this run's non-link weights/best.pt"
        )
    declared_hash = _required_string(
        checkpoint.get("sha256"), label="checkpoint.sha256", max_length=64
    )
    actual_hash = sha256_file(resolved_checkpoint)
    if declared_hash != actual_hash:
        raise ArtifactSpecBuildError("training checkpoint SHA-256 does not match best.pt")
    declared_size = checkpoint.get("sizeBytes")
    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size <= 0
        or declared_size != resolved_checkpoint.stat().st_size
    ):
        raise ArtifactSpecBuildError("training checkpoint size does not match best.pt")

    data_config_value = _required_string(
        configuration.get("dataConfig"), label="configuration.dataConfig"
    )
    data_config = Path(data_config_value)
    if not data_config.is_absolute() or is_link_like(data_config) or not data_config.is_file():
        raise ArtifactSpecBuildError("training data config is unavailable or linked")
    if sha256_file(data_config) != configuration.get("dataConfigSha256"):
        raise ArtifactSpecBuildError("training data config SHA-256 does not match")
    _validate_data_config_class_map(data_config)
    try:
        aggregate = verify_prepared_dataset(data_config.resolve(strict=True))
    except DatasetIntegrityError as error:
        raise ArtifactSpecBuildError(
            f"prepared training dataset integrity failed: {error}"
        ) from error
    if aggregate != configuration.get("datasetAggregateSha256"):
        raise ArtifactSpecBuildError("training dataset aggregate does not match the manifest")
    preparation = _load_preparation_manifest(data_config.parent / "preparation.manifest.json")
    canonical = _required_mapping(preparation.get("canonical"), label="prepared canonical")
    if canonical.get("classMap") != _CLASS_MAP:
        raise ArtifactSpecBuildError("prepared training dataset class map is not canonical")
    resolved_device = _required_string(
        configuration.get("resolvedDevice"),
        label="configuration.resolvedDevice",
        max_length=128,
    )
    return (
        resolved_checkpoint,
        actual_hash,
        run_name,
        image_size,
        resolved_device,
    )


def _load_preparation_manifest(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactSpecBuildError("preparation manifest must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ArtifactSpecBuildError("preparation manifest root must be an object")
    return value


def _validate_data_config_class_map(path: Path) -> None:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError as error:
        raise ArtifactSpecBuildError("PyYAML is unavailable") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ArtifactSpecBuildError("training data config must be valid UTF-8 YAML") from error
    if not isinstance(value, Mapping):
        raise ArtifactSpecBuildError("training data config root must be an object")
    names = value.get("names")
    if isinstance(names, list) and all(isinstance(item, str) for item in names):
        normalized = {str(index): item for index, item in enumerate(names)}
    elif isinstance(names, Mapping):
        normalized = {str(key): item for key, item in names.items()}
    else:
        raise ArtifactSpecBuildError("training data config class map is invalid")
    if normalized != _CLASS_MAP or value.get("nc") not in (None, 5):
        raise ArtifactSpecBuildError("training data config class map is not canonical")


def build_artifact_spec(
    training_manifest: Path,
    output: Path,
    *,
    artifact_id: str,
    source_url: str,
    artifact_license: str,
    license_reviewed: bool,
    confidence_threshold: float,
    iou_threshold: float,
    device: str,
) -> Path:
    """Verify training evidence and atomically publish one strict artifact spec."""

    manifest_path, manifest = _load_manifest(training_manifest)
    output = _new_output_file(output)
    source_url = _public_source_url(source_url)
    artifact_license = _reviewed_license(artifact_license, license_reviewed)
    checkpoint, checksum, run_name, image_size, trained_device = _verified_checkpoint(
        manifest_path, manifest
    )
    if not device.strip() or device != device.strip() or len(device) > 128 or "\x00" in device:
        raise ArtifactSpecBuildError("device must be a bounded non-blank value")
    if device == "trained":
        device = trained_device
    payload = {
        "artifactId": artifact_id,
        "version": run_name,
        "modelFamily": "yolo11s",
        "artifactPath": str(checkpoint),
        "sha256": checksum,
        "sourceUrl": source_url,
        "license": artifact_license,
        "classMap": _CLASS_MAP,
        "confidenceThreshold": confidence_threshold,
        "iouThreshold": iou_threshold,
        "imageSize": [image_size, image_size],
        "device": device,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        load_artifact_spec(temporary)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def run(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = build_parser().parse_args(arguments)
        output = build_artifact_spec(
            parsed.training_manifest,
            parsed.output,
            artifact_id=parsed.artifact_id,
            source_url=parsed.source_url,
            artifact_license=parsed.artifact_license,
            license_reviewed=parsed.license_reviewed,
            confidence_threshold=parsed.confidence_threshold,
            iou_threshold=parsed.iou_threshold,
            device=parsed.device,
        )
    except KeyboardInterrupt:
        print("artifact spec generation interrupted; no output was published", file=sys.stderr)
        return 130
    except (ArtifactSpecBuildError, DatasetIntegrityError, OSError, RuntimeError) as error:
        print(f"artifact spec generation failed: {error}", file=sys.stderr)
        return 1
    print(f"artifact spec complete: {output}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
