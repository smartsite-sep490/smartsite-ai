"""Local-only, reproducible launcher for official Ultralytics YOLO11s PPE training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Protocol

from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP
from smartsite_ai.runtime_device import validate_runtime_device
from smartsite_ai.training.dataset_integrity import DatasetIntegrityError, verify_prepared_dataset

_MAX_DATA_CONFIG_BYTES = 256 * 1024
_RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_MANIFEST_FILENAME = "training.manifest.json"


class TrainingConfigurationError(ValueError):
    """Raised when local training inputs are unsafe or inconsistent."""


class TrainingExecutionError(RuntimeError):
    """Raised when the provider does not produce the required successful run output."""


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    """Validated immutable inputs for one new local training run."""

    data_config: Path
    dataset_aggregate_sha256: str
    base_weights: Path
    output_root: Path
    run_name: str
    run_dir: Path
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    seed: int
    requested_device: str
    device: str


class TrainingProvider(Protocol):
    def train(self, configuration: TrainingConfiguration) -> None:
        """Train exactly one model and return only after provider output is durable."""


@dataclass(frozen=True, slots=True)
class TrainingServices:
    """Injected side-effect seams used by focused tests."""

    provider_factory: Callable[[], TrainingProvider]
    runtime_facts: Callable[[], Mapping[str, object]]
    git_facts: Callable[[], Mapping[str, object]]
    resolve_device: Callable[[str], str]
    now: Callable[[], datetime]
    monotonic: Callable[[], float]


def _bounded_integer(label: str, minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from error
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must satisfy {minimum} <= value <= {maximum}"
            )
        return number

    return parse


def _batch_size(value: str) -> int:
    return _bounded_integer("batch", 1, 512)(value)


def _device(value: str) -> str:
    try:
        return validate_runtime_device(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing Torch or Ultralytics."""

    parser = argparse.ArgumentParser(
        prog="smartsite-ai-train-ppe",
        description="Fine-tune an explicit local official YOLO11s checkpoint for SmartSite PPE.",
    )
    parser.add_argument("--data", type=Path, required=True, help="absolute local data.yaml path")
    parser.add_argument(
        "--base-weights", type=Path, required=True, help="absolute local official yolo11s .pt path"
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="absolute ignored training output directory"
    )
    parser.add_argument("--name", required=True, help="new run directory name")
    parser.add_argument("--epochs", type=_bounded_integer("epochs", 1, 1_000), default=100)
    parser.add_argument("--imgsz", type=_bounded_integer("imgsz", 128, 4_096), default=640)
    parser.add_argument("--batch", type=_batch_size, default=16)
    parser.add_argument("--patience", type=_bounded_integer("patience", 0, 1_000), default=50)
    parser.add_argument("--seed", type=_bounded_integer("seed", 0, 2**32 - 1), default=0)
    parser.add_argument("--device", type=_device, default="auto")
    return parser


def _absolute_regular_file(path: Path, *, label: str, suffix: str) -> Path:
    if not path.is_absolute():
        raise TrainingConfigurationError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingConfigurationError(
            f"{label} must identify an existing regular file"
        ) from error
    if path.is_symlink() or not resolved.is_file():
        raise TrainingConfigurationError(f"{label} must identify a non-symlink regular file")
    if resolved.suffix.lower() != suffix:
        raise TrainingConfigurationError(f"{label} must use the {suffix} suffix")
    return resolved


def _load_data_config(path: Path) -> Mapping[str, object]:
    try:
        if path.stat().st_size > _MAX_DATA_CONFIG_BYTES:
            raise TrainingConfigurationError("data config exceeds 256 KiB")
        text = path.read_text(encoding="utf-8")
    except TrainingConfigurationError:
        raise
    except (OSError, UnicodeError) as error:
        raise TrainingConfigurationError("data config must be readable UTF-8 YAML") from error
    lowered = text.lower()
    if "download:" in lowered or "http://" in lowered or "https://" in lowered:
        raise TrainingConfigurationError("data config must not declare downloads or URLs")
    try:
        import yaml

        parsed = yaml.safe_load(text)
    except ImportError as error:
        raise TrainingConfigurationError(
            "PyYAML is unavailable; install the pinned vision or cuda126 profile"
        ) from error
    except Exception as error:
        raise TrainingConfigurationError("data config is not valid YAML") from error
    if not isinstance(parsed, Mapping):
        raise TrainingConfigurationError("data config root must be a mapping")
    return parsed


def _normalized_class_map(value: object) -> tuple[tuple[int, str], ...]:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise TrainingConfigurationError("data config names list must contain only strings")
        return tuple(enumerate(value))
    if not isinstance(value, Mapping):
        raise TrainingConfigurationError("data config must declare names as a list or mapping")
    normalized: list[tuple[int, str]] = []
    for raw_id, raw_name in value.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, int | str):
            raise TrainingConfigurationError("data config class IDs must be integers")
        try:
            class_id = int(raw_id)
        except ValueError as error:
            raise TrainingConfigurationError("data config class IDs must be integers") from error
        if str(class_id) != str(raw_id) or not isinstance(raw_name, str):
            raise TrainingConfigurationError("data config class map is invalid")
        normalized.append((class_id, raw_name))
    return tuple(sorted(normalized))


def _validate_class_map(data: Mapping[str, object]) -> None:
    class_map = _normalized_class_map(data.get("names"))
    if class_map != CANONICAL_PPE_CLASS_MAP:
        raise TrainingConfigurationError(
            "data config names must exactly equal 0 Person, 1 Hardhat, 2 NO-Hardhat, "
            "3 Safety Vest, 4 NO-Safety Vest"
        )
    nc = data.get("nc")
    if nc is not None and (isinstance(nc, bool) or nc != len(CANONICAL_PPE_CLASS_MAP)):
        raise TrainingConfigurationError("data config nc must equal 5 when declared")


def _validate_dataset_paths(data: Mapping[str, object], data_config: Path) -> None:
    root_value = data.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise TrainingConfigurationError("data config path must be a non-blank local path")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise TrainingConfigurationError(
            "data config path must be absolute for deterministic Ultralytics resolution"
        )
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingConfigurationError(
            "data config dataset root does not exist locally"
        ) from error
    if not root.is_dir():
        raise TrainingConfigurationError("data config dataset root must be a local directory")

    for split in ("train", "val", "test"):
        values = data.get(split)
        if isinstance(values, str):
            split_paths = [values]
        elif isinstance(values, list) and values and all(isinstance(item, str) for item in values):
            split_paths = values
        else:
            raise TrainingConfigurationError(
                f"data config {split} must declare one or more local paths"
            )
        for value in split_paths:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                candidate = candidate.resolve(strict=True)
            except (OSError, RuntimeError, ValueError) as error:
                raise TrainingConfigurationError(
                    f"data config {split} path does not exist locally"
                ) from error
            if not candidate.is_dir() and not candidate.is_file():
                raise TrainingConfigurationError(
                    f"data config {split} path must be a local file or directory"
                )
            if candidate.is_file():
                try:
                    contents = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    raise TrainingConfigurationError(
                        f"data config {split} index must be readable UTF-8"
                    ) from error
                lowered = contents.lower()
                if "http://" in lowered or "https://" in lowered:
                    raise TrainingConfigurationError(
                        f"data config {split} index must not contain remote URLs"
                    )


def _validate_output_root(path: Path) -> Path:
    if not path.is_absolute():
        raise TrainingConfigurationError("output root must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingConfigurationError("output root must be an existing directory") from error
    if path.is_symlink() or not resolved.is_dir():
        raise TrainingConfigurationError("output root must be a non-symlink directory")
    _require_ignored_repository_path(resolved)
    return resolved


def _require_ignored_repository_path(path: Path) -> None:
    """Require an in-repository output root to be covered by Git ignore rules."""

    try:
        root_result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if root_result.returncode != 0:
        return
    repository_root = Path(root_result.stdout.strip()).resolve()
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
        raise TrainingConfigurationError("in-repository output root must be ignored by Git")


def preflight_arguments(args: argparse.Namespace) -> TrainingConfiguration:
    """Validate all paths and invariants before loading a provider or creating output."""

    data_config = _absolute_regular_file(args.data, label="data config", suffix=".yaml")
    base_weights = _absolute_regular_file(args.base_weights, label="base weights", suffix=".pt")
    data = _load_data_config(data_config)
    _validate_class_map(data)
    _validate_dataset_paths(data, data_config)
    try:
        dataset_aggregate_sha256 = verify_prepared_dataset(data_config)
    except DatasetIntegrityError as error:
        raise TrainingConfigurationError(f"prepared dataset integrity failed: {error}") from error
    output_root = _validate_output_root(args.output_root)
    if not isinstance(args.name, str) or not _RUN_NAME_PATTERN.fullmatch(args.name):
        raise TrainingConfigurationError(
            "name must be 1-64 characters using only letters, numbers, dot, underscore, or hyphen"
        )
    run_dir = output_root / args.name
    if run_dir.exists() or run_dir.is_symlink():
        raise TrainingConfigurationError("final training run directory already exists")
    if args.imgsz % 32 != 0:
        raise TrainingConfigurationError("imgsz must be a multiple of 32")
    return TrainingConfiguration(
        data_config=data_config,
        dataset_aggregate_sha256=dataset_aggregate_sha256,
        base_weights=base_weights,
        output_root=output_root,
        run_name=args.name,
        run_dir=run_dir,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        patience=args.patience,
        seed=args.seed,
        requested_device=args.device,
        device=args.device,
    )


def _default_provider_factory() -> TrainingProvider:
    from smartsite_ai.training.ultralytics_provider import UltralyticsTrainingProvider

    return UltralyticsTrainingProvider()


def _distribution_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


def _default_runtime_facts() -> Mapping[str, object]:
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ultralytics": _distribution_version("ultralytics"),
        "torch": _distribution_version("torch"),
    }
    try:
        import torch

        facts["cudaAvailable"] = bool(torch.cuda.is_available())
        facts["cudaRuntime"] = torch.version.cuda
        facts["cudaDevice"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        facts.update(cudaAvailable=False, cudaRuntime=None, cudaDevice=None)
    return facts


def _default_git_facts() -> Mapping[str, object]:
    code_directory = Path(__file__).resolve().parent

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(code_directory), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    try:
        sha = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commitSha": None, "dirty": None}
    if sha.returncode != 0 or status.returncode != 0:
        return {"available": False, "commitSha": None, "dirty": None}
    return {
        "available": True,
        "commitSha": sha.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def _default_resolve_device(configured: str) -> str:
    from smartsite_ai.realtime import resolve_realtime_device

    try:
        import torch
    except ImportError:
        return resolve_realtime_device(
            configured,
            cuda_available=False,
            cuda_device_count=0,
        )
    return resolve_realtime_device(
        configured,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _default_services() -> TrainingServices:
    return TrainingServices(
        provider_factory=_default_provider_factory,
        runtime_facts=_default_runtime_facts,
        git_facts=_default_git_facts,
        resolve_device=_default_resolve_device,
        now=lambda: datetime.now(UTC),
        monotonic=time.monotonic,
    )


def _manifest(
    configuration: TrainingConfiguration,
    *,
    command: Sequence[str],
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
    runtime: Mapping[str, object],
    git: Mapping[str, object],
    checkpoint: Path,
    data_config_sha256: str,
    base_weights_sha256: str,
) -> dict[str, object]:
    return {
        "schemaVersion": "1.0.0",
        "status": "COMPLETE",
        "command": {"arguments": list(command)},
        "configuration": {
            "dataConfig": str(configuration.data_config),
            "dataConfigSha256": data_config_sha256,
            "datasetAggregateSha256": configuration.dataset_aggregate_sha256,
            "baseWeights": str(configuration.base_weights),
            "baseWeightsSha256": base_weights_sha256,
            "outputRoot": str(configuration.output_root),
            "name": configuration.run_name,
            "epochs": configuration.epochs,
            "imageSize": configuration.image_size,
            "batch": configuration.batch_size,
            "patience": configuration.patience,
            "seed": configuration.seed,
            "requestedDevice": configuration.requested_device,
            "resolvedDevice": configuration.device,
            "deterministic": True,
            "automaticMixedPrecision": False,
            "dataLoaderWorkers": 0,
        },
        "source": dict(git),
        "runtime": dict(runtime),
        "startedAt": started_at.isoformat(),
        "completedAt": completed_at.isoformat(),
        "elapsedSeconds": elapsed_seconds,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "sizeBytes": checkpoint.stat().st_size,
        },
    }


def execute_training(
    configuration: TrainingConfiguration,
    *,
    command: Sequence[str],
    services: TrainingServices,
) -> Path:
    """Run training and atomically publish a COMPLETE manifest after verifying best.pt."""

    try:
        resolved_device = services.resolve_device(configuration.requested_device)
    except ValueError as error:
        raise TrainingConfigurationError(f"device is unavailable: {error}") from error
    configuration = replace(configuration, device=resolved_device)
    try:
        before_dataset_aggregate = verify_prepared_dataset(configuration.data_config)
    except DatasetIntegrityError as error:
        raise TrainingConfigurationError(f"prepared dataset integrity failed: {error}") from error
    if before_dataset_aggregate != configuration.dataset_aggregate_sha256:
        raise TrainingConfigurationError("prepared dataset changed after preflight")
    data_config_sha256 = _sha256(configuration.data_config)
    base_weights_sha256 = _sha256(configuration.base_weights)
    started_at = services.now()
    started = services.monotonic()
    git = services.git_facts()
    provider = services.provider_factory()
    provider.train(configuration)
    if _sha256(configuration.data_config) != data_config_sha256:
        raise TrainingExecutionError("data config changed during training")
    if _sha256(configuration.base_weights) != base_weights_sha256:
        raise TrainingExecutionError("base weights changed during training")
    try:
        after_dataset_aggregate = verify_prepared_dataset(configuration.data_config)
    except DatasetIntegrityError as error:
        raise TrainingExecutionError(
            f"prepared dataset integrity failed after training: {error}"
        ) from error
    if after_dataset_aggregate != before_dataset_aggregate:
        raise TrainingExecutionError("prepared dataset changed during training")
    checkpoint = configuration.run_dir / "weights" / "best.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise TrainingExecutionError("training did not produce a non-empty weights/best.pt")
    completed_at = services.now()
    manifest_path = configuration.run_dir / _MANIFEST_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise TrainingExecutionError("provider unexpectedly created the training manifest path")
    payload = _manifest(
        configuration,
        command=command,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=max(0.0, services.monotonic() - started),
        runtime=services.runtime_facts(),
        git=git,
        checkpoint=checkpoint,
        data_config_sha256=data_config_sha256,
        base_weights_sha256=base_weights_sha256,
    )
    _atomic_json(manifest_path, payload)
    return manifest_path


def run(argv: Sequence[str] | None = None, *, services: TrainingServices | None = None) -> int:
    """Run the CLI with injectable provider seams and stable exit codes."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = build_parser().parse_args(arguments)
        configuration = preflight_arguments(parsed)
        manifest_path = execute_training(
            configuration,
            command=("smartsite-ai-train-ppe", *arguments),
            services=services or _default_services(),
        )
    except KeyboardInterrupt:
        print("training interrupted; no COMPLETE manifest was written", file=sys.stderr)
        return 130
    except (TrainingConfigurationError, TrainingExecutionError, RuntimeError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 1
    print(f"training complete: {manifest_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
