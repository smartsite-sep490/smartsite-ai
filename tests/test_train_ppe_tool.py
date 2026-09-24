import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from smartsite_ai.tools.train_ppe import (
    TrainingConfigurationError,
    TrainingServices,
    build_parser,
    preflight_arguments,
    run,
)


def _write_inputs(tmp_path: Path, *, names: str | None = None) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    for split in ("train", "validation", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
    data = tmp_path / "data.yaml"
    data.write_text(
        "\n".join(
            (
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "nc: 5",
                names or "names: [Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest]",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    weights = tmp_path / "yolo11s.pt"
    weights.write_bytes(b"local official base weights fixture")
    output_root = tmp_path / "runs"
    output_root.mkdir(exist_ok=True)
    return data, weights, output_root


def _argv(tmp_path: Path, **overrides: object) -> list[str]:
    data, weights, output_root = _write_inputs(tmp_path)
    values: dict[str, object] = {
        "data": data,
        "weights": weights,
        "output_root": output_root,
        "name": "ppe-run-001",
        "epochs": 25,
        "imgsz": 640,
        "batch": 8,
        "patience": 10,
        "seed": 42,
        "device": "cuda:0",
    }
    values.update(overrides)
    return [
        "--data",
        str(values["data"]),
        "--base-weights",
        str(values["weights"]),
        "--output-root",
        str(values["output_root"]),
        "--name",
        str(values["name"]),
        "--epochs",
        str(values["epochs"]),
        "--imgsz",
        str(values["imgsz"]),
        "--batch",
        str(values["batch"]),
        "--patience",
        str(values["patience"]),
        "--seed",
        str(values["seed"]),
        "--device",
        str(values["device"]),
    ]


class SuccessfulProvider:
    def __init__(self) -> None:
        self.received = []

    def train(self, configuration: object) -> None:
        self.received.append(configuration)
        checkpoint = configuration.run_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fine-tuned checkpoint")


def _sequence(values: list[object]) -> Iterator[object]:
    yield from values


def _services(provider: object) -> TrainingServices:
    moments = _sequence(
        [
            datetime(2026, 9, 25, 1, 2, 3, tzinfo=UTC),
            datetime(2026, 9, 25, 1, 3, 4, tzinfo=UTC),
        ]
    )
    monotonic = _sequence([10.0, 71.25])
    return TrainingServices(
        provider_factory=lambda: provider,
        runtime_facts=lambda: {
            "python": "3.12.13",
            "ultralytics": "8.4.155",
            "torch": "2.14.0+cu126",
            "cudaAvailable": True,
            "cudaRuntime": "12.6",
            "cudaDevice": "NVIDIA GeForce RTX 4060",
        },
        git_facts=lambda: {
            "available": True,
            "commitSha": "a" * 40,
            "dirty": False,
        },
        resolve_device=lambda requested: "cuda:0" if requested == "auto" else requested,
        now=lambda: next(moments),
        monotonic=lambda: next(monotonic),
    )


def test_import_does_not_load_vision_or_network_stacks() -> None:
    script = (
        "import sys; import smartsite_ai.tools.train_ppe; "
        "forbidden={'torch','ultralytics','requests','httpx'}; "
        "loaded=forbidden.intersection(sys.modules); assert not loaded, loaded"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr


def test_preflight_normalizes_absolute_local_inputs(tmp_path: Path) -> None:
    configuration = preflight_arguments(build_parser().parse_args(_argv(tmp_path)))

    assert configuration.data_config == (tmp_path / "data.yaml").resolve()
    assert configuration.base_weights == (tmp_path / "yolo11s.pt").resolve()
    assert configuration.run_dir == (tmp_path / "runs" / "ppe-run-001").resolve()
    assert configuration.epochs == 25
    assert configuration.image_size == 640
    assert configuration.batch_size == 8
    assert configuration.patience == 10
    assert configuration.seed == 42
    assert configuration.requested_device == "cuda:0"
    assert configuration.device == "cuda:0"


@pytest.mark.parametrize(
    "names",
    [
        "names: [Person, Hardhat, Safety Vest, NO-Hardhat, NO-Safety Vest]",
        "names: [Person, Hardhat, NO-Hardhat, Safety Vest]",
        "names: [Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Helmet]",
    ],
)
def test_preflight_requires_exact_canonical_class_order(tmp_path: Path, names: str) -> None:
    arguments = _argv(tmp_path)
    _write_inputs(tmp_path, names=names)

    with pytest.raises(TrainingConfigurationError, match="exactly equal"):
        preflight_arguments(build_parser().parse_args(arguments))


def test_preflight_rejects_downloadable_data_config(tmp_path: Path) -> None:
    arguments = _argv(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text(
        data.read_text(encoding="utf-8") + "download: https://example.invalid/data.zip\n",
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigurationError, match="downloads or URLs"):
        preflight_arguments(build_parser().parse_args(arguments))


def test_preflight_rejects_missing_local_dataset_split(tmp_path: Path) -> None:
    arguments = _argv(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text(
        data.read_text(encoding="utf-8").replace("test: images/test", "test: images/missing"),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigurationError, match="test path does not exist locally"):
        preflight_arguments(build_parser().parse_args(arguments))


def test_preflight_rejects_relative_input_and_output_paths(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigurationError, match="data config must be an absolute"):
        preflight_arguments(build_parser().parse_args(_argv(tmp_path, data=Path("data.yaml"))))

    with pytest.raises(TrainingConfigurationError, match="output root must be an absolute"):
        preflight_arguments(build_parser().parse_args(_argv(tmp_path, output_root=Path("runs"))))


def test_preflight_rejects_existing_final_run_directory(tmp_path: Path) -> None:
    arguments = _argv(tmp_path)
    (tmp_path / "runs" / "ppe-run-001").mkdir()

    with pytest.raises(TrainingConfigurationError, match="already exists"):
        preflight_arguments(build_parser().parse_args(arguments))


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--epochs", "0"),
        ("--imgsz", "4097"),
        ("--batch", "0"),
        ("--patience", "1001"),
        ("--seed", "-1"),
        ("--device", "mps"),
        ("--device", "cuda:999"),
    ],
)
def test_parser_rejects_unsafe_training_values(tmp_path: Path, option: str, value: str) -> None:
    arguments = _argv(tmp_path)
    index = arguments.index(option)
    arguments[index + 1] = value

    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(arguments)

    assert raised.value.code == 2


def test_preflight_requires_image_size_multiple_of_32(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigurationError, match="multiple of 32"):
        preflight_arguments(build_parser().parse_args(_argv(tmp_path, imgsz=650)))


def test_success_writes_atomic_complete_manifest_with_checkpoint_hash(tmp_path: Path) -> None:
    provider = SuccessfulProvider()
    arguments = _argv(tmp_path)

    result = run(arguments, services=_services(provider))

    assert result == 0
    assert len(provider.received) == 1
    manifest_path = tmp_path / "runs" / "ppe-run-001" / "training.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "runs" / "ppe-run-001" / "weights" / "best.pt"
    assert manifest["status"] == "COMPLETE"
    assert manifest["schemaVersion"] == "1.0.0"
    assert manifest["configuration"]["deterministic"] is True
    assert manifest["configuration"]["requestedDevice"] == "cuda:0"
    assert manifest["configuration"]["resolvedDevice"] == "cuda:0"
    assert (
        manifest["configuration"]["dataConfigSha256"]
        == hashlib.sha256((tmp_path / "data.yaml").read_bytes()).hexdigest()
    )
    assert (
        manifest["configuration"]["baseWeightsSha256"]
        == hashlib.sha256(b"local official base weights fixture").hexdigest()
    )
    assert manifest["source"] == {
        "available": True,
        "commitSha": "a" * 40,
        "dirty": False,
    }
    assert manifest["runtime"]["cudaDevice"] == "NVIDIA GeForce RTX 4060"
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(b"fine-tuned checkpoint").hexdigest()
    assert manifest["checkpoint"]["path"] == str(checkpoint.resolve())
    assert manifest["elapsedSeconds"] == 61.25
    assert not list(manifest_path.parent.glob(".training.manifest.json.*"))


def test_auto_device_is_resolved_before_provider_and_recorded(tmp_path: Path) -> None:
    provider = SuccessfulProvider()
    arguments = _argv(tmp_path, device="auto")

    assert run(arguments, services=_services(provider)) == 0

    assert provider.received[0].requested_device == "auto"
    assert provider.received[0].device == "cuda:0"
    manifest = json.loads(
        (tmp_path / "runs" / "ppe-run-001" / "training.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["configuration"]["requestedDevice"] == "auto"
    assert manifest["configuration"]["resolvedDevice"] == "cuda:0"


def test_numeric_cuda_device_is_resolved_before_provider(tmp_path: Path) -> None:
    provider = SuccessfulProvider()
    services = _services(provider)
    services = TrainingServices(
        provider_factory=services.provider_factory,
        runtime_facts=services.runtime_facts,
        git_facts=services.git_facts,
        resolve_device=lambda requested: f"cuda:{requested}",
        now=services.now,
        monotonic=services.monotonic,
    )

    assert run(_argv(tmp_path, device="0"), services=services) == 0
    assert provider.received[0].requested_device == "0"
    assert provider.received[0].device == "cuda:0"


def test_unavailable_explicit_cuda_is_a_controlled_failure(tmp_path: Path) -> None:
    provider = SuccessfulProvider()
    services = _services(provider)
    services = TrainingServices(
        provider_factory=services.provider_factory,
        runtime_facts=services.runtime_facts,
        git_facts=services.git_facts,
        resolve_device=lambda _requested: (_ for _ in ()).throw(ValueError("CUDA unavailable")),
        now=services.now,
        monotonic=services.monotonic,
    )

    assert run(_argv(tmp_path), services=services) == 1
    assert not provider.received
    assert not (tmp_path / "runs" / "ppe-run-001").exists()


class FailingProvider:
    def train(self, configuration: object) -> None:
        configuration.run_dir.mkdir(parents=True)
        raise RuntimeError("training provider failure")


class InterruptingProvider:
    def train(self, configuration: object) -> None:
        configuration.run_dir.mkdir(parents=True)
        raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("provider", "expected_code"),
    [(FailingProvider(), 1), (InterruptingProvider(), 130)],
)
def test_failure_or_interruption_never_writes_complete_manifest(
    tmp_path: Path, provider: object, expected_code: int
) -> None:
    result = run(_argv(tmp_path), services=_services(provider))

    assert result == expected_code
    assert not (tmp_path / "runs" / "ppe-run-001" / "training.manifest.json").exists()


class MissingCheckpointProvider:
    def train(self, configuration: object) -> None:
        configuration.run_dir.mkdir(parents=True)


def test_successful_provider_return_without_best_checkpoint_is_failure(tmp_path: Path) -> None:
    result = run(_argv(tmp_path), services=_services(MissingCheckpointProvider()))

    assert result == 1
    assert not (tmp_path / "runs" / "ppe-run-001" / "training.manifest.json").exists()


def test_manifest_preserves_runtime_mapping_as_json(tmp_path: Path) -> None:
    provider = SuccessfulProvider()
    services = _services(provider)

    assert run(_argv(tmp_path), services=services) == 0
    payload: Mapping[str, object] = json.loads(
        (tmp_path / "runs" / "ppe-run-001" / "training.manifest.json").read_text(encoding="utf-8")
    )

    assert isinstance(payload["runtime"], dict)
