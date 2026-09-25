from pathlib import Path

import pytest

from smartsite_ai.tools.train_ppe import TrainingConfiguration, TrainingExecutionError
from smartsite_ai.training.ultralytics_provider import UltralyticsTrainingProvider


class FakeNetwork:
    def __init__(self, *, scale: str = "s") -> None:
        self.yaml = {"yaml_file": "yolo11.yaml", "scale": scale}


class FakeModel:
    task = "detect"
    names = {0: "person"}

    def __init__(self, *, scale: str = "s", failure: Exception | None = None) -> None:
        self.model = FakeNetwork(scale=scale)
        self.failure = failure
        self.arguments: dict[str, object] | None = None

    def train(self, **arguments: object) -> object:
        self.arguments = arguments
        if self.failure is not None:
            raise self.failure
        return object()


class CacheCreatingModel(FakeModel):
    def train(self, **arguments: object) -> object:
        data = Path(str(arguments["data"]))
        (data.parent / "labels.cache").write_bytes(b"generated cache")
        return super().train(**arguments)


def _configuration(tmp_path: Path) -> TrainingConfiguration:
    return TrainingConfiguration(
        data_config=(tmp_path / "data.yaml").resolve(),
        dataset_aggregate_sha256="a" * 64,
        base_weights=(tmp_path / "yolo11s.pt").resolve(),
        output_root=(tmp_path / "runs").resolve(),
        run_name="run-001",
        run_dir=(tmp_path / "runs" / "run-001").resolve(),
        epochs=80,
        image_size=640,
        batch_size=16,
        patience=20,
        seed=7,
        requested_device="cuda:0",
        device="cuda:0",
    )


def test_provider_runs_deterministic_official_yolo11s_training(tmp_path: Path) -> None:
    model = FakeModel()
    provider = UltralyticsTrainingProvider(
        model_factory=lambda _path: model,
        version_factory=lambda: "8.4.155",
    )

    provider.train(_configuration(tmp_path))

    assert model.arguments == {
        "data": str((tmp_path / "data.yaml").resolve()),
        "project": str((tmp_path / "runs").resolve()),
        "name": "run-001",
        "exist_ok": False,
        "epochs": 80,
        "imgsz": 640,
        "batch": 16,
        "patience": 20,
        "seed": 7,
        "device": "cuda:0",
        "deterministic": True,
        "resume": False,
        "plots": False,
        "cache": False,
    }


def test_provider_rejects_non_s_variant_before_training(tmp_path: Path) -> None:
    model = FakeModel(scale="n")
    provider = UltralyticsTrainingProvider(
        model_factory=lambda _path: model,
        version_factory=lambda: "8.4.155",
    )

    with pytest.raises(TrainingExecutionError, match="YOLO11s"):
        provider.train(_configuration(tmp_path))

    assert model.arguments is None


@pytest.mark.parametrize("failure", [None, RuntimeError("provider failed")])
def test_provider_removes_generated_label_cache_on_success_and_failure(
    tmp_path: Path, failure: Exception | None
) -> None:
    model = CacheCreatingModel(failure=failure)
    provider = UltralyticsTrainingProvider(
        model_factory=lambda _path: model,
        version_factory=lambda: "8.4.155",
    )

    if failure is None:
        provider.train(_configuration(tmp_path))
    else:
        with pytest.raises(TrainingExecutionError, match="Ultralytics training failed"):
            provider.train(_configuration(tmp_path))

    assert not (tmp_path / "labels.cache").exists()


def test_provider_rejects_unpinned_ultralytics_before_loading(tmp_path: Path) -> None:
    loaded = False

    def model_factory(_path: Path) -> FakeModel:
        nonlocal loaded
        loaded = True
        return FakeModel()

    provider = UltralyticsTrainingProvider(
        model_factory=model_factory,
        version_factory=lambda: "8.4.154",
    )

    with pytest.raises(TrainingExecutionError, match="must equal pinned version"):
        provider.train(_configuration(tmp_path))

    assert not loaded


def test_provider_translates_provider_failure_without_leaking_message(tmp_path: Path) -> None:
    model = FakeModel(failure=RuntimeError("token=secret"))
    provider = UltralyticsTrainingProvider(
        model_factory=lambda _path: model,
        version_factory=lambda: "8.4.155",
    )

    with pytest.raises(TrainingExecutionError, match="Ultralytics training failed") as raised:
        provider.train(_configuration(tmp_path))

    assert "secret" not in str(raised.value)
