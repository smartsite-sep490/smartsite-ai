import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.provider_validation import (
    PINNED_ULTRALYTICS_VERSION,
    ProviderValidationAdapter,
    ProviderValidationArguments,
    ProviderValidationError,
    ProviderValidationMetrics,
)
from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP


def _arguments(tmp_path: Path, **overrides: object) -> ProviderValidationArguments:
    values: dict[str, object] = {
        "model_path": (tmp_path / "model.pt").resolve(),
        "data_config_path": (tmp_path / "data.yaml").resolve(),
        "split": "test",
        "image_size": (640, 640),
        "confidence_threshold": 0.001,
        "iou_threshold": 0.70,
        "max_detections": 300,
        "batch_size": 4,
        "workers": 2,
        "device": "cuda:0",
        "seed": 0,
        **overrides,
    }
    return ProviderValidationArguments.model_validate(values)


def _result(**overrides: object) -> dict[str, object]:
    return {
        "provider_name": "ultralytics",
        "provider_version": PINNED_ULTRALYTICS_VERSION,
        "class_map": CANONICAL_PPE_CLASS_MAP,
        "ap50": 0.81,
        "ap50_95": 0.56,
        **overrides,
    }


class FakeProvider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[ProviderValidationArguments] = []

    def validate(self, arguments: ProviderValidationArguments) -> object:
        self.calls.append(arguments)
        return self.result


def test_import_does_not_load_cv_torch_or_ultralytics() -> None:
    script = (
        "import sys; import smartsite_ai.evaluation.provider_validation; "
        "assert 'cv2' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'ultralytics' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_adapter_returns_metrics_with_the_exact_arguments_passed_to_provider(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    provider = FakeProvider(_result())

    report = ProviderValidationAdapter(provider).validate(arguments)

    assert provider.calls == [arguments]
    assert provider.calls[0] is arguments
    assert report.arguments is arguments
    assert report.metrics.ap50 == pytest.approx(0.81)
    assert report.metrics.ap50_95 == pytest.approx(0.56)
    assert report.metrics.class_map == CANONICAL_PPE_CLASS_MAP
    assert report.model_dump(mode="json") == {
        "arguments": {
            "model_path": str(arguments.model_path),
            "data_config_path": str(arguments.data_config_path),
            "task": "detect",
            "mode": "val",
            "split": "test",
            "image_size": [640, 640],
            "confidence_threshold": 0.001,
            "iou_threshold": 0.7,
            "max_detections": 300,
            "batch_size": 4,
            "workers": 2,
            "device": "cuda:0",
            "seed": 0,
            "deterministic": True,
            "plots": False,
            "save_json": False,
        },
        "metrics": {
            "provider_name": "ultralytics",
            "provider_version": PINNED_ULTRALYTICS_VERSION,
            "class_map": [
                [class_id, class_name] for class_id, class_name in CANONICAL_PPE_CLASS_MAP
            ],
            "ap50": 0.81,
            "ap50_95": 0.56,
        },
    }


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model_path": Path("relative.pt")}, "absolute"),
        ({"data_config_path": Path("relative.yaml")}, "absolute"),
        ({"image_size": (0, 640)}, "greater than or equal"),
        ({"image_size": (640,)}, "at least 2 items"),
        ({"confidence_threshold": float("nan")}, "finite number"),
        ({"iou_threshold": 1.1}, "less than or equal"),
        ({"max_detections": 0}, "greater than or equal"),
        ({"batch_size": 0}, "greater than or equal"),
        ({"workers": 257}, "less than or equal"),
        ({"device": " "}, "non-blank"),
        ({"device": "x" * 129}, "at most 128"),
        ({"seed": -1}, "greater than or equal"),
        ({"plots": True}, "Input should be False"),
        ({"save_json": True}, "Input should be False"),
    ],
)
def test_arguments_are_strict_and_bounded(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _arguments(tmp_path, **overrides)


def test_arguments_reject_extra_fields_and_are_frozen(tmp_path: Path) -> None:
    payload = _arguments(tmp_path).model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderValidationArguments.model_validate(payload)

    arguments = _arguments(tmp_path)
    with pytest.raises(ValidationError, match="frozen"):
        arguments.device = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"provider_name": "other"}, "provider_name"),
        ({"provider_version": "8.4.154"}, "provider_version"),
        ({"class_map": CANONICAL_PPE_CLASS_MAP[:-1]}, "class_map"),
        ({"class_map": tuple(reversed(CANONICAL_PPE_CLASS_MAP))}, "class_map"),
        ({"ap50": float("nan")}, "finite number"),
        ({"ap50": -0.01}, "greater than or equal"),
        ({"ap50": 1.01}, "less than or equal"),
        ({"ap50_95": float("inf")}, "finite number"),
        ({"ap50": 0.50, "ap50_95": 0.51}, "must not exceed"),
        ({"extra": True}, "Extra inputs are not permitted"),
    ],
)
def test_adapter_rejects_invalid_or_contradictory_provider_results(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    provider = FakeProvider(_result(**overrides))

    with pytest.raises(ProviderValidationError, match=match):
        ProviderValidationAdapter(provider).validate(_arguments(tmp_path))


def test_metrics_are_strict_and_frozen() -> None:
    metrics = ProviderValidationMetrics.model_validate(_result())

    with pytest.raises(ValidationError, match="frozen"):
        metrics.ap50 = 0.1  # type: ignore[misc]


def test_provider_failure_is_safely_translated_without_swallowing_cause(tmp_path: Path) -> None:
    class FailingProvider:
        def validate(self, arguments: ProviderValidationArguments) -> object:
            del arguments
            raise RuntimeError("provider internals")

    with pytest.raises(ProviderValidationError, match="provider validation failed") as exc_info:
        ProviderValidationAdapter(FailingProvider()).validate(_arguments(tmp_path))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "provider internals" not in str(exc_info.value)


def test_control_flow_exceptions_are_not_translated(tmp_path: Path) -> None:
    class InterruptedProvider:
        def validate(self, arguments: ProviderValidationArguments) -> object:
            del arguments
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ProviderValidationAdapter(InterruptedProvider()).validate(_arguments(tmp_path))
