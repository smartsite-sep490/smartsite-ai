import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest

from smartsite_ai.tools.evaluate import (
    EvaluationServices,
    build_parser,
    preflight_arguments,
    run,
)

PPE_REGION_ID = "10000000-0000-0000-0000-000000000001"


def _input_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "dataset_manifest": tmp_path / "dataset.manifest.json",
        "artifact_spec": tmp_path / "artifact.json",
        "episodes_index": tmp_path / "episodes.jsonl",
        "region_configuration": tmp_path / "regions.json",
        "provider_data_config": tmp_path / "provider-data.yaml",
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    return paths


def _argv(tmp_path: Path, **overrides: object) -> list[str]:
    paths = _input_files(tmp_path)
    values: dict[str, object] = {
        **paths,
        "split": "test",
        "match_iou": "0.50",
        "report_dir": tmp_path / "run-001",
        "ppe_region_id": PPE_REGION_ID,
    }
    values.update(overrides)
    return [
        "--dataset-manifest",
        str(values["dataset_manifest"]),
        "--artifact-spec",
        str(values["artifact_spec"]),
        "--episodes-index",
        str(values["episodes_index"]),
        "--region-configuration",
        str(values["region_configuration"]),
        "--ppe-region-id",
        str(values["ppe_region_id"]),
        "--provider-data-config",
        str(values["provider_data_config"]),
        "--split",
        str(values["split"]),
        "--match-iou",
        str(values["match_iou"]),
        "--report-dir",
        str(values["report_dir"]),
    ]


def test_parser_requires_every_locked_contract_argument() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args([])

    assert raised.value.code == 2


@pytest.mark.parametrize("value", ["train", "validation", "test"])
def test_parser_accepts_each_manifest_split(tmp_path: Path, value: str) -> None:
    args = build_parser().parse_args(_argv(tmp_path, split=value))

    assert args.split == value


@pytest.mark.parametrize("value", ["unknown", "TEST", ""])
def test_parser_rejects_invalid_split(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(_argv(tmp_path, split=value))

    assert raised.value.code == 2


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf", "-inf"])
def test_parser_rejects_invalid_or_non_finite_match_iou(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(_argv(tmp_path, match_iou=value))

    assert raised.value.code == 2


def test_preflight_normalizes_paths_and_declares_outputs(tmp_path: Path) -> None:
    args = build_parser().parse_args(_argv(tmp_path) + ["--annotated"])

    configuration = preflight_arguments(args)

    assert configuration.dataset_manifest.is_absolute()
    assert configuration.dataset_manifest == (tmp_path / "dataset.manifest.json").resolve()
    assert configuration.artifact_spec == (tmp_path / "artifact.json").resolve()
    assert configuration.report_dir == (tmp_path / "run-001").resolve()
    assert configuration.ppe_region_id == UUID(PPE_REGION_ID)
    assert configuration.annotated_dir == configuration.report_dir / "annotated"
    assert configuration.predictions_path == configuration.report_dir / "predictions.jsonl"
    assert configuration.accuracy_report_path == configuration.report_dir / "accuracy.report.json"
    assert configuration.candidate_report_path == configuration.report_dir / "candidate.report.json"
    assert configuration.summary_path == configuration.report_dir / "summary.md"


def test_run_passes_normalized_configuration_to_injected_service(tmp_path: Path) -> None:
    received = []

    def execute(configuration: object) -> None:
        received.append(configuration)
        configuration.predictions_path.write_text(
            '{"detections":[],"frameId":"frame-1"}\n', encoding="utf-8"
        )
        configuration.accuracy_report_path.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        configuration.candidate_report_path.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        configuration.summary_path.write_text("complete\n", encoding="utf-8")

    result = run(_argv(tmp_path), services=EvaluationServices(execute=execute))

    assert result == 0
    assert len(received) == 1
    assert received[0].report_dir == (tmp_path / "run-001").resolve()
    assert received[0].report_dir.is_dir()


def test_preflight_rejects_missing_or_non_regular_input(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    argv = _argv(tmp_path, dataset_manifest=missing)

    assert run(argv, services=EvaluationServices(execute=lambda _configuration: None)) == 2


def test_preflight_rejects_existing_report_directory(tmp_path: Path) -> None:
    report_dir = tmp_path / "already-exists"
    report_dir.mkdir()
    invoked = False

    def execute(_configuration: object) -> None:
        nonlocal invoked
        invoked = True

    result = run(
        _argv(tmp_path, report_dir=report_dir),
        services=EvaluationServices(execute=execute),
    )

    assert result == 2
    assert not invoked


def test_preflight_rejects_report_path_that_aliases_an_input(tmp_path: Path) -> None:
    paths = _input_files(tmp_path)
    args = build_parser().parse_args(
        _argv(
            tmp_path,
            dataset_manifest=paths["dataset_manifest"],
            report_dir=paths["dataset_manifest"],
        )
    )

    with pytest.raises(ValueError, match="report directory"):
        preflight_arguments(args)


def test_run_detects_hard_link_output_alias(tmp_path: Path) -> None:
    def alias_input(configuration: object) -> None:
        os.link(configuration.dataset_manifest, configuration.predictions_path)
        configuration.accuracy_report_path.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        configuration.candidate_report_path.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        configuration.summary_path.write_text("complete\n", encoding="utf-8")

    result = run(_argv(tmp_path), services=EvaluationServices(execute=alias_input))

    assert result == 1
    assert not (tmp_path / "run-001" / "predictions.jsonl").exists()
    assert (tmp_path / "run-001" / "evaluation.incomplete.json").is_file()


def test_import_does_not_load_vision_or_network_stacks() -> None:
    script = (
        "import sys; import smartsite_ai.tools.evaluate; "
        "forbidden = {'cv2', 'torch', 'ultralytics', 'httpx', 'requests'}; "
        "loaded = forbidden.intersection(sys.modules); assert not loaded, loaded"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_interruption_removes_known_partial_outputs_and_writes_incomplete_status(
    tmp_path: Path,
) -> None:
    def interrupt(configuration: object) -> None:
        configuration.predictions_path.write_text("partial", encoding="utf-8")
        assert configuration.annotated_dir is not None
        configuration.annotated_dir.mkdir()
        (configuration.annotated_dir / "partial.jpg").write_bytes(b"partial")
        raise KeyboardInterrupt

    result = run(
        _argv(tmp_path) + ["--annotated"],
        services=EvaluationServices(execute=interrupt),
    )
    report_dir = tmp_path / "run-001"
    status = json.loads((report_dir / "evaluation.incomplete.json").read_text(encoding="utf-8"))

    assert result == 130
    assert status == {
        "failure": {
            "errorType": "KeyboardInterrupt",
            "message": "evaluation interrupted",
        },
        "schemaVersion": "1.0.0",
        "status": "INCOMPLETE",
    }
    assert not (report_dir / "predictions.jsonl").exists()
    assert not (report_dir / "annotated").exists()


def test_controlled_failure_writes_sanitized_incomplete_status(tmp_path: Path) -> None:
    def fail(_configuration: object) -> None:
        raise RuntimeError("provider token=super-secret")

    result = run(_argv(tmp_path), services=EvaluationServices(execute=fail))
    status_path = tmp_path / "run-001" / "evaluation.incomplete.json"
    serialized = status_path.read_text(encoding="utf-8")
    status = json.loads(serialized)

    assert result == 1
    assert status["status"] == "INCOMPLETE"
    assert status["failure"] == {
        "errorType": "RuntimeError",
        "message": "provider token=***",
    }
    assert "super-secret" not in serialized


def test_default_service_runs_real_preflight_without_optional_vision_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smartsite_ai.evaluation.runtime_adapters as runtime_adapters

    monkeypatch.setattr(runtime_adapters, "find_spec", lambda _package: None)
    result = run(_argv(tmp_path))
    status_path = tmp_path / "run-001" / "evaluation.incomplete.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert result == 1
    assert status["status"] == "INCOMPLETE"
    assert status["failure"]["errorType"] == "ArtifactValidationError"


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("predictions.jsonl", "not-json\n"),
        ("accuracy.report.json", '{"status":"INCOMPLETE"}\n'),
        ("candidate.report.json", "{}\n"),
        ("summary.md", "   \n"),
    ],
)
def test_run_rejects_malformed_or_incomplete_executor_outputs(
    tmp_path: Path, filename: str, contents: str
) -> None:
    def execute(configuration: object) -> None:
        configuration.predictions_path.write_text(
            '{"detections":[],"frameId":"frame-1"}\n', encoding="utf-8"
        )
        configuration.accuracy_report_path.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        configuration.candidate_report_path.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
        configuration.summary_path.write_text("complete\n", encoding="utf-8")
        (configuration.report_dir / filename).write_text(contents, encoding="utf-8")

    result = run(_argv(tmp_path), services=EvaluationServices(execute=execute))

    assert result == 1
    assert (tmp_path / "run-001" / "evaluation.incomplete.json").is_file()
