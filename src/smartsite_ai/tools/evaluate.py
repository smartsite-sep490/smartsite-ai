"""Side-effect-free CLI contract and preflight for the PPE evaluation gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from smartsite_ai.evaluation.report import safe_failure_details

_SPLITS = ("train", "validation", "test")
_OUTPUT_FILENAMES = (
    "predictions.jsonl",
    "accuracy.report.json",
    "candidate.report.json",
    "summary.md",
)
_INCOMPLETE_REPORT_FILENAME = "evaluation.incomplete.json"


class EvaluationExecutionUnavailableError(RuntimeError):
    """Raised until the real scoring orchestrator is explicitly wired."""


class EvaluationOutputValidationError(RuntimeError):
    """Raised when an executor returns without a complete, isolated output set."""


@dataclass(frozen=True, slots=True)
class EvaluationRunConfiguration:
    """Normalized, preflighted paths and immutable CLI evaluation settings."""

    dataset_manifest: Path
    artifact_spec: Path
    episodes_index: Path
    region_configuration: Path
    ppe_region_id: UUID
    provider_data_config: Path
    split: Literal["train", "validation", "test"]
    match_iou: float
    report_dir: Path
    predictions_path: Path
    accuracy_report_path: Path
    candidate_report_path: Path
    summary_path: Path
    incomplete_report_path: Path
    annotated_dir: Path | None


EvaluationExecutor = Callable[[EvaluationRunConfiguration], None]


@dataclass(frozen=True, slots=True)
class EvaluationServices:
    """Injectable execution seam; importing this module never loads a vision provider."""

    execute: EvaluationExecutor


def _match_iou(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("match IoU must be a number") from error
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("match IoU must be finite and satisfy 0 < value <= 1")
    return number


def _canonical_uuid(value: str) -> UUID:
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("PPE region ID must be a canonical UUID") from error
    if str(parsed) != value.lower():
        raise argparse.ArgumentTypeError("PPE region ID must be a canonical UUID")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation parser without importing OpenCV, Torch, or Ultralytics."""

    parser = argparse.ArgumentParser(
        prog="smartsite-ai-evaluate",
        description="Run the reproducible SmartSite PPE evaluation gate from local inputs.",
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--artifact-spec", type=Path, required=True)
    parser.add_argument("--episodes-index", type=Path, required=True)
    parser.add_argument("--region-configuration", type=Path, required=True)
    parser.add_argument("--ppe-region-id", type=_canonical_uuid, required=True)
    parser.add_argument("--provider-data-config", type=Path, required=True)
    parser.add_argument("--split", choices=_SPLITS, required=True)
    parser.add_argument("--match-iou", type=_match_iou, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--annotated",
        action="store_true",
        help="export annotated evidence outside the timed evaluation path",
    )
    return parser


def _required_regular_file(path: Path, *, argument: str) -> Path:
    try:
        normalized = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{argument} must identify an existing regular file") from error
    if not normalized.is_file():
        raise ValueError(f"{argument} must identify an existing regular file")
    return normalized


def _normalized_new_report_dir(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("report directory must not be a symbolic link")
    try:
        normalized = expanded.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("report directory path is invalid") from error
    if normalized.exists() or normalized.is_symlink():
        raise ValueError("report directory already exists")
    if not normalized.parent.is_dir():
        raise ValueError("report directory parent must already exist")
    return normalized


def _identifies_same_path(first: Path, second: Path) -> bool:
    try:
        if first.exists() and second.exists() and os.path.samefile(first, second):
            return True
    except OSError:
        pass
    first_text = os.path.normcase(os.path.abspath(first))
    second_text = os.path.normcase(os.path.abspath(second))
    return first_text == second_text


def preflight_arguments(args: argparse.Namespace) -> EvaluationRunConfiguration:
    """Normalize paths and reject unsafe/colliding run inputs before creating outputs."""

    if args.split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")
    if (
        isinstance(args.match_iou, bool)
        or not isinstance(args.match_iou, int | float)
        or not math.isfinite(args.match_iou)
        or not 0.0 < args.match_iou <= 1.0
    ):
        raise ValueError("match IoU must be finite and satisfy 0 < value <= 1")
    if not isinstance(args.ppe_region_id, UUID):
        raise ValueError("PPE region ID must be a canonical UUID")

    inputs = {
        "dataset manifest": _required_regular_file(
            args.dataset_manifest, argument="dataset manifest"
        ),
        "artifact spec": _required_regular_file(args.artifact_spec, argument="artifact spec"),
        "episodes index": _required_regular_file(args.episodes_index, argument="episodes index"),
        "region configuration": _required_regular_file(
            args.region_configuration, argument="region configuration"
        ),
        "provider data config": _required_regular_file(
            args.provider_data_config, argument="provider data config"
        ),
    }
    report_dir = _normalized_new_report_dir(args.report_dir)
    output_paths = tuple(report_dir / filename for filename in _OUTPUT_FILENAMES) + (
        report_dir / _INCOMPLETE_REPORT_FILENAME,
    )
    if args.annotated:
        output_paths += (report_dir / "annotated",)

    for input_name, input_path in inputs.items():
        if _identifies_same_path(report_dir, input_path):
            raise ValueError(f"report directory aliases {input_name}")
        for output_path in output_paths:
            if _identifies_same_path(output_path, input_path):
                raise ValueError(f"evaluation output aliases {input_name}")

    return EvaluationRunConfiguration(
        dataset_manifest=inputs["dataset manifest"],
        artifact_spec=inputs["artifact spec"],
        episodes_index=inputs["episodes index"],
        region_configuration=inputs["region configuration"],
        ppe_region_id=args.ppe_region_id,
        provider_data_config=inputs["provider data config"],
        split=args.split,
        match_iou=args.match_iou,
        report_dir=report_dir,
        predictions_path=output_paths[0],
        accuracy_report_path=output_paths[1],
        candidate_report_path=output_paths[2],
        summary_path=output_paths[3],
        incomplete_report_path=output_paths[4],
        annotated_dir=report_dir / "annotated" if args.annotated else None,
    )


def _default_execute(configuration: EvaluationRunConfiguration) -> None:
    """Build explicit local providers and run the real evaluation orchestration."""

    from smartsite_ai.domain.regions import CameraRegionConfiguration
    from smartsite_ai.evaluation.dataset import load_evaluation_dataset
    from smartsite_ai.evaluation.execution import (
        EvaluationExecutionMetadata,
        EvaluationExecutionRequest,
        EvaluationExecutionServices,
        execute_evaluation,
    )
    from smartsite_ai.evaluation.provider_validation import (
        ProviderValidationAdapter,
        ProviderValidationArguments,
    )
    from smartsite_ai.evaluation.report import (
        CommandArgument,
        RuntimeReportMetadata,
    )
    from smartsite_ai.evaluation.runtime_adapters import (
        OpenCvEvaluationMedia,
        OpenCvOverlayRenderer,
        UltralyticsValidationProvider,
        ultralytics_runtime_sandbox,
    )
    from smartsite_ai.inference.loading import load_yolo11_detector
    from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
    from smartsite_ai.pipelines import (
        Mf05Mf06Pipeline,
        PpePipeline,
        RestrictedZonePipeline,
        TemporalPpeCandidateGate,
    )
    from smartsite_ai.tracking import IoUPersonTracker

    runtime_stack = ExitStack()
    runtime_stack.enter_context(ultralytics_runtime_sandbox())
    try:
        detector, runner, artifact, _class_map = load_yolo11_detector(
            configuration.artifact_spec,
            runner_factory=UltralyticsYoloRunner,
        )
    except BaseException:
        runtime_stack.close()
        raise

    def execute_loaded() -> None:
        if artifact.model_family != "yolo11s":
            raise ValueError("the SmartSite evaluation gate requires modelFamily yolo11s")

        region_configuration = CameraRegionConfiguration.from_wire_bytes(
            configuration.region_configuration.read_bytes()
        )
        region_ids = {region.region_id.casefold() for region in region_configuration.regions}
        if str(configuration.ppe_region_id).casefold() not in region_ids:
            raise ValueError("the selected PPE region is absent from the camera configuration")

        media = OpenCvEvaluationMedia()
        overlay_renderer = OpenCvOverlayRenderer() if configuration.annotated_dir else None

        class _FrameReader:
            def __init__(self) -> None:
                self._sequences: dict[str, int] = {}

            def read(self, dataset_root: Path, frame: Any) -> Any:
                sequence = (
                    frame.frame_index
                    if frame.frame_index is not None
                    else self._sequences.get(frame.media_path, 0)
                )
                self._sequences[frame.media_path] = sequence + 1
                envelope, _drawable = media.load(
                    frame,
                    dataset_root=dataset_root,
                    camera_external_id=region_configuration.camera_external_id,
                    session_id=uuid5(NAMESPACE_URL, f"smartsite-evaluation:{frame.media_path}"),
                    sequence_number=sequence,
                )
                return envelope

        def write_evidence(frame: Any, source: Any, plan: Any) -> None:
            assert configuration.annotated_dir is not None
            assert overlay_renderer is not None
            stable_name = uuid5(NAMESPACE_URL, f"smartsite-evidence:{frame.frame_id}").hex
            rendered = overlay_renderer.render(source, plan)
            overlay_renderer.write(configuration.annotated_dir / f"{stable_name}.png", rendered)

        provider_arguments = ProviderValidationArguments(
            model_path=artifact.resolved_path,
            data_config_path=configuration.provider_data_config,
            split="val" if configuration.split == "validation" else configuration.split,
            image_size=artifact.image_size,
            confidence_threshold=0.001,
            iou_threshold=0.70,
            max_detections=300,
            batch_size=1,
            workers=0,
            device=artifact.device,
            seed=490,
        )
        provider_adapter = ProviderValidationAdapter(UltralyticsValidationProvider())
        metadata = EvaluationExecutionMetadata(
            git=_git_metadata(),
            runtime=RuntimeReportMetadata(
                python_version=platform.python_version(),
                platform=platform.platform(),
                device=artifact.device,
                package_versions=_runtime_package_versions(),
                hardware={"machine": platform.machine() or "unknown"},
            ),
            command_arguments=(
                CommandArgument(name="split", value=configuration.split),
                CommandArgument(name="matchIou", value=str(configuration.match_iou)),
                CommandArgument(name="ppeRegionId", value=str(configuration.ppe_region_id)),
                CommandArgument(
                    name="annotated", value=str(configuration.annotated_dir is not None)
                ),
            ),
        )
        request = EvaluationExecutionRequest(
            dataset_manifest_path=configuration.dataset_manifest,
            split=configuration.split,
            episodes_index_path=configuration.episodes_index,
            region_configuration_path=configuration.region_configuration,
            artifact=artifact,
            match_iou=configuration.match_iou,
            onset_tolerance=timedelta(milliseconds=500),
            predictions_path=configuration.predictions_path,
            accuracy_report_path=configuration.accuracy_report_path,
            candidate_report_path=configuration.candidate_report_path,
            summary_path=configuration.summary_path,
        )
        services = EvaluationExecutionServices(
            dataset_loader=load_evaluation_dataset,
            region_loader=lambda _path: region_configuration,
            frame_reader=_FrameReader(),
            detector=detector,
            pipeline=Mf05Mf06Pipeline(
                tracker=IoUPersonTracker(),
                ppe=PpePipeline(),
                zones=RestrictedZonePipeline(),
                ppe_region_id=str(configuration.ppe_region_id),
            ),
            temporal_gate=TemporalPpeCandidateGate(),
            provider_validation=lambda: provider_adapter.validate(provider_arguments),
            evidence_writer=write_evidence if configuration.annotated_dir else None,
        )
        try:
            if configuration.annotated_dir is not None:
                configuration.annotated_dir.mkdir()
            asyncio.run(execute_evaluation(request, metadata=metadata, services=services))
        finally:
            media.close()

    try:
        execute_loaded()
    finally:
        try:
            runner.close()
        finally:
            runtime_stack.close()


def _git_metadata() -> Any:
    from smartsite_ai.evaluation.report import GitReportMetadata

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvaluationExecutionUnavailableError(
            "Git metadata is unavailable for a reproducible evaluation"
        ) from error
    return GitReportMetadata(commit_sha=commit, dirty_worktree=dirty)


def _runtime_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("smartsite-ai", "ultralytics", "torch", "opencv-python"):
        try:
            versions[package] = distribution_version(package)
        except PackageNotFoundError:
            continue
    return versions


def _cleanup_partial_outputs(configuration: EvaluationRunConfiguration) -> None:
    for path in (
        configuration.predictions_path,
        configuration.accuracy_report_path,
        configuration.candidate_report_path,
        configuration.summary_path,
    ):
        with suppress(OSError):
            path.unlink(missing_ok=True)
    if configuration.annotated_dir is not None and configuration.annotated_dir.exists():
        with suppress(OSError):
            shutil.rmtree(configuration.annotated_dir)


def _validate_completed_outputs(configuration: EvaluationRunConfiguration) -> None:
    input_paths = (
        configuration.dataset_manifest,
        configuration.artifact_spec,
        configuration.episodes_index,
        configuration.region_configuration,
        configuration.provider_data_config,
    )
    output_paths = (
        configuration.predictions_path,
        configuration.accuracy_report_path,
        configuration.candidate_report_path,
        configuration.summary_path,
    )
    for output_path in output_paths:
        if output_path.is_symlink() or not output_path.is_file():
            raise EvaluationOutputValidationError(
                f"executor did not produce required output: {output_path.name}"
            )
        if any(_identifies_same_path(output_path, input_path) for input_path in input_paths):
            raise EvaluationOutputValidationError(
                f"executor output aliases an input: {output_path.name}"
            )
    if configuration.annotated_dir is not None and (
        configuration.annotated_dir.is_symlink() or not configuration.annotated_dir.is_dir()
    ):
        raise EvaluationOutputValidationError("executor did not produce annotated evidence")

    _validate_predictions_output(configuration.predictions_path)
    _validate_complete_report_output(configuration.accuracy_report_path, label="accuracy")
    _validate_complete_report_output(configuration.candidate_report_path, label="candidate")
    try:
        summary = configuration.summary_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EvaluationOutputValidationError("summary output is not readable UTF-8") from error
    if not summary.strip():
        raise EvaluationOutputValidationError("summary output must not be blank")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationOutputValidationError(f"{label} output is not valid JSON") from error
    if not isinstance(value, dict):
        raise EvaluationOutputValidationError(f"{label} output must be a JSON object")
    return value


def _validate_complete_report_output(path: Path, *, label: str) -> None:
    report = _read_json_object(path, label=f"{label} report")
    if report.get("status") != "COMPLETE":
        raise EvaluationOutputValidationError(f"{label} report must have COMPLETE status")


def _validate_predictions_output(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvaluationOutputValidationError("predictions output is not readable UTF-8") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationOutputValidationError(
                f"predictions output contains a blank row at line {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationOutputValidationError(
                f"predictions output contains invalid JSON at line {line_number}"
            ) from error
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("frameId"), str)
            or not isinstance(row.get("detections"), list)
        ):
            raise EvaluationOutputValidationError(
                f"predictions output row {line_number} violates the frame contract"
            )


def _write_incomplete_status(
    configuration: EvaluationRunConfiguration,
    error: BaseException,
) -> None:
    details = safe_failure_details(error)
    payload: dict[str, Any] = {
        "failure": details.model_dump(mode="json", by_alias=True),
        "schemaVersion": "1.0.0",
        "status": "INCOMPLETE",
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{configuration.incomplete_report_path.name}.",
            suffix=".tmp",
            dir=configuration.report_dir,
        )
        temporary_path = Path(raw_path)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, configuration.incomplete_report_path)
        temporary_path.unlink(missing_ok=True)
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _record_failure(
    configuration: EvaluationRunConfiguration,
    error: BaseException,
) -> None:
    _cleanup_partial_outputs(configuration)
    try:
        _write_incomplete_status(configuration, error)
    except OSError:
        print("error: could not persist incomplete evaluation status", file=sys.stderr)


def run(
    argv: Sequence[str] | None = None,
    *,
    services: EvaluationServices | None = None,
) -> int:
    """Preflight and execute an evaluation through an explicit injected service."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configuration = preflight_arguments(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        configuration.report_dir.mkdir(mode=0o700)
    except OSError:
        print("error: could not create a new report directory", file=sys.stderr)
        return 2

    active_services = services or EvaluationServices(execute=_default_execute)
    try:
        active_services.execute(configuration)
        _validate_completed_outputs(configuration)
        return 0
    except KeyboardInterrupt:
        interrupted = KeyboardInterrupt("evaluation interrupted")
        _record_failure(configuration, interrupted)
        print("error: evaluation interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        details = safe_failure_details(error)
        _record_failure(configuration, error)
        print(f"error: {details.message}", file=sys.stderr)
        return 1


def main() -> None:
    """Console-script entry point for the evaluation gate."""

    raise SystemExit(run())


__all__ = [
    "EvaluationExecutionUnavailableError",
    "EvaluationOutputValidationError",
    "EvaluationRunConfiguration",
    "EvaluationServices",
    "build_parser",
    "main",
    "preflight_arguments",
    "run",
]
