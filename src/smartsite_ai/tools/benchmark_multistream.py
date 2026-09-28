"""Benchmark one shared YOLO11s runner across 1/2/3 local video replays."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from smartsite_ai.benchmark.multistream import (
    BenchmarkError,
    InferenceSynchronizerProtocol,
    ReplaySourceProtocol,
    ResourceMonitorProtocol,
    run_multistream_scenario,
)
from smartsite_ai.benchmark.runtime import (
    OpenCvReplaySource,
    ProcessResourceMonitor,
    TorchSynchronizer,
    git_facts,
    hardware_facts,
    resolve_torch_device,
    runtime_facts,
)
from smartsite_ai.evaluation.runtime_adapters import ultralytics_runtime_sandbox
from smartsite_ai.inference.loading import RunnerFactory, load_yolo11_detector
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.training.dataset_integrity import is_link_like

_MAX_INPUT_BYTES = 20 * 1024 * 1024 * 1024
_MAX_DURATION_SECONDS = 3_600.0


@dataclass(frozen=True, slots=True)
class BenchmarkServices:
    runtime_context: Callable[[], AbstractContextManager[object]]
    runner_factory: RunnerFactory
    torch_factory: Callable[[], ModuleType]
    source_factory: Callable[[Path], ReplaySourceProtocol]
    synchronizer_factory: Callable[[ModuleType, int | None], InferenceSynchronizerProtocol]
    resource_monitor_factory: Callable[[ModuleType, int | None], ResourceMonitorProtocol]
    runtime_facts: Callable[[ModuleType], Mapping[str, object]]
    hardware_facts: Callable[[ModuleType], Mapping[str, object]]
    git_facts: Callable[[Path], Mapping[str, object]]


def _default_services() -> BenchmarkServices:
    return BenchmarkServices(
        runtime_context=ultralytics_runtime_sandbox,
        runner_factory=UltralyticsYoloRunner,
        torch_factory=_import_torch,
        source_factory=OpenCvReplaySource,
        synchronizer_factory=TorchSynchronizer,
        resource_monitor_factory=ProcessResourceMonitor,
        runtime_facts=runtime_facts,
        hardware_facts=hardware_facts,
        git_facts=git_facts,
    )


def _import_torch() -> ModuleType:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - selected runtime extra
        raise BenchmarkError("Torch runtime is unavailable") from error
    return torch


def _finite_duration(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("duration must be numeric") from error
    if not math.isfinite(number) or number < 0 or number > _MAX_DURATION_SECONDS:
        raise argparse.ArgumentTypeError(
            f"duration must be finite and between 0 and {_MAX_DURATION_SECONDS:g} seconds"
        )
    return number


def _positive_duration(value: str) -> float:
    number = _finite_duration(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return number


def _positive_fps(value: str) -> float:
    number = _positive_duration(value)
    if number > 240:
        raise argparse.ArgumentTypeError("target FPS must not exceed 240")
    return number


def _stream_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stream count must be an integer") from error
    if count not in {1, 2, 3}:
        raise argparse.ArgumentTypeError("stream count must be 1, 2, or 3")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartsite-ai-benchmark-multistream",
        description=(
            "Measure a verified local YOLO11 detector with one shared runner and concurrent "
            "local video replay. The report records observations only; it applies no pass "
            "threshold."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help=(
            "absolute local video path; repeat for distinct streams or provide one synthetic replay"
        ),
    )
    parser.add_argument("--artifact-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new atomic JSON report path")
    parser.add_argument(
        "--stream-counts",
        type=_stream_count,
        nargs="+",
        default=(1, 2, 3),
        help="unique scenario counts to run (default: 1 2 3)",
    )
    parser.add_argument("--warmup-seconds", type=_positive_duration, default=5.0)
    parser.add_argument("--measurement-seconds", type=_positive_duration, default=30.0)
    parser.add_argument("--target-fps", type=_positive_fps, default=None)
    parser.add_argument("--resource-sample-interval-seconds", type=_positive_duration, default=0.1)
    return parser


def _regular_input(path: Path, *, label: str, suffixes: frozenset[str]) -> Path:
    if not path.is_absolute():
        raise BenchmarkError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BenchmarkError(f"{label} does not exist") from error
    if is_link_like(path) or not resolved.is_file() or resolved.suffix.casefold() not in suffixes:
        raise BenchmarkError(f"{label} must be a supported regular local file")
    if resolved.stat().st_size > _MAX_INPUT_BYTES:
        raise BenchmarkError(f"{label} exceeds the size limit")
    return resolved


def _new_output(path: Path, protected: Sequence[Path]) -> Path:
    if not path.is_absolute():
        raise BenchmarkError("output path must be absolute")
    resolved = path.resolve(strict=False)
    if resolved.suffix.casefold() != ".json":
        raise BenchmarkError("output path must use the .json suffix")
    if resolved.exists() or resolved.is_symlink():
        raise BenchmarkError("output path must not already exist or be link-like")
    if not resolved.parent.is_dir():
        raise BenchmarkError("output parent directory does not exist")
    if any(os.path.normcase(str(resolved)) == os.path.normcase(str(item)) for item in protected):
        raise BenchmarkError("output path must not replace an input or configuration file")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _scenario_inputs(inputs: Sequence[dict[str, object]], count: int) -> list[dict[str, object]]:
    if len(inputs) == 1:
        return [inputs[0] for _ in range(count)]
    if len(inputs) < count:
        raise BenchmarkError(
            "provide one input for synthetic replay or at least the maximum requested stream count"
        )
    return list(inputs[:count])


def _verified_git_facts(facts: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(facts)
    sha = normalized.get("commitSha")
    if (
        normalized.get("available") is not True
        or normalized.get("dirty") is not False
        or not isinstance(sha, str)
        or len(sha) != 40
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise BenchmarkError("benchmark requires an available clean Git commit")
    return normalized


def _verified_hardware_facts(
    facts: Mapping[str, object], *, selected_cuda_device: int | None
) -> dict[str, object]:
    normalized = dict(facts)
    if selected_cuda_device is None:
        return normalized
    devices = normalized.get("cudaDevices")
    if not isinstance(devices, list):
        raise BenchmarkError("CUDA hardware provenance is unavailable")
    selected = next(
        (
            device
            for device in devices
            if isinstance(device, Mapping) and device.get("index") == selected_cuda_device
        ),
        None,
    )
    if not isinstance(selected, Mapping) or not isinstance(selected.get("driverVersion"), str):
        raise BenchmarkError("NVIDIA driver version is unavailable for the selected CUDA device")
    return normalized


def _execute_complete(
    args: argparse.Namespace,
    *,
    services: BenchmarkServices,
    output: Path,
    artifact_spec_path: Path,
    inputs: Sequence[dict[str, object]],
    started_at: datetime,
) -> dict[str, object]:
    stream_counts = tuple(args.stream_counts)
    if len(set(stream_counts)) != len(stream_counts):
        raise BenchmarkError("stream counts must be unique")
    if tuple(sorted(stream_counts)) != stream_counts:
        raise BenchmarkError("stream counts must be in ascending order")
    max_count = max(stream_counts)
    if len(inputs) not in {1, max_count}:
        raise BenchmarkError(
            "provide one synthetic replay input or exactly the maximum requested stream count"
        )
    _scenario_inputs(inputs, max_count)

    artifact_spec_sha256 = _sha256(artifact_spec_path)
    torch_module = services.torch_factory()
    repository_root = Path(__file__).resolve().parents[3]
    starting_git = _verified_git_facts(services.git_facts(repository_root))
    scenarios: list[dict[str, object]] = []
    runner = None
    with services.runtime_context():
        detector, runner, artifact, _class_map = load_yolo11_detector(
            artifact_spec_path,
            runner_factory=services.runner_factory,
        )
        try:
            device_index = resolve_torch_device(torch_module, artifact.device)
            hardware = _verified_hardware_facts(
                services.hardware_facts(torch_module), selected_cuda_device=device_index
            )
            synchronizer = services.synchronizer_factory(torch_module, device_index)
            for count in stream_counts:
                scenario_inputs = _scenario_inputs(inputs, count)
                media_hashes = [str(item["sha256"]) for item in scenario_inputs]
                result = asyncio.run(
                    run_multistream_scenario(
                        detector=detector,
                        input_paths=[Path(str(item["path"])) for item in scenario_inputs],
                        source_factory=services.source_factory,
                        synchronizer=synchronizer,
                        resource_monitor=services.resource_monitor_factory(
                            torch_module, device_index
                        ),
                        warmup_seconds=args.warmup_seconds,
                        measurement_seconds=args.measurement_seconds,
                        target_fps=args.target_fps,
                        resource_sample_interval_seconds=args.resource_sample_interval_seconds,
                    )
                )
                result["syntheticConcurrentReplay"] = len(set(media_hashes)) < len(media_hashes)
                result["inputSha256"] = media_hashes
                scenarios.append(result)
        finally:
            if runner is not None:
                runner.close()
    if _sha256(artifact_spec_path) != artifact_spec_sha256:
        raise BenchmarkError("artifact spec changed during benchmark")
    if _sha256(artifact.resolved_path) != artifact.actual_sha256:
        raise BenchmarkError("model checkpoint changed during benchmark")
    for item in inputs:
        if _sha256(Path(str(item["path"]))) != item["sha256"]:
            raise BenchmarkError("benchmark input changed during benchmark")
    finishing_git = _verified_git_facts(services.git_facts(repository_root))
    if finishing_git != starting_git:
        raise BenchmarkError("Git provenance changed during benchmark")

    finished_at = datetime.now(UTC)
    return {
        "schemaVersion": "1.0.0",
        "status": "COMPLETE",
        "benchmarkType": "YOLO11S_LOCAL_SHARED_RUNNER_VIDEO_REPLAY",
        "reportOnlyBaseline": True,
        "thresholdsApplied": False,
        "startedAtUtc": started_at.isoformat(),
        "finishedAtUtc": finished_at.isoformat(),
        "configuration": {
            "streamCounts": list(stream_counts),
            "warmupSeconds": args.warmup_seconds,
            "measurementSeconds": args.measurement_seconds,
            "targetFps": args.target_fps,
            "resourceSampleIntervalSeconds": args.resource_sample_interval_seconds,
            "latencyScope": "scheduled-local-replay-to-normalized-detection",
            "latencyPopulation": (
                "all normalized-detection completions started from scheduled slots in the "
                "measurement window, including completions after that window"
            ),
            "percentileEstimator": "linear-interpolation-at-rank-(n-1)*q",
            "latencyMetricDefinitions": {
                "decodeAndPack": "OpenCV frame decode plus packed BGR24 byte conversion",
                "inferenceQueueWait": "wait for the one shared serialized detector lane",
                "detector": (
                    "CUDA-synchronized detector call including preprocessing, provider "
                    "inference/postprocessing, and normalized contract construction"
                ),
                "pipeline": "decode-and-pack start through normalized detection completion",
                "endToEnd": "scheduled replay deadline through normalized detection completion",
            },
            "throughputScope": "frames completed within the fixed measurement window",
            "dropDefinition": (
                "scheduled replay frames skipped after a missed processing deadline or whose "
                "normalized detection completed after the measurement window"
            ),
            "sharedRunner": True,
            "serializedInference": True,
            "cudaSynchronizedInferenceLatency": device_index is not None,
            "deviceSynchronization": "cuda" if device_index is not None else "not-required-cpu",
            "localOnly": True,
        },
        "artifact": {
            "artifactSpecPath": str(artifact_spec_path),
            "artifactSpecSha256": artifact_spec_sha256,
            "checkpointPath": str(artifact.resolved_path),
            "checkpointSha256": artifact.actual_sha256,
            "artifactId": artifact.artifact_id,
            "version": artifact.version,
            "modelFamily": artifact.model_family,
            "device": artifact.device,
            "confidenceThreshold": artifact.confidence_threshold,
            "iouThreshold": artifact.iou_threshold,
            "imageSize": list(artifact.image_size),
            "classMap": {str(key): value for key, value in artifact.class_map},
            "sourceUrl": artifact.source_url,
            "license": artifact.license,
        },
        "inputs": list(inputs),
        "scenarios": scenarios,
        "runtime": dict(services.runtime_facts(torch_module)),
        "hardware": hardware,
        "git": finishing_git,
        "outputPath": str(output),
    }


def run(
    argv: Sequence[str] | None = None,
    *,
    services: BenchmarkServices | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    started_at = datetime.now(UTC)
    output: Path | None = None
    try:
        artifact_spec = _regular_input(
            args.artifact_spec, label="artifact spec", suffixes=frozenset({".json"})
        )
        videos = [
            _regular_input(
                path,
                label=f"benchmark input {index + 1}",
                suffixes=frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"}),
            )
            for index, path in enumerate(args.input)
        ]
        if len(set(videos)) != len(videos):
            raise BenchmarkError("explicit benchmark input paths must be unique")
        output = _new_output(args.output, [artifact_spec, *videos])
        inputs = [
            {"path": str(path), "sha256": _sha256(path), "sizeBytes": path.stat().st_size}
            for path in videos
        ]
        report = _execute_complete(
            args,
            services=services or _default_services(),
            output=output,
            artifact_spec_path=artifact_spec,
            inputs=inputs,
            started_at=started_at,
        )
        _atomic_json(output, report)
        print(f"multistream benchmark COMPLETE: {output}")
        return 0
    except KeyboardInterrupt:
        failure: BaseException = BenchmarkError("benchmark interrupted")
        exit_code = 130
    except Exception as error:
        failure = error
        exit_code = 1
    if output is not None:
        incomplete = {
            "schemaVersion": "1.0.0",
            "status": "INCOMPLETE",
            "benchmarkType": "YOLO11S_LOCAL_SHARED_RUNNER_VIDEO_REPLAY",
            "reportOnlyBaseline": True,
            "thresholdsApplied": False,
            "startedAtUtc": started_at.isoformat(),
            "finishedAtUtc": datetime.now(UTC).isoformat(),
            "failure": {"type": type(failure).__name__, "message": str(failure)},
        }
        try:
            _atomic_json(output, incomplete)
        except OSError as write_error:
            print(f"multistream benchmark failed and report could not be written: {write_error}")
    print(f"multistream benchmark INCOMPLETE: {type(failure).__name__}: {failure}")
    return exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()


__all__ = ["BenchmarkServices", "build_parser", "main", "run"]
