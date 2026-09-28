import asyncio
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

from smartsite_ai.benchmark.multistream import (
    DecodedFrame,
    ReplaySourceFacts,
    run_multistream_scenario,
)
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.tools.benchmark_multistream import BenchmarkServices, run


class FakeSource:
    instances: list["FakeSource"] = []

    def __init__(self, _path: Path) -> None:
        self.facts = ReplaySourceFacts(width=1, height=1, fps=100.0, frame_count=10)
        self.closed = False
        self.skipped = 0
        self.instances.append(self)

    def read(self) -> DecodedFrame:
        return DecodedFrame(payload=b"\x00\x00\x00", decode_seconds=0.001)

    def skip(self, frame_count: int) -> None:
        self.skipped += frame_count

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    def __init__(self, *, delay: float = 0.0, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.maximum_active = 0

    async def detect(self, frame):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("synthetic detector failure")
            return DetectionBatch.from_frame(
                frame,
                model_artifact_id="model-1",
                model_version="v1",
                model_sha256="a" * 64,
                detections=(),
            )
        finally:
            self.active -= 1


class FakeSynchronizer:
    def __init__(self) -> None:
        self.calls = 0

    def synchronize(self) -> None:
        self.calls += 1


class FakeResources:
    def __init__(self, *_args: object) -> None:
        self.started = 0
        self.samples = 0
        self.finished = 0

    def start(self) -> None:
        self.started += 1

    def sample(self) -> None:
        self.samples += 1

    def finish(self):
        self.finished += 1
        return {
            "measurementScope": "fixed-scheduled-window",
            "measurementWallSeconds": 0.08,
            "rss": {"peakSampledBytes": 123},
            "cpu": {"processSeconds": 0.01},
            "vram": {"available": False},
        }


def test_three_streams_share_one_serial_inference_lane_and_report_metrics(tmp_path: Path) -> None:
    FakeSource.instances.clear()
    detector = FakeDetector(delay=0.008)
    synchronizer = FakeSynchronizer()
    resources = FakeResources()

    result = asyncio.run(
        run_multistream_scenario(
            detector=detector,
            input_paths=[tmp_path / f"stream-{index}.mp4" for index in range(3)],
            source_factory=FakeSource,
            synchronizer=synchronizer,
            resource_monitor=resources,
            warmup_seconds=0.02,
            measurement_seconds=0.08,
            target_fps=100.0,
            resource_sample_interval_seconds=0.005,
        )
    )

    assert detector.maximum_active == 1
    assert result["streamCount"] == 3
    assert resources.started == resources.finished == 1
    assert resources.samples > 0
    assert synchronizer.calls >= 2
    assert result["executionElapsedSeconds"] >= result["measurementSeconds"]
    for stream in result["streams"]:
        assert stream["processedFrames"] > 0
        assert stream["scheduledFrames"] <= 8
        assert stream["scheduledFrames"] == stream["processedFrames"] + stream["droppedFrames"]
        assert stream["latency"]["decodeAndPack"] == {
            "p50Ms": 1.0,
            "p95Ms": 1.0,
            "p99Ms": 1.0,
        }
        assert stream["latency"]["detector"]["p95Ms"] is not None
        assert stream["latency"]["pipeline"]["p99Ms"] is not None
        assert stream["latency"]["endToEnd"]["p50Ms"] is not None
        assert stream["latency"]["sampleCount"] == stream["completedInferenceFrames"]


def test_overload_advances_replay_source_past_skipped_schedules(tmp_path: Path) -> None:
    FakeSource.instances.clear()

    result = asyncio.run(
        run_multistream_scenario(
            detector=FakeDetector(delay=0.03),
            input_paths=[tmp_path / "overloaded.mp4"],
            source_factory=FakeSource,
            synchronizer=FakeSynchronizer(),
            resource_monitor=FakeResources(),
            warmup_seconds=0,
            measurement_seconds=0.08,
            target_fps=100.0,
            resource_sample_interval_seconds=0.005,
        )
    )

    skipped = sum(source.skipped for source in FakeSource.instances)
    assert skipped > 0
    assert result["aggregate"]["droppedFrames"] >= skipped


def test_completion_after_measurement_window_is_dropped_not_reported_as_throughput(
    tmp_path: Path,
) -> None:
    result = asyncio.run(
        run_multistream_scenario(
            detector=FakeDetector(delay=0.05),
            input_paths=[tmp_path / "slow.mp4"],
            source_factory=FakeSource,
            synchronizer=FakeSynchronizer(),
            resource_monitor=FakeResources(),
            warmup_seconds=0,
            measurement_seconds=0.02,
            target_fps=50.0,
            resource_sample_interval_seconds=0.005,
        )
    )

    stream = result["streams"][0]
    assert stream["scheduledFrames"] == 1
    assert stream["processedFrames"] == 0
    assert stream["droppedFrames"] == 1
    assert stream["lateCompletedFrames"] == 1
    assert stream["completedInferenceFrames"] == 1
    assert stream["effectiveFps"] == 0
    assert stream["latency"]["detector"]["p50Ms"] is not None
    assert stream["latency"]["sampleCount"] == 1
    assert result["executionElapsedSeconds"] > result["measurementSeconds"]
    assert result["resources"]["measurementScope"] == "fixed-scheduled-window"


class FakeRunner:
    instances: list["FakeRunner"] = []
    fail = False

    def __init__(self) -> None:
        self.artifact = None
        self.closed = 0
        self.instances.append(self)

    @property
    def provider_metadata(self):
        return {
            "providerName": "fake",
            "providerVersion": "1.0",
            "architecture": "yolo11",
            "variant": "s",
            "task": "detect",
            "classMap": {
                "0": "Person",
                "1": "Hardhat",
                "2": "NO-Hardhat",
                "3": "Safety Vest",
                "4": "NO-Safety Vest",
            },
        }

    def load(self, artifact) -> None:
        self.artifact = artifact

    def predict(self, _frame):
        if self.fail:
            raise RuntimeError("synthetic provider failure")
        return ()

    def close(self) -> None:
        self.closed += 1


def _fake_torch() -> ModuleType:
    module = ModuleType("fake_torch")
    module.version = SimpleNamespace(cuda=None)
    module.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    return module


def _services() -> BenchmarkServices:
    return BenchmarkServices(
        runtime_context=lambda: nullcontext(),
        runner_factory=FakeRunner,
        torch_factory=_fake_torch,
        source_factory=FakeSource,
        synchronizer_factory=lambda _torch, _device: FakeSynchronizer(),
        resource_monitor_factory=lambda _torch, _device: FakeResources(),
        runtime_facts=lambda _torch: {"python": "test"},
        hardware_facts=lambda _torch: {"machine": "test"},
        git_facts=lambda _path: {"available": True, "commitSha": "b" * 40, "dirty": False},
    )


def _benchmark_files(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake local video")
    model = tmp_path / "best.pt"
    model.write_bytes(b"fake model")
    spec = tmp_path / "artifact.json"
    spec.write_text(
        json.dumps(
            {
                "artifactId": "smartsite-yolo11s-ppe",
                "version": "test-v1",
                "modelFamily": "yolo11s",
                "artifactPath": str(model.resolve()),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "sourceUrl": "https://example.com/model",
                "license": "test-only",
                "classMap": {
                    "0": "Person",
                    "1": "Hardhat",
                    "2": "NO-Hardhat",
                    "3": "Safety Vest",
                    "4": "NO-Safety Vest",
                },
                "confidenceThreshold": 0.25,
                "iouThreshold": 0.45,
                "imageSize": [640, 640],
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    return video, spec


def _arguments(video: Path, spec: Path, output: Path) -> list[str]:
    return [
        "--input",
        str(video.resolve()),
        "--artifact-spec",
        str(spec.resolve()),
        "--output",
        str(output.resolve()),
        "--stream-counts",
        "1",
        "2",
        "3",
        "--warmup-seconds",
        "0.01",
        "--measurement-seconds",
        "0.03",
        "--target-fps",
        "30",
        "--resource-sample-interval-seconds",
        "0.005",
    ]


def test_cli_writes_atomic_complete_report_with_exact_provenance(tmp_path: Path) -> None:
    FakeRunner.instances.clear()
    FakeRunner.fail = False
    video, spec = _benchmark_files(tmp_path)
    output = tmp_path / "benchmark.json"

    assert run(_arguments(video, spec, output), services=_services()) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETE"
    assert report["reportOnlyBaseline"] is True
    assert report["thresholdsApplied"] is False
    assert report["configuration"]["cudaSynchronizedInferenceLatency"] is False
    assert report["configuration"]["deviceSynchronization"] == "not-required-cpu"
    assert report["artifact"]["modelFamily"] == "yolo11s"
    assert (
        report["artifact"]["checkpointSha256"]
        == hashlib.sha256((tmp_path / "best.pt").read_bytes()).hexdigest()
    )
    assert [item["streamCount"] for item in report["scenarios"]] == [1, 2, 3]
    assert report["scenarios"][0]["syntheticConcurrentReplay"] is False
    assert report["scenarios"][1]["syntheticConcurrentReplay"] is True
    assert report["scenarios"][2]["syntheticConcurrentReplay"] is True
    assert FakeRunner.instances[0].closed == 1
    assert not list(tmp_path.glob(".benchmark.json.*.tmp"))


def test_cli_atomically_writes_incomplete_report_and_closes_runner_on_failure(
    tmp_path: Path,
) -> None:
    FakeRunner.instances.clear()
    FakeRunner.fail = True
    video, spec = _benchmark_files(tmp_path)
    output = tmp_path / "benchmark.json"

    assert run(_arguments(video, spec, output), services=_services()) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "INCOMPLETE"
    assert report["thresholdsApplied"] is False
    assert report["failure"]["type"] == "BenchmarkError"
    assert "detector failed" in report["failure"]["message"]
    assert FakeRunner.instances[0].closed == 1
    assert not list(tmp_path.glob(".benchmark.json.*.tmp"))
