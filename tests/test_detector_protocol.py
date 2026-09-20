import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope

ROOT = Path(__file__).resolve().parents[1]
SESSION = UUID("00000000-0000-4000-8000-000000000001")


def make_frame() -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="stream-01",
        session_id=SESSION,
        camera_external_id="cam-01",
        captured_at=datetime(2026, 9, 20, 12, tzinfo=UTC),
        width=1,
        height=1,
        sequence_number=3,
        payload=bytes(3),
    )


class FakeDetector:
    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="fake-model",
            model_version="v1",
            model_sha256="f" * 64,
            detections=(),
        )


def test_detector_protocol_is_structural_and_async() -> None:
    detector: DetectorProtocol = FakeDetector()
    frame = make_frame()

    assert isinstance(detector, DetectorProtocol)
    result = asyncio.run(detector.detect(frame))

    assert isinstance(result, DetectionBatch)
    assert result.session_id == frame.session_id
    assert result.detections == ()


def test_importing_detector_protocol_does_not_load_provider_runtime() -> None:
    environment = os.environ.copy()
    python_path = str(ROOT / "src")
    if existing := environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, existing))
    environment["PYTHONPATH"] = python_path
    code = (
        "import sys; "
        "import smartsite_ai.inference.protocol; "
        "loaded = {'ultralytics', 'torch', 'cv2'} & set(sys.modules); "
        "assert not loaded, loaded"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
