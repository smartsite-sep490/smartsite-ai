import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from smartsite_ai.ingestion.envelope import FrameEnvelope

CAMERA = UUID("11111111-1111-4111-8111-111111111111")
NOW = datetime(2026, 10, 27, tzinfo=UTC)


def credential_payload():
    return {
        "schemaVersion": "1.0.0",
        "cameraId": str(CAMERA),
        "workerGenerationId": str(uuid4()),
        "streamSessionId": str(uuid4()),
        "secret": "x" * 43,
        "expiresAt": "2026-10-27T01:00:00.000Z",
    }


def test_credential_load_is_bounded_scoped_expiring_and_redacted(tmp_path: Path):
    from smartsite_ai.integrations.worker_generation import load_worker_generation

    payload = credential_payload()
    path = tmp_path / "credential.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    binding = load_worker_generation(path, CAMERA, now=NOW)
    assert payload["secret"] not in repr(binding)
    for bad in [
        {**payload, "cameraId": str(uuid4())},
        {**payload, "secret": "secret"},
        {**payload, "expiresAt": NOW.isoformat()},
        {**payload, "unknown": True},
    ]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError) as exc:
            load_worker_generation(path, CAMERA, now=NOW)
        assert payload["secret"] not in str(exc.value)
    path.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError):
        load_worker_generation(path, CAMERA, now=NOW)
    path.write_text(json.dumps(payload)[:-1] + ', "cameraId": "' + str(CAMERA) + '"}')
    with pytest.raises(ValueError):
        load_worker_generation(path, CAMERA, now=NOW)


class FakeSource:
    source_id = "synthetic"
    is_connected = False

    def __init__(self):
        self.session = uuid4()
        self.sequence = 0
        self.opens = 0

    async def connect(self):
        self.opens += 1
        self.is_connected = True

    async def close(self):
        self.is_connected = False

    async def read_frame(self):
        frame = FrameEnvelope(
            stream_id="synthetic",
            camera_external_id="CAM",
            session_id=self.session,
            captured_at=NOW,
            sequence_number=self.sequence,
            width=1,
            height=1,
            payload=b"123",
        )
        self.sequence += 1
        return frame


@pytest.mark.anyio
async def test_bound_source_uses_issued_session_once_and_preserves_sequence():
    from smartsite_ai.integrations.worker_generation import (
        GenerationBoundSource,
        WorkerGenerationCredential,
    )

    binding = WorkerGenerationCredential.model_validate(credential_payload())
    source = FakeSource()
    bound = GenerationBoundSource(source, binding, clock=lambda: NOW)
    await bound.connect()
    for index in range(2):
        frame = await bound.read_frame()
        assert frame.session_id == UUID(binding.stream_session_id)
        assert frame.sequence_number == index
    await bound.close()
    with pytest.raises(RuntimeError):
        await bound.connect()
    assert source.opens == 1


@pytest.mark.anyio
async def test_bound_source_rejects_changed_underlying_session_and_expired_binding():
    from smartsite_ai.integrations.worker_generation import (
        GenerationBoundSource,
        WorkerGenerationCredential,
    )

    binding = WorkerGenerationCredential.model_validate(credential_payload())
    source = FakeSource()
    current = [NOW]
    bound = GenerationBoundSource(source, binding, clock=lambda: current[0])
    await bound.connect()
    await bound.read_frame()
    source.session = uuid4()
    with pytest.raises(RuntimeError):
        await bound.read_frame()
    current[0] = NOW + timedelta(hours=1)
    with pytest.raises(RuntimeError):
        await bound.read_frame()
    await bound.close()
    assert not source.is_connected


@pytest.mark.anyio
async def test_actual_stream_worker_stops_bound_generation_on_connection_loss():
    from smartsite_ai.ingestion.config import StreamConfig
    from smartsite_ai.ingestion.source import SourceReadError
    from smartsite_ai.ingestion.status import StreamState
    from smartsite_ai.ingestion.worker import StreamWorker
    from smartsite_ai.integrations.worker_generation import (
        GenerationBoundSource,
        WorkerGenerationCredential,
    )

    class LostSource(FakeSource):
        async def read_frame(self):
            await asyncio.sleep(0)
            if self.sequence:
                raise SourceReadError("Synthetic connection loss")
            return await super().read_frame()

    binding = WorkerGenerationCredential.model_validate(credential_payload())
    source = LostSource()
    stream = StreamWorker(
        StreamConfig(
            stream_id="synthetic",
            camera_external_id="CAM",
            source_url="fake://synthetic",
            max_consecutive_failures=1,
        ),
        source=GenerationBoundSource(source, binding, clock=lambda: NOW),
    )
    await stream.start()
    await asyncio.wait_for(stream._loop_task, 2)
    assert stream.state is StreamState.ERROR
    assert source.opens == 1
    assert not source.is_connected
    await stream.stop()
