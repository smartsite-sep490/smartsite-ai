import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from smartsite_ai.config import Settings
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import SourceConnectionError, SourceReadError
from smartsite_ai.live_source_recovery import LiveSourceRecovery


def test_recovery_is_opt_in_and_configuration_bounded():
    assert Settings(_env_file=None).realtime_live_reconnect_attempts == 0
    for invalid in (-1, 4):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, realtime_live_reconnect_attempts=invalid)


def test_failures_exhaust_budget_and_successful_reads_alone_replenish_it():
    async def scenario():
        delays = []
        recovery = LiveSourceRecovery(3, sleeper=AsyncMock(side_effect=delays.append))
        source = MagicMock()
        source.connect = AsyncMock(side_effect=SourceConnectionError("private-url"))
        assert not await recovery.reconnect(source)
        assert source.connect.await_count == 3
        assert len(delays) == 3
        assert 0.8 <= delays[0] <= 1.2
        assert 1.6 <= delays[1] <= 2.4
        assert 3.2 <= delays[2] <= 4.0
        for _ in range(29):
            recovery.observed_frame()
        assert not await recovery.reconnect(source)
        # reconnect resets the success streak; need 30 NEW reads, not polls.
        for _ in range(30):
            recovery.observed_frame()
        source.connect.side_effect = None
        assert await recovery.reconnect(source)
        assert source.connect.await_count == 4

    asyncio.run(scenario())


def test_cancelled_recovery_does_not_reconnect_source():
    async def scenario():
        async def cancelled(_delay):
            raise asyncio.CancelledError

        source = MagicMock()
        source.connect = AsyncMock()
        with pytest.raises(asyncio.CancelledError):
            await LiveSourceRecovery(3, sleeper=cancelled).reconnect(source)
        source.connect.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("read_error", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_stream_recovers_new_session_without_model_reload(monkeypatch, read_error, enabled):
    import smartsite_ai.realtime as realtime
    from smartsite_ai.inference.models import DetectionBatch

    session = 0
    read_count = 0
    sequence = 0
    source = MagicMock()
    media = MagicMock()
    resets = []
    original_reset = realtime.RealtimeStreamGateTracker.reset

    def reset(tracker):
        resets.append(session)
        original_reset(tracker)

    monkeypatch.setattr(realtime.RealtimeStreamGateTracker, "reset", reset)

    async def connect():
        nonlocal session, sequence
        if session == 1:
            assert resets == [1]
            media.clear.assert_called_once()
        session += 1
        sequence = 0

    async def read():
        nonlocal read_count, sequence
        read_count += 1
        if session == 1 and read_count > 1:
            if read_error:
                raise SourceReadError("private-source")
            return None
        envelope = FrameEnvelope(
            stream_id="s",
            camera_external_id="camera-1",
            session_id=UUID(int=session),
            captured_at=datetime.now(UTC),
            sequence_number=sequence,
            width=4,
            height=2,
            payload=bytes([0, 0, 255]) * 8,
        )
        sequence += 1
        return envelope

    source.connect = connect
    source.read_frame = read
    source.close = AsyncMock()
    detector, runner, pipeline = MagicMock(), MagicMock(), MagicMock()

    async def detect(envelope):
        return DetectionBatch.from_frame(
            envelope,
            model_artifact_id="fake",
            model_version="test",
            model_sha256="a" * 64,
            detections=(),
        )

    detector.detect = AsyncMock(side_effect=detect)
    pipeline.observe.return_value = None
    pipeline.occupied_zone_track_regions.return_value = frozenset()
    loader = MagicMock(return_value=(detector, runner, pipeline, None))
    monkeypatch.setattr(realtime, "_load_realtime_stack", loader)
    monkeypatch.setattr(realtime, "OpenCvFrameSource", lambda config: source)
    monkeypatch.setattr(
        realtime,
        "LiveSourceRecovery",
        lambda attempts: LiveSourceRecovery(attempts, sleeper=AsyncMock()),
    )
    sent = []

    async def send(payload):
        sent.append(payload)
        if payload.get("type") == "frame" and len([v for v in sent if v["type"] == "frame"]) == 3:
            raise WebSocketDisconnect()

    socket = MagicMock()
    socket.accept, socket.close = AsyncMock(), AsyncMock()
    socket.send_json = send
    asyncio.run(
        realtime.stream_realtime(
            socket,
            Settings(
                _env_file=None,
                realtime_source="0",
                realtime_live_reconnect_attempts=3 if enabled else 0,
            ),
            media_hub=media,
        )
    )
    assert [v["type"] for v in sent] == (
        ["frame", "error", "frame", "frame"] if enabled else ["frame", "error"]
    )
    frames = [v for v in sent if v["type"] == "frame"]
    if enabled:
        assert frames[0]["sessionId"] != frames[1]["sessionId"]
        assert frames[0]["sequenceNumber"] == frames[1]["sequenceNumber"] == "0"
    assert session == (2 if enabled else 1)
    assert resets == ([1] if enabled else [])
    loader.assert_called_once()
    runner.close.assert_called_once()
    source.close.assert_awaited_once()
    assert "private" not in str(sent)


def test_replay_uses_existing_loop_not_live_recovery(tmp_path, monkeypatch):
    import smartsite_ai.realtime as realtime

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake clip, source mocked")
    source, detector, runner, pipeline, socket = (MagicMock() for _ in range(5))
    source.connect = AsyncMock(side_effect=[None, WebSocketDisconnect()])
    source.read_frame = AsyncMock(return_value=None)
    source.close = AsyncMock()
    socket.accept, socket.send_json, socket.close = AsyncMock(), AsyncMock(), AsyncMock()
    budgets = []

    def recovery(attempts):
        budgets.append(attempts)
        return LiveSourceRecovery(attempts, sleeper=AsyncMock())

    monkeypatch.setattr(realtime, "LiveSourceRecovery", recovery)
    monkeypatch.setattr(
        realtime, "_load_realtime_stack", lambda settings: (detector, runner, pipeline, None)
    )
    monkeypatch.setattr(realtime, "OpenCvFrameSource", lambda config: source)
    asyncio.run(
        realtime.stream_realtime(
            socket,
            Settings(_env_file=None, realtime_source=str(clip), realtime_live_reconnect_attempts=3),
        )
    )
    assert budgets == [0]
    assert source.connect.await_count == 2
    source.close.assert_awaited_once()
    runner.close.assert_called_once()
    socket.send_json.assert_not_awaited()
