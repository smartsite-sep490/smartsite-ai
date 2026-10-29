"""Configuration-only worker ownership and observer behavior with synthetic frames."""

import asyncio
import importlib
import importlib.util
import sys
import threading
from datetime import timedelta
from uuid import uuid4

import pytest
from configuration_capture_fakes import NOW, FakeSource, finish, frame, until


def capture_pair(source=None, *, kind="LIVE", clock=None, encoder=None):
    name = "smartsite_ai.ingestion.configuration_capture"
    assert importlib.util.find_spec(name) is not None, "Capture-only owner is not implemented"
    module = importlib.import_module(name)
    source = source or FakeSource()
    lease = module.ConfigurationSourceLease(source, "camera-a", source_kind=kind)
    options = {"clock": clock or (lambda: NOW), "stop_timeout": 0.03}
    if encoder is not None:
        options["encoder"] = encoder
    return module.ConfigurationCapture(lease, **options), source, lease, module


def test_observing_never_connects_or_imports_a_detector():
    before = set(sys.modules)
    capture, source, _, _ = capture_pair()
    assert capture.latest() is None
    assert capture.state == "IDLE"
    assert source.connects == source.reads == 0
    assert not ({"torch", "cv2", "ultralytics"} & (set(sys.modules) - before))


@pytest.mark.parametrize(
    "field,value",
    [("camera_external_id", "other"), ("source_kind", "REPLAY"), ("source", FakeSource())],
)
def test_owner_binding_cannot_be_retargeted_after_construction(field, value):
    _, _, lease, _ = capture_pair()
    with pytest.raises(AttributeError):
        setattr(lease, field, value)


@pytest.mark.parametrize("camera", ["camera\nsecret", "camera\ud800", "", "x" * 129])
def test_source_owner_rejects_camera_id_that_canonical_contract_would_reject(camera):
    _, _, _, module = capture_pair()
    with pytest.raises(ValueError, match="identity"):
        module.ConfigurationSourceLease(FakeSource(), camera, source_kind="LIVE")


@pytest.mark.parametrize("kind", ["LIVE", "REPLAY"])
def test_explicit_start_yields_purpose_frame_and_observers_share_one_source(kind):
    async def scenario():
        capture, source, _, _ = capture_pair(kind=kind)
        await asyncio.gather(capture.start(), capture.start())
        await source.reading.wait()
        source.items.put_nowait(frame())
        await until(lambda: capture.latest() is not None)
        snapshots = [capture.latest() for _ in range(30)]
        assert source.connects == 1
        assert all(snapshot == snapshots[0] for snapshot in snapshots)
        snapshot = snapshots[0]
        assert set(snapshot) == {"schemaVersion", "purpose", "sourceKind", "frame"}
        assert snapshot["purpose"] == "REGION_CONFIGURATION"
        assert snapshot["sourceKind"] == kind
        assert snapshot["frame"]["sequenceNumber"] == "0"
        assert snapshot["frame"]["cameraExternalId"] == "camera-a"
        assert snapshot["frame"]["capturedAt"] == NOW.isoformat()
        snapshot["frame"]["cameraExternalId"] = "tampered"
        assert capture.latest()["frame"]["cameraExternalId"] == "camera-a"
        await finish(capture, source)
        assert source.closes == 1
        assert capture.latest() is None
        with pytest.raises(RuntimeError):
            await capture.start()

    asyncio.run(scenario())


def test_slow_encoder_resumes_with_newest_pending_frame_not_a_history_queue():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        encoded = []
        image = importlib.import_module("smartsite_ai.configuration_image")

        def encode(value):
            encoded.append(value.sequence_number)
            started.set()
            assert release.wait(2)
            return image.encode_configuration_image(value)

        capture, source, _, _ = capture_pair(encoder=encode)
        try:
            await capture.start()
            await source.reading.wait()
            source.items.put_nowait(frame())
            await until(started.is_set)
            for seq in range(1, 101):
                source.items.put_nowait(frame(seq))
            await until(lambda: source.reads == 102)
            release.set()
            await until(
                lambda: (
                    capture.latest() is not None
                    and capture.latest()["frame"]["sequenceNumber"] == "100"
                )
            )
            assert encoded == [0, 100]
        finally:
            release.set()
            await finish(capture, source)

    asyncio.run(scenario())


def test_partial_connection_failure_closes_once_and_never_retries_on_observation():
    async def scenario():
        class PartialSource(FakeSource):
            async def connect(self):
                self.connects += 1
                raise RuntimeError("private connection detail")

        capture, source, _, _ = capture_pair(PartialSource())
        await capture.start()
        await until(lambda: source.closes == 1)
        assert capture.reason == "source_failed" and capture.latest() is None
        assert await capture.stop()
        assert await capture.stop()
        assert source.connects == source.closes == 1

    asyncio.run(scenario())


def test_encoder_failure_clears_image_and_does_not_release_a_held_native_read():
    async def scenario():
        def encode(_):
            raise RuntimeError("private encoder detail")

        capture, source, _, _ = capture_pair(encoder=encode)
        await capture.start()
        await source.reading.wait()
        source.items.put_nowait(frame())
        await until(lambda: capture.state == "FAILED")
        assert capture.latest() is None and capture.reason == "image_encoding_failed"
        assert not await capture.stop()
        assert source.closes == 0
        await finish(capture, source)

    asyncio.run(scenario())


def test_second_owner_cannot_connect_until_first_io_and_close_are_drained():
    async def scenario():
        capture, source, lease, module = capture_pair()
        second = module.ConfigurationCapture(lease, clock=lambda: NOW, stop_timeout=0.03)
        await capture.start()
        await source.reading.wait()
        with pytest.raises(RuntimeError):
            await second.start()
        assert not await capture.stop()
        assert capture.state == "STOPPING"
        assert capture.latest() is None
        assert source.closes == 0  # never release a handle while native read may use it
        with pytest.raises(RuntimeError):
            await second.start()
        source.close_gate.clear()
        source.items.put_nowait(frame())  # late read cannot publish
        await until(lambda: source.closes == 1)
        assert not await capture.stop()
        with pytest.raises(RuntimeError):
            await second.start()
        source.close_gate.set()
        assert await capture.stop()
        await second.start()
        assert source.connects == 1  # task has not run yet
        await until(lambda: source.connects == 2)
        await finish(second, source)

    asyncio.run(scenario())


def test_stop_during_connect_does_not_release_before_late_allocation():
    async def scenario():
        source = FakeSource()
        source.connect_gate.clear()
        capture, _, lease, module = capture_pair(source)
        await capture.start()
        await source.connected.wait()
        assert not await capture.stop()
        assert source.closes == 0
        other = module.ConfigurationCapture(lease)
        with pytest.raises(RuntimeError):
            await other.start()
        source.connect_gate.set()
        assert await capture.stop()
        assert source.closes == 1 and source.reads == 0

    asyncio.run(scenario())


def test_close_failure_keeps_ownership_unresolved_and_hides_secret():
    async def scenario():
        source = FakeSource()
        source.close_error = True
        capture, _, lease, module = capture_pair(source)
        await capture.start()
        await source.reading.wait()
        source.items.put_nowait(None)
        await until(lambda: capture.state == "FAILED")
        assert not await capture.stop()
        assert capture.reason == "source_release_failed"
        assert "private" not in repr(capture.latest()) + capture.reason
        with pytest.raises(RuntimeError):
            await module.ConfigurationCapture(lease).start()
        assert source.connects == source.closes == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", [None, RuntimeError("rtsp://private-secret")])
def test_eof_and_source_failure_clear_pixels_without_request_restart(terminal):
    async def scenario():
        capture, source, _, _ = capture_pair()
        await capture.start()
        await source.reading.wait()
        source.items.put_nowait(frame())
        await until(lambda: capture.latest() is not None)
        source.items.put_nowait(terminal)
        await until(lambda: source.closes == 1)
        assert capture.latest() is None
        for _ in range(10):
            assert capture.latest() is None
        assert source.connects == 1
        assert capture.reason in ("source_ended", "source_failed")
        assert await capture.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"camera_external_id": "wrong-camera"},
        {"stream_id": "wrong-stream"},
        {"sequence_number": 1},
    ],
)
def test_first_frame_identity_or_sequence_failure_is_terminal(changes):
    async def scenario():
        capture, source, _, _ = capture_pair()
        await capture.start()
        await source.reading.wait()
        source.items.put_nowait(frame(**changes))
        await until(lambda: source.closes == 1)
        assert capture.latest() is None
        assert capture.reason == "frame_identity_invalid"
        assert await capture.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("changes", [{"session_id": uuid4()}, {"sequence_number": 0}])
def test_midstream_session_change_or_nonincreasing_sequence_clears_image(changes):
    async def scenario():
        capture, source, _, _ = capture_pair()
        await capture.start()
        await source.reading.wait()
        source.items.put_nowait(frame())
        await until(lambda: capture.latest() is not None)
        source.items.put_nowait(frame(**changes))
        await until(lambda: source.closes == 1)
        assert capture.latest() is None
        assert capture.reason == "frame_identity_invalid"
        assert await capture.stop()

    asyncio.run(scenario())


def test_freshness_is_rechecked_on_read_without_retimestamping():
    async def scenario():
        now = [NOW]
        capture, source, _, _ = capture_pair(clock=lambda: now[0])
        await capture.start()
        await source.reading.wait()
        source.items.put_nowait(frame())
        await until(lambda: capture.latest() is not None)
        now[0] += timedelta(seconds=2, microseconds=1)
        assert capture.latest() is None
        now[0] = NOW - timedelta(seconds=1, microseconds=1)
        assert capture.latest() is None
        await finish(capture, source)

    asyncio.run(scenario())


def test_slow_encoder_keeps_only_latest_pending_frame_and_stop_waits_for_it():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        encoded = []
        image = importlib.import_module("smartsite_ai.configuration_image")

        def encode(value):
            encoded.append(value.sequence_number)
            started.set()
            assert release.wait(2)
            return image.encode_configuration_image(value)

        capture, source, _, _ = capture_pair(encoder=encode)
        try:
            await capture.start()
            await source.reading.wait()
            source.items.put_nowait(frame())
            await until(started.is_set)
            for seq in range(1, 101):
                source.items.put_nowait(frame(seq))
            await until(lambda: source.reads == 102)
            assert encoded == [0]
            assert capture.pending_frames == 1
            assert not await capture.stop()
            assert capture.latest() is None
            source.items.put_nowait(None)
            assert not await capture.stop()
        finally:
            release.set()
            await finish(capture, source)
        assert encoded == [0]  # pending frame is discarded on stop

    asyncio.run(scenario())


def test_cancelling_stop_waiter_cannot_cancel_or_release_source_io():
    async def scenario():
        capture, source, _, _ = capture_pair()
        await capture.start()
        await source.reading.wait()
        waiting = asyncio.create_task(capture.stop())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert source.closes == 0 and capture.latest() is None
        await finish(capture, source)
        assert source.closes == 1

    asyncio.run(scenario())
