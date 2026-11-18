import asyncio
import json
import socket
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import SecretStr
from test_camera_media_observer import until
from test_multi_camera_runtime import _clean_result, _StopPoller

from smartsite_ai.config import Settings
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.queue import PreviewFrameQueue
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.testing import FakeFrameSource, make_test_frame
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.runtime.media_host import run_with_camera_preview
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.runtime.supervisor import CameraSession, RuntimePlan
from smartsite_ai.tools.run_camera_runtime import build_parser, run_manifest


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_media_host_shares_one_fake_capture_and_closes_all_owners(cancel: bool) -> None:
    frame = make_test_frame("media", 0)
    source = FakeFrameSource("media", [frame], block_when_exhausted=True)
    queue = PreviewFrameQueue()
    stream = StreamWorker(
        StreamConfig(
            stream_id="media", camera_external_id="ext-media", source_url="0", is_live=True
        ),
        source=source,
        preview_frames=queue,
    )
    observer = CameraPreviewObserver(
        "ext-media",
        queue,
        source_kind="LIVE",
        is_streaming=lambda: stream.state == StreamState.STREAMING,
    )
    observers = {}
    stop, bound = asyncio.Event(), asyncio.Event()
    ports, lifecycle = [], []

    class Worker:
        async def run(self):
            await stream.start()
            try:
                assert await stream.get_frame() is frame
                await stop.wait()
                return replace(_clean_result(), frames_processed=1)
            finally:
                await stream.stop()

    async def preflight():
        return None

    async def open_model():
        lifecycle.append("open_fake_model")
        return cast(DetectorProtocol, object())

    async def close_model():
        lifecycle.append("close_fake_model")

    def sessions(_detector):
        observers["ext-media"] = observer
        return (CameraSession("media", "ext-media", Worker(), _StopPoller(), preview=observer),)

    def ready(port):
        ports.append(port)
        bound.set()

    task = asyncio.create_task(
        run_with_camera_preview(
            RuntimePlan(preflight, open_model, close_model, sessions),
            observers,
            SecretStr("x" * 32),
            host="127.0.0.1",
            port=0,
            on_bound=ready,
        )
    )
    try:
        await asyncio.wait_for(bound.wait(), 2)
        await until(lambda: observer.latest() is not None)
        async with (
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{ports[0]}", trust_env=False, timeout=2
            ) as client,
            client.stream(
                "GET",
                "/v1/realtime/video-stream?cameraExternalId=ext-media",
                headers={"Authorization": "Bearer " + "x" * 32},
            ) as response,
        ):
            assert response.status_code == 200
            payload = json.loads(await response.aiter_lines().__anext__())
            assert payload["sequenceNumber"] == "0"
            assert payload["sessionId"] == str(frame.session_id)
            assert set(payload) == {
                "type",
                "mediaVersion",
                "cameraExternalId",
                "sessionId",
                "sequenceNumber",
                "capturedAt",
                "width",
                "height",
                "imageDataUrl",
            }
        assert source.connect_calls == 1
        assert source.close_calls == 0
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            stop.set()
            report = await task
            assert report.model_closed
            assert report.cameras[0].frames_processed == 1
    finally:
        stop.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert source.close_calls == 1
    assert observer.latest() is None
    assert lifecycle == ["open_fake_model", "close_fake_model"]
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", ports[0])) != 0


@pytest.mark.anyio
async def test_preview_cli_is_opt_in_and_requires_its_server_only_token() -> None:
    assert build_parser().parse_args(["--manifest", "absent.json"]).preview_port is None
    assert (
        build_parser()
        .parse_args(["--manifest", "absent.json", "--preview-port", "8123"])
        .preview_port
        == 8123
    )
    with pytest.raises(CameraRuntimeError, match="Preview service token"):
        await run_manifest(
            Path("absent.json"),
            settings=Settings(_env_file=None, preview_service_token=None),
            preview_port=8123,
        )
    with pytest.raises(CameraRuntimeError, match="Preview port"):
        await run_manifest(Path("absent.json"), settings=Settings(_env_file=None), preview_port=0)


@pytest.mark.anyio
async def test_preview_bind_failure_cannot_start_a_model_or_camera() -> None:
    async def forbidden():
        raise AssertionError("Port failure must precede runtime preflight/model activation")

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        plan = RuntimePlan(forbidden, forbidden, forbidden, lambda _: ())
        with pytest.raises(CameraRuntimeError, match="bind"):
            await run_with_camera_preview(
                plan, {}, SecretStr("x" * 32), host="127.0.0.1", port=occupied.getsockname()[1]
            )
