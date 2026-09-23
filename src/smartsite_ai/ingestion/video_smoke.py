"""CLI smoke test for a local video or RTSP source through the ingestion worker."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Absolute video path, camera index, or RTSP URL")
    parser.add_argument("--stream-id", default="video-smoke")
    parser.add_argument("--camera-external-id", default="video-smoke-camera")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Treat EOF as a reconnectable disconnect (use for RTSP, not finite files)",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    source_url = str(Path(args.source).expanduser()) if "://" not in args.source else args.source
    config = StreamConfig(
        stream_id=args.stream_id,
        camera_external_id=args.camera_external_id,
        source_url=source_url,
        is_live=args.live,
        target_fps=args.target_fps,
        max_consecutive_failures=3 if args.live else None,
        reconnect_initial_delay=0.1,
        reconnect_max_delay=1.0,
        reconnect_jitter=0.0,
    )
    source = OpenCvFrameSource(config)
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    consumed = 0
    try:
        while args.max_frames <= 0 or consumed < args.max_frames:
            try:
                frame = await asyncio.wait_for(worker.get_frame(), timeout=10.0)
            except (QueueClosedError, TimeoutError):
                break
            consumed += 1
            if consumed == 1:
                print(f"first frame: {frame.width}x{frame.height} format={frame.pixel_format}")
    finally:
        await worker.stop()

    status = worker.snapshot()
    print(
        json.dumps(
            {
                "state": status.state.value,
                "frames_consumed": consumed,
                "frames_enqueued": status.metrics.frames_enqueued,
                "frames_dropped": status.metrics.frames_dropped,
                "read_errors": status.metrics.read_errors,
                "connection_errors": status.metrics.connection_errors,
                "reconnect_attempts": status.metrics.reconnect_attempts,
                "sanitized_source": status.sanitized_url,
            },
            indent=2,
        )
    )
    return 0 if consumed > 0 and status.state == StreamState.STOPPED else 1


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
