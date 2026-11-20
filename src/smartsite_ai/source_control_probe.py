"""Explicit bounded source capture child; no model, camera or process opens on import."""

import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path


async def capture_source(source: str, is_live: bool) -> dict[str, object]:
    """Keep the sensitive source in a filtered child environment, never command arguments."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT"}
    }
    env["SMARTSITE_AI_SOURCE_TEST_SOURCE"] = source
    env["SMARTSITE_AI_SOURCE_TEST_LIVE"] = "true" if is_live else "false"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(Path(__file__)),
        "--capture",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=400_001,
        creationflags=int(getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)),
    )
    try:
        async with asyncio.timeout(10):
            if process.stdout is None:
                raise RuntimeError("SOURCE_UNAVAILABLE")
            data = bytearray()
            while True:
                chunk = await process.stdout.read(32_768)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > 400_000:
                    raise RuntimeError("SOURCE_UNAVAILABLE")
            await process.wait()
            if process.returncode != 0:
                raise RuntimeError("SOURCE_UNAVAILABLE")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise RuntimeError("SOURCE_UNAVAILABLE")
            return result
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                async with asyncio.timeout(1):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()


async def _capture() -> dict[str, object]:
    from PIL import Image

    from smartsite_ai.ingestion.config import StreamConfig
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    config = StreamConfig(
        stream_id="explicit-source-test",
        camera_external_id="explicit-source-test",
        source_url=os.environ["SMARTSITE_AI_SOURCE_TEST_SOURCE"],
        is_live=os.environ["SMARTSITE_AI_SOURCE_TEST_LIVE"] == "true",
        pace_replay=False,
    )
    source = OpenCvFrameSource(config)
    try:
        await source.connect()
        frame = await source.read_frame()
        if frame is None or frame.width * frame.height > 8_000_000:
            raise RuntimeError("SOURCE_UNAVAILABLE")
        image = Image.frombytes("RGB", (frame.width, frame.height), frame.payload, "raw", "BGR")
        image.thumbnail((960, 540))
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=75)
            data = output.getvalue()
        if len(data) > 250_000:
            raise RuntimeError("SOURCE_UNAVAILABLE")
        return {
            "width": image.width,
            "height": image.height,
            "capturedAt": frame.captured_at_iso,
            "imageDataUrl": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        }
    finally:
        await source.close()


if __name__ == "__main__":
    try:
        if sys.argv[1:] != ["--capture"]:
            raise RuntimeError("INVALID_SOURCE_TEST")
        print(json.dumps(asyncio.run(_capture())), flush=True)
    except Exception:
        print('{"code":"SOURCE_UNAVAILABLE"}', flush=True)
        raise SystemExit(1) from None
