import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from smartsite_ai.source_activation import SourceActivationError
from smartsite_ai.source_process_driver import SourceProcessDriver
from smartsite_ai.source_resume_models import ResumeBinding, ResumeConfiguration, ResumeManifest

SCOPE = ("11111111-1111-4111-8111-111111111111", "camera-a")
CAMERA = str(uuid4())


def _generation() -> dict[str, str]:
    return {
        "workerGenerationId": str(uuid4()),
        "streamSessionId": str(uuid4()),
        "secret": "z" * 48,
        "expiresAt": "2099-01-01T00:00:00+00:00",
    }


def _write_stop_aware_child(script: Path, marker: Path, stop_file: Path) -> None:
    script.write_text(
        "import json, os, pathlib, time\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        f"stop_file = pathlib.Path({str(stop_file)!r})\n"
        "marker.write_text(os.environ['SMARTSITE_AI_SOURCE_RUNTIME_GENERATION'])\n"
        "while not stop_file.exists():\n"
        "    time.sleep(0.02)\n"
    )


def test_generation_id_prefers_owned_generation_after_start_and_falls_back_on_clean_stop(
    tmp_path,
):
    seed_a = str(uuid4())
    generation_file = tmp_path / "generation.json"
    seed_bytes = json.dumps({"workerGenerationId": seed_a}).encode("utf-8")
    generation_file.write_bytes(seed_bytes)

    stop_file = tmp_path / "STOP"
    marker = tmp_path / "ready.json"
    child = tmp_path / "capture_child.py"
    _write_stop_aware_child(child, marker, stop_file)

    async def ready(identifier):
        return (
            marker.exists() and json.loads(marker.read_text())["workerGenerationId"] == identifier
        )

    async def run():
        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=generation_file,
            environment={},
            ready=ready,
        )
        assert await driver.generation_id(SCOPE) == seed_a

        gen_b = _generation()
        await driver.start(SCOPE, "0", gen_b)
        try:
            assert await driver.is_running(SCOPE, gen_b["workerGenerationId"])
            assert generation_file.read_bytes() == seed_bytes
            assert await driver.generation_id(SCOPE) == gen_b["workerGenerationId"]
        finally:
            await driver.stop(SCOPE)

        assert generation_file.read_bytes() == seed_bytes
        assert await driver.generation_id(SCOPE) == seed_a

    asyncio.run(run())


def test_generation_id_prefers_owned_generation_after_validated_resume(tmp_path):
    seed_a = str(uuid4())
    generation_file = tmp_path / "generation.json"
    seed_bytes = json.dumps({"workerGenerationId": seed_a}).encode("utf-8")
    generation_file.write_bytes(seed_bytes)

    stop_file = tmp_path / "STOP"
    marker = tmp_path / "infer_ready.json"
    capture_child = tmp_path / "capture_child.py"
    infer_child = tmp_path / "infer_child.py"
    _write_stop_aware_child(capture_child, tmp_path / "capture_ready.json", stop_file)
    _write_stop_aware_child(infer_child, marker, stop_file)

    gen_c = _generation()
    loaded = ResumeConfiguration(configurationVersion=12, viewRevision=7, bundleSha256="a" * 64)
    manifest_c = ResumeManifest(
        generationBinding=ResumeBinding(
            cameraId=CAMERA,
            workerGenerationId=gen_c["workerGenerationId"],
            streamSessionId=gen_c["streamSessionId"],
        ),
        configuration=loaded,
    )

    async def ready(identifier):
        return (
            marker.exists() and json.loads(marker.read_text())["workerGenerationId"] == identifier
        )

    async def load(_manifest, _generation):
        return loaded

    async def run():
        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(capture_child)),
            resume_command=(str(Path(sys.executable)), "-I", str(infer_child)),
            stop_file=stop_file,
            generation_file=generation_file,
            environment={},
            ready=ready,
            resume_camera_id=CAMERA,
            resume_manifest_supported=True,
            resume_configuration_loader=load,
        )
        await driver.start_resume(SCOPE, "0", gen_c, manifest_c)
        try:
            assert await driver.is_running(SCOPE, gen_c["workerGenerationId"])
            assert generation_file.read_bytes() == seed_bytes
            assert await driver.generation_id(SCOPE) == gen_c["workerGenerationId"]
        finally:
            await driver.stop(SCOPE)

    asyncio.run(run())


def test_generation_id_rejects_unknown_scope(tmp_path):
    seed_a = str(uuid4())
    generation_file = tmp_path / "generation.json"
    generation_file.write_text(json.dumps({"workerGenerationId": seed_a}))

    async def ready(_):
        return True

    driver = SourceProcessDriver(
        scope=SCOPE,
        command=(str(tmp_path / "capture-cli"),),
        stop_file=tmp_path / "STOP",
        generation_file=generation_file,
        environment={},
        ready=ready,
    )

    async def run():
        with pytest.raises(SourceActivationError):
            await driver.generation_id(("other-site", "camera-a"))
        with pytest.raises(SourceActivationError):
            await driver.generation_id((SCOPE[0], "other-camera"))

    asyncio.run(run())


def test_generation_id_fails_closed_under_reconciliation_latch(tmp_path, monkeypatch):
    seed_a = str(uuid4())
    generation_file = tmp_path / "generation.json"
    seed_bytes = json.dumps({"workerGenerationId": seed_a}).encode("utf-8")
    generation_file.write_bytes(seed_bytes)

    class _FakeHungProcess:
        def __init__(self):
            self.pid = 7272
            self.returncode = None
            self._terminated = asyncio.Event()

        def kill(self):
            self.returncode = -9
            self._terminated.set()

        async def wait(self):
            await self._terminated.wait()
            return self.returncode

    async def fake_exec(*_cmd, **_kwargs):
        return _FakeHungProcess()

    real_timeout = asyncio.timeout
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        asyncio, "timeout", lambda d: real_timeout(0.02 if d is not None and d >= 1 else d)
    )
    monkeypatch.setattr(
        asyncio,
        "wait_for",
        lambda aw, timeout=None: real_wait_for(
            aw, timeout=0.02 if timeout is not None and timeout >= 1 else timeout
        ),
    )

    async def ready(_):
        return True

    async def run():
        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(tmp_path / "capture-cli"), "--manifest", str(tmp_path / "capture.json")),
            stop_file=tmp_path / "STOP",
            generation_file=generation_file,
            environment={},
            ready=ready,
        )
        gen_b = _generation()
        await driver.start(SCOPE, "0", gen_b)
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.stop(SCOPE)

        assert generation_file.read_bytes() == seed_bytes
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.generation_id(SCOPE)

    asyncio.run(run())
