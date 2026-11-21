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


def _resume_bundle() -> tuple[ResumeConfiguration, ResumeManifest, dict[str, str]]:
    generation = _generation()
    configuration = ResumeConfiguration(
        configurationVersion=12, viewRevision=7, bundleSha256="a" * 64
    )
    manifest = ResumeManifest(
        generationBinding=ResumeBinding(
            cameraId=CAMERA,
            workerGenerationId=generation["workerGenerationId"],
            streamSessionId=generation["streamSessionId"],
        ),
        configuration=configuration,
    )
    return configuration, manifest, generation


def _write_stop_aware_child(script: Path, marker: Path, stop_file: Path, mode: str) -> None:
    script.write_text(
        "import json, os, pathlib, time\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        f"stop_file = pathlib.Path({str(stop_file)!r})\n"
        "payload = {\n"
        f"    'mode': {mode!r},\n"
        "    'source': os.environ.get('SMARTSITE_AI_SOURCE_RUNTIME_SOURCE'),\n"
        "    'generation': json.loads(os.environ['SMARTSITE_AI_SOURCE_RUNTIME_GENERATION']),\n"
        "    'expected': os.environ.get('SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION'),\n"
        "}\n"
        "marker.write_text(json.dumps(payload))\n"
        "while not stop_file.exists():\n"
        "    time.sleep(0.02)\n"
    )


def test_resume_command_requires_nonempty_absolute_executable(tmp_path):
    async def ready(_):
        return True

    for invalid in [(), ("relative/python", "-V"), ("smartsite-ai-camera-runtime",)]:
        with pytest.raises(ValueError, match="absolute"):
            SourceProcessDriver(
                scope=SCOPE,
                command=(str(Path(sys.executable)),),
                resume_command=invalid,
                stop_file=tmp_path / "stop",
                generation_file=tmp_path / "generation.json",
                environment={},
                ready=ready,
            )


def test_omitted_resume_command_keeps_default_single_command_behavior(tmp_path):
    async def run():
        stop_file = tmp_path / "stop"
        marker = tmp_path / "default_marker.json"
        child = tmp_path / "default_child.py"
        _write_stop_aware_child(child, marker, stop_file, "default")
        loaded, manifest, new_generation = _resume_bundle()

        async def ready(identifier):
            return (
                marker.exists()
                and json.loads(marker.read_text())["generation"]["workerGenerationId"] == identifier
            )

        async def load(_manifest, _generation):
            return loaded

        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=tmp_path / "generation.json",
            environment={},
            ready=ready,
            resume_camera_id=CAMERA,
            resume_manifest_supported=True,
            resume_configuration_loader=load,
        )
        await driver.start_resume(SCOPE, "0", new_generation, manifest)
        try:
            assert await driver.is_running(SCOPE, new_generation["workerGenerationId"])
            observed = json.loads(marker.read_text())
            assert observed["mode"] == "default"
            assert json.loads(observed["expected"]) == manifest.model_dump()
        finally:
            await driver.stop(SCOPE)
        assert driver.process is None and driver.current_source is None

    asyncio.run(run())


def test_start_runs_capture_command_and_start_resume_routes_validated_inference_command(tmp_path):
    async def run():
        stop_file = tmp_path / "stop"
        capture_marker = tmp_path / "capture_marker.json"
        infer_marker = tmp_path / "infer_marker.json"
        capture_child = tmp_path / "capture_child.py"
        infer_child = tmp_path / "infer_child.py"
        _write_stop_aware_child(capture_child, capture_marker, stop_file, "capture")
        _write_stop_aware_child(infer_child, infer_marker, stop_file, "inference")

        initial_generation = _generation()
        loaded, manifest, resume_generation = _resume_bundle()

        async def ready(identifier):
            for path in (infer_marker, capture_marker):
                if (
                    path.exists()
                    and json.loads(path.read_text())["generation"]["workerGenerationId"]
                    == identifier
                ):
                    return True
            return False

        async def load(_manifest, _generation):
            return loaded

        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(capture_child)),
            resume_command=(str(Path(sys.executable)), "-I", str(infer_child)),
            stop_file=stop_file,
            generation_file=tmp_path / "generation.json",
            environment={},
            ready=ready,
            resume_camera_id=CAMERA,
            resume_manifest_supported=True,
            resume_configuration_loader=load,
        )

        # Ordinary start launches capture CLI without inference command or resume manifest.
        await driver.start(SCOPE, "0", initial_generation)
        try:
            assert await driver.is_running(SCOPE, initial_generation["workerGenerationId"])
            capture_data = json.loads(capture_marker.read_text())
            assert capture_data["mode"] == "capture"
            assert capture_data["expected"] is None
            assert not infer_marker.exists()
        finally:
            await driver.stop(SCOPE)
        assert driver.process is None and driver.current_source is None

        capture_marker.unlink()

        # Invalid/missing bundle refuses start_resume without spawning inference command.
        wrong = manifest.model_copy(
            update={"configuration": loaded.model_copy(update={"configurationVersion": 13})}
        )
        with pytest.raises(ValueError, match="configuration changed"):
            await driver.start_resume(SCOPE, "0", resume_generation, wrong)
        with pytest.raises(SourceActivationError):
            await driver.start_resume(SCOPE, "0", resume_generation, None)
        assert driver.process is None and not infer_marker.exists()

        # Valid resume selects explicitly configured inference command and passes exact bundle.
        await driver.start_resume(SCOPE, "0", resume_generation, manifest)
        try:
            assert await driver.is_running(SCOPE, resume_generation["workerGenerationId"])
            infer_data = json.loads(infer_marker.read_text())
            assert infer_data["mode"] == "inference"
            assert json.loads(infer_data["expected"]) == manifest.model_dump()
            assert not capture_marker.exists()
        finally:
            await driver.stop(SCOPE)
        assert driver.process is None and driver.current_source is None

    asyncio.run(run())


def test_spontaneous_nonzero_exit_before_stop_keeps_existing_reap_semantics(tmp_path):
    async def run():
        child = tmp_path / "crash_child.py"
        child.write_text("raise SystemExit(7)\n")
        replacement = tmp_path / "healthy_child.py"
        marker = tmp_path / "healthy.json"
        stop_file = tmp_path / "stop"
        _write_stop_aware_child(replacement, marker, stop_file, "replacement")

        async def ready(identifier):
            return (
                marker.exists()
                and json.loads(marker.read_text())["generation"]["workerGenerationId"] == identifier
            )

        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=tmp_path / "generation.json",
            environment={},
            ready=ready,
        )
        await driver.start(SCOPE, "0", _generation())
        process = driver.process
        assert process is not None
        async with asyncio.timeout(5):
            await process.wait()
        assert process.returncode == 7
        assert driver.current_source is None
        await driver.stop(SCOPE)
        assert driver.process is None

        driver.command = (str(Path(sys.executable)), "-I", str(replacement))
        next_gen = _generation()
        await driver.start(SCOPE, "0", next_gen)
        try:
            assert await driver.is_running(SCOPE, next_gen["workerGenerationId"])
        finally:
            await driver.stop(SCOPE)
        assert driver.process is None

    asyncio.run(run())


def test_forced_kill_on_stop_timeout_requires_reconciliation_and_latches_sticky(
    tmp_path, monkeypatch
):
    class _FakeHungProcess:
        def __init__(self):
            self.pid = 4242
            self.returncode = None
            self.kill_calls = 0
            self._terminated = asyncio.Event()

        def kill(self):
            self.kill_calls += 1
            self.returncode = -9
            self._terminated.set()

        async def wait(self):
            await self._terminated.wait()
            return self.returncode

    spawned: list[tuple[tuple[str, ...], _FakeHungProcess]] = []

    async def fake_create_subprocess_exec(*cmd, **_kwargs):
        proc = _FakeHungProcess()
        spawned.append((cmd, proc))
        return proc

    real_timeout = asyncio.timeout
    real_wait_for = asyncio.wait_for

    def fast_timeout(delay):
        return real_timeout(0.02 if delay is not None and delay >= 1 else delay)

    def fast_wait_for(aw, timeout=None):
        capped = 0.02 if timeout is not None and timeout >= 1 else timeout
        return real_wait_for(aw, timeout=capped)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "timeout", fast_timeout)
    monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)

    async def run():
        loaded, manifest, next_generation = _resume_bundle()
        first_generation = _generation()

        async def ready(_identifier):
            return True

        async def load(_manifest, _generation):
            return loaded

        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(tmp_path / "capture-cli"), "--manifest", str(tmp_path / "capture.json")),
            stop_file=tmp_path / "STOP",
            generation_file=tmp_path / "generation.json",
            environment={},
            ready=ready,
            resume_camera_id=CAMERA,
            resume_manifest_supported=True,
            resume_configuration_loader=load,
        )

        await driver.start(SCOPE, "0", first_generation)
        assert len(spawned) == 1
        hung_proc = spawned[0][1]
        assert await driver.is_running(SCOPE, first_generation["workerGenerationId"])

        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.stop(SCOPE)

        # Forced kill evidence and current generation identity must remain retained.
        assert hung_proc.kill_calls >= 1
        assert hung_proc.returncode is not None and hung_proc.returncode != 0
        assert driver.process is hung_proc
        assert driver._owned_generation == first_generation
        assert not await driver.is_running(SCOPE, first_generation["workerGenerationId"])

        # Retrying stop must fail closed cleanly and never clear the sticky reconciliation latch.
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.stop(SCOPE)
        assert driver.process is hung_proc
        assert driver._owned_generation == first_generation

        # Replacement via ordinary start or inference start_resume must remain blocked.
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.start(SCOPE, "0", _generation())
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.start_resume(SCOPE, "0", next_generation, manifest)
        assert len(spawned) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure_mode", ["kill_raises", "wait_cancelled"])
def test_failed_or_cancelled_forced_kill_retains_live_child_and_blocks_is_running_and_restart(
    tmp_path, monkeypatch, failure_mode
):
    class _FakeUnkillableProcess:
        def __init__(self):
            self.pid = 5151
            self.returncode = None
            self.kill_calls = 0

        def kill(self):
            self.kill_calls += 1
            if failure_mode == "kill_raises":
                raise PermissionError("Access denied")

        async def wait(self):
            if failure_mode == "wait_cancelled" and self.kill_calls > 0:
                raise asyncio.CancelledError()
            await asyncio.sleep(30)
            return self.returncode

    spawned: list[_FakeUnkillableProcess] = []

    async def fake_create_subprocess_exec(*_cmd, **_kwargs):
        proc = _FakeUnkillableProcess()
        spawned.append(proc)
        return proc

    real_timeout = asyncio.timeout
    real_wait_for = asyncio.wait_for

    def fast_timeout(delay):
        return real_timeout(0.02 if delay is not None and delay >= 1 else delay)

    def fast_wait_for(aw, timeout=None):
        capped = 0.02 if timeout is not None and timeout >= 1 else timeout
        return real_wait_for(aw, timeout=capped)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "timeout", fast_timeout)
    monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)

    async def run():
        loaded, manifest, next_generation = _resume_bundle()
        first_generation = _generation()

        async def ready(_identifier):
            return True

        async def load(_manifest, _generation):
            return loaded

        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(tmp_path / "capture-cli"), "--manifest", str(tmp_path / "capture.json")),
            resume_command=(
                str(tmp_path / "infer-cli"),
                "--manifest",
                str(tmp_path / "infer.json"),
            ),
            stop_file=tmp_path / "STOP",
            generation_file=tmp_path / "generation.json",
            environment={},
            ready=ready,
            resume_camera_id=CAMERA,
            resume_manifest_supported=True,
            resume_configuration_loader=load,
        )

        await driver.start(SCOPE, "0", first_generation)
        proc = spawned[0]
        assert await driver.is_running(SCOPE, first_generation["workerGenerationId"])

        expected_exc = PermissionError if failure_mode == "kill_raises" else asyncio.CancelledError
        with pytest.raises(expected_exc):
            await driver.stop(SCOPE)

        # Even if kill failed or wait was cancelled and child stayed alive (returncode is None),
        # reconciliation latch stays armed: is_running is False, evidence kept, restart blocked.
        assert proc.returncode is None
        assert driver.process is proc
        assert driver._owned_generation == first_generation
        assert not await driver.is_running(SCOPE, first_generation["workerGenerationId"])

        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.stop(SCOPE)
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.start(SCOPE, "0", _generation())
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await driver.start_resume(SCOPE, "0", next_generation, manifest)
        assert len(spawned) == 1

    asyncio.run(run())
