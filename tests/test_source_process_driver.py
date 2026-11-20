import asyncio
import json
import sys
from pathlib import Path

from smartsite_ai.source_process_driver import SourceProcessDriver


def test_owned_process_is_stopped_before_replacement_without_database_environment(tmp_path):
    async def run():
        output = tmp_path / "ready.json"
        stop = tmp_path / "stop"
        generation = tmp_path / "generation.json"
        generation.write_text(json.dumps({"workerGenerationId": "previous"}))
        child = tmp_path / "child.py"
        child.write_text(
            "import json,os,time,pathlib\n"
            f"out=pathlib.Path({str(output)!r}); stop=pathlib.Path({str(stop)!r})\n"
            "assert 'DATABASE_URL' not in os.environ\n"
            "out.write_text(os.environ['SMARTSITE_AI_SOURCE_RUNTIME_GENERATION'])\n"
            "while not stop.exists(): time.sleep(.02)\n"
        )

        async def ready(identifier):
            return (
                output.exists()
                and json.loads(output.read_text())["workerGenerationId"] == identifier
            )

        driver = SourceProcessDriver(
            scope=("site", "camera"),
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop,
            generation_file=generation,
            environment={},
            ready=ready,
        )
        assert await driver.generation_id(("site", "camera")) == "previous"
        await driver.start(
            ("site", "camera"),
            "rtsp://secret-password@192.168.1.2/live",
            {"workerGenerationId": "next"},
        )
        process = driver.process
        try:
            assert await driver.is_running(("site", "camera"), "next")
            assert driver.current_source is not None
        finally:
            await driver.stop(("site", "camera"))
        assert process.returncode == 0
        assert driver.process is None and driver.current_source is None

    asyncio.run(run())
