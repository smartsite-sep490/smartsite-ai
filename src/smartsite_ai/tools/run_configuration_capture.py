"""Explicit bounded loopback host for model-free camera configuration images."""

import argparse
import asyncio
import json
import os
import signal
import socket
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from smartsite_ai.configuration_runtime import ConfigurationCaptureRuntime
from smartsite_ai.tools.configuration_host_config import (
    ConfigurationHostError,
    HostConfiguration,
    load_host_configuration,
)


def _emit(value: dict[str, object]) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def _runtime(config: HostConfiguration) -> ConfigurationCaptureRuntime:
    return ConfigurationCaptureRuntime(
        stream=config.stream,
        source_kind=config.source_kind,
        resource_id=config.resource_id,
        lock_directory=config.lock_directory,
        service_token=config.token,
    )


class _OperatorServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        # The CLI requests server exit itself. Uvicorn must not re-raise a signal
        # before the independent capture worker has drained its native resources.
        yield


def _server(app: FastAPI) -> uvicorn.Server:
    return _OperatorServer(
        uvicorn.Config(app, host="127.0.0.1", log_level="error", access_log=False, lifespan="off")
    )


async def _drain_owned(operation: Awaitable[bool]) -> tuple[bool, bool]:
    cancelled = False
    task = asyncio.ensure_future(operation)
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.cancelled():
                return False, True
            cancelled = True


async def serve_host(
    config: HostConfiguration,
    *,
    runtime_factory: Callable[[HostConfiguration], ConfigurationCaptureRuntime] = _runtime,
    server_factory: Callable[[FastAPI], uvicorn.Server] = _server,
    emit: Callable[[dict[str, object]], None] = _emit,
    stop_requested: asyncio.Event | None = None,
) -> dict[str, object]:
    """Bind first, explicitly start once and prove native release before returning."""
    if config.duration_seconds > 3600 and stop_requested is None:
        raise ConfigurationHostError("CONFIGURATION_HOST_INVALID_CONFIG")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", config.port))
        listener.setblocking(False)
    except OSError:
        listener.close()
        raise ConfigurationHostError("CONFIGURATION_HOST_BIND_UNAVAILABLE") from None
    port = listener.getsockname()[1]
    runtime, server, watcher = None, None, None
    failure, cancelled = False, False
    termination_reason = "server_returned"
    result: dict[str, object] = {}
    try:
        emit({"state": "CONFIGURATION_HOST_BOUND_NOT_READY", "port": port})
        runtime = runtime_factory(config)
        server = server_factory(runtime.app)
        await runtime.start()

        async def watch() -> None:
            nonlocal termination_reason
            deadline = asyncio.get_running_loop().time() + config.duration_seconds
            while not server.should_exit:
                if runtime.state not in ("IDLE", "RUNNING"):
                    termination_reason = "capture_terminal"
                elif stop_requested is not None and stop_requested.is_set():
                    termination_reason = "operator_stop"
                elif asyncio.get_running_loop().time() >= deadline:
                    termination_reason = "deadline_elapsed"
                else:
                    await asyncio.sleep(0.025)
                    continue
                server.should_exit = True
                return

        watcher = asyncio.create_task(watch())
        await server.serve(sockets=[listener])
        failure = runtime.state == "FAILED"
    except asyncio.CancelledError:
        termination_reason = "host_cancelled"
        cancelled = True
        raise
    except (Exception, SystemExit):
        termination_reason = "host_failed"
        raise ConfigurationHostError("CONFIGURATION_HOST_SERVER_FAILED") from None
    finally:
        # Snapshot before stop changes lifecycle state; only known codes may reach logs.
        capture_state = runtime.state if runtime is not None else None
        capture_reason = runtime.reason if runtime is not None else None
        if capture_state not in (None, "IDLE", "RUNNING", "STOPPING", "STOPPED", "FAILED"):
            capture_state = "UNKNOWN"
        if capture_reason not in (
            None,
            "source_ended",
            "frame_identity_invalid",
            "image_encoding_failed",
            "capture_cancelled",
            "source_failed",
            "source_release_failed",
        ):
            capture_reason = "unknown"
        if server is not None:
            server.should_exit = True
        if watcher is not None:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        try:
            released = True
            if runtime is not None:
                released, stop_cancelled = await _drain_owned(runtime.stop())
                cancelled = cancelled or stop_cancelled
            if not released:
                emit({"state": "CONFIGURATION_HOST_DRAIN_PENDING", "nativeReleaseProved": False})
                released, drain_cancelled = await _drain_owned(runtime.wait_for_release())
                cancelled = cancelled or drain_cancelled
            result = {
                "state": "CONFIGURATION_HOST_STOPPED"
                if released
                else "CONFIGURATION_HOST_STOP_UNRESOLVED",
                "port": port,
                "nativeReleaseProved": released,
                "terminationReason": termination_reason,
                "captureStateBeforeStop": capture_state,
                "captureReasonBeforeStop": capture_reason,
            }
            emit(result)
            if not released:
                raise ConfigurationHostError("CONFIGURATION_HOST_STOP_UNRESOLVED")
            current = asyncio.current_task()
            if cancelled or (current is not None and current.cancelling()):
                raise asyncio.CancelledError
        finally:
            listener.close()
    if failure:
        raise ConfigurationHostError("CONFIGURATION_HOST_CAPTURE_FAILED")
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationHostError("CONFIGURATION_HOST_INVALID_CONFIG")


def _validated_stop_file(path: Path | None, manifest: Path) -> Path | None:
    if path is None:
        return None
    try:
        if (
            not path.is_absolute()
            or path.name != "STOP"
            or path.parent.resolve(strict=True) != manifest.parent.resolve(strict=True)
            or path.is_symlink()
            or path.exists()
        ):
            raise ValueError("Fresh operator-local STOP path required")
        return path.parent.resolve(strict=True) / "STOP"
    except (OSError, ValueError):
        raise ConfigurationHostError("CONFIGURATION_HOST_INVALID_CONFIG") from None


async def _watch_stop_file(path: Path, requested: asyncio.Event) -> None:
    try:
        while not requested.is_set():
            if path.exists():
                requested.set()
                return
            await asyncio.sleep(0.25)
    except OSError:
        requested.set()
        raise ConfigurationHostError("CONFIGURATION_HOST_INVALID_CONFIG") from None


async def _command(config: HostConfiguration, *, stop_file: Path | None = None) -> None:
    loop, requested = asyncio.get_running_loop(), asyncio.Event()
    previous = {}
    watcher = None
    try:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is not None:
                previous[number] = signal.signal(
                    number, lambda _number, _frame: loop.call_soon_threadsafe(requested.set)
                )
        if stop_file is not None:
            watcher = asyncio.create_task(_watch_stop_file(stop_file, requested))
        await serve_host(config, stop_requested=requested)
    finally:
        try:
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--stop-file", type=Path)
        args = parser.parse_args(argv)
        stop_file = _validated_stop_file(args.stop_file, args.manifest)
        config = (
            load_host_configuration(args.manifest, os.environ, controlled_stop=True)
            if stop_file
            else load_host_configuration(args.manifest, os.environ)
        )
        asyncio.run(_command(config, stop_file=stop_file) if stop_file else _command(config))
    except ConfigurationHostError as error:
        _emit({"state": "CONFIGURATION_HOST_FAILED", "code": error.code})
        return 1
    except KeyboardInterrupt:
        _emit({"state": "CONFIGURATION_HOST_FAILED", "code": "CONFIGURATION_HOST_INTERRUPTED"})
        return 130
    except Exception:
        _emit({"state": "CONFIGURATION_HOST_FAILED", "code": "CONFIGURATION_HOST_SERVER_FAILED"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
