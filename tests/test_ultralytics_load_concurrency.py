"""Exercise provider guard ownership without real models or provider imports."""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from smartsite_ai.inference import ultralytics_runner as runner


@pytest.mark.parametrize("first_fails", [False, True])
def test_overlapping_loads_preserve_offline_guards(monkeypatch, first_fails) -> None:
    modules = {}
    for name in (
        "ultralytics",
        "ultralytics.nn",
        "ultralytics.nn.tasks",
        "ultralytics.utils",
        "ultralytics.utils.checks",
        "ultralytics.utils.downloads",
    ):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    tasks = modules["ultralytics.nn.tasks"]
    utils = modules["ultralytics.utils"]
    checks = modules["ultralytics.utils.checks"]
    downloads = modules["ultralytics.utils.downloads"]

    def original_download(*args, **kwargs):
        raise AssertionError("network must remain unavailable during construction")

    def original_requirements(*args, **kwargs):
        assert kwargs.get("install") is False
        return True

    downloads.attempt_download_asset = original_download
    tasks.check_requirements = checks.check_requirements = original_requirements
    utils.AUTOINSTALL = checks.AUTOINSTALL = True
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.setattr(runner, "_require_exact_local_file", lambda path: None)
    started = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    second_attempted = threading.Event()

    def construct(path):
        index = 0 if path.name == "a.pt" else 1
        started[index].set()
        assert release[index].wait(3), "test did not release the constructor"
        if index == 0 and first_fails:
            raise RuntimeError("first checkpoint failed")
        return object()

    modules["ultralytics"].YOLO = construct

    def second_load():
        second_attempted.set()
        return runner._load_exact_ultralytics_model(Path("b.pt"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runner._load_exact_ultralytics_model, Path("a.pt"))
        try:
            assert started[0].wait(1)
            second = pool.submit(second_load)
            assert second_attempted.wait(1)
            assert not started[1].wait(0.1), "constructors overlap global provider guards"
            release[0].set()
            if first_fails:
                with pytest.raises(RuntimeError, match="checkpoint failed"):
                    first.result(timeout=1)
            else:
                first.result(timeout=1)
            assert started[1].wait(1)
            assert downloads.attempt_download_asset is not original_download
            assert checks.check_requirements is not original_requirements
            assert tasks.check_requirements is checks.check_requirements
            assert not utils.AUTOINSTALL and not checks.AUTOINSTALL
            assert os.environ["YOLO_AUTOINSTALL"] == "false"
            assert checks.check_requirements("example") is True
            release[1].set()
            second.result(timeout=1)
        finally:
            for event in release:
                event.set()

    assert downloads.attempt_download_asset is original_download
    assert tasks.check_requirements is checks.check_requirements is original_requirements
    assert utils.AUTOINSTALL and checks.AUTOINSTALL
    assert os.environ["YOLO_AUTOINSTALL"] == "true"
