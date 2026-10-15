"""Actual filesystem locks on the test directory; never a camera/model process."""

import importlib
import importlib.util
import os
import subprocess
import sys
from unittest.mock import patch
from uuid import UUID

import pytest

RESOURCE = UUID("e98f6475-0660-4c71-a466-4c93d1739ee0")


def lease(directory):
    name = "smartsite_ai.ingestion.configuration_process_lease"
    assert importlib.util.find_spec(name) is not None, "Configuration process ownership is missing"
    return importlib.import_module(name).ConfigurationProcessLease(directory, RESOURCE)


def test_nonblocking_same_process_contention_and_release_preserve_lock_inode(tmp_path):
    first, other = lease(tmp_path), lease(tmp_path)
    assert list(tmp_path.iterdir()) == []
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="ownership"):
            other.acquire()
        path = next(tmp_path.iterdir())
        assert path.stat().st_size == 0
        original = path.stat().st_ino
    finally:
        first.release()
    other.acquire()
    other.release()
    other.release()
    assert path.stat().st_ino == original


def test_actual_second_python_process_is_refused_then_can_acquire_after_release(tmp_path):
    owner = lease(tmp_path)
    code = """
from pathlib import Path
from uuid import UUID
import sys
from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
guard = ConfigurationProcessLease(Path(sys.argv[1]), UUID(sys.argv[2]))
try:
    guard.acquire()
except RuntimeError:
    print('BUSY')
else:
    guard.release()
    print('OWNED')
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def child():
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path), str(RESOURCE)],
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip()

    owner.acquire()
    try:
        assert child() == "BUSY"
    finally:
        owner.release()
    assert child() == "OWNED"


def test_guard_refuses_missing_directory_without_creating_it(tmp_path):
    guard = lease(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="ownership"):
        guard.acquire()
    assert not (tmp_path / "missing").exists()


def test_network_directory_after_resolution_is_refused_before_opening_any_file(tmp_path):
    from pathlib import Path

    guard = lease(tmp_path)
    resolved = type("ResolvedNetworkDirectory", (), {"__str__": lambda _: "\\\\server\\share"})()
    with (
        patch.object(Path, "resolve", return_value=resolved),
        patch("os.open") as opener,
        pytest.raises(RuntimeError, match="ownership"),
    ):
        guard.acquire()
    opener.assert_not_called()


def test_refuses_nonempty_lock_file_without_modifying_its_contents(tmp_path):
    path = tmp_path / f"configuration-capture-{RESOURCE}.lock"
    path.write_bytes(b"unrelated preserved file")
    with pytest.raises(RuntimeError, match="ownership"):
        lease(tmp_path).acquire()
    assert path.read_bytes() == b"unrelated preserved file"
