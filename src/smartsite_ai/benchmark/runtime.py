"""Concrete local replay and resource adapters for the benchmark harness."""

from __future__ import annotations

import ctypes
import importlib.metadata
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from ctypes import wintypes
from pathlib import Path
from time import perf_counter, process_time
from types import ModuleType
from typing import Any, cast

from smartsite_ai.benchmark.multistream import BenchmarkError, DecodedFrame, ReplaySourceFacts

_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})


class OpenCvReplaySource:
    """Decode and loop one finite local video without rendering or writing output."""

    def __init__(self, path: Path) -> None:
        resolved = _regular_local_video(path)
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - selected runtime extra
            raise BenchmarkError("OpenCV runtime is unavailable") from error
        capture = cv2.VideoCapture(str(resolved))
        if not capture.isOpened():
            capture.release()
            raise BenchmarkError(f"could not open benchmark video {resolved.name}")
        try:
            width = _positive_integer(capture.get(cv2.CAP_PROP_FRAME_WIDTH), "video width")
            height = _positive_integer(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), "video height")
            frame_count = _positive_integer(
                capture.get(cv2.CAP_PROP_FRAME_COUNT), "video frame count"
            )
            fps = _positive_float(capture.get(cv2.CAP_PROP_FPS), "video FPS")
        except Exception:
            capture.release()
            raise
        self._path = resolved
        self._cv2 = cv2
        self._capture = capture
        self._facts = ReplaySourceFacts(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
        self._closed = False

    @property
    def facts(self) -> ReplaySourceFacts:
        return self._facts

    def read(self) -> DecodedFrame:
        if self._closed:
            raise BenchmarkError("benchmark video source is closed")
        started = perf_counter()
        ok, frame = self._capture.read()
        if not ok:
            if not self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0.0):
                raise BenchmarkError(f"could not rewind benchmark video {self._path.name}")
            ok, frame = self._capture.read()
        if not ok or frame is None:
            raise BenchmarkError(f"could not decode benchmark video {self._path.name}")
        shape = getattr(frame, "shape", None)
        if shape != (self._facts.height, self._facts.width, 3):
            raise BenchmarkError("decoded video frame shape changed during benchmark")
        try:
            payload = bytes(frame.tobytes())
        except Exception as error:
            raise BenchmarkError("could not pack decoded frame as BGR24") from error
        elapsed = max(perf_counter() - started, 0.0)
        return DecodedFrame(payload=payload, decode_seconds=elapsed)

    def skip(self, frame_count: int) -> None:
        if self._closed:
            raise BenchmarkError("benchmark video source is closed")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise BenchmarkError("skip frame count must be a non-negative integer")
        for _ in range(frame_count):
            if self._capture.grab():
                continue
            if not self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0.0):
                raise BenchmarkError(f"could not rewind benchmark video {self._path.name}")
            if not self._capture.grab():
                raise BenchmarkError(f"could not skip benchmark video {self._path.name}")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()


class TorchSynchronizer:
    """Synchronize the selected CUDA device at explicit measurement boundaries."""

    def __init__(self, torch_module: ModuleType, device_index: int | None) -> None:
        self._torch = torch_module
        self._device_index = device_index

    def synchronize(self) -> None:
        if self._device_index is not None:
            self._torch.cuda.synchronize(self._device_index)


class ProcessResourceMonitor:
    """Sample process RSS/CPU and report synchronized Torch allocator VRAM."""

    def __init__(self, torch_module: ModuleType, device_index: int | None) -> None:
        self._torch = torch_module
        self._device_index = device_index
        self._started_wall: float | None = None
        self._started_cpu: float | None = None
        self._start_rss = 0
        self._peak_rss = 0

    def start(self) -> None:
        if self._started_wall is not None:
            raise BenchmarkError("resource monitor cannot be started twice")
        if self._device_index is not None:
            self._torch.cuda.synchronize(self._device_index)
            self._torch.cuda.reset_peak_memory_stats(self._device_index)
        self._start_rss = _current_rss_bytes()
        self._peak_rss = self._start_rss
        self._started_cpu = process_time()
        self._started_wall = perf_counter()

    def sample(self) -> None:
        if self._started_wall is None:
            raise BenchmarkError("resource monitor has not started")
        self._peak_rss = max(self._peak_rss, _current_rss_bytes())

    def finish(self) -> Mapping[str, object]:
        if self._started_wall is None or self._started_cpu is None:
            raise BenchmarkError("resource monitor has not started")
        self.sample()
        wall_seconds = max(perf_counter() - self._started_wall, 0.0)
        cpu_seconds = max(process_time() - self._started_cpu, 0.0)
        end_rss = _current_rss_bytes()
        self._peak_rss = max(self._peak_rss, end_rss)
        vram: dict[str, object] = {
            "available": False,
            "measurementScope": "torchAllocatorOnly",
            "deviceIndices": [],
            "aggregateAllocatedBytes": 0,
            "aggregateReservedBytes": 0,
            "peakAllocatedBytes": 0,
            "peakReservedBytes": 0,
        }
        if self._device_index is not None:
            vram.update(
                available=True,
                deviceIndices=[self._device_index],
                aggregateAllocatedBytes=int(self._torch.cuda.memory_allocated(self._device_index)),
                aggregateReservedBytes=int(self._torch.cuda.memory_reserved(self._device_index)),
                peakAllocatedBytes=int(self._torch.cuda.max_memory_allocated(self._device_index)),
                peakReservedBytes=int(self._torch.cuda.max_memory_reserved(self._device_index)),
            )
        return {
            "measurementScope": "fixed-scheduled-window",
            "measurementWallSeconds": round(wall_seconds, 6),
            "rss": {
                "startBytes": self._start_rss,
                "endBytes": end_rss,
                "peakSampledBytes": self._peak_rss,
            },
            "cpu": {
                "processSeconds": round(cpu_seconds, 6),
                "averageProcessPercentOneCoreScale": round(
                    100.0 * cpu_seconds / wall_seconds if wall_seconds > 0 else 0.0,
                    6,
                ),
                "percentScale": "one-logical-core-equals-100-percent",
                "logicalCpuCount": os.cpu_count(),
            },
            "vram": vram,
        }


def resolve_torch_device(torch_module: ModuleType, configured: str) -> int | None:
    """Resolve the device used by the provider for synchronization and VRAM metrics."""

    normalized = configured.casefold()
    if normalized == "cpu":
        return None
    if normalized == "auto":
        return 0 if bool(torch_module.cuda.is_available()) else None
    if normalized in {"cuda", "0"}:
        index = 0
    elif normalized.startswith("cuda:"):
        index = int(normalized.split(":", 1)[1])
    elif normalized.isdecimal():
        index = int(normalized)
    else:
        raise BenchmarkError(f"unsupported benchmark device selector {configured!r}")
    if not bool(torch_module.cuda.is_available()) or index >= int(torch_module.cuda.device_count()):
        raise BenchmarkError(f"CUDA device {index} is unavailable")
    return index


def runtime_facts(torch_module: ModuleType) -> dict[str, object]:
    packages: dict[str, str] = {}
    for package in ("opencv-python", "torch", "torchvision", "ultralytics"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cudaRuntime": getattr(torch_module.version, "cuda", None),
    }


def hardware_facts(torch_module: ModuleType) -> dict[str, object]:
    cuda_devices: list[dict[str, object]] = []
    if bool(torch_module.cuda.is_available()):
        driver_versions = _nvidia_driver_versions()
        for index in range(int(torch_module.cuda.device_count())):
            properties = torch_module.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": str(torch_module.cuda.get_device_name(index)),
                    "totalMemoryBytes": int(properties.total_memory),
                    "driverVersion": driver_versions.get(index),
                }
            )
    return {
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "logicalCpuCount": os.cpu_count(),
        "physicalMemoryBytes": _physical_memory_bytes(),
        "cudaDevices": cuda_devices,
    }


def git_facts(code_directory: Path) -> dict[str, object]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(code_directory), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    if sha.returncode != 0 or status.returncode != 0:
        return {"available": False, "commitSha": None, "dirty": None}
    return {
        "available": True,
        "commitSha": sha.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def _nvidia_driver_versions() -> dict[int, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    versions: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdecimal() or not fields[1]:
            continue
        versions[int(fields[0])] = fields[1]
    return versions


def _regular_local_video(path: Path) -> Path:
    if not path.is_absolute():
        raise BenchmarkError("benchmark video path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BenchmarkError("benchmark video does not exist") from error
    if (
        path.is_symlink()
        or not resolved.is_file()
        or resolved.suffix.casefold() not in _VIDEO_SUFFIXES
    ):
        raise BenchmarkError("benchmark video must be a supported regular local file")
    return resolved


def _positive_float(value: object, label: str) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"{label} is invalid") from error
    if not math.isfinite(number) or number <= 0:
        raise BenchmarkError(f"{label} is invalid")
    return number


def _positive_integer(value: object, label: str) -> int:
    number = _positive_float(value, label)
    if not number.is_integer():
        raise BenchmarkError(f"{label} is invalid")
    return int(number)


def _current_rss_bytes() -> int:
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise BenchmarkError("could not read process RSS")
        return int(counters.WorkingSetSize)
    proc_statm = Path("/proc/self/statm")
    if proc_statm.is_file():
        try:
            resident_pages = int(proc_statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError) as error:
            raise BenchmarkError("could not read process RSS") from error
    raise BenchmarkError("current process RSS is unavailable on this platform")


def _physical_memory_bytes() -> int | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(MemoryStatus),)
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


__all__ = [
    "OpenCvReplaySource",
    "ProcessResourceMonitor",
    "TorchSynchronizer",
    "git_facts",
    "hardware_facts",
    "resolve_torch_device",
    "runtime_facts",
]
