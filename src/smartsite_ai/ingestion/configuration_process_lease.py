"""Cooperating-process exclusion for one explicitly mapped physical-source UUID."""

import os
import stat
from pathlib import Path
from uuid import UUID


class ConfigurationProcessLease:
    """Nonblocking OS lock, retained until explicit proven source release.

    All aliases of a physical source must use the same UUID and private local
    directory. This is not a hardware alias detector or a distributed lock.
    Never delete/replace these lock files while an owner may still be alive.
    No destructor releases a possibly still-owned native resource.
    """

    def __init__(self, directory: Path, resource_id: UUID) -> None:
        if not directory.is_absolute() or not isinstance(resource_id, UUID):
            raise ValueError("An absolute local lock directory and resource UUID are required")
        if str(directory).startswith("\\\\"):
            raise ValueError("Configuration source locks require a local directory")
        self._directory = directory
        self._filename = f"configuration-capture-{resource_id}.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        """Acquire before constructing a source; never wait or truncate a file."""
        if self._fd is not None:
            raise RuntimeError("Configuration source process ownership is unavailable")
        descriptor: int | None = None
        try:
            directory = self._directory.resolve(strict=True)
            if str(directory).startswith("\\\\") or not directory.is_dir():
                raise OSError("Lock parent is not a directory")
            path = directory / self._filename
            if path.is_symlink():
                raise OSError("Lock path must not be a symbolic link")
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != 0:
                raise OSError("Lock path must be a dedicated empty file")
            if os.name == "nt":
                import msvcrt

                # Windows explicitly supports locking a region beyond EOF.
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            elif os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise OSError("No supported local file lock implementation")
            current = path.lstat()
            if stat.S_ISLNK(current.st_mode) or (opened.st_dev, opened.st_ino) != (
                current.st_dev,
                current.st_ino,
            ):
                raise OSError("Lock path changed during acquisition")
        except (OSError, RuntimeError):
            if descriptor is not None:
                os.close(descriptor)
            raise RuntimeError("Configuration source process ownership is unavailable") from None
        self._fd = descriptor

    def release(self) -> None:
        """Only the capture lifecycle may call this after proven native release."""
        if self._fd is not None:
            descriptor, self._fd = self._fd, None
            os.close(descriptor)
