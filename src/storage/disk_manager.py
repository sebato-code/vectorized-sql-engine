"""DiskManager implementation for fixed-size page storage.

Manages reading, writing, and atomic allocation of 4096-byte pages on disk
with file bounds checking and robust I/O error handling.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Optional, Union

PAGE_SIZE: int = 4096


class DiskManagerError(Exception):
    """Base exception for all DiskManager errors."""


class PageOutOfBoundsError(DiskManagerError, IndexError, ValueError):
    """Raised when accessing a page_id that is outside valid file bounds."""


class DiskManagerIOError(DiskManagerError, OSError):
    """Raised when an I/O operation fails on the underlying disk file."""


class DiskManagerClosedError(DiskManagerError, RuntimeError):
    """Raised when an operation is performed on a closed DiskManager instance."""


class InvalidPageDataError(DiskManagerError, ValueError):
    """Raised when page data does not match the fixed PAGE_SIZE or is invalid."""


class DiskManager:
    """Manages low-level disk I/O operations with fixed-size pages.

    Features:
    - Fixed 4096-byte pages mapped linearly to disk offsets (page_id * 4096).
    - Thread-safe atomic page allocation (allocate_page).
    - Strict file boundary and argument validation.
    - Robust I/O error handling wrapping OS-level faults.
    - Context manager support for deterministic resource cleanup.
    """

    PAGE_SIZE: int = PAGE_SIZE

    def __init__(self, db_file: Union[str, Path]) -> None:
        """Initialize the DiskManager with the specified database file path.

        If the file does not exist, it is created. If it exists, its length
        must be an exact multiple of PAGE_SIZE.

        Args:
            db_file: Path to the database file.

        Raises:
            ValueError: If db_file path is empty.
            IsADirectoryError: If db_file points to an existing directory.
            DiskManagerIOError: If opening or validating the file fails due to I/O errors.
        """
        if not db_file or not str(db_file).strip():
            raise ValueError("Database file path cannot be empty.")

        self._path: Path = Path(db_file)
        self._lock: threading.RLock = threading.RLock()
        self._is_closed: bool = False

        try:
            if self._path.exists():
                if self._path.is_dir():
                    raise IsADirectoryError(f"Target path is a directory: {self._path}")
                self._file = open(self._path, "r+b")
            else:
                if self._path.parent and not self._path.parent.exists():
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self._path, "w+b")

            # Determine number of pages from existing file size
            self._file.seek(0, os.SEEK_END)
            file_size = self._file.tell()

            if file_size % self.PAGE_SIZE != 0:
                self._file.close()
                self._is_closed = True
                raise DiskManagerIOError(
                    f"Corrupted database file '{self._path}': size {file_size} "
                    f"is not a multiple of PAGE_SIZE ({self.PAGE_SIZE})"
                )

            self._num_pages: int = file_size // self.PAGE_SIZE

        except (IsADirectoryError, ValueError):
            raise
        except OSError as exc:
            raise DiskManagerIOError(f"Failed to open database file '{self._path}': {exc}") from exc

    @property
    def file_path(self) -> Path:
        """Return the path to the managed database file."""
        return self._path

    @property
    def num_pages(self) -> int:
        """Return the number of pages currently allocated in the database file."""
        with self._lock:
            return self._num_pages

    @property
    def file_size(self) -> int:
        """Return the logical size of the database file in bytes."""
        with self._lock:
            return self._num_pages * self.PAGE_SIZE

    @property
    def is_closed(self) -> bool:
        """Return True if the DiskManager is closed, False otherwise."""
        with self._lock:
            return self._is_closed or self._file.closed

    def _check_open(self) -> None:
        """Verify that the DiskManager is open and ready for operations.

        Raises:
            DiskManagerClosedError: If the manager has been closed.
        """
        if self._is_closed or self._file.closed:
            raise DiskManagerClosedError("DiskManager is closed.")

    def _validate_page_id(self, page_id: int) -> None:
        """Validate that page_id is a valid integer within allocated file bounds.

        Args:
            page_id: Page identifier to validate.

        Raises:
            TypeError: If page_id is not an integer.
            PageOutOfBoundsError: If page_id is negative or >= num_pages.
        """
        if not isinstance(page_id, int) or isinstance(page_id, bool):
            raise TypeError(f"page_id must be an integer, got {type(page_id).__name__}")

        if page_id < 0:
            raise PageOutOfBoundsError(f"page_id must be non-negative, got {page_id}")

        if page_id >= self._num_pages:
            raise PageOutOfBoundsError(
                f"page_id {page_id} is out of bounds (num_pages={self._num_pages})"
            )

    def allocate_page(self) -> int:
        """Atomically allocate a new page in the database file.

        Appends a new 4096-byte page initialized with zero bytes, flushes the
        write to disk, and increments the page count.

        Returns:
            The newly allocated page_id (0-indexed).

        Raises:
            DiskManagerClosedError: If the manager is closed.
            DiskManagerIOError: If an I/O error occurs while extending the file.
        """
        with self._lock:
            self._check_open()
            allocated_page_id = self._num_pages
            offset = allocated_page_id * self.PAGE_SIZE
            try:
                self._file.seek(offset)
                self._file.write(b"\x00" * self.PAGE_SIZE)
                self._file.flush()
                self._num_pages += 1
                return allocated_page_id
            except OSError as exc:
                raise DiskManagerIOError(
                    f"Failed to allocate page {allocated_page_id}: {exc}"
                ) from exc

    def read_page(self, page_id: int) -> bytes:
        """Read a fixed-size 4096-byte page from disk.

        Args:
            page_id: 0-indexed identifier of the page to read.

        Returns:
            bytes containing the 4096-byte page content.

        Raises:
            TypeError: If page_id is not an integer.
            PageOutOfBoundsError: If page_id < 0 or page_id >= num_pages.
            DiskManagerClosedError: If the manager is closed.
            DiskManagerIOError: If reading fails or an unexpected EOF is reached.
        """
        with self._lock:
            self._check_open()
            self._validate_page_id(page_id)
            offset = page_id * self.PAGE_SIZE
            try:
                self._file.seek(offset)
                data = self._file.read(self.PAGE_SIZE)
                if len(data) != self.PAGE_SIZE:
                    raise DiskManagerIOError(
                        f"Unexpected short read on page {page_id}: expected {self.PAGE_SIZE} "
                        f"bytes, got {len(data)}"
                    )
                return data
            except OSError as exc:
                raise DiskManagerIOError(
                    f"Failed to read page {page_id} at offset {offset}: {exc}"
                ) from exc

    def write_page(self, page_id: int, page_data: Union[bytes, bytearray, memoryview]) -> None:
        """Write a fixed-size 4096-byte page to disk.

        Args:
            page_id: 0-indexed identifier of the page to write.
            page_data: 4096-byte data buffer to write.

        Raises:
            TypeError: If page_id is not an integer or page_data is not bytes-like.
            InvalidPageDataError: If len(page_data) != PAGE_SIZE.
            PageOutOfBoundsError: If page_id < 0 or page_id >= num_pages.
            DiskManagerClosedError: If the manager is closed.
            DiskManagerIOError: If writing to disk fails.
        """
        if not isinstance(page_data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"page_data must be bytes, bytearray, or memoryview, got {type(page_data).__name__}"
            )

        if len(page_data) != self.PAGE_SIZE:
            raise InvalidPageDataError(
                f"page_data must be exactly {self.PAGE_SIZE} bytes, got {len(page_data)}"
            )

        data_to_write = bytes(page_data)

        with self._lock:
            self._check_open()
            self._validate_page_id(page_id)
            offset = page_id * self.PAGE_SIZE
            try:
                self._file.seek(offset)
                self._file.write(data_to_write)
                self._file.flush()
            except OSError as exc:
                raise DiskManagerIOError(
                    f"Failed to write page {page_id} at offset {offset}: {exc}"
                ) from exc

    def sync(self) -> None:
        """Flush internal buffers and force write to disk using os.fsync.

        Raises:
            DiskManagerClosedError: If the manager is closed.
            DiskManagerIOError: If syncing fails.
        """
        with self._lock:
            self._check_open()
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError as exc:
                raise DiskManagerIOError(f"Failed to sync disk file '{self._path}': {exc}") from exc

    def flush(self) -> None:
        """Alias for sync."""
        self.sync()

    def close(self) -> None:
        """Close the database file after syncing any pending writes.

        Safe to call multiple times.
        """
        with self._lock:
            if not self._is_closed:
                try:
                    if not self._file.closed:
                        try:
                            self._file.flush()
                            os.fsync(self._file.fileno())
                        except OSError:
                            pass
                        self._file.close()
                finally:
                    self._is_closed = True

    def __enter__(self) -> DiskManager:
        """Enter context manager."""
        self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close the DiskManager."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation of the DiskManager."""
        status = "closed" if self.is_closed else f"pages={self.num_pages}"
        return f"DiskManager(path={str(self._path)!r}, {status}, page_size={self.PAGE_SIZE})"
