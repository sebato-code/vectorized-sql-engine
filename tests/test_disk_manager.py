"""Exhaustive unit test suite for DiskManager."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.storage.disk_manager import (
    PAGE_SIZE,
    DiskManager,
    DiskManagerClosedError,
    DiskManagerError,
    DiskManagerIOError,
    InvalidPageDataError,
    PageOutOfBoundsError,
)


class TestDiskManagerBasic:
    """Basic initialization and property checks."""

    def test_init_new_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_init.db"
        dm = DiskManager(db_path)
        try:
            assert dm.file_path == db_path
            assert dm.num_pages == 0
            assert dm.file_size == 0
            assert not dm.is_closed
            assert "pages=0" in repr(dm)
        finally:
            dm.close()

    def test_init_creates_nested_directories(self, tmp_path: Path) -> None:
        db_path = tmp_path / "deep" / "nested" / "dir" / "test.db"
        assert not db_path.parent.exists()

        dm = DiskManager(db_path)
        try:
            assert db_path.exists()
            assert dm.num_pages == 0
        finally:
            dm.close()

    def test_init_with_empty_or_invalid_path(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            DiskManager("")
        with pytest.raises(ValueError, match="cannot be empty"):
            DiskManager("   ")

    def test_init_with_directory_path(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            DiskManager(tmp_path)


class TestPageAllocation:
    """Tests for atomic page allocation."""

    def test_allocate_page_sequential(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_alloc.db"
        with DiskManager(db_path) as dm:
            p0 = dm.allocate_page()
            p1 = dm.allocate_page()
            p2 = dm.allocate_page()

            assert p0 == 0
            assert p1 == 1
            assert p2 == 2
            assert dm.num_pages == 3
            assert dm.file_size == 3 * PAGE_SIZE

    def test_allocated_page_initialized_to_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_zero.db"
        with DiskManager(db_path) as dm:
            pid = dm.allocate_page()
            data = dm.read_page(pid)
            assert data == b"\x00" * PAGE_SIZE

    def test_concurrent_page_allocation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_concurrent_alloc.db"
        num_threads = 10
        pages_per_thread = 20
        total_pages = num_threads * pages_per_thread

        with DiskManager(db_path) as dm:
            def allocate_worker() -> list[int]:
                allocated = []
                for _ in range(pages_per_thread):
                    allocated.append(dm.allocate_page())
                return allocated

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(allocate_worker) for _ in range(num_threads)]
                results = [f.result() for f in futures]

            all_pages = [pid for sublist in results for pid in sublist]
            assert len(all_pages) == total_pages
            assert len(set(all_pages)) == total_pages
            assert set(all_pages) == set(range(total_pages))
            assert dm.num_pages == total_pages
            assert dm.file_size == total_pages * PAGE_SIZE


class TestReadWritePages:
    """Tests for read and write operations."""

    def test_write_and_read_single_page(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_rw_single.db"
        with DiskManager(db_path) as dm:
            pid = dm.allocate_page()
            payload = b"A" * PAGE_SIZE
            dm.write_page(pid, payload)

            read_data = dm.read_page(pid)
            assert read_data == payload

    def test_write_and_read_multiple_pages(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_rw_multi.db"
        with DiskManager(db_path) as dm:
            payloads = {}
            for i in range(5):
                pid = dm.allocate_page()
                char = bytes([ord("A") + i])
                payload = char * PAGE_SIZE
                dm.write_page(pid, payload)
                payloads[pid] = payload

            # Read back in non-sequential order
            for pid in [4, 1, 3, 0, 2]:
                assert dm.read_page(pid) == payloads[pid]

    def test_overwrite_page(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_overwrite.db"
        with DiskManager(db_path) as dm:
            pid0 = dm.allocate_page()
            pid1 = dm.allocate_page()

            data_v1 = b"1" * PAGE_SIZE
            data_v2 = b"2" * PAGE_SIZE
            other_data = b"X" * PAGE_SIZE

            dm.write_page(pid0, data_v1)
            dm.write_page(pid1, other_data)

            assert dm.read_page(pid0) == data_v1

            dm.write_page(pid0, data_v2)
            assert dm.read_page(pid0) == data_v2
            assert dm.read_page(pid1) == other_data

    def test_write_with_bytearray_and_memoryview(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_buffer_types.db"
        with DiskManager(db_path) as dm:
            p0 = dm.allocate_page()
            p1 = dm.allocate_page()

            ba = bytearray(b"B" * PAGE_SIZE)
            mv = memoryview(b"M" * PAGE_SIZE)

            dm.write_page(p0, ba)
            dm.write_page(p1, mv)

            assert dm.read_page(p0) == bytes(ba)
            assert dm.read_page(p1) == bytes(mv)

    def test_write_binary_random_data(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_random_binary.db"
        random_bytes = os.urandom(PAGE_SIZE)
        with DiskManager(db_path) as dm:
            pid = dm.allocate_page()
            dm.write_page(pid, random_bytes)
            assert dm.read_page(pid) == random_bytes

    def test_concurrent_read_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_concurrent_rw.db"
        num_pages = 16

        with DiskManager(db_path) as dm:
            page_ids = [dm.allocate_page() for _ in range(num_pages)]

            def rw_worker(pid: int) -> bool:
                expected = bytes([(pid * 17 + i) % 256 for i in range(PAGE_SIZE)])
                dm.write_page(pid, expected)
                actual = dm.read_page(pid)
                return actual == expected

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(rw_worker, page_ids))

            assert all(results)


class TestPersistenceAndReopen:
    """Tests for file persistence and re-opening existing database files."""

    def test_reopen_existing_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_persist.db"
        data0 = b"0" * PAGE_SIZE
        data1 = b"1" * PAGE_SIZE

        # Write and close
        with DiskManager(db_path) as dm1:
            dm1.allocate_page()
            dm1.allocate_page()
            dm1.write_page(0, data0)
            dm1.write_page(1, data1)

        # Reopen and verify
        with DiskManager(db_path) as dm2:
            assert dm2.num_pages == 2
            assert dm2.file_size == 2 * PAGE_SIZE
            assert dm2.read_page(0) == data0
            assert dm2.read_page(1) == data1

            # Allocate new page on existing file
            p2 = dm2.allocate_page()
            assert p2 == 2
            assert dm2.num_pages == 3


class TestBoundsChecking:
    """Tests for file boundary checks on read and write."""

    def test_read_empty_file_out_of_bounds(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_empty_bounds.db"
        with DiskManager(db_path) as dm:
            with pytest.raises(PageOutOfBoundsError) as exc_info:
                dm.read_page(0)
            assert issubclass(PageOutOfBoundsError, (IndexError, ValueError))
            assert isinstance(exc_info.value, IndexError)

    def test_read_beyond_num_pages(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_read_bounds.db"
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            dm.allocate_page()

            # Within bounds: 0, 1
            dm.read_page(0)
            dm.read_page(1)

            # Exactly at boundary (num_pages = 2)
            with pytest.raises(PageOutOfBoundsError):
                dm.read_page(2)

            # Well beyond boundary
            with pytest.raises(PageOutOfBoundsError):
                dm.read_page(100)

    def test_read_negative_page_id(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_read_neg.db"
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            with pytest.raises(PageOutOfBoundsError):
                dm.read_page(-1)
            with pytest.raises(PageOutOfBoundsError):
                dm.read_page(-50)

    def test_write_beyond_num_pages(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_write_bounds.db"
        data = b"W" * PAGE_SIZE
        with DiskManager(db_path) as dm:
            dm.allocate_page()

            # Within bounds: 0
            dm.write_page(0, data)

            # At boundary (num_pages = 1)
            with pytest.raises(PageOutOfBoundsError):
                dm.write_page(1, data)

            # Beyond boundary
            with pytest.raises(PageOutOfBoundsError):
                dm.write_page(5, data)

    def test_write_negative_page_id(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_write_neg.db"
        data = b"W" * PAGE_SIZE
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            with pytest.raises(PageOutOfBoundsError):
                dm.write_page(-1, data)

    def test_invalid_page_id_types(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_types.db"
        data = b"X" * PAGE_SIZE
        with DiskManager(db_path) as dm:
            dm.allocate_page()

            for invalid_id in ["0", 1.5, None, True, False]:
                with pytest.raises(TypeError):
                    dm.read_page(invalid_id)  # type: ignore
                with pytest.raises(TypeError):
                    dm.write_page(invalid_id, data)  # type: ignore


class TestDataValidation:
    """Tests for validating page data payload."""

    def test_write_undersized_page_raises_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_undersized.db"
        with DiskManager(db_path) as dm:
            pid = dm.allocate_page()

            with pytest.raises(InvalidPageDataError) as exc_info:
                dm.write_page(pid, b"short")
            assert issubclass(InvalidPageDataError, ValueError)
            assert isinstance(exc_info.value, ValueError)

            with pytest.raises(InvalidPageDataError):
                dm.write_page(pid, b"")

            with pytest.raises(InvalidPageDataError):
                dm.write_page(pid, b"x" * (PAGE_SIZE - 1))

    def test_write_oversized_page_raises_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_oversized.db"
        with DiskManager(db_path) as dm:
            pid = dm.allocate_page()

            with pytest.raises(InvalidPageDataError):
                dm.write_page(pid, b"x" * (PAGE_SIZE + 1))

            with pytest.raises(InvalidPageDataError):
                dm.write_page(pid, b"x" * (PAGE_SIZE * 2))

    def test_write_invalid_data_type(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_invalid_datatype.db"
        with DiskManager(db_path) as dm:
            pid = dm.allocate_page()

            with pytest.raises(TypeError):
                dm.write_page(pid, "a" * PAGE_SIZE)  # type: ignore

            with pytest.raises(TypeError):
                dm.write_page(pid, [0] * PAGE_SIZE)  # type: ignore


class TestIOErrorsAndCorruption:
    """Tests for I/O error conditions, corruption detection, and fault handling."""

    def test_corrupted_file_not_aligned_to_page_size(self, tmp_path: Path) -> None:
        db_path = tmp_path / "corrupted.db"
        # Write unaligned data (e.g. 100 bytes)
        with open(db_path, "wb") as f:
            f.write(b"X" * 100)

        with pytest.raises(DiskManagerIOError, match="not a multiple of PAGE_SIZE"):
            DiskManager(db_path)

    def test_corrupted_file_partial_page(self, tmp_path: Path) -> None:
        db_path = tmp_path / "corrupted_partial.db"
        # Write 1 full page + 50 extra bytes
        with open(db_path, "wb") as f:
            f.write(b"X" * (PAGE_SIZE + 50))

        with pytest.raises(DiskManagerIOError, match="not a multiple of PAGE_SIZE"):
            DiskManager(db_path)

    def test_short_read_detection(self, tmp_path: Path) -> None:
        db_path = tmp_path / "short_read.db"
        dm = DiskManager(db_path)
        pid = dm.allocate_page()
        dm.close()

        # Externally truncate file to cause short read
        with open(db_path, "r+b") as f:
            f.truncate(100)

        # Force dm file to reopen or manually simulate short read
        dm2 = object.__new__(DiskManager)
        dm2._path = db_path
        dm2._is_closed = False
        dm2._num_pages = 1
        dm2._lock = MagicMock()
        dm2._lock.__enter__.return_value = None
        dm2._lock.__exit__.return_value = None
        dm2._file = open(db_path, "rb")

        try:
            with pytest.raises(DiskManagerIOError, match="short read"):
                dm2.read_page(pid)
        finally:
            dm2._file.close()

    def test_io_error_on_seek_or_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_io_err.db"
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            with patch.object(dm._file, "write", side_effect=OSError("Disk full")):
                with pytest.raises(DiskManagerIOError, match="Failed to write page"):
                    dm.write_page(0, b"x" * PAGE_SIZE)

    def test_io_error_on_read(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_read_io_err.db"
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            with patch.object(dm._file, "read", side_effect=OSError("Hardware I/O error")):
                with pytest.raises(DiskManagerIOError, match="Failed to read page"):
                    dm.read_page(0)

    def test_io_error_on_allocate(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_alloc_io_err.db"
        with DiskManager(db_path) as dm:
            with patch.object(dm._file, "write", side_effect=OSError("No space left on device")):
                with pytest.raises(DiskManagerIOError, match="Failed to allocate page"):
                    dm.allocate_page()

    def test_io_error_on_sync(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_sync_io_err.db"
        with DiskManager(db_path) as dm:
            with patch("os.fsync", side_effect=OSError("Sync failed")):
                with pytest.raises(DiskManagerIOError, match="Failed to sync disk file"):
                    dm.sync()


class TestLifecycleAndContextManager:
    """Tests for context management, close behavior, and post-close guards."""

    def test_context_manager(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_cm.db"
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            assert not dm.is_closed
        assert dm.is_closed

    def test_operations_after_close_raise_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_closed.db"
        dm = DiskManager(db_path)
        dm.allocate_page()
        dm.close()

        assert dm.is_closed
        assert "closed" in repr(dm)

        with pytest.raises(DiskManagerClosedError):
            dm.read_page(0)

        with pytest.raises(DiskManagerClosedError):
            dm.write_page(0, b"x" * PAGE_SIZE)

        with pytest.raises(DiskManagerClosedError):
            dm.allocate_page()

        with pytest.raises(DiskManagerClosedError):
            dm.sync()

        with pytest.raises(DiskManagerClosedError):
            with dm:
                pass

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_idempotent_close.db"
        dm = DiskManager(db_path)
        dm.allocate_page()
        dm.close()
        dm.close()
        dm.close()
        assert dm.is_closed

    def test_flush_alias_for_sync(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_flush.db"
        with DiskManager(db_path) as dm:
            dm.allocate_page()
            dm.flush()  # should not raise
