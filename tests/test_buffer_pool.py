"""Exhaustive unit test suite for BufferPoolManager."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.storage.buffer_pool import (
    BufferPoolError,
    BufferPoolFullError,
    BufferPoolManager,
    ClockReplacer,
    LRUReplacer,
    Page,
    PageNotFoundError,
    PagePinnedError,
    Replacer,
)
from src.storage.disk_manager import (
    PAGE_SIZE,
    DiskManager,
    PageOutOfBoundsError,
)


@pytest.fixture
def disk_manager(tmp_path: Path) -> DiskManager:
    """Fixture providing a fresh DiskManager with 10 pre-allocated pages."""
    db_file = tmp_path / "test_bpm.db"
    dm = DiskManager(db_file)
    for i in range(10):
        pid = dm.allocate_page()
        # Initialize each page with a distinctive pattern
        pattern = f"Page_{pid:04d}_initial_data".encode().ljust(PAGE_SIZE, b"\x00")
        dm.write_page(pid, pattern)
    return dm


class TestPageModel:
    """Tests for Page data structure."""

    def test_page_initialization_defaults(self) -> None:
        p = Page()
        assert p.page_id == -1
        assert p.pin_count == 0
        assert not p.is_dirty
        assert len(p.data) == PAGE_SIZE
        assert p.get_data() == b"\x00" * PAGE_SIZE

    def test_page_initialization_with_data(self) -> None:
        data = b"X" * PAGE_SIZE
        p = Page(page_id=42, data=data)
        assert p.page_id == 42
        assert p.pin_count == 0
        assert not p.is_dirty
        assert p.get_data() == data

    def test_page_invalid_data_size(self) -> None:
        with pytest.raises(ValueError, match="Page data must be 4096 bytes"):
            Page(page_id=1, data=b"short")

    def test_page_set_data(self) -> None:
        p = Page(page_id=1)
        new_data = b"Y" * PAGE_SIZE
        p.set_data(new_data, mark_dirty=True)
        assert p.get_data() == new_data
        assert p.is_dirty

    def test_page_set_data_without_mark_dirty(self) -> None:
        p = Page(page_id=1)
        new_data = b"Z" * PAGE_SIZE
        p.set_data(new_data, mark_dirty=False)
        assert p.get_data() == new_data
        assert not p.is_dirty

    def test_page_reset_memory(self) -> None:
        p = Page(page_id=5, data=b"A" * PAGE_SIZE)
        p.pin_count = 3
        p.is_dirty = True
        p.reset_memory()
        assert p.page_id == -1
        assert p.pin_count == 0
        assert not p.is_dirty
        assert p.get_data() == b"\x00" * PAGE_SIZE

    def test_page_repr(self) -> None:
        p = Page(page_id=7)
        assert "Page(id=7, pin_count=0, dirty=False)" in repr(p)


class TestReplacers:
    """Tests for eviction replacer algorithms."""

    def test_clock_replacer_basic(self) -> None:
        replacer = ClockReplacer(pool_size=3)
        assert replacer.size() == 0
        assert replacer.victim() is None

        # Unpin frames 0, 1, 2
        replacer.unpin(0)
        replacer.unpin(1)
        replacer.unpin(2)
        assert replacer.size() == 3

        # First sweep: ref_bits are cleared from True to False, then victim chosen
        v0 = replacer.victim()
        assert v0 == 0
        assert replacer.size() == 2

        v1 = replacer.victim()
        assert v1 == 1
        assert replacer.size() == 1

        v2 = replacer.victim()
        assert v2 == 2
        assert replacer.size() == 0
        assert replacer.victim() is None

    def test_clock_replacer_second_chance(self) -> None:
        replacer = ClockReplacer(pool_size=3)
        replacer.unpin(0)
        replacer.unpin(1)

        # Pin frame 0 (now in use, not evictable)
        replacer.pin(0)
        assert replacer.size() == 1

        # Only frame 1 is evictable
        assert replacer.victim() == 1
        assert replacer.size() == 0
        assert replacer.victim() is None

    def test_clock_replacer_invalid_pool_size(self) -> None:
        with pytest.raises(ValueError):
            ClockReplacer(pool_size=0)
        with pytest.raises(ValueError):
            ClockReplacer(pool_size=-1)

    def test_lru_replacer_order(self) -> None:
        replacer = LRUReplacer(pool_size=3)
        replacer.unpin(0)
        replacer.unpin(1)
        replacer.unpin(2)

        # Unpin 0 again to make it most recently used
        replacer.unpin(0)

        # Least recently used should be 1, then 2, then 0
        assert replacer.victim() == 1
        assert replacer.victim() == 2
        assert replacer.victim() == 0
        assert replacer.victim() is None

    def test_lru_replacer_pin(self) -> None:
        replacer = LRUReplacer(pool_size=3)
        replacer.unpin(0)
        replacer.unpin(1)
        replacer.pin(0)
        assert replacer.size() == 1
        assert replacer.victim() == 1


class TestBufferPoolManagerBasic:
    """Basic initialization and property checks."""

    def test_bpm_initialization(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=5)
        assert bpm.pool_size == 5
        assert bpm.num_free_frames == 5
        assert len(bpm.page_table) == 0
        assert bpm.disk_manager is disk_manager
        assert "pool_size=5" in repr(bpm)

    def test_bpm_invalid_pool_size(self, disk_manager: DiskManager) -> None:
        with pytest.raises(ValueError, match="pool_size must be positive"):
            BufferPoolManager(disk_manager, pool_size=0)
        with pytest.raises(ValueError, match="pool_size must be positive"):
            BufferPoolManager(disk_manager, pool_size=-3)

    def test_bpm_invalid_replacer(self, disk_manager: DiskManager) -> None:
        with pytest.raises(ValueError, match="Unsupported replacer policy"):
            BufferPoolManager(disk_manager, pool_size=5, replacer="unknown_replacer")

    def test_bpm_lru_replacer_config(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3, replacer="lru")
        assert isinstance(bpm.replacer, LRUReplacer)


class TestFetchAndUnpinPage:
    """Acceptance Criterion 1: fetch_page and unpin_page manage pin_count and dirty flag."""

    def test_fetch_page_increments_pin_count(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)

        # First fetch: loads from disk, pin_count = 1
        p0 = bpm.fetch_page(0)
        assert p0.page_id == 0
        assert p0.pin_count == 1
        assert not p0.is_dirty
        assert bpm.num_free_frames == 2

        # Second fetch of same page: cached, increments pin_count to 2
        p0_again = bpm.fetch_page(0)
        assert p0_again is p0
        assert p0.pin_count == 2
        assert bpm.num_free_frames == 2

        # Third fetch: increments to 3
        bpm.fetch_page(0)
        assert p0.pin_count == 3

    def test_unpin_page_decrements_pin_count(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        bpm.fetch_page(0)
        bpm.fetch_page(0)
        p0 = bpm.fetch_page(0)
        assert p0.pin_count == 3

        # First unpin
        ok = bpm.unpin_page(0, is_dirty=False)
        assert ok is True
        assert p0.pin_count == 2
        assert not p0.is_dirty

        # Second unpin
        ok = bpm.unpin_page(0, is_dirty=False)
        assert ok is True
        assert p0.pin_count == 1
        assert not p0.is_dirty

        # Third unpin: pin_count reaches 0
        ok = bpm.unpin_page(0, is_dirty=False)
        assert ok is True
        assert p0.pin_count == 0
        assert not p0.is_dirty

    def test_unpin_page_marks_dirty(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        p0 = bpm.fetch_page(0)
        assert not p0.is_dirty

        # Modify page data and unpin with is_dirty=True
        p0.data[:5] = b"HELLO"
        ok = bpm.unpin_page(0, is_dirty=True)
        assert ok is True
        assert p0.pin_count == 0
        assert p0.is_dirty is True

    def test_unpin_preserves_dirty_flag_when_is_dirty_is_false(self, disk_manager: DiskManager) -> None:
        """If page was already marked dirty, subsequent unpin with is_dirty=False retains dirty."""
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        p0 = bpm.fetch_page(0)
        bpm.fetch_page(0)
        assert p0.pin_count == 2

        # First unpin marks dirty
        bpm.unpin_page(0, is_dirty=True)
        assert p0.pin_count == 1
        assert p0.is_dirty is True

        # Second unpin with is_dirty=False MUST NOT reset dirty flag to False
        bpm.unpin_page(0, is_dirty=False)
        assert p0.pin_count == 0
        assert p0.is_dirty is True

    def test_unpin_page_already_unpinned(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        p0 = bpm.fetch_page(0)
        bpm.unpin_page(0)
        assert p0.pin_count == 0

        # Unpinning again when pin_count is 0 should return False
        ok = bpm.unpin_page(0)
        assert ok is False
        assert p0.pin_count == 0

    def test_unpin_nonexistent_page(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        ok = bpm.unpin_page(999, is_dirty=False)
        assert ok is False

    def test_fetch_and_unpin_type_validation(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        with pytest.raises(TypeError):
            bpm.fetch_page("0")  # type: ignore
        with pytest.raises(TypeError):
            bpm.fetch_page(True)  # type: ignore
        with pytest.raises(TypeError):
            bpm.unpin_page("0")  # type: ignore


class TestEvictionPolicyAndDiskFlush:
    """Acceptance Criterion 2: Eviction only evicts pin_count == 0 and flushes dirty to disk."""

    def test_eviction_only_targets_unpinned_pages(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        # Pin page 0 (pin_count=1)
        p0 = bpm.fetch_page(0)
        # Fetch page 1 and then unpin it (pin_count=0)
        p1 = bpm.fetch_page(1)
        bpm.unpin_page(1, is_dirty=False)

        assert p0.pin_count == 1
        assert p1.pin_count == 0

        # Now fetch page 2: pool is full, but page 1 has pin_count == 0
        p2 = bpm.fetch_page(2)
        assert p2.page_id == 2

        # Page 0 must STILL be present and pinned in buffer pool
        assert 0 in bpm.page_table
        # Page 1 must have been evicted
        assert 1 not in bpm.page_table
        assert 2 in bpm.page_table

    def test_eviction_flushes_dirty_victim_to_disk(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        # Fetch page 0, modify it, and unpin with is_dirty=True
        p0 = bpm.fetch_page(0)
        updated_data = b"MODIFIED_PAGE_0_DATA".ljust(PAGE_SIZE, b"\x00")
        p0.set_data(updated_data, mark_dirty=True)
        bpm.unpin_page(0, is_dirty=True)

        # Fetch page 1 and unpin it (clean)
        bpm.fetch_page(1)
        bpm.unpin_page(1, is_dirty=False)

        # Both pages 0 and 1 are in pool with pin_count == 0
        # Now fetch page 2 and page 3 to force eviction of both pages 0 and 1
        bpm.fetch_page(2)
        bpm.unpin_page(2, is_dirty=False)
        bpm.fetch_page(3)

        # Page 0 has been evicted from memory
        assert 0 not in bpm.page_table

        # Verify disk has the updated data for page 0!
        disk_data = disk_manager.read_page(0)
        assert disk_data == updated_data

    def test_eviction_does_not_write_clean_pages_to_disk(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=1)

        # Fetch page 0 and unpin without modifying (clean)
        bpm.fetch_page(0)
        bpm.unpin_page(0, is_dirty=False)

        # Spy on disk_manager.write_page
        with patch.object(disk_manager, "write_page", wraps=disk_manager.write_page) as mock_write:
            # Fetch page 1, evicting clean page 0
            bpm.fetch_page(1)
            mock_write.assert_not_called()

    def test_roundtrip_persistence_after_eviction(self, disk_manager: DiskManager) -> None:
        """Page modified, evicted to disk, then re-fetched matches modified content."""
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        p0 = bpm.fetch_page(0)
        marker = b"ROUNDTRIP_PERSISTENCE_TEST_BYTES".ljust(PAGE_SIZE, b"\x01")
        p0.set_data(marker, mark_dirty=True)
        bpm.unpin_page(0, is_dirty=True)

        # Evict page 0 by fetching pages 1 and 2
        bpm.fetch_page(1)
        bpm.unpin_page(1)
        bpm.fetch_page(2)
        bpm.unpin_page(2)

        assert 0 not in bpm.page_table

        # Re-fetch page 0: must reload from disk with marker content
        p0_reloaded = bpm.fetch_page(0)
        assert p0_reloaded.get_data() == marker

    def test_clock_replacer_order_in_eviction(self, disk_manager: DiskManager) -> None:
        """Verify Clock sweep second-chance behavior during frame eviction."""
        bpm = BufferPoolManager(disk_manager, pool_size=3, replacer="clock")

        # Load 0, 1, 2
        bpm.fetch_page(0)
        bpm.fetch_page(1)
        bpm.fetch_page(2)

        bpm.unpin_page(0)
        bpm.unpin_page(1)
        bpm.unpin_page(2)

        # Evict one: with clock starting at 0, ref_bits cleared, page 0 evicted
        bpm.fetch_page(3)
        assert 0 not in bpm.page_table
        assert 3 in bpm.page_table

    def test_lru_replacer_order_in_eviction(self, disk_manager: DiskManager) -> None:
        """Verify LRU replacer evicts least recently unpinned page."""
        bpm = BufferPoolManager(disk_manager, pool_size=3, replacer="lru")

        bpm.fetch_page(0)
        bpm.fetch_page(1)
        bpm.fetch_page(2)

        # Unpin in order: 0, 1, 2
        bpm.unpin_page(0)
        bpm.unpin_page(1)
        bpm.unpin_page(2)

        # Re-fetch page 0 and unpin it to make it the most recently used
        bpm.fetch_page(0)
        bpm.unpin_page(0)

        # Evict: page 1 should be evicted first (least recently used among 1, 2, 0)
        bpm.fetch_page(3)
        assert 1 not in bpm.page_table
        assert 0 in bpm.page_table
        assert 2 in bpm.page_table
        assert 3 in bpm.page_table


class TestBufferPoolFullError:
    """Acceptance Criterion 3: BufferPoolFullError when all frames are pinned."""

    def test_fetch_page_raises_when_all_frames_pinned(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        # Pin all frames
        bpm.fetch_page(0)
        bpm.fetch_page(1)

        # All 2 frames are pinned with pin_count == 1
        with pytest.raises(BufferPoolFullError, match="all 2 frames are pinned"):
            bpm.fetch_page(2)

    def test_new_page_raises_when_all_frames_pinned(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        bpm.fetch_page(0)
        bpm.fetch_page(1)

        with pytest.raises(BufferPoolFullError, match="all 2 frames are pinned"):
            bpm.new_page()

    def test_cached_pages_still_fetchable_when_pool_full(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        p0 = bpm.fetch_page(0)
        p1 = bpm.fetch_page(1)

        # Attempting to fetch an uncached page raises
        with pytest.raises(BufferPoolFullError):
            bpm.fetch_page(2)

        # But already cached pages can still be fetched (incrementing pin_count)
        p0_again = bpm.fetch_page(0)
        assert p0_again is p0
        assert p0.pin_count == 2

    def test_unpinning_one_page_allows_subsequent_fetch(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=2)

        bpm.fetch_page(0)
        bpm.fetch_page(1)

        with pytest.raises(BufferPoolFullError):
            bpm.fetch_page(2)

        # Unpin page 1
        bpm.unpin_page(1)

        # Now fetch_page(2) should succeed by evicting page 1
        p2 = bpm.fetch_page(2)
        assert p2.page_id == 2
        assert 1 not in bpm.page_table
        assert 2 in bpm.page_table


class TestFlushAndPersistence:
    """Tests for explicit flush operations."""

    def test_flush_page_writes_to_disk_and_clears_dirty(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        p0 = bpm.fetch_page(0)
        new_content = b"EXPLICIT_FLUSH_DATA".ljust(PAGE_SIZE, b"\x02")
        p0.set_data(new_content, mark_dirty=True)

        assert p0.is_dirty

        # Explicitly flush page 0
        ok = bpm.flush_page(0)
        assert ok is True
        assert not p0.is_dirty  # Dirty flag cleared
        assert p0.pin_count == 1  # Still pinned in pool

        # Verify disk received data
        assert disk_manager.read_page(0) == new_content

    def test_flush_page_not_in_pool(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        ok = bpm.flush_page(999)
        assert ok is False

    def test_flush_all_pages(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)

        p0 = bpm.fetch_page(0)
        p1 = bpm.fetch_page(1)
        p0.set_data(b"PAGE_0_DIRTY".ljust(PAGE_SIZE, b"\x00"), mark_dirty=True)
        p1.set_data(b"PAGE_1_DIRTY".ljust(PAGE_SIZE, b"\x00"), mark_dirty=True)

        bpm.flush_all_pages()

        assert not p0.is_dirty
        assert not p1.is_dirty
        assert disk_manager.read_page(0) == b"PAGE_0_DIRTY".ljust(PAGE_SIZE, b"\x00")
        assert disk_manager.read_page(1) == b"PAGE_1_DIRTY".ljust(PAGE_SIZE, b"\x00")

    def test_context_manager_flushes_on_exit(self, disk_manager: DiskManager) -> None:
        with BufferPoolManager(disk_manager, pool_size=3) as bpm:
            p0 = bpm.fetch_page(0)
            p0.set_data(b"CONTEXT_MANAGER_FLUSH".ljust(PAGE_SIZE, b"\x00"), mark_dirty=True)
            bpm.unpin_page(0, is_dirty=True)

        # On exit, all dirty pages are flushed
        assert disk_manager.read_page(0) == b"CONTEXT_MANAGER_FLUSH".ljust(PAGE_SIZE, b"\x00")


class TestNewAndDeletePage:
    """Tests for new_page and delete_page operations."""

    def test_new_page_allocates_and_pins(self, disk_manager: DiskManager) -> None:
        initial_num_pages = disk_manager.num_pages
        bpm = BufferPoolManager(disk_manager, pool_size=3)

        newPage = bpm.new_page()
        assert newPage.page_id == initial_num_pages
        assert newPage.pin_count == 1
        assert not newPage.is_dirty
        assert newPage.get_data() == b"\x00" * PAGE_SIZE
        assert disk_manager.num_pages == initial_num_pages + 1
        assert newPage.page_id in bpm.page_table

    def test_delete_page_success(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        p0 = bpm.fetch_page(0)
        bpm.unpin_page(0)

        # Page 0 is in pool and unpinned
        assert 0 in bpm.page_table
        ok = bpm.delete_page(0)
        assert ok is True
        assert 0 not in bpm.page_table
        assert bpm.num_free_frames == 3

    def test_delete_pinned_page_fails(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        bpm.fetch_page(0)  # Pinned

        ok = bpm.delete_page(0)
        assert ok is False
        assert 0 in bpm.page_table

    def test_delete_nonexistent_page(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        ok = bpm.delete_page(999)
        assert ok is True

    def test_get_page_helper(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        assert bpm.get_page(0) is None

        p0 = bpm.fetch_page(0)
        cached = bpm.get_page(0)
        assert cached is p0
        # get_page does not modify pin_count
        assert cached.pin_count == 1


class TestConcurrency:
    """Thread safety tests for BufferPoolManager."""

    def test_concurrent_fetches_and_unpins(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=5)
        num_threads = 10
        iterations = 50

        def worker(thread_idx: int) -> None:
            for it in range(iterations):
                pid = (thread_idx + it) % 10
                try:
                    page = bpm.fetch_page(pid)
                    # Read/write small portion
                    page.data[0] = (page.data[0] + 1) % 256
                    bpm.unpin_page(pid, is_dirty=True)
                except BufferPoolFullError:
                    # Expected occasionally under heavy contention if all frames pinned
                    pass

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        # Check internal consistency
        bpm.flush_all_pages()
        for pid in range(10):
            data = disk_manager.read_page(pid)
            assert len(data) == PAGE_SIZE


class TestErrorHandling:
    """Tests for exception and error scenarios."""

    def test_fetch_invalid_page_bounds(self, disk_manager: DiskManager) -> None:
        bpm = BufferPoolManager(disk_manager, pool_size=3)
        initial_free = bpm.num_free_frames

        # Request page beyond disk bounds
        with pytest.raises(PageOutOfBoundsError):
            bpm.fetch_page(9999)

        # Ensure no frame was leaked
        assert bpm.num_free_frames == initial_free
        assert 9999 not in bpm.page_table
