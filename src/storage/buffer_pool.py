"""BufferPoolManager implementation for in-memory page frame management.

Provides fixed-capacity buffer pool caching between disk and memory, supporting:
- Frame allocation and eviction using Clock sweep or LRU replacers.
- Page pinning / unpinning with dirty page tracking.
- Dirty page flushing to disk via DiskManager.
- Concurrency protection and bounds validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

from src.storage.disk_manager import PAGE_SIZE, DiskManager, DiskManagerError


class BufferPoolError(Exception):
    """Base exception for all BufferPoolManager errors."""


class BufferPoolFullError(BufferPoolError, RuntimeError):
    """Raised when all frames in the buffer pool are pinned and none can be evicted."""


class PageNotFoundError(BufferPoolError, KeyError):
    """Raised when an operation references a page not present in the buffer pool."""


class PagePinnedError(BufferPoolError, RuntimeError):
    """Raised when an invalid operation is performed on a pinned page."""


class Page:
    """Represents an in-memory page frame.

    Attributes:
        page_id: Identifier of the page on disk (-1 if unassigned).
        pin_count: Number of concurrent threads/users currently pinning this page.
        is_dirty: True if in-memory content has been modified and differs from disk.
        data: Bytearray of size PAGE_SIZE holding the raw page content.
    """

    def __init__(
        self,
        page_id: int = -1,
        data: Optional[Union[bytes, bytearray, memoryview]] = None,
    ) -> None:
        """Initialize an in-memory Page frame.

        Args:
            page_id: 0-indexed page ID, or -1 if the frame is empty.
            data: Optional initial page data (must be PAGE_SIZE bytes if provided).
        """
        self.page_id: int = page_id
        self.pin_count: int = 0
        self.is_dirty: bool = False

        if data is None:
            self.data: bytearray = bytearray(PAGE_SIZE)
        else:
            if len(data) != PAGE_SIZE:
                raise ValueError(f"Page data must be {PAGE_SIZE} bytes, got {len(data)}")
            self.data = bytearray(data)

    def get_data(self) -> bytes:
        """Return a read-only bytes snapshot of the page data."""
        return bytes(self.data)

    def set_data(
        self,
        new_data: Union[bytes, bytearray, memoryview],
        mark_dirty: bool = True,
    ) -> None:
        """Replace page content and optionally mark the page dirty.

        Args:
            new_data: New page data of length PAGE_SIZE.
            mark_dirty: If True, mark the page as dirty.
        """
        if len(new_data) != PAGE_SIZE:
            raise ValueError(f"Page data must be exactly {PAGE_SIZE} bytes, got {len(new_data)}")
        self.data[:] = new_data
        if mark_dirty:
            self.is_dirty = True

    def reset_memory(self) -> None:
        """Clear page contents and reset state to unassigned."""
        self.page_id = -1
        self.pin_count = 0
        self.is_dirty = False
        self.data = bytearray(PAGE_SIZE)

    def __repr__(self) -> str:
        return (
            f"Page(id={self.page_id}, pin_count={self.pin_count}, "
            f"dirty={self.is_dirty})"
        )


class Replacer(ABC):
    """Abstract base class for buffer pool frame eviction policies."""

    @abstractmethod
    def victim(self) -> Optional[int]:
        """Select and remove a victim frame eligible for eviction.

        Returns:
            The selected frame_id, or None if no frames are eligible for eviction.
        """

    @abstractmethod
    def pin(self, frame_id: int) -> None:
        """Mark a frame as pinned, removing it from eviction candidacy."""

    @abstractmethod
    def unpin(self, frame_id: int) -> None:
        """Mark a frame as unpinned (pin_count == 0), making it an eviction candidate."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of frames currently eligible for eviction."""

    def __len__(self) -> int:
        return self.size()


class ClockReplacer(Replacer):
    """Clock sweep (Second Chance) frame eviction policy.

    Maintains a circular buffer of frame usage bits. The clock hand sweeps
    through candidate frames whose pin_count is 0:
    - If the reference bit is 1, it is cleared to 0 (second chance).
    - If the reference bit is 0, the frame is chosen as the victim.
    """

    def __init__(self, pool_size: int) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        self._pool_size: int = pool_size
        self._in_replacer: List[bool] = [False] * pool_size
        self._ref_bits: List[bool] = [False] * pool_size
        self._hand: int = 0
        self._count: int = 0

    def victim(self) -> Optional[int]:
        """Find victim frame using Clock sweep policy."""
        if self._count == 0:
            return None

        # Sweep circularly until a frame with ref_bit == False is found
        while True:
            curr = self._hand
            self._hand = (self._hand + 1) % self._pool_size

            if self._in_replacer[curr]:
                if self._ref_bits[curr]:
                    # Second chance: clear bit and advance hand
                    self._ref_bits[curr] = False
                else:
                    # Found victim with ref_bit == False
                    self._in_replacer[curr] = False
                    self._ref_bits[curr] = False
                    self._count -= 1
                    return curr

    def pin(self, frame_id: int) -> None:
        """Remove frame from replacer (it is now pinned)."""
        if 0 <= frame_id < self._pool_size and self._in_replacer[frame_id]:
            self._in_replacer[frame_id] = False
            self._ref_bits[frame_id] = False
            self._count -= 1

    def unpin(self, frame_id: int) -> None:
        """Add frame to replacer (pin_count == 0)."""
        if 0 <= frame_id < self._pool_size:
            if not self._in_replacer[frame_id]:
                self._in_replacer[frame_id] = True
                self._count += 1
            self._ref_bits[frame_id] = True

    def size(self) -> int:
        return self._count


class LRUReplacer(Replacer):
    """Least Recently Used (LRU) frame eviction policy.

    Tracks unpinned frames in order of last access / unpin.
    The least recently unpinned frame is evicted first.
    """

    def __init__(self, pool_size: int) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        self._pool_size: int = pool_size
        self._entries: OrderedDict[int, None] = OrderedDict()

    def victim(self) -> Optional[int]:
        """Evict least recently used unpinned frame."""
        if not self._entries:
            return None
        victim_frame, _ = self._entries.popitem(last=False)
        return victim_frame

    def pin(self, frame_id: int) -> None:
        """Remove frame from LRU tracking."""
        self._entries.pop(frame_id, None)

    def unpin(self, frame_id: int) -> None:
        """Add or update frame in LRU tracking."""
        if 0 <= frame_id < self._pool_size:
            if frame_id in self._entries:
                self._entries.move_to_end(frame_id)
            else:
                self._entries[frame_id] = None

    def size(self) -> int:
        return len(self._entries)


class BufferPoolManager:
    """Manages a pool of in-memory page frames backed by a DiskManager.

    Responsibilities:
    - Maintain a fixed number of in-memory frames.
    - Translate disk page IDs to in-memory frame indices via a page table.
    - Track page pin counts and dirty flags.
    - Evict unpinned frames (pin_count == 0) using Clock sweep or LRU replacer.
    - Write dirty pages back to disk via DiskManager upon eviction or explicit flush.
    - Thread-safe concurrency control across all operations.
    """

    def __init__(
        self,
        disk_manager: DiskManager,
        pool_size: int = 10,
        replacer: Union[str, Replacer, None] = "clock",
    ) -> None:
        """Initialize BufferPoolManager.

        Args:
            disk_manager: Instance of DiskManager used for disk page I/O.
            pool_size: Number of page frames to maintain in memory.
            replacer: Eviction policy - "clock" (default), "lru", or custom Replacer instance.

        Raises:
            ValueError: If pool_size <= 0 or replacer name is invalid.
        """
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive integer, got {pool_size}")

        self._disk_manager: DiskManager = disk_manager
        self._pool_size: int = pool_size
        self._lock: threading.RLock = threading.RLock()

        # In-memory frames and routing table
        self._frames: List[Page] = [Page() for _ in range(pool_size)]
        self._page_table: Dict[int, int] = {}  # page_id -> frame_id
        self._free_list: List[int] = list(range(pool_size))

        # Eviction replacer setup
        if replacer is None or (isinstance(replacer, str) and replacer.lower() == "clock"):
            self._replacer: Replacer = ClockReplacer(pool_size)
        elif isinstance(replacer, str) and replacer.lower() == "lru":
            self._replacer = LRUReplacer(pool_size)
        elif isinstance(replacer, Replacer):
            self._replacer = replacer
        else:
            raise ValueError(f"Unsupported replacer policy: {replacer}")

    @property
    def pool_size(self) -> int:
        """Return total frame capacity of the buffer pool."""
        return self._pool_size

    @property
    def disk_manager(self) -> DiskManager:
        """Return underlying DiskManager instance."""
        return self._disk_manager

    @property
    def replacer(self) -> Replacer:
        """Return current eviction replacer instance."""
        return self._replacer

    @property
    def num_free_frames(self) -> int:
        """Return number of currently unused frames."""
        with self._lock:
            return len(self._free_list)

    @property
    def page_table(self) -> Dict[int, int]:
        """Return a copy of the current page_id -> frame_id mapping."""
        with self._lock:
            return dict(self._page_table)

    def _find_available_frame(self) -> Optional[int]:
        """Find an available frame from free list or evict an unpinned frame.

        Must be called with self._lock held.

        Returns:
            frame_id to use, or None if all frames are pinned.
        """
        # 1. Prefer unallocated frames from the free list
        if self._free_list:
            return self._free_list.pop(0)

        # 2. Select an unpinned victim frame via eviction replacer
        victim_frame_id = self._replacer.victim()
        if victim_frame_id is None:
            return None

        victim_page = self._frames[victim_frame_id]

        # 3. If victim is dirty, flush it to disk first
        if victim_page.is_dirty:
            self._disk_manager.write_page(victim_page.page_id, victim_page.data)
            victim_page.is_dirty = False

        # 4. Remove victim mapping from page table
        if victim_page.page_id in self._page_table:
            del self._page_table[victim_page.page_id]

        return victim_frame_id

    def fetch_page(self, page_id: int) -> Page:
        """Fetch a page into memory by page_id and pin it.

        If the page is already cached in memory, increments its pin_count.
        Otherwise, acquires a frame (evicting an unpinned page if necessary),
        reads page data from disk, sets pin_count to 1, and marks it clean.

        Args:
            page_id: 0-indexed page ID to fetch.

        Returns:
            The Page frame holding the requested page.

        Raises:
            TypeError: If page_id is not an integer.
            BufferPoolFullError: If all frames in the buffer pool are pinned.
            DiskManagerError: If reading the page from disk fails.
        """
        if not isinstance(page_id, int) or isinstance(page_id, bool):
            raise TypeError(f"page_id must be an integer, got {type(page_id).__name__}")

        with self._lock:
            # Case 1: Page is already cached in buffer pool
            if page_id in self._page_table:
                frame_id = self._page_table[page_id]
                page = self._frames[frame_id]
                page.pin_count += 1
                self._replacer.pin(frame_id)
                return page

            # Case 2: Page needs to be loaded into an available frame
            frame_id = self._find_available_frame()
            if frame_id is None:
                raise BufferPoolFullError(
                    f"BufferPool is full: all {self._pool_size} frames are pinned; "
                    f"cannot fetch page {page_id}."
                )

            # Read page data from disk
            try:
                data = self._disk_manager.read_page(page_id)
            except Exception:
                # On read error, restore frame to free list and propagate
                self._free_list.append(frame_id)
                raise

            # Populate frame
            page = self._frames[frame_id]
            page.page_id = page_id
            page.data = bytearray(data)
            page.pin_count = 1
            page.is_dirty = False

            self._page_table[page_id] = frame_id
            self._replacer.pin(frame_id)
            return page

    def unpin_page(self, page_id: int, is_dirty: bool = False) -> bool:
        """Unpin a page in the buffer pool.

        Decrements page pin_count. If is_dirty is True, marks the page dirty.
        When pin_count drops to 0, notifies replacer that frame can be evicted.

        Args:
            page_id: Page identifier to unpin.
            is_dirty: If True, mark page as dirty (does not clear dirty if False).

        Returns:
            True if unpinned successfully, False if page not in pool or already pin_count == 0.
        """
        if not isinstance(page_id, int) or isinstance(page_id, bool):
            raise TypeError(f"page_id must be an integer, got {type(page_id).__name__}")

        with self._lock:
            if page_id not in self._page_table:
                return False

            frame_id = self._page_table[page_id]
            page = self._frames[frame_id]

            if page.pin_count <= 0:
                return False

            if is_dirty:
                page.is_dirty = True

            page.pin_count -= 1
            if page.pin_count == 0:
                self._replacer.unpin(frame_id)

            return True

    def flush_page(self, page_id: int) -> bool:
        """Flush a page to disk regardless of pin_count if it is dirty.

        Args:
            page_id: Identifier of page to flush.

        Returns:
            True if page was flushed (or clean), False if page not in buffer pool.

        Raises:
            DiskManagerError: If disk write fails.
        """
        with self._lock:
            if page_id not in self._page_table:
                return False

            frame_id = self._page_table[page_id]
            page = self._frames[frame_id]

            if page.is_dirty:
                self._disk_manager.write_page(page.page_id, page.data)
                page.is_dirty = False

            return True

    def flush_all_pages(self) -> None:
        """Flush all dirty pages currently in the buffer pool to disk."""
        with self._lock:
            for page_id, frame_id in list(self._page_table.items()):
                page = self._frames[frame_id]
                if page.is_dirty:
                    self._disk_manager.write_page(page.page_id, page.data)
                    page.is_dirty = False
            self._disk_manager.flush()

    def new_page(self) -> Page:
        """Allocate a new page on disk and load it into the buffer pool.

        Returns:
            The newly created Page frame, with pin_count=1 and is_dirty=False.

        Raises:
            BufferPoolFullError: If all frames are pinned.
            DiskManagerError: If disk allocation fails.
        """
        with self._lock:
            frame_id = self._find_available_frame()
            if frame_id is None:
                raise BufferPoolFullError(
                    f"BufferPool is full: all {self._pool_size} frames are pinned; "
                    "cannot allocate new page."
                )

            try:
                page_id = self._disk_manager.allocate_page()
                data = self._disk_manager.read_page(page_id)
            except Exception:
                self._free_list.append(frame_id)
                raise

            page = self._frames[frame_id]
            page.page_id = page_id
            page.data = bytearray(data)
            page.pin_count = 1
            page.is_dirty = False

            self._page_table[page_id] = frame_id
            self._replacer.pin(frame_id)
            return page

    def delete_page(self, page_id: int) -> bool:
        """Delete a page from the buffer pool and release its frame.

        Args:
            page_id: Identifier of page to remove.

        Returns:
            True if removed (or page was not present), False if page is pinned.
        """
        with self._lock:
            if page_id not in self._page_table:
                return True

            frame_id = self._page_table[page_id]
            page = self._frames[frame_id]

            if page.pin_count > 0:
                return False

            self._replacer.pin(frame_id)
            del self._page_table[page_id]
            page.reset_memory()
            self._free_list.append(frame_id)
            return True

    def get_page(self, page_id: int) -> Optional[Page]:
        """Look up a cached page in the buffer pool without altering its pin_count.

        Args:
            page_id: Identifier of page.

        Returns:
            Page object if cached, None otherwise.
        """
        with self._lock:
            frame_id = self._page_table.get(page_id)
            if frame_id is not None:
                return self._frames[frame_id]
            return None

    def close(self) -> None:
        """Flush all dirty pages to disk."""
        self.flush_all_pages()

    def __enter__(self) -> BufferPoolManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"BufferPoolManager(pool_size={self._pool_size}, "
                f"cached_pages={len(self._page_table)}, "
                f"free_frames={len(self._free_list)}, "
                f"evictable={self._replacer.size()})"
            )
