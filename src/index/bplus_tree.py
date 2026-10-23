"""BPlusTree index implementation backed by BufferPoolManager pages.

Provides a disk-backed B+ Tree supporting:
- Logarithmic exact search by key.
- Insertion with automatic node splitting (splits) for both leaf and internal nodes.
- Sequential range scan (range_scan) traversing pointers between leaf pages.
- Memory and frame management through BufferPoolManager page pinning and unpinning.
"""

from __future__ import annotations

import bisect
from enum import IntEnum
from pathlib import Path
import pickle
import struct
import threading
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

from src.storage.buffer_pool import BufferPoolManager, Page
from src.storage.disk_manager import PAGE_SIZE

HEADER_FORMAT: str = ">4s B H q q q H I"
HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)  # 37 bytes
MAGIC_BYTES: bytes = b"BPT1"


class BPlusTreeError(Exception):
    """Base exception for all BPlusTree errors."""


class DuplicateKeyError(BPlusTreeError, KeyError):
    """Raised when inserting a duplicate key when replace=False."""


class KeyNotFoundError(BPlusTreeError, KeyError):
    """Raised when accessing a non-existent key via __getitem__ or __delitem__."""


class InvalidNodeError(BPlusTreeError, ValueError):
    """Raised when serialized page data is invalid or corrupted."""


class NodeType(IntEnum):
    """Node classification in the B+ Tree."""

    LEAF = 0
    INTERNAL = 1


class BPlusTreeNode:
    """In-memory representation of a B+ Tree node serialized to a BufferPool page frame.

    Page binary layout (4096 bytes):
    - Offset 0  (4 bytes): Magic bytes b"BPT1"
    - Offset 4  (1 byte) : Node type (0 = LEAF, 1 = INTERNAL)
    - Offset 5  (2 bytes): num_keys (unsigned short)
    - Offset 7  (8 bytes): parent_page_id (signed int64, -1 if root)
    - Offset 15 (8 bytes): next_page_id (signed int64, -1 if none; leaf chain)
    - Offset 23 (8 bytes): prev_page_id (signed int64, -1 if none; leaf chain)
    - Offset 31 (2 bytes): max_keys (unsigned short)
    - Offset 33 (4 bytes): payload_length (unsigned int32)
    - Offset 37 (N bytes): pickled payload (keys + values for leaf; keys + children for internal)
    - Remainder          : Zero padded up to PAGE_SIZE
    """

    def __init__(
        self,
        page_id: int,
        is_leaf: bool,
        parent_page_id: int = -1,
        next_page_id: int = -1,
        prev_page_id: int = -1,
        keys: Optional[List[Any]] = None,
        values: Optional[List[Any]] = None,
        children: Optional[List[int]] = None,
        max_keys: int = 32,
    ) -> None:
        self.page_id: int = page_id
        self.is_leaf: bool = is_leaf
        self.parent_page_id: int = parent_page_id
        self.next_page_id: int = next_page_id
        self.prev_page_id: int = prev_page_id
        self.keys: List[Any] = keys if keys is not None else []
        self.values: List[Any] = values if values is not None else []
        self.children: List[int] = children if children is not None else []
        self.max_keys: int = max_keys

    @property
    def node_type(self) -> NodeType:
        return NodeType.LEAF if self.is_leaf else NodeType.INTERNAL

    def to_bytes(self, buffer: Union[bytearray, memoryview]) -> None:
        """Serialize node metadata and entries into the provided page buffer."""
        if self.is_leaf:
            payload = pickle.dumps((self.keys, self.values), protocol=5)
        else:
            payload = pickle.dumps((self.keys, self.children), protocol=5)

        total_length = HEADER_SIZE + len(payload)
        if total_length > PAGE_SIZE:
            raise InvalidNodeError(
                f"Serialized node {self.page_id} size {total_length} exceeds PAGE_SIZE {PAGE_SIZE}"
            )

        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_BYTES,
            1 if self.is_leaf else 0,
            len(self.keys),
            self.parent_page_id,
            self.next_page_id,
            self.prev_page_id,
            self.max_keys,
            len(payload),
        )
        buffer[:HEADER_SIZE] = header
        buffer[HEADER_SIZE:total_length] = payload
        buffer[total_length:PAGE_SIZE] = b"\x00" * (PAGE_SIZE - total_length)

    @classmethod
    def from_bytes(
        cls, page_id: int, data: Union[bytes, bytearray, memoryview]
    ) -> BPlusTreeNode:
        """Deserialize a BPlusTreeNode from raw page bytes."""
        if len(data) < HEADER_SIZE:
            raise InvalidNodeError(
                f"Page data length {len(data)} is smaller than HEADER_SIZE {HEADER_SIZE}"
            )

        magic, is_leaf_int, num_keys, parent_id, next_id, prev_id, max_keys, payload_len = (
            struct.unpack_from(HEADER_FORMAT, data, 0)
        )

        if magic != MAGIC_BYTES:
            raise InvalidNodeError(
                f"Invalid BPlusTree header magic {magic!r} on page {page_id}"
            )

        if HEADER_SIZE + payload_len > PAGE_SIZE:
            raise InvalidNodeError(
                f"Corrupted page {page_id}: payload length {payload_len} exceeds available space"
            )

        payload_bytes = bytes(data[HEADER_SIZE : HEADER_SIZE + payload_len])
        is_leaf = bool(is_leaf_int)

        try:
            if is_leaf:
                keys, values = pickle.loads(payload_bytes)
                children: List[int] = []
            else:
                keys, children = pickle.loads(payload_bytes)
                values = []
        except Exception as exc:
            raise InvalidNodeError(
                f"Failed to deserialize page {page_id} payload: {exc}"
            ) from exc

        return cls(
            page_id=page_id,
            is_leaf=is_leaf,
            parent_page_id=parent_id,
            next_page_id=next_id,
            prev_page_id=prev_id,
            keys=keys,
            values=values,
            children=children,
            max_keys=max_keys,
        )

    def __repr__(self) -> str:
        role = "Leaf" if self.is_leaf else "Internal"
        return (
            f"BPlusTreeNode({role}, id={self.page_id}, parent={self.parent_page_id}, "
            f"keys={len(self.keys)}, next={self.next_page_id})"
        )


class BPlusTree:
    """Disk-backed B+ Tree index supported by BufferPoolManager pages.

    Features:
    - Root page allocation and dynamic root page updates during splits.
    - Leaf split and internal node split (upward propagation).
    - Exact key search with O(log N) page fetches.
    - Sequential range scan (range_scan) traversing leaf next_page_id pointers.
    - Deterministic page pinning/unpinning ensuring zero frame leaks.
    - Context manager support with dirty page flushing.
    """

    def __init__(
        self,
        buffer_pool_manager: BufferPoolManager,
        root_page_id: Optional[int] = None,
        max_keys: int = 32,
        internal_max_keys: Optional[int] = None,
    ) -> None:
        """Initialize BPlusTree.

        Args:
            buffer_pool_manager: BufferPoolManager instance managing page frames.
            root_page_id: Page ID of existing tree root. If None, allocates a new root leaf.
            max_keys: Maximum keys per leaf node before splitting (must be >= 2).
            internal_max_keys: Maximum keys per internal node before splitting (must be >= 2).
                               Defaults to max_keys.
        """
        if buffer_pool_manager is None:
            raise ValueError("buffer_pool_manager cannot be None")

        if max_keys < 2:
            raise ValueError(f"max_keys must be >= 2, got {max_keys}")

        if internal_max_keys is None:
            internal_max_keys = max_keys
        elif internal_max_keys < 2:
            raise ValueError(f"internal_max_keys must be >= 2, got {internal_max_keys}")

        self._bpm: BufferPoolManager = buffer_pool_manager
        self._max_keys: int = max_keys
        self._internal_max_keys: int = internal_max_keys
        self._lock: threading.RLock = threading.RLock()
        self._size: int = 0

        with self._lock:
            if root_page_id is not None:
                self._root_page_id: int = root_page_id
                self._size = self._compute_size()
            else:
                # Allocate initial root leaf page
                root_page = self._bpm.new_page()
                self._root_page_id = root_page.page_id
                root_node = BPlusTreeNode(
                    page_id=self._root_page_id,
                    is_leaf=True,
                    parent_page_id=-1,
                    next_page_id=-1,
                    prev_page_id=-1,
                    max_keys=self._max_keys,
                )
                root_node.to_bytes(root_page.data)
                self._bpm.unpin_page(root_page.page_id, is_dirty=True)
                self._size = 0

    @property
    def buffer_pool_manager(self) -> BufferPoolManager:
        """Return the underlying BufferPoolManager instance."""
        return self._bpm

    @property
    def root_page_id(self) -> int:
        """Return page ID of current root node."""
        with self._lock:
            return self._root_page_id

    @property
    def max_keys(self) -> int:
        """Return maximum keys per leaf node."""
        return self._max_keys

    @property
    def internal_max_keys(self) -> int:
        """Return maximum keys per internal node."""
        return self._internal_max_keys

    def _read_node(self, page_id: int) -> BPlusTreeNode:
        """Fetch and deserialize a node page, releasing pin immediately."""
        page = self._bpm.fetch_page(page_id)
        try:
            return BPlusTreeNode.from_bytes(page_id, page.data)
        finally:
            self._bpm.unpin_page(page_id, is_dirty=False)

    def _write_node(self, node: BPlusTreeNode) -> None:
        """Serialize a node into its page buffer and unpin with is_dirty=True."""
        page = self._bpm.fetch_page(node.page_id)
        try:
            node.to_bytes(page.data)
        finally:
            self._bpm.unpin_page(node.page_id, is_dirty=True)

    def _compute_size(self) -> int:
        """Traverse all leaf pages to compute total key count."""
        if self._root_page_id < 0:
            return 0
        curr_id = self._root_page_id
        curr = self._read_node(curr_id)
        while not curr.is_leaf:
            if not curr.children:
                return 0
            curr_id = curr.children[0]
            curr = self._read_node(curr_id)

        total = 0
        while curr_id != -1:
            leaf = self._read_node(curr_id)
            total += len(leaf.keys)
            curr_id = leaf.next_page_id
        return total

    def _find_leaf(self, key: Any) -> BPlusTreeNode:
        """Traverse from root to the leaf node that should contain key."""
        curr_id = self._root_page_id
        curr = self._read_node(curr_id)
        while not curr.is_leaf:
            idx = bisect.bisect_right(curr.keys, key)
            curr_id = curr.children[idx]
            curr = self._read_node(curr_id)
        return curr

    def search(self, key: Any) -> Optional[Any]:
        """Perform exact search for key in the index.

        Args:
            key: Comparable search key.

        Returns:
            The associated value if found, or None.
        """
        if key is None:
            return None

        with self._lock:
            leaf = self._find_leaf(key)
            idx = bisect.bisect_left(leaf.keys, key)
            if idx < len(leaf.keys) and leaf.keys[idx] == key:
                return leaf.values[idx]
            return None

    def insert(self, key: Any, value: Any, replace: bool = True) -> bool:
        """Insert a key-value pair into the B+ Tree index.

        If the target leaf exceeds max_keys, splits the leaf and promotes the
        split key upwards, splitting internal nodes up to the root if necessary.

        Args:
            key: Comparable index key.
            value: Associated value/payload.
            replace: If True, overwrite value if key already exists.
                     If False, raise DuplicateKeyError.

        Returns:
            True if a new key was inserted, False if an existing key was updated.

        Raises:
            ValueError: If key is None.
            DuplicateKeyError: If replace is False and key already exists.
        """
        if key is None:
            raise ValueError("Key cannot be None")

        with self._lock:
            leaf = self._find_leaf(key)
            idx = bisect.bisect_left(leaf.keys, key)

            # Key already exists
            if idx < len(leaf.keys) and leaf.keys[idx] == key:
                if not replace:
                    raise DuplicateKeyError(f"Key {key!r} already exists in BPlusTree")
                leaf.values[idx] = value
                self._write_node(leaf)
                return False

            # Insert new entry
            leaf.keys.insert(idx, key)
            leaf.values.insert(idx, value)
            self._size += 1

            if len(leaf.keys) > leaf.max_keys:
                self._split_leaf(leaf)
            else:
                self._write_node(leaf)

            return True

    def _split_leaf(self, leaf: BPlusTreeNode) -> None:
        """Split an overfull leaf node and propagate split key to parent."""
        right_page = self._bpm.new_page()
        right_page_id = right_page.page_id
        self._bpm.unpin_page(right_page_id, is_dirty=False)

        mid = len(leaf.keys) // 2
        right_keys = leaf.keys[mid:]
        right_values = leaf.values[mid:]

        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]

        split_key = right_keys[0]
        old_next_id = leaf.next_page_id

        right_leaf = BPlusTreeNode(
            page_id=right_page_id,
            is_leaf=True,
            parent_page_id=leaf.parent_page_id,
            next_page_id=old_next_id,
            prev_page_id=leaf.page_id,
            keys=right_keys,
            values=right_values,
            max_keys=leaf.max_keys,
        )
        leaf.next_page_id = right_page_id

        if old_next_id != -1:
            old_next = self._read_node(old_next_id)
            old_next.prev_page_id = right_page_id
            self._write_node(old_next)

        self._write_node(leaf)
        self._write_node(right_leaf)

        self._insert_into_parent(leaf.page_id, split_key, right_page_id)

    def _insert_into_parent(
        self, left_child_id: int, key: Any, right_child_id: int
    ) -> None:
        """Insert a promoted split key and child pointer into parent node."""
        left_node = self._read_node(left_child_id)
        parent_id = left_node.parent_page_id

        if parent_id == -1:
            # Left child was the root; allocate a new root internal node
            new_root_page = self._bpm.new_page()
            new_root_id = new_root_page.page_id
            self._bpm.unpin_page(new_root_id, is_dirty=False)

            new_root = BPlusTreeNode(
                page_id=new_root_id,
                is_leaf=False,
                parent_page_id=-1,
                next_page_id=-1,
                prev_page_id=-1,
                keys=[key],
                children=[left_child_id, right_child_id],
                max_keys=self._internal_max_keys,
            )

            left_node.parent_page_id = new_root_id
            self._write_node(left_node)

            right_node = self._read_node(right_child_id)
            right_node.parent_page_id = new_root_id
            self._write_node(right_node)

            self._write_node(new_root)
            self._root_page_id = new_root_id
            return

        parent = self._read_node(parent_id)
        child_idx = parent.children.index(left_child_id)

        parent.keys.insert(child_idx, key)
        parent.children.insert(child_idx + 1, right_child_id)

        right_node = self._read_node(right_child_id)
        if right_node.parent_page_id != parent_id:
            right_node.parent_page_id = parent_id
            self._write_node(right_node)

        if len(parent.keys) > parent.max_keys:
            self._split_internal(parent)
        else:
            self._write_node(parent)

    def _split_internal(self, node: BPlusTreeNode) -> None:
        """Split an overfull internal node and promote median key."""
        right_page = self._bpm.new_page()
        right_page_id = right_page.page_id
        self._bpm.unpin_page(right_page_id, is_dirty=False)

        mid = len(node.keys) // 2
        promote_key = node.keys[mid]

        left_keys = node.keys[:mid]
        left_children = node.children[: mid + 1]

        right_keys = node.keys[mid + 1 :]
        right_children = node.children[mid + 1 :]

        node.keys = left_keys
        node.children = left_children

        right_node = BPlusTreeNode(
            page_id=right_page_id,
            is_leaf=False,
            parent_page_id=node.parent_page_id,
            next_page_id=-1,
            prev_page_id=-1,
            keys=right_keys,
            children=right_children,
            max_keys=node.max_keys,
        )

        for child_id in right_children:
            child = self._read_node(child_id)
            child.parent_page_id = right_page_id
            self._write_node(child)

        self._write_node(node)
        self._write_node(right_node)

        self._insert_into_parent(node.page_id, promote_key, right_page_id)

    def range_scan(
        self,
        start_key: Optional[Any] = None,
        end_key: Optional[Any] = None,
        inclusive_start: bool = True,
        inclusive_end: bool = True,
    ) -> List[Tuple[Any, Any]]:
        """Perform a sequential range scan traversing leaf pointers (next_page_id).

        Finds the initial leaf via tree traversal, then follows horizontal
        next_page_id leaf links to collect entries within [start_key, end_key].

        Args:
            start_key: Lower bound of scan. If None, begins at smallest key in tree.
            end_key: Upper bound of scan. If None, continues until largest key in tree.
            inclusive_start: If True, include keys == start_key.
            inclusive_end: If True, include keys == end_key.

        Returns:
            List of (key, value) pairs in sorted ascending order.
        """
        with self._lock:
            if start_key is not None and end_key is not None and start_key > end_key:
                return []

            if start_key is None:
                curr_id = self._root_page_id
                curr = self._read_node(curr_id)
                while not curr.is_leaf:
                    curr_id = curr.children[0]
                    curr = self._read_node(curr_id)
                start_leaf = curr
                start_idx = 0
            else:
                start_leaf = self._find_leaf(start_key)
                if inclusive_start:
                    start_idx = bisect.bisect_left(start_leaf.keys, start_key)
                else:
                    start_idx = bisect.bisect_right(start_leaf.keys, start_key)

            results: List[Tuple[Any, Any]] = []
            curr_id = start_leaf.page_id
            curr_idx = start_idx

            while curr_id != -1:
                leaf = self._read_node(curr_id)

                stop = False
                for i in range(curr_idx, len(leaf.keys)):
                    k = leaf.keys[i]
                    v = leaf.values[i]

                    if start_key is not None:
                        if inclusive_start and k < start_key:
                            continue
                        if not inclusive_start and k <= start_key:
                            continue

                    if end_key is not None:
                        if inclusive_end and k > end_key:
                            stop = True
                            break
                        if not inclusive_end and k >= end_key:
                            stop = True
                            break

                    results.append((k, v))

                if stop:
                    break

                curr_id = leaf.next_page_id
                curr_idx = 0

            return results

    def delete(self, key: Any) -> bool:
        """Delete key from the tree.

        Args:
            key: Key to delete.

        Returns:
            True if key was removed, False if not found.
        """
        if key is None:
            return False

        with self._lock:
            leaf = self._find_leaf(key)
            idx = bisect.bisect_left(leaf.keys, key)
            if idx < len(leaf.keys) and leaf.keys[idx] == key:
                leaf.keys.pop(idx)
                leaf.values.pop(idx)
                self._write_node(leaf)
                self._size -= 1
                return True
            return False

    def get_leaves(self) -> List[int]:
        """Return list of leaf page IDs by following next_page_id links."""
        with self._lock:
            curr_id = self._root_page_id
            curr = self._read_node(curr_id)
            while not curr.is_leaf:
                curr_id = curr.children[0]
                curr = self._read_node(curr_id)

            leaves = []
            while curr_id != -1:
                leaves.append(curr_id)
                leaf = self._read_node(curr_id)
                curr_id = leaf.next_page_id
            return leaves

    def get_height(self) -> int:
        """Return current height of the B+ Tree (1 for root leaf)."""
        with self._lock:
            h = 1
            curr = self._read_node(self._root_page_id)
            while not curr.is_leaf:
                h += 1
                curr = self._read_node(curr.children[0])
            return h

    def flush(self) -> None:
        """Flush all dirty pages in the buffer pool to disk."""
        self._bpm.flush_all_pages()

    def is_empty(self) -> bool:
        """Return True if the tree contains no keys."""
        return len(self) == 0

    def keys(self) -> List[Any]:
        """Return all keys in ascending order."""
        return [k for k, _ in self.range_scan()]

    def values(self) -> List[Any]:
        """Return all values in ascending key order."""
        return [v for _, v in self.range_scan()]

    def items(self) -> List[Tuple[Any, Any]]:
        """Return all (key, value) pairs in ascending key order."""
        return self.range_scan()

    def get(self, key: Any, default: Any = None) -> Any:
        """Return value for key, or default if key not present."""
        val = self.search(key)
        if val is None:
            if not self.__contains__(key):
                return default
        return val

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not None

    def __getitem__(self, key: Any) -> Any:
        val = self.search(key)
        if val is None:
            if not self.__contains__(key):
                raise KeyNotFoundError(f"Key {key!r} not found in BPlusTree")
        return val

    def __setitem__(self, key: Any, value: Any) -> None:
        self.insert(key, value, replace=True)

    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyNotFoundError(f"Key {key!r} not found in BPlusTree")

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        return iter(self.range_scan())

    def __enter__(self) -> BPlusTree:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.flush()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"BPlusTree(root_page_id={self._root_page_id}, "
                f"height={self.get_height()}, size={len(self)}, "
                f"max_keys={self._max_keys})"
            )
