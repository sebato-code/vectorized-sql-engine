"""Exhaustive unit test suite for BPlusTree index backed by BufferPoolManager."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import pytest

from src.index.bplus_tree import (
    BPlusTree,
    BPlusTreeError,
    BPlusTreeNode,
    DuplicateKeyError,
    InvalidNodeError,
    KeyNotFoundError,
    NodeType,
)
from src.storage.buffer_pool import BufferPoolManager, Page
from src.storage.disk_manager import PAGE_SIZE, DiskManager


@pytest.fixture
def disk_manager(tmp_path: Path) -> DiskManager:
    """Fixture providing a fresh DiskManager."""
    db_file = tmp_path / "test_bplus_tree.db"
    return DiskManager(db_file)


@pytest.fixture
def buffer_pool(disk_manager: DiskManager) -> BufferPoolManager:
    """Fixture providing a BufferPoolManager with 10 frames."""
    return BufferPoolManager(disk_manager, pool_size=10)


@pytest.fixture
def small_buffer_pool(disk_manager: DiskManager) -> BufferPoolManager:
    """Fixture providing a BufferPoolManager with minimal frames (3 frames)."""
    return BufferPoolManager(disk_manager, pool_size=3)


class TestBPlusTreeNode:
    """Tests for BPlusTreeNode serialization and metadata."""

    def test_leaf_node_serialization_roundtrip(self) -> None:
        node = BPlusTreeNode(
            page_id=1,
            is_leaf=True,
            parent_page_id=0,
            next_page_id=2,
            prev_page_id=-1,
            keys=[10, 20, 30],
            values=["v10", "v20", "v30"],
            max_keys=3,
        )
        assert node.node_type == NodeType.LEAF
        assert "Leaf" in repr(node)

        buffer = bytearray(PAGE_SIZE)
        node.to_bytes(buffer)

        deserialized = BPlusTreeNode.from_bytes(1, buffer)
        assert deserialized.page_id == 1
        assert deserialized.is_leaf is True
        assert deserialized.parent_page_id == 0
        assert deserialized.next_page_id == 2
        assert deserialized.prev_page_id == -1
        assert deserialized.keys == [10, 20, 30]
        assert deserialized.values == ["v10", "v20", "v30"]
        assert deserialized.children == []
        assert deserialized.max_keys == 3

    def test_internal_node_serialization_roundtrip(self) -> None:
        node = BPlusTreeNode(
            page_id=0,
            is_leaf=False,
            parent_page_id=-1,
            next_page_id=-1,
            prev_page_id=-1,
            keys=[20],
            children=[1, 2],
            max_keys=3,
        )
        assert node.node_type == NodeType.INTERNAL
        assert "Internal" in repr(node)

        buffer = bytearray(PAGE_SIZE)
        node.to_bytes(buffer)

        deserialized = BPlusTreeNode.from_bytes(0, buffer)
        assert deserialized.page_id == 0
        assert deserialized.is_leaf is False
        assert deserialized.parent_page_id == -1
        assert deserialized.next_page_id == -1
        assert deserialized.prev_page_id == -1
        assert deserialized.keys == [20]
        assert deserialized.children == [1, 2]
        assert deserialized.values == []

    def test_deserialization_invalid_magic(self) -> None:
        corrupted_data = bytearray(PAGE_SIZE)
        with pytest.raises(InvalidNodeError, match="Invalid BPlusTree header magic"):
            BPlusTreeNode.from_bytes(1, corrupted_data)

    def test_deserialization_short_data(self) -> None:
        with pytest.raises(InvalidNodeError, match="smaller than HEADER_SIZE"):
            BPlusTreeNode.from_bytes(1, b"too_short")


class TestBPlusTreeInitialization:
    """Tests for BPlusTree creation and argument validation."""

    def test_default_initialization(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        assert tree.root_page_id >= 0
        assert tree.max_keys == 4
        assert tree.internal_max_keys == 4
        assert tree.buffer_pool_manager is buffer_pool
        assert len(tree) == 0
        assert tree.is_empty()
        assert tree.get_height() == 1
        assert "BPlusTree" in repr(tree)

    def test_invalid_parameters(self, buffer_pool: BufferPoolManager) -> None:
        with pytest.raises(ValueError, match="buffer_pool_manager cannot be None"):
            BPlusTree(None)  # type: ignore

        with pytest.raises(ValueError, match="max_keys must be >= 2"):
            BPlusTree(buffer_pool, max_keys=1)

        with pytest.raises(ValueError, match="internal_max_keys must be >= 2"):
            BPlusTree(buffer_pool, max_keys=3, internal_max_keys=1)


class TestBPlusTreeInsertionAndSearch:
    """Acceptance criteria: Insertion with splits and exact search by key."""

    def test_single_insert_and_search(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        assert tree.search(10) is None

        inserted = tree.insert(10, "value_10")
        assert inserted is True
        assert len(tree) == 1
        assert not tree.is_empty()
        assert tree.search(10) == "value_10"
        assert tree.search(20) is None

    def test_multiple_inserts_within_leaf_capacity(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tree.insert(30, "thirty")
        tree.insert(10, "ten")
        tree.insert(20, "twenty")

        assert len(tree) == 3
        assert tree.get_height() == 1
        assert tree.search(10) == "ten"
        assert tree.search(20) == "twenty"
        assert tree.search(30) == "thirty"

    def test_leaf_split_creates_internal_root(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        initial_root = tree.root_page_id

        # Insert 3 keys (fills leaf)
        tree.insert(10, "10")
        tree.insert(20, "20")
        tree.insert(30, "30")
        assert tree.get_height() == 1

        # 4th key triggers leaf split and new root creation
        tree.insert(40, "40")
        assert tree.get_height() == 2
        assert tree.root_page_id != initial_root

        # Verify all keys are retrievable
        for k in [10, 20, 30, 40]:
            assert tree.search(k) == str(k)

        # Verify leaves list
        leaves = tree.get_leaves()
        assert len(leaves) == 2

    def test_cascading_internal_node_splits(self, buffer_pool: BufferPoolManager) -> None:
        # With max_keys=2, splits occur very frequently
        tree = BPlusTree(buffer_pool, max_keys=2, internal_max_keys=2)

        n = 30
        for i in range(n):
            tree.insert(i, f"val_{i}")

        assert len(tree) == n
        assert tree.get_height() >= 3

        for i in range(n):
            assert tree.search(i) == f"val_{i}", f"Missing key {i}"

        # Non-existent keys
        assert tree.search(-1) is None
        assert tree.search(n + 10) is None

    def test_duplicate_key_replace(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        assert tree.insert(10, "old") is True
        assert tree.search(10) == "old"
        assert len(tree) == 1

        # Replace existing key
        assert tree.insert(10, "new", replace=True) is False
        assert tree.search(10) == "new"
        assert len(tree) == 1

    def test_duplicate_key_without_replace_raises(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tree.insert(10, "first")
        with pytest.raises(DuplicateKeyError, match="already exists"):
            tree.insert(10, "second", replace=False)

    def test_invalid_key_none(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        with pytest.raises(ValueError, match="Key cannot be None"):
            tree.insert(None, "val")
        assert tree.search(None) is None

    def test_dictionary_interface(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tree[1] = "apple"
        tree[2] = "banana"

        assert tree[1] == "apple"
        assert tree[2] == "banana"
        assert 1 in tree
        assert 2 in tree
        assert 3 not in tree
        assert tree.get(1) == "apple"
        assert tree.get(3, "default") == "default"

        with pytest.raises(KeyNotFoundError):
            _ = tree[3]


class TestBPlusTreeRangeScan:
    """Acceptance criteria: Sequential range scan traversing pointers between leaf pages."""

    def test_empty_tree_range_scan(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        assert tree.range_scan() == []
        assert tree.range_scan(10, 20) == []

    def test_full_range_scan_unbounded(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        items = [(i, f"v_{i}") for i in range(25)]
        for k, v in items:
            tree.insert(k, v)

        scan_result = tree.range_scan()
        assert scan_result == items
        assert tree.keys() == [k for k, _ in items]
        assert tree.values() == [v for _, v in items]
        assert tree.items() == items
        assert list(tree) == items

    def test_bounded_range_scan_inclusive(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for i in range(50):
            tree.insert(i, f"v_{i}")

        result = tree.range_scan(15, 25, inclusive_start=True, inclusive_end=True)
        expected = [(i, f"v_{i}") for i in range(15, 26)]
        assert result == expected

    def test_bounded_range_scan_exclusive(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for i in range(50):
            tree.insert(i, f"v_{i}")

        result = tree.range_scan(15, 25, inclusive_start=False, inclusive_end=False)
        expected = [(i, f"v_{i}") for i in range(16, 25)]
        assert result == expected

    def test_half_bounded_range_scans(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for i in range(30):
            tree.insert(i, f"v_{i}")

        # start_key only
        from_20 = tree.range_scan(start_key=20)
        assert from_20 == [(i, f"v_{i}") for i in range(20, 30)]

        # end_key only
        to_10 = tree.range_scan(end_key=10)
        assert to_10 == [(i, f"v_{i}") for i in range(11)]

    def test_range_scan_with_nonexistent_boundary_keys(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for i in [10, 20, 30, 40, 50]:
            tree.insert(i, f"val_{i}")

        # Boundaries not present in tree
        result = tree.range_scan(15, 35)
        assert result == [(20, "val_20"), (30, "val_30")]

        # Range before smallest key
        assert tree.range_scan(1, 5) == []

        # Range after largest key
        assert tree.range_scan(60, 100) == []

    def test_range_scan_inverted_bounds(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tree.insert(10, "a")
        tree.insert(20, "b")
        assert tree.range_scan(50, 10) == []

    def test_range_scan_traverses_leaf_linked_list(self, buffer_pool: BufferPoolManager) -> None:
        """Explicitly verify that range_scan steps through next_page_id leaf links."""
        tree = BPlusTree(buffer_pool, max_keys=2)
        for i in range(20):
            tree.insert(i, i * 100)

        leaves = tree.get_leaves()
        assert len(leaves) >= 10

        # Verify leaves are correctly chained
        for idx in range(len(leaves) - 1):
            curr_leaf = tree._read_node(leaves[idx])
            assert curr_leaf.next_page_id == leaves[idx + 1]
            next_leaf = tree._read_node(leaves[idx + 1])
            assert next_leaf.prev_page_id == leaves[idx]

        last_leaf = tree._read_node(leaves[-1])
        assert last_leaf.next_page_id == -1

        # Scan that traverses across all these leaves
        scan = tree.range_scan(2, 18)
        assert scan == [(i, i * 100) for i in range(2, 19)]


class TestKeyTypesAndInsertionOrders:
    """Tests for diverse data types and insertion order permutations."""

    def test_descending_insertion_order(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for k in range(100, 0, -1):
            tree.insert(k, f"desc_{k}")

        assert len(tree) == 100
        for k in range(1, 101):
            assert tree.search(k) == f"desc_{k}"

        assert tree.range_scan(25, 30) == [(k, f"desc_{k}") for k in range(25, 31)]

    def test_random_permutation_insertions(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        keys = list(range(150))
        random.seed(12345)
        random.shuffle(keys)

        for k in keys:
            tree.insert(k, k * 2)

        assert len(tree) == 150
        for k in keys:
            assert tree.search(k) == k * 2

        sorted_items = [(k, k * 2) for k in range(150)]
        assert tree.range_scan() == sorted_items

    def test_string_keys(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        words = ["delta", "alpha", "charlie", "bravo", "echo", "foxtrot", "golf"]
        for w in words:
            tree.insert(w, len(w))

        assert len(tree) == len(words)
        for w in words:
            assert tree.search(w) == len(w)

        assert tree.range_scan("bravo", "echo") == [
            ("bravo", 5),
            ("charlie", 7),
            ("delta", 5),
            ("echo", 4),
        ]

    def test_tuple_record_id_keys(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tuples = [(1, 0), (1, 1), (2, 0), (2, 5), (3, 2)]
        for t in tuples:
            tree.insert(t, f"rec_{t}")

        for t in tuples:
            assert tree.search(t) == f"rec_{t}"

        assert tree.range_scan((1, 1), (2, 5)) == [
            ((1, 1), "rec_(1, 1)"),
            ((2, 0), "rec_(2, 0)"),
            ((2, 5), "rec_(2, 5)"),
        ]


class TestBufferPoolIntegrationAndEviction:
    """Verifies that BPlusTree operates without leaking page pins and survives eviction."""

    def test_no_frame_pin_leaks_after_operations(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for i in range(50):
            tree.insert(i, i * 10)

        # After inserts, all frames in buffer pool must have pin_count == 0
        for pid, fid in buffer_pool.page_table.items():
            frame = buffer_pool._frames[fid]
            assert frame.pin_count == 0, f"Frame {fid} for page {pid} leaked pin ({frame.pin_count})"

        # After search
        for i in range(50):
            tree.search(i)
            for pid, fid in buffer_pool.page_table.items():
                frame = buffer_pool._frames[fid]
                assert frame.pin_count == 0

        # After range scan
        tree.range_scan()
        for pid, fid in buffer_pool.page_table.items():
            frame = buffer_pool._frames[fid]
            assert frame.pin_count == 0

    def test_stress_with_minimal_buffer_pool(self, small_buffer_pool: BufferPoolManager) -> None:
        """Buffer pool with only 3 frames handling tree with dozens of pages."""
        tree = BPlusTree(small_buffer_pool, max_keys=2)
        n = 80
        for i in range(n):
            tree.insert(i, f"payload_{i}")

        assert len(tree) == n
        for i in range(n):
            assert tree.search(i) == f"payload_{i}"

        scan = tree.range_scan(30, 40)
        assert scan == [(i, f"payload_{i}") for i in range(30, 41)]

    def test_persistence_reload_from_disk(self, tmp_path: Path) -> None:
        db_file = tmp_path / "persistence_tree.db"

        # Step 1: Create and populate tree
        dm1 = DiskManager(db_file)
        bpm1 = BufferPoolManager(dm1, pool_size=5)
        with BPlusTree(bpm1, max_keys=3) as tree1:
            for i in range(60):
                tree1.insert(i, f"val_{i}")
            root_id = tree1.root_page_id

        dm1.close()

        # Step 2: Reopen from disk with separate DiskManager and BufferPoolManager
        dm2 = DiskManager(db_file)
        bpm2 = BufferPoolManager(dm2, pool_size=5)
        tree2 = BPlusTree(bpm2, root_page_id=root_id, max_keys=3)

        assert len(tree2) == 60
        for i in range(60):
            assert tree2.search(i) == f"val_{i}"

        assert tree2.range_scan(20, 25) == [(i, f"val_{i}") for i in range(20, 26)]
        dm2.close()


class TestDeletion:
    """Tests for key deletion from the index."""

    def test_delete_existing_key(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        for i in range(10):
            tree.insert(i, i * 10)

        assert len(tree) == 10
        deleted = tree.delete(5)
        assert deleted is True
        assert len(tree) == 9
        assert tree.search(5) is None
        assert 5 not in tree
        assert 5 not in tree.keys()

    def test_delete_nonexistent_key(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tree.insert(1, "one")
        assert tree.delete(999) is False
        assert tree.delete(None) is False
        assert len(tree) == 1

    def test_delitem_interface(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        tree["key1"] = "val1"
        del tree["key1"]
        assert "key1" not in tree

        with pytest.raises(KeyNotFoundError):
            del tree["key1"]


class TestConcurrency:
    """Thread safety tests for BPlusTree."""

    def test_concurrent_inserts_and_reads(self, buffer_pool: BufferPoolManager) -> None:
        tree = BPlusTree(buffer_pool, max_keys=3)
        num_threads = 6
        items_per_thread = 20

        def worker(thread_idx: int) -> None:
            for i in range(items_per_thread):
                key = thread_idx * items_per_thread + i
                tree.insert(key, f"val_{key}")
                # Mixed read
                assert tree.search(key) == f"val_{key}"

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(num_threads)]
            for f in futures:
                f.result()

        total = num_threads * items_per_thread
        assert len(tree) == total
        for k in range(total):
            assert tree.search(k) == f"val_{k}"
