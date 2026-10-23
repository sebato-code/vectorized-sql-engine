"""Exhaustive unit and integration tests for vectorized IndexScan operator."""

from __future__ import annotations

from pathlib import Path
import pytest
from typing import Any, Dict, List

from src.index.bplus_tree import BPlusTree
from src.query.engine import (
    Aggregate,
    Avg,
    Col,
    ColumnNotFoundError,
    Count,
    Filter,
    HashJoin,
    IndexScan,
    Limit,
    Max,
    Min,
    OrderBy,
    Project,
    Sum,
    VectorBatch,
    col,
)
from src.storage.buffer_pool import BufferPoolManager
from src.storage.disk_manager import DiskManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def disk_manager(tmp_path: Path) -> DiskManager:
    """Fixture providing a fresh DiskManager."""
    db_file = tmp_path / "test_index_scan.db"
    return DiskManager(db_file)


@pytest.fixture
def buffer_pool(disk_manager: DiskManager) -> BufferPoolManager:
    """Fixture providing a BufferPoolManager with 10 frames."""
    return BufferPoolManager(disk_manager, pool_size=10)


@pytest.fixture
def small_buffer_pool(disk_manager: DiskManager) -> BufferPoolManager:
    """Fixture providing a BufferPoolManager with minimal frames (3 frames)."""
    return BufferPoolManager(disk_manager, pool_size=3)


@pytest.fixture
def populated_tree(buffer_pool: BufferPoolManager) -> BPlusTree:
    """Fixture providing a BPlusTree populated with 100 integer keys (0..99)."""
    tree = BPlusTree(buffer_pool, max_keys=4)
    for i in range(100):
        tree.insert(i, f"val_{i:03d}")
    return tree


@pytest.fixture
def dict_tree(buffer_pool: BufferPoolManager) -> BPlusTree:
    """Fixture providing a BPlusTree storing row dictionaries as values."""
    tree = BPlusTree(buffer_pool, max_keys=4)
    for i in range(50):
        tree.insert(
            i,
            {
                "id": i,
                "name": f"user_{i}",
                "age": 20 + (i % 30),
                "salary": 1000.0 * (i + 1),
            },
        )
    return tree


# ============================================================================
# 1. Point Lookup Tests
# ============================================================================


class TestIndexScanPointLookup:
    """Unit tests for exact key point lookup."""

    def test_point_lookup_existing_key_leaf_only(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=10)
        tree.insert(1, "one")
        tree.insert(2, "two")
        tree.insert(3, "three")

        scan = IndexScan(tree, key=2)
        assert scan.is_point_lookup is True
        assert scan.is_range_scan is False
        assert scan.point_key == 2

        batch = scan.next()
        assert batch is not None
        assert isinstance(batch, VectorBatch)
        assert batch.num_rows == 1
        assert batch.column("key") == [2]
        assert batch.column("value") == ["two"]

        # Second next() returns EOF
        assert scan.next() is None

    def test_point_lookup_multilevel_tree(self, populated_tree: BPlusTree) -> None:
        assert populated_tree.get_height() > 1

        scan = IndexScan(populated_tree, key=42)
        batches = scan.execute()
        assert len(batches) == 1
        assert batches[0].num_rows == 1
        assert batches[0].column("key") == [42]
        assert batches[0].column("value") == ["val_042"]

    def test_point_lookup_non_existing_key(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, key=999)
        batches = scan.execute()
        assert len(batches) == 0

        scan.open()
        assert scan.next() is None

    def test_point_lookup_boundary_keys(self, populated_tree: BPlusTree) -> None:
        # Min key (0)
        scan_min = IndexScan(populated_tree, key=0)
        rows_min = scan_min.execute_all_rows()
        assert len(rows_min) == 1
        assert rows_min[0] == {"key": 0, "value": "val_000"}

        # Max key (99)
        scan_max = IndexScan(populated_tree, key=99)
        rows_max = scan_max.execute_all_rows()
        assert len(rows_max) == 1
        assert rows_max[0] == {"key": 99, "value": "val_099"}

    def test_point_lookup_with_none_value(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        tree.insert(10, None)

        scan = IndexScan(tree, key=10)
        rows = scan.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {"key": 10, "value": None}

    def test_point_lookup_zero_and_negative_keys(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        tree.insert(-10, "neg_10")
        tree.insert(-1, "neg_1")
        tree.insert(0, "zero")
        tree.insert(5, "five")

        scan = IndexScan(tree, key=-10)
        rows = scan.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {"key": -10, "value": "neg_10"}

        scan_zero = IndexScan(tree, key=0)
        rows_zero = scan_zero.execute_all_rows()
        assert len(rows_zero) == 1
        assert rows_zero[0] == {"key": 0, "value": "zero"}

    def test_point_lookup_string_keys(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        tree.insert("alice", 100)
        tree.insert("bob", 200)
        tree.insert("charlie", 300)

        scan = IndexScan(tree, key="bob")
        rows = scan.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {"key": "bob", "value": 200}

    def test_point_lookup_positional_and_keyword_aliases(
        self, populated_tree: BPlusTree
    ) -> None:
        # Positional key
        scan1 = IndexScan(populated_tree, 25)
        assert scan1.is_point_lookup is True
        assert scan1.execute_all_rows()[0]["key"] == 25

        # point_key keyword
        scan2 = IndexScan(populated_tree, point_key=25)
        assert scan2.is_point_lookup is True
        assert scan2.execute_all_rows()[0]["key"] == 25

        # exact_key keyword
        scan3 = IndexScan(populated_tree, exact_key=25)
        assert scan3.is_point_lookup is True
        assert scan3.execute_all_rows()[0]["key"] == 25

    def test_point_lookup_custom_column_names(
        self, populated_tree: BPlusTree
    ) -> None:
        scan = IndexScan(
            populated_tree,
            key=15,
            key_column="user_id",
            value_column="tag",
        )
        assert scan.columns == ["user_id", "tag"]
        rows = scan.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {"user_id": 15, "tag": "val_015"}

    def test_point_lookup_with_projection(
        self, populated_tree: BPlusTree
    ) -> None:
        scan = IndexScan(populated_tree, key=15, projection=["value"])
        assert scan.columns == ["value"]
        rows = scan.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {"value": "val_015"}


# ============================================================================
# 2. Range Scan Tests
# ============================================================================


class TestIndexScanRangeScan:
    """Unit tests for key range scanning."""

    def test_range_scan_closed_interval(self, populated_tree: BPlusTree) -> None:
        # Range: [10, 15] inclusive
        scan = IndexScan(
            populated_tree,
            low_key=10,
            high_key=15,
            inclusive_low=True,
            inclusive_high=True,
        )
        assert scan.is_range_scan is True
        assert scan.is_point_lookup is False

        rows = scan.execute_all_rows()
        assert len(rows) == 6
        assert [r["key"] for r in rows] == [10, 11, 12, 13, 14, 15]
        assert [r["value"] for r in rows] == [f"val_{i:03d}" for i in range(10, 16)]

    def test_range_scan_open_interval(self, populated_tree: BPlusTree) -> None:
        # Range: (10, 15) exclusive
        scan = IndexScan(
            populated_tree,
            low_key=10,
            high_key=15,
            inclusive_low=False,
            inclusive_high=False,
        )
        rows = scan.execute_all_rows()
        assert len(rows) == 4
        assert [r["key"] for r in rows] == [11, 12, 13, 14]

    def test_range_scan_half_open_intervals(
        self, populated_tree: BPlusTree
    ) -> None:
        # [10, 15)
        scan_left = IndexScan(
            populated_tree,
            low_key=10,
            high_key=15,
            inclusive_low=True,
            inclusive_high=False,
        )
        rows_left = scan_left.execute_all_rows()
        assert [r["key"] for r in rows_left] == [10, 11, 12, 13, 14]

        # (10, 15]
        scan_right = IndexScan(
            populated_tree,
            low_key=10,
            high_key=15,
            inclusive_low=False,
            inclusive_high=True,
        )
        rows_right = scan_right.execute_all_rows()
        assert [r["key"] for r in rows_right] == [11, 12, 13, 14, 15]

    def test_range_scan_lower_bound_only(
        self, populated_tree: BPlusTree
    ) -> None:
        # Range: >= 90
        scan = IndexScan(populated_tree, low_key=90)
        rows = scan.execute_all_rows()
        assert len(rows) == 10
        assert [r["key"] for r in rows] == list(range(90, 100))

    def test_range_scan_upper_bound_only(
        self, populated_tree: BPlusTree
    ) -> None:
        # Range: <= 9
        scan = IndexScan(populated_tree, high_key=9)
        rows = scan.execute_all_rows()
        assert len(rows) == 10
        assert [r["key"] for r in rows] == list(range(0, 10))

    def test_range_scan_full_index(self, populated_tree: BPlusTree) -> None:
        # Scan entire tree (low_key=None, high_key=None)
        scan = IndexScan(populated_tree)
        rows = scan.execute_all_rows()
        assert len(rows) == 100
        assert [r["key"] for r in rows] == list(range(100))

    def test_range_scan_inverted_bounds(self, populated_tree: BPlusTree) -> None:
        # low > high
        scan = IndexScan(populated_tree, low_key=50, high_key=10)
        rows = scan.execute_all_rows()
        assert len(rows) == 0

    def test_range_scan_empty_subrange(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        tree.insert(10, "ten")
        tree.insert(20, "twenty")
        tree.insert(30, "thirty")

        # Range [12, 18] matches no keys
        scan = IndexScan(tree, low_key=12, high_key=18)
        assert scan.execute_all_rows() == []

    def test_range_scan_single_key_match(
        self, populated_tree: BPlusTree
    ) -> None:
        scan = IndexScan(populated_tree, low_key=42, high_key=42)
        rows = scan.execute_all_rows()
        assert len(rows) == 1
        assert rows[0]["key"] == 42

        # Exclusive single key match should yield 0 rows
        scan_ex = IndexScan(
            populated_tree,
            low_key=42,
            high_key=42,
            inclusive_low=False,
            inclusive_high=True,
        )
        assert scan_ex.execute_all_rows() == []

    def test_range_scan_all_keys_smaller_or_larger(
        self, populated_tree: BPlusTree
    ) -> None:
        # Range beyond max key
        scan_high = IndexScan(populated_tree, low_key=200, high_key=300)
        assert scan_high.execute_all_rows() == []

        # Range before min key
        scan_low = IndexScan(populated_tree, low_key=-50, high_key=-10)
        assert scan_low.execute_all_rows() == []

    def test_range_scan_positional_arguments(
        self, populated_tree: BPlusTree
    ) -> None:
        # IndexScan(tree, 10, 14) -> low=10, high=14
        scan = IndexScan(populated_tree, 10, 14)
        assert scan.is_range_scan is True
        rows = scan.execute_all_rows()
        assert [r["key"] for r in rows] == [10, 11, 12, 13, 14]

    def test_range_scan_keyword_aliases(
        self, populated_tree: BPlusTree
    ) -> None:
        # low / high keywords
        scan1 = IndexScan(populated_tree, low=10, high=12)
        assert [r["key"] for r in scan1.execute_all_rows()] == [10, 11, 12]

        # start_key / end_key keywords with inclusive_start / inclusive_end
        scan2 = IndexScan(
            populated_tree,
            start_key=10,
            end_key=12,
            inclusive_start=False,
            inclusive_end=False,
        )
        assert [r["key"] for r in scan2.execute_all_rows()] == [11]


# ============================================================================
# 3. Batching & Vectorization Tests
# ============================================================================


class TestIndexScanBatching:
    """Tests for VectorBatch chunking and vectorized emission."""

    def test_batch_size_smaller_than_result(
        self, populated_tree: BPlusTree
    ) -> None:
        # 40 items in range [10, 49], batch_size=10 -> exactly 4 batches of 10
        scan = IndexScan(populated_tree, low_key=10, high_key=49, batch_size=10)
        batches = scan.execute()

        assert len(batches) == 4
        for b in batches:
            assert b.num_rows == 10
            assert isinstance(b, VectorBatch)

        all_keys = [k for b in batches for k in b.column("key")]
        assert all_keys == list(range(10, 50))

    def test_batch_size_with_remainder(
        self, populated_tree: BPlusTree
    ) -> None:
        # 25 items in range [10, 34], batch_size=10 -> 2 batches of 10 + 1 batch of 5
        scan = IndexScan(populated_tree, low_key=10, high_key=34, batch_size=10)
        batches = scan.execute()

        assert len(batches) == 3
        assert [b.num_rows for b in batches] == [10, 10, 5]

        all_keys = [k for b in batches for k in b.column("key")]
        assert all_keys == list(range(10, 35))

    def test_batch_size_1(self, populated_tree: BPlusTree) -> None:
        # 5 items with batch_size=1 -> 5 batches of 1 row
        scan = IndexScan(populated_tree, low_key=1, high_key=5, batch_size=1)
        batches = scan.execute()

        assert len(batches) == 5
        for b in batches:
            assert b.num_rows == 1

    def test_batch_size_larger_than_result(
        self, populated_tree: BPlusTree
    ) -> None:
        # 20 items with batch_size=500 -> 1 batch of 20
        scan = IndexScan(populated_tree, low_key=10, high_key=29, batch_size=500)
        batches = scan.execute()

        assert len(batches) == 1
        assert batches[0].num_rows == 20

    def test_empty_scan_returns_no_batches(
        self, populated_tree: BPlusTree
    ) -> None:
        scan = IndexScan(populated_tree, low_key=500, high_key=600)
        batches = scan.execute()
        assert len(batches) == 0


# ============================================================================
# 4. Schema, Dict Unpacking & Projections
# ============================================================================


class TestIndexScanSchemaAndData:
    """Tests for columnar data formatting, dictionary payloads, and projections."""

    def test_dict_payload_expansion(self, dict_tree: BPlusTree) -> None:
        scan = IndexScan(dict_tree, low_key=0, high_key=4, key_column="user_id")
        batches = scan.execute()

        assert len(batches) == 1
        b = batches[0]
        assert b.num_rows == 5
        assert set(b.columns) == {"user_id", "id", "name", "age", "salary"}
        assert b.column("user_id") == [0, 1, 2, 3, 4]
        assert b.column("name") == ["user_0", "user_1", "user_2", "user_3", "user_4"]
        assert b.column("age") == [20, 21, 22, 23, 24]

    def test_dict_payload_projection(self, dict_tree: BPlusTree) -> None:
        scan = IndexScan(
            dict_tree,
            low_key=10,
            high_key=12,
            projection=["name", "salary"],
        )
        assert scan.columns == ["name", "salary"]

        batches = scan.execute()
        assert len(batches) == 1
        b = batches[0]
        assert b.columns == ["name", "salary"]
        assert b.column("name") == ["user_10", "user_11", "user_12"]
        assert b.column("salary") == [11000.0, 12000.0, 13000.0]

    def test_columns_spec_override(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(
            populated_tree,
            low_key=1,
            high_key=3,
            columns=["row_id", "val_payload"],
        )
        assert scan.columns == ["row_id", "val_payload"]
        batch = scan.next()
        assert batch is not None
        assert batch.columns == ["row_id", "val_payload"]
        assert batch.column("row_id") == [1, 2, 3]
        assert batch.column("val_payload") == ["val_001", "val_002", "val_003"]

    def test_tuple_payload_with_columns_spec(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        tree.insert(1, ("alice", 25))
        tree.insert(2, ("bob", 30))

        scan = IndexScan(tree, columns=["id", "name", "age"])
        rows = scan.execute_all_rows()
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "alice", "age": 25}
        assert rows[1] == {"id": 2, "name": "bob", "age": 30}

    def test_invalid_projection_raises_column_not_found(
        self, populated_tree: BPlusTree
    ) -> None:
        scan = IndexScan(
            populated_tree,
            low_key=1,
            high_key=5,
            projection=["non_existent_column"],
        )
        with pytest.raises(ColumnNotFoundError):
            scan.execute()


# ============================================================================
# 5. Volcano Lifecycle & Execution Protocols
# ============================================================================


class TestIndexScanVolcanoLifecycle:
    """Tests verifying Volcano iterator contract and protocols."""

    def test_lifecycle_open_next_close(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, low_key=10, high_key=12)
        assert scan.is_open is False

        scan.open()
        assert scan.is_open is True

        b = scan.next()
        assert b is not None
        assert b.num_rows == 3

        assert scan.next() is None

        scan.close()
        assert scan.is_open is False

    def test_multiple_open_restarts_scan(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, low_key=5, high_key=7)

        # First run
        rows1 = scan.execute_all_rows()
        assert len(rows1) == 3

        # Second run on same operator instance
        scan.open()
        rows2 = []
        while True:
            batch = scan.next()
            if batch is None:
                break
            rows2.extend(batch.to_rows())
        scan.close()

        assert rows1 == rows2

    def test_next_without_explicit_open(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, key=10)
        assert scan.is_open is False

        # Calling next() implicitly opens
        batch = scan.next()
        assert scan.is_open is True
        assert batch is not None
        assert batch.num_rows == 1
        scan.close()

    def test_next_after_eof(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, key=10)
        scan.open()
        assert scan.next() is not None
        assert scan.next() is None
        # Subsequent next() continues returning None
        assert scan.next() is None
        assert scan.next() is None
        scan.close()

    def test_python_iteration_protocol(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, low_key=1, high_key=6, batch_size=2)
        batches = [b for b in scan]

        assert len(batches) == 3
        for b in batches:
            assert b.num_rows == 2

    def test_multiple_python_iterations(self, populated_tree: BPlusTree) -> None:
        scan = IndexScan(populated_tree, low_key=1, high_key=4)

        run1 = [b.num_rows for b in scan]
        run2 = [b.num_rows for b in scan]

        assert run1 == run2 == [4]

    def test_context_manager_protocol(self, populated_tree: BPlusTree) -> None:
        with IndexScan(populated_tree, low_key=10, high_key=15) as scan:
            assert scan.is_open is True
            rows = [r for b in scan for r in b.to_rows()]
            assert len(rows) == 6

        assert scan.is_open is False

    def test_repr_point_lookup_and_range_scan(
        self, populated_tree: BPlusTree
    ) -> None:
        scan_point = IndexScan(populated_tree, key=42)
        assert "IndexScan" in repr(scan_point)
        assert "key=42" in repr(scan_point)

        scan_range = IndexScan(populated_tree, low_key=10, high_key=20)
        assert "IndexScan" in repr(scan_range)
        assert "low_key=10" in repr(scan_range)
        assert "high_key=20" in repr(scan_range)


# ============================================================================
# 6. Validation and Error Handling
# ============================================================================


class TestIndexScanValidation:
    """Tests for parameter validation and defensive error handling."""

    def test_none_index_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="index cannot be None"):
            IndexScan(None)  # type: ignore[arg-type]

    def test_invalid_index_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="index must be a BPlusTree instance"):
            IndexScan("not_a_tree")  # type: ignore[arg-type]

    def test_invalid_batch_size_raises_value_error(
        self, populated_tree: BPlusTree
    ) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            IndexScan(populated_tree, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            IndexScan(populated_tree, batch_size=-5)


# ============================================================================
# 7. Integration Tests with Disk, BufferPool & Engine Pipelines
# ============================================================================


class TestIndexScanIntegration:
    """Integration tests with disk paging, eviction, mutations, and query pipelines."""

    def test_index_scan_with_disk_eviction(
        self, small_buffer_pool: BufferPoolManager
    ) -> None:
        """Scan over a multi-page B+ Tree with only 3 buffer pool frames."""
        tree = BPlusTree(small_buffer_pool, max_keys=3)
        # Insert 150 items to force multiple leaf/internal page splits and evictions
        for i in range(150):
            tree.insert(i, f"item_{i:04d}")

        # Scan full range with batch_size=15
        scan = IndexScan(tree, low_key=20, high_key=79, batch_size=15)
        batches = scan.execute()

        assert len(batches) == 4
        all_keys = [k for b in batches for k in b.column("key")]
        assert all_keys == list(range(20, 80))

    def test_index_scan_reflects_tree_mutations(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        tree = BPlusTree(buffer_pool, max_keys=4)
        tree.insert(1, "one")
        tree.insert(2, "two")

        scan = IndexScan(tree)
        assert [r["key"] for r in scan.execute_all_rows()] == [1, 2]

        # Mutate tree: insert and delete
        tree.insert(3, "three")
        tree.delete(1)

        # Re-running scan reflects updated state
        assert [r["key"] for r in scan.execute_all_rows()] == [2, 3]

    def test_pipeline_index_scan_with_filter(
        self, populated_tree: BPlusTree
    ) -> None:
        # IndexScan -> Filter (where key % 2 == 0 and key >= 20 and key <= 30)
        scan = IndexScan(populated_tree, low_key=20, high_key=30)
        filter_op = Filter(scan, (col("key") % 2) == 0)

        rows = filter_op.execute_all_rows()
        assert [r["key"] for r in rows] == [20, 22, 24, 26, 28, 30]

    def test_pipeline_index_scan_with_project(
        self, populated_tree: BPlusTree
    ) -> None:
        # IndexScan -> Project
        scan = IndexScan(populated_tree, low_key=1, high_key=3)
        proj_op = Project(
            scan,
            {
                "id": Col("key"),
                "doubled_id": Col("key") * 2,
            },
        )
        rows = proj_op.execute_all_rows()
        assert rows == [
            {"id": 1, "doubled_id": 2},
            {"id": 2, "doubled_id": 4},
            {"id": 3, "doubled_id": 6},
        ]

    def test_pipeline_index_scan_with_limit(
        self, populated_tree: BPlusTree
    ) -> None:
        # Wide index scan restricted by Limit
        scan = IndexScan(populated_tree, low_key=0, high_key=99, batch_size=5)
        limit_op = Limit(scan, limit=8, offset=3)

        rows = limit_op.execute_all_rows()
        assert len(rows) == 8
        assert [r["key"] for r in rows] == list(range(3, 11))

    def test_pipeline_index_scan_with_orderby(
        self, populated_tree: BPlusTree
    ) -> None:
        scan = IndexScan(populated_tree, low_key=10, high_key=14)
        order_op = OrderBy(scan, by="key", ascending=False)

        rows = order_op.execute_all_rows()
        assert [r["key"] for r in rows] == [14, 13, 12, 11, 10]

    def test_pipeline_index_scan_hash_join(
        self, buffer_pool: BufferPoolManager
    ) -> None:
        # Two BPlusTree indexes: users and scores
        users_tree = BPlusTree(buffer_pool, max_keys=4)
        users_tree.insert(1, "Alice")
        users_tree.insert(2, "Bob")
        users_tree.insert(3, "Charlie")

        scores_tree = BPlusTree(buffer_pool, max_keys=4)
        scores_tree.insert(1, 95)
        scores_tree.insert(2, 88)
        scores_tree.insert(4, 70)

        scan_users = IndexScan(
            users_tree, key_column="user_id", value_column="user_name"
        )
        scan_scores = IndexScan(
            scores_tree, key_column="user_id", value_column="score"
        )

        join_op = HashJoin(
            left=scan_users,
            right=scan_scores,
            left_on="user_id",
            right_on="user_id",
            join_type="INNER",
        )
        rows = join_op.execute_all_rows()
        assert len(rows) == 2
        assert rows[0] == {
            "user_id": 1,
            "user_name": "Alice",
            "score": 95,
            "right_user_id": 1,
        }
        assert rows[1] == {
            "user_id": 2,
            "user_name": "Bob",
            "score": 88,
            "right_user_id": 2,
        }

    def test_pipeline_index_scan_aggregate(
        self, populated_tree: BPlusTree
    ) -> None:
        # IndexScan -> Aggregate
        scan = IndexScan(populated_tree, low_key=1, high_key=10)
        agg_op = Aggregate(
            scan,
            aggregates=[
                Count(alias="cnt"),
                Sum("key", alias="total_key"),
                Avg("key", alias="avg_key"),
                Min("key", alias="min_key"),
                Max("key", alias="max_key"),
            ],
        )
        rows = agg_op.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {
            "cnt": 10,
            "total_key": 55,
            "avg_key": 5.5,
            "min_key": 1,
            "max_key": 10,
        }

    def test_pipeline_full_complex_query(
        self, dict_tree: BPlusTree
    ) -> None:
        # Complex pipeline: IndexScan -> Filter -> Project -> Limit
        scan = IndexScan(dict_tree, low_key=0, high_key=40, batch_size=10)
        filter_op = Filter(scan, Col("age") >= 25)
        proj_op = Project(
            filter_op,
            {
                "name": Col("name"),
                "age_next_year": Col("age") + 1,
                "annual_k": Col("salary") / 1000,
            },
        )
        limit_op = Limit(proj_op, limit=3)

        rows = limit_op.execute_all_rows()
        assert len(rows) == 3
        for r in rows:
            assert "name" in r
            assert r["age_next_year"] >= 26
            assert r["annual_k"] > 0
