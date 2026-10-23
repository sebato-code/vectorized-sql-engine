"""Comprehensive test suite for the Rule-Based Query Optimizer (RBO).

Verifies:
1. Optimizer configuration, catalog management, and index registration.
2. Predicate Pushdown:
   - Pushing filters through INNER JOINs to left and right relations.
   - Pushing filters with table aliases and qualified/unqualified columns.
   - Handling cross-table join predicates (retained at join level).
   - Eliminating redundant LogicalFilter nodes when all conjuncts are pushed.
   - Pushing through multi-table chained joins (3+ tables).
   - Pushing through LEFT OUTER JOINs (left-table filters and ON-clause pushdown).
   - Pushing through LogicalSort and LogicalProjection.
   - Preserving disjunctive predicates spanning multiple tables.
3. Index Selection:
   - Converting equality filters (=) to LogicalIndexScan.
   - Inverted equality predicates (Literal = Col).
   - Range comparison filters (<, <=, >, >=).
   - Two-sided bounded range (col >= low AND col <= high) to BETWEEN.
   - Multi-predicate filters with residual filters over LogicalIndexScan.
   - IN predicates.
   - Fallback to sequential scan when no matching index exists.
   - Integration with real BPlusTree index instances.
4. Projection Pruning:
   - Pruning unreferenced columns from simple SELECT scans.
   - Retaining columns referenced in WHERE, JOIN ON, and ORDER BY clauses.
   - Multi-table join column pruning.
   - Handling wildcard Star (*) and table wildcards (table.*).
   - Pruning on LogicalIndexScan while preserving index key column.
   - Eliminating redundant nested projections.
5. End-to-End Pipeline:
   - Full parser -> optimizer pipeline on complex SQL queries.
   - Idempotency and plan representation formatting.
"""

from pathlib import Path
from typing import Any, List
import pytest

from src.index.bplus_tree import BPlusTree
from src.query.optimizer import (
    Catalog,
    IndexInfo,
    IndexScan,
    IndexScanNode,
    IndexSelectionRule,
    LogicalIndexScan,
    OptimizerError,
    OptimizerRule,
    PredicatePushdownRule,
    ProjectionPruningRule,
    RuleApplicationError,
    RuleBasedOptimizer,
    combine_conjuncts,
    get_plan_tables,
    get_referenced_columns,
    optimize,
    optimize_plan,
    split_conjuncts,
)
from src.query.parser import (
    BinaryOp,
    BinaryOpType,
    ColumnRef,
    JoinType,
    Literal,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalPlan,
    LogicalProjection,
    LogicalScan,
    LogicalSort,
    OrderByItem,
    SelectItem,
    Star,
    parse_logical_plan,
)
from src.storage.buffer_pool import BufferPoolManager
from src.storage.disk_manager import DiskManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_catalog() -> Catalog:
    """Catalog with schemas and indexes for users, orders, and products."""
    cat = Catalog()
    cat.add_table("users", ["id", "name", "age", "email", "status", "department_id"])
    cat.add_table("orders", ["order_id", "user_id", "amount", "status", "created_at"])
    cat.add_table("products", ["product_id", "title", "price", "category"])

    cat.add_index("users", "id", index_name="idx_users_id", is_unique=True)
    cat.add_index("users", "email", index_name="idx_users_email", is_unique=True)
    cat.add_index("users", "age", index_name="idx_users_age")
    cat.add_index("orders", "order_id", index_name="idx_orders_id", is_unique=True)
    cat.add_index("orders", "user_id", index_name="idx_orders_user_id")
    cat.add_index("products", "product_id", index_name="idx_products_id", is_unique=True)
    return cat


@pytest.fixture
def optimizer(sample_catalog: Catalog) -> RuleBasedOptimizer:
    """Standard RuleBasedOptimizer initialized with sample catalog."""
    return RuleBasedOptimizer(catalog=sample_catalog)


@pytest.fixture
def real_bplus_tree(tmp_path: Path) -> BPlusTree:
    """Instantiate a real BPlusTree backed by BufferPoolManager."""
    db_file = tmp_path / "test_optimizer_btree.db"
    dm = DiskManager(db_file)
    bpm = BufferPoolManager(dm, pool_size=10)
    tree = BPlusTree(bpm, max_keys=4)
    # Insert sample records
    for uid in range(1, 50):
        tree.insert(uid, f"user_row_{uid}")
    return tree


# ============================================================================
# 1. Optimizer Basics & Catalog Tests
# ============================================================================


class TestOptimizerBasics:
    """Tests for RuleBasedOptimizer initialization, catalog, and configuration."""

    def test_default_initialization(self) -> None:
        opt = RuleBasedOptimizer()
        assert len(opt.rules) == 3
        rule_names = [r.name for r in opt.rules]
        assert rule_names == [
            "PredicatePushdownRule",
            "IndexSelectionRule",
            "ProjectionPruningRule",
        ]

    def test_selective_rules_initialization(self) -> None:
        opt = RuleBasedOptimizer(
            predicate_pushdown=True,
            index_selection=False,
            projection_pruning=False,
        )
        assert len(opt.rules) == 1
        assert opt.rules[0].name == "PredicatePushdownRule"

    def test_custom_rule_addition(self) -> None:
        class DummyRule(OptimizerRule):
            def apply(self, plan: LogicalPlan, optimizer: RuleBasedOptimizer) -> LogicalPlan:
                return plan

        opt = RuleBasedOptimizer()
        opt.add_rule(DummyRule())
        assert len(opt.rules) == 4
        assert opt.rules[-1].name == "DummyRule"

    def test_catalog_table_management(self) -> None:
        cat = Catalog()
        cat.add_table("items", ["id", "sku", "qty"])
        assert cat.has_table("items")
        assert not cat.has_table("unknown")
        assert cat.get_table_columns("items") == ["id", "sku", "qty"]
        assert cat.get_table_columns("unknown") is None

    def test_catalog_index_management(self) -> None:
        cat = Catalog()
        cat.add_index("items", "sku", index_name="idx_items_sku")
        assert cat.has_index("items", "sku")
        assert not cat.has_index("items", "qty")
        assert cat.get_index_name("items", "sku") == "idx_items_sku"
        info = cat.get_index_info("items", "sku")
        assert info is not None
        assert info.table_name == "items"
        assert info.column_name == "sku"

    def test_catalog_flexible_index_loading(self) -> None:
        # Dict formats
        cat1 = Catalog(indexes={"users.id": "idx1", ("users", "email"): "idx2"})
        assert cat1.has_index("users", "id")
        assert cat1.has_index("users", "email")

        # Nested dict format
        cat2 = Catalog(indexes={"users": {"id": None, "age": None}})
        assert cat2.has_index("users", "id")
        assert cat2.has_index("users", "age")

        # List/set formats
        cat3 = Catalog(indexes=[("orders", "order_id"), "orders.amount"])
        assert cat3.has_index("orders", "order_id")
        assert cat3.has_index("orders", "amount")

    def test_optimizer_register_aliases(self) -> None:
        opt = RuleBasedOptimizer()
        opt.register_table("customers", ["cid", "name"])
        opt.register_index("customers", "cid", index_name="idx_cust")
        assert opt.has_index("customers", "cid")
        assert opt.get_index_name("customers", "cid") == "idx_cust"
        assert opt.get_table_columns("customers") == ["cid", "name"]

    def test_convenience_optimize_function(self) -> None:
        sql = "SELECT id, name FROM users WHERE id = 10"
        plan = parse_logical_plan(sql)
        optimized = optimize(plan, indexes={"users": ["id"]})
        assert isinstance(optimized, LogicalProjection)
        # Scan was transformed into LogicalIndexScan
        assert isinstance(optimized.input, LogicalIndexScan)

    def test_logical_index_scan_properties(self) -> None:
        scan = LogicalIndexScan(
            table_name="users",
            index_column="id",
            operator="=",
            value=42,
            alias="u",
            columns=["id", "name"],
            index_name="idx_u_id",
        )
        assert scan.table == "users"
        assert scan.column_name == "id"
        assert scan.key_column == "id"
        assert scan.key == 42
        assert scan.keys == 42
        assert scan.values == 42
        assert scan.op == "="
        assert scan.children == []
        assert "idx_u_id" in scan.pretty()
        assert "LogicalIndexScan" in str(scan)

        # Aliases
        assert IndexScan is LogicalIndexScan
        assert IndexScanNode is LogicalIndexScan


# ============================================================================
# 2. Predicate Pushdown Tests
# ============================================================================


class TestPredicatePushdown:
    """Tests for Predicate Pushdown transformation rule."""

    def test_pushdown_single_predicate_to_left(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.age > 21
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # Left child of join has the filter pushed down
        assert isinstance(join_node.left, LogicalFilter)
        assert join_node.left.condition.left == ColumnRef("age", "u")
        assert isinstance(join_node.left.input, LogicalScan)
        assert join_node.left.input.table_name == "users"

        # Right child has NO filter
        assert isinstance(join_node.right, LogicalScan)
        assert join_node.right.table_name == "orders"

    def test_pushdown_single_predicate_to_right(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.amount > 500
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # Left child has NO filter
        assert isinstance(join_node.left, LogicalScan)
        assert join_node.left.table_name == "users"

        # Right child has the filter pushed down
        assert isinstance(join_node.right, LogicalFilter)
        assert join_node.right.condition.left == ColumnRef("amount", "o")
        assert isinstance(join_node.right.input, LogicalScan)
        assert join_node.right.input.table_name == "orders"

    def test_pushdown_conjunctive_predicates_to_both_sides(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.age >= 18 AND o.amount < 1000
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # Left side filter
        assert isinstance(join_node.left, LogicalFilter)
        assert join_node.left.condition.left == ColumnRef("age", "u")
        assert join_node.left.input.table_name == "users"

        # Right side filter
        assert isinstance(join_node.right, LogicalFilter)
        assert join_node.right.condition.left == ColumnRef("amount", "o")
        assert join_node.right.input.table_name == "orders"

    def test_pushdown_with_join_predicate_retention(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.age > 25 AND u.id = o.user_id
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        # Because u.id = o.user_id refers to both, it remains in a LogicalFilter above the join
        assert isinstance(optimized.input, LogicalFilter)
        join_node = optimized.input.input
        assert isinstance(join_node, LogicalJoin)

        # Left child still receives u.age > 25 pushdown
        assert isinstance(join_node.left, LogicalFilter)
        assert join_node.left.condition.left == ColumnRef("age", "u")

    def test_pushdown_multi_table_chained_joins(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount, p.title
        FROM users u
        JOIN orders o ON u.id = o.user_id
        JOIN products p ON o.order_id = p.product_id
        WHERE u.status = 'active' AND o.amount > 50 AND p.price < 200
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        outer_join = optimized.input
        assert isinstance(outer_join, LogicalJoin)

        # p.price < 200 pushed to products (right child of outer join)
        assert isinstance(outer_join.right, LogicalFilter)
        assert outer_join.right.input.table_name == "products"

        # Inner join is left child
        inner_join = outer_join.left
        assert isinstance(inner_join, LogicalJoin)

        # u.status = 'active' pushed to users
        assert isinstance(inner_join.left, LogicalFilter)
        assert inner_join.left.input.table_name == "users"

        # o.amount > 50 pushed to orders
        assert isinstance(inner_join.right, LogicalFilter)
        assert inner_join.right.input.table_name == "orders"

    def test_pushdown_left_outer_join(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id AND o.status = 'completed'
        WHERE u.age > 30
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)
        assert join_node.join_type == JoinType.LEFT

        # u.age > 30 pushed to left table users
        assert isinstance(join_node.left, LogicalFilter)
        assert join_node.left.input.table_name == "users"

        # ON clause o.status = 'completed' pushed to right table orders
        assert isinstance(join_node.right, LogicalFilter)
        assert join_node.right.input.table_name == "orders"

    def test_pushdown_through_sort_operator(self, optimizer: RuleBasedOptimizer) -> None:
        # Construct: Filter on top of Sort on top of Scan
        scan = LogicalScan(table_name="users")
        sort = LogicalSort(input=scan, order_by=[OrderByItem(ColumnRef("age"), ascending=False)])
        filter_node = LogicalFilter(
            input=sort,
            condition=BinaryOp(ColumnRef("age"), ">", Literal(18)),
        )

        optimized = optimizer.pushdown_predicates(filter_node)
        # Filter should now be below Sort
        assert isinstance(optimized, LogicalSort)
        assert isinstance(optimized.input, LogicalFilter)
        assert isinstance(optimized.input.input, LogicalScan)

    def test_pushdown_through_projection_operator(self, optimizer: RuleBasedOptimizer) -> None:
        # Construct: Filter on top of Projection on top of Scan
        scan = LogicalScan(table_name="users")
        proj = LogicalProjection(
            input=scan,
            expressions=[SelectItem(ColumnRef("id")), SelectItem(ColumnRef("age"))],
        )
        filter_node = LogicalFilter(
            input=proj,
            condition=BinaryOp(ColumnRef("age"), ">", Literal(21)),
        )

        optimized = optimizer.pushdown_predicates(filter_node)
        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalFilter)
        assert isinstance(optimized.input.input, LogicalScan)

    def test_disjunctive_cross_table_predicate_retained_above_join(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.age > 25 OR o.amount > 100
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        # Filter remains above join because OR cannot be split across relations
        assert isinstance(optimized.input, LogicalFilter)
        assert isinstance(optimized.input.input, LogicalJoin)

    def test_pushdown_unqualified_columns_with_catalog(self, optimizer: RuleBasedOptimizer) -> None:
        # Unqualified 'age' belongs to users, 'amount' belongs to orders in catalog
        sql = """
        SELECT name, amount
        FROM users
        JOIN orders ON users.id = orders.user_id
        WHERE age > 18 AND amount < 50
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.pushdown_predicates(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # age pushed to users
        assert isinstance(join_node.left, LogicalFilter)
        assert join_node.left.input.table_name == "users"

        # amount pushed to orders
        assert isinstance(join_node.right, LogicalFilter)
        assert join_node.right.input.table_name == "orders"


# ============================================================================
# 3. Index Selection Tests
# ============================================================================


class TestIndexSelection:
    """Tests for Index Selection converting LogicalFilter into LogicalIndexScan."""

    def test_equality_index_selection(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users WHERE id = 10"
        plan = parse_logical_plan(sql)
        # id has index in sample_catalog
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalIndexScan)
        idx_scan = optimized.input
        assert idx_scan.table_name == "users"
        assert idx_scan.index_column == "id"
        assert idx_scan.operator == "="
        assert idx_scan.value == 10
        assert idx_scan.index_name == "idx_users_id"

    def test_inverted_equality_index_selection(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users WHERE 42 = id"
        plan = parse_logical_plan(sql)
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalIndexScan)
        idx_scan = optimized.input
        assert idx_scan.index_column == "id"
        assert idx_scan.operator == "="
        assert idx_scan.value == 42

    def test_range_index_selection_greater_than(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users WHERE age > 30"
        plan = parse_logical_plan(sql)
        # age has index in sample_catalog
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalIndexScan)
        idx_scan = optimized.input
        assert idx_scan.index_column == "age"
        assert idx_scan.operator == ">"
        assert idx_scan.value == 30
        assert idx_scan.start_key == 30
        assert idx_scan.inclusive_start is False

    def test_range_index_selection_less_than_equals(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users WHERE age <= 65"
        plan = parse_logical_plan(sql)
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalIndexScan)
        idx_scan = optimized.input
        assert idx_scan.index_column == "age"
        assert idx_scan.operator == "<="
        assert idx_scan.value == 65
        assert idx_scan.end_key == 65
        assert idx_scan.inclusive_end is True

    def test_bounded_range_between_index_selection(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users WHERE age >= 20 AND age <= 40"
        plan = parse_logical_plan(sql)
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalIndexScan)
        idx_scan = optimized.input
        assert idx_scan.index_column == "age"
        assert idx_scan.operator == "BETWEEN"
        assert idx_scan.value == (20, 40)
        assert idx_scan.start_key == 20
        assert idx_scan.end_key == 40
        assert idx_scan.inclusive_start is True
        assert idx_scan.inclusive_end is True

    def test_filter_with_indexed_and_non_indexed_predicates(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        # id has index, status does NOT have index
        sql = "SELECT name FROM users WHERE id = 15 AND status = 'active'"
        plan = parse_logical_plan(sql)
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        # Residual filter wraps LogicalIndexScan
        assert isinstance(optimized.input, LogicalFilter)
        residual_filter = optimized.input
        assert residual_filter.condition.left == ColumnRef("status")

        assert isinstance(residual_filter.input, LogicalIndexScan)
        idx_scan = residual_filter.input
        assert idx_scan.index_column == "id"
        assert idx_scan.value == 15

    def test_no_index_available_retains_filter_scan(self, optimizer: RuleBasedOptimizer) -> None:
        # status has NO index
        sql = "SELECT name FROM users WHERE status = 'pending'"
        plan = parse_logical_plan(sql)
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalFilter)
        assert isinstance(optimized.input.input, LogicalScan)

    def test_index_selection_with_table_alias(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT u.name FROM users u WHERE u.email = 'alice@example.com'"
        plan = parse_logical_plan(sql)
        optimized = optimizer.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalIndexScan)
        idx_scan = optimized.input
        assert idx_scan.table_name == "users"
        assert idx_scan.alias == "u"
        assert idx_scan.index_column == "email"
        assert idx_scan.value == "alice@example.com"

    def test_index_selection_with_real_bplus_tree(
        self, real_bplus_tree: BPlusTree
    ) -> None:
        cat = Catalog()
        cat.add_table("users", ["id", "data"])
        cat.add_index("users", "id", index=real_bplus_tree, index_name="bplus_users_id")

        opt = RuleBasedOptimizer(catalog=cat)
        plan = parse_logical_plan("SELECT data FROM users WHERE id = 25")
        optimized = opt.select_indexes(plan)

        assert isinstance(optimized, LogicalProjection)
        idx_scan = optimized.input
        assert isinstance(idx_scan, LogicalIndexScan)
        assert idx_scan.index is real_bplus_tree
        # Verify real tree search matches
        assert idx_scan.index.search(25) == "user_row_25"


# ============================================================================
# 4. Projection Pruning Tests
# ============================================================================


class TestProjectionPruning:
    """Tests for Projection Pruning eliminating unreferenced columns."""

    def test_simple_select_pruning(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users"
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        scan = optimized.input
        assert isinstance(scan, LogicalScan)
        assert scan.columns == ["name"]

    def test_select_multiple_columns_pruning(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT id, email FROM users"
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        scan = optimized.input
        assert isinstance(scan, LogicalScan)
        assert "id" in scan.columns
        assert "email" in scan.columns
        assert "age" not in scan.columns
        assert "status" not in scan.columns

    def test_where_columns_retained_in_scan(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users WHERE age > 30"
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        filter_node = optimized.input
        assert isinstance(filter_node, LogicalFilter)
        scan = filter_node.input
        assert isinstance(scan, LogicalScan)
        # Both projected column 'name' and filter column 'age' must be retained
        assert "name" in scan.columns
        assert "age" in scan.columns
        assert "email" not in scan.columns

    def test_join_columns_pruning(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # users needs 'name' (projection) and 'id' (join condition)
        u_scan = join_node.left
        assert isinstance(u_scan, LogicalScan)
        assert "id" in u_scan.columns
        assert "name" in u_scan.columns
        assert "age" not in u_scan.columns
        assert "email" not in u_scan.columns

        # orders needs 'amount' (projection) and 'user_id' (join condition)
        o_scan = join_node.right
        assert isinstance(o_scan, LogicalScan)
        assert "amount" in o_scan.columns
        assert "user_id" in o_scan.columns
        assert "status" not in o_scan.columns

    def test_order_by_columns_retained(self, optimizer: RuleBasedOptimizer) -> None:
        sql = "SELECT name FROM users ORDER BY age DESC"
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        sort_node = optimized.input
        assert isinstance(sort_node, LogicalSort)
        scan = sort_node.input
        assert isinstance(scan, LogicalScan)
        assert "name" in scan.columns
        assert "age" in scan.columns

    def test_star_wildcard_retains_all_catalog_columns(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        sql = "SELECT * FROM users"
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        scan = optimized.input
        assert isinstance(scan, LogicalScan)
        catalog_cols = optimizer.get_table_columns("users")
        assert scan.columns == catalog_cols

    def test_table_wildcard_pruning_with_join(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.*, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.prune_projections(plan)

        assert isinstance(optimized, LogicalProjection)
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # users should retain all catalog columns due to u.*
        u_scan = join_node.left
        assert u_scan.columns == optimizer.get_table_columns("users")

        # orders only needs amount and user_id
        o_scan = join_node.right
        assert "amount" in o_scan.columns
        assert "user_id" in o_scan.columns
        assert "status" not in o_scan.columns

    def test_projection_pruning_on_index_scan(self, optimizer: RuleBasedOptimizer) -> None:
        # Pre-build an IndexScan with extra columns
        idx_scan = LogicalIndexScan(
            table_name="users",
            index_column="id",
            operator="=",
            value=10,
            columns=["id", "name", "email", "age"],
        )
        proj = LogicalProjection(
            input=idx_scan,
            expressions=[SelectItem(ColumnRef("name"))],
        )

        optimized = optimizer.prune_projections(proj)
        assert isinstance(optimized, LogicalProjection)
        scan = optimized.input
        assert isinstance(scan, LogicalIndexScan)
        # Must retain 'name' (projected) AND 'id' (index key)
        assert "name" in scan.columns
        assert "id" in scan.columns
        assert "email" not in scan.columns

    def test_eliminate_nested_projections(self, optimizer: RuleBasedOptimizer) -> None:
        scan = LogicalScan(table_name="users")
        inner_proj = LogicalProjection(
            input=scan,
            expressions=[SelectItem(ColumnRef("id")), SelectItem(ColumnRef("name"))],
        )
        outer_proj = LogicalProjection(
            input=inner_proj,
            expressions=[SelectItem(ColumnRef("name"))],
        )

        optimized = optimizer.prune_projections(outer_proj)
        assert isinstance(optimized, LogicalProjection)
        # Inner projection collapsed directly into scan
        assert isinstance(optimized.input, LogicalScan)
        assert optimized.input.columns == ["name"]


# ============================================================================
# 5. Combined End-to-End Pipeline Tests
# ============================================================================


class TestCombinedOptimizerPipeline:
    """Tests executing the entire Rule-Based Optimizer pipeline end-to-end."""

    def test_join_with_pushdown_index_selection_and_pruning(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.id = 42 AND o.status = 'shipped'
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.optimize(plan)

        # 1. Top is LogicalProjection
        assert isinstance(optimized, LogicalProjection)

        # 2. Join node with eliminated top filter
        join_node = optimized.input
        assert isinstance(join_node, LogicalJoin)

        # 3. Left child: users filter was pushed down, matched index on 'id',
        # and converted to LogicalIndexScan!
        left_child = join_node.left
        assert isinstance(left_child, LogicalIndexScan)
        assert left_child.table_name == "users"
        assert left_child.index_column == "id"
        assert left_child.value == 42
        # Columns pruned to required: name and id
        assert "name" in left_child.columns
        assert "id" in left_child.columns
        assert "email" not in left_child.columns

        # 4. Right child: orders filter on 'status' was pushed down.
        # Since 'status' has no index, remains LogicalFilter over LogicalScan
        right_child = join_node.right
        assert isinstance(right_child, LogicalFilter)
        assert right_child.condition.left == ColumnRef("status", "o")
        assert isinstance(right_child.input, LogicalScan)
        # Orders columns pruned to: amount, user_id, status
        assert "amount" in right_child.input.columns
        assert "user_id" in right_child.input.columns
        assert "status" in right_child.input.columns
        assert "created_at" not in right_child.input.columns

    def test_three_way_join_complete_pipeline(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, p.title
        FROM users u
        JOIN orders o ON u.id = o.user_id
        JOIN products p ON o.order_id = p.product_id
        WHERE u.id = 10 AND p.product_id = 99
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.optimize(plan)

        assert isinstance(optimized, LogicalProjection)
        outer_join = optimized.input
        assert isinstance(outer_join, LogicalJoin)

        # products matched index on product_id -> LogicalIndexScan
        assert isinstance(outer_join.right, LogicalIndexScan)
        assert outer_join.right.table_name == "products"
        assert outer_join.right.index_column == "product_id"
        assert outer_join.right.value == 99

        # users matched index on id -> LogicalIndexScan
        inner_join = outer_join.left
        assert isinstance(inner_join, LogicalJoin)
        assert isinstance(inner_join.left, LogicalIndexScan)
        assert inner_join.left.table_name == "users"
        assert inner_join.left.index_column == "id"
        assert inner_join.left.value == 10

        # orders has no filter -> LogicalScan with pruned columns
        assert isinstance(inner_join.right, LogicalScan)
        assert "order_id" in inner_join.right.columns
        assert "user_id" in inner_join.right.columns

    def test_query_with_limit_and_order_by(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT name, age
        FROM users
        WHERE age >= 21 AND age <= 35
        ORDER BY age DESC
        LIMIT 5
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.optimize(plan)

        # Plan hierarchy: Limit -> Projection -> Sort -> IndexScan
        assert isinstance(optimized, LogicalLimit)
        assert optimized.limit == 5

        proj = optimized.input
        assert isinstance(proj, LogicalProjection)

        sort_node = proj.input
        assert isinstance(sort_node, LogicalSort)

        idx_scan = sort_node.input
        assert isinstance(idx_scan, LogicalIndexScan)
        assert idx_scan.index_column == "age"
        assert idx_scan.operator == "BETWEEN"
        assert idx_scan.value == (21, 35)
        assert "name" in idx_scan.columns
        assert "age" in idx_scan.columns

    def test_pipeline_idempotency(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.name, o.amount
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.id = 5
        """
        plan = parse_logical_plan(sql)
        opt1 = optimizer.optimize(plan)
        opt2 = optimizer.optimize(opt1)

        # Second optimization pass should preserve the exact same structure
        assert str(opt1) == str(opt2)
        assert opt1.pretty() == opt2.pretty()

    def test_expression_helper_functions(self) -> None:
        # split_conjuncts & combine_conjuncts
        c1 = BinaryOp(ColumnRef("a"), "=", Literal(1))
        c2 = BinaryOp(ColumnRef("b"), "=", Literal(2))
        c3 = BinaryOp(ColumnRef("c"), "=", Literal(3))

        combined = combine_conjuncts([c1, c2, c3])
        assert combined is not None
        split = split_conjuncts(combined)
        assert len(split) == 3
        assert split == [c1, c2, c3]

        assert combine_conjuncts([]) is None

        # get_referenced_columns
        cols = get_referenced_columns(combined)
        assert cols == {(None, "a"), (None, "b"), (None, "c")}

        # get_plan_tables
        scan_a = LogicalScan(table_name="table_a", alias="ta")
        scan_b = LogicalScan(table_name="table_b")
        join = LogicalJoin(left=scan_a, right=scan_b)
        tbls = get_plan_tables(join)
        assert tbls == {"table_a": "table_a", "ta": "table_a", "table_b": "table_b"}

    def test_pushdown_right_join(self, optimizer: RuleBasedOptimizer) -> None:
        scan_l = LogicalScan(table_name="users", alias="u")
        scan_r = LogicalScan(table_name="orders", alias="o")
        join = LogicalJoin(
            left=scan_l,
            right=scan_r,
            join_type=JoinType.RIGHT,
            condition=BinaryOp(ColumnRef("id", "u"), "=", ColumnRef("user_id", "o")),
        )
        filter_node = LogicalFilter(
            input=join,
            condition=BinaryOp(ColumnRef("amount", "o"), ">", Literal(100)),
        )
        optimized = optimizer.pushdown_predicates(filter_node)
        assert isinstance(optimized, LogicalJoin)
        assert isinstance(optimized.right, LogicalFilter)
        assert optimized.right.input.table_name == "orders"

    def test_pushdown_full_outer_join_retains_predicates_above_join(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        scan_l = LogicalScan(table_name="users", alias="u")
        scan_r = LogicalScan(table_name="orders", alias="o")
        join = LogicalJoin(
            left=scan_l,
            right=scan_r,
            join_type=JoinType.FULL,
            condition=BinaryOp(ColumnRef("id", "u"), "=", ColumnRef("user_id", "o")),
        )
        filter_node = LogicalFilter(
            input=join,
            condition=BinaryOp(ColumnRef("age", "u"), ">", Literal(18)),
        )
        optimized = optimizer.pushdown_predicates(filter_node)
        assert isinstance(optimized, LogicalFilter)
        assert isinstance(optimized.input, LogicalJoin)

    def test_pushdown_cross_join(self, optimizer: RuleBasedOptimizer) -> None:
        scan_l = LogicalScan(table_name="users", alias="u")
        scan_r = LogicalScan(table_name="orders", alias="o")
        join = LogicalJoin(
            left=scan_l,
            right=scan_r,
            join_type=JoinType.CROSS,
        )
        filter_node = LogicalFilter(
            input=join,
            condition=BinaryOp(
                BinaryOp(ColumnRef("age", "u"), ">", Literal(21)),
                BinaryOpType.AND,
                BinaryOp(ColumnRef("amount", "o"), "<", Literal(500)),
            ),
        )
        optimized = optimizer.pushdown_predicates(filter_node)
        assert isinstance(optimized, LogicalJoin)
        assert isinstance(optimized.left, LogicalFilter)
        assert isinstance(optimized.right, LogicalFilter)

    def test_index_selection_in_list(self, optimizer: RuleBasedOptimizer) -> None:
        scan = LogicalScan(table_name="users")
        in_expr = BinaryOp(
            ColumnRef("id"),
            "IN",
            Literal([1, 2, 3]),
        )
        filter_node = LogicalFilter(input=scan, condition=in_expr)
        optimized = optimizer.select_indexes(filter_node)
        assert isinstance(optimized, LogicalIndexScan)
        assert optimized.index_column == "id"
        assert optimized.operator == "IN"
        assert optimized.value == [1, 2, 3]

    def test_projection_pruning_arithmetic_and_function_expressions(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        from src.query.parser import FunctionCall
        scan = LogicalScan(table_name="products")
        proj = LogicalProjection(
            input=scan,
            expressions=[
                SelectItem(
                    BinaryOp(ColumnRef("price"), "*", Literal(1.2)),
                    alias="taxed_price",
                ),
                SelectItem(
                    FunctionCall(name="COUNT", args=[ColumnRef("title")]),
                    alias="cnt",
                ),
            ],
        )
        optimized = optimizer.prune_projections(proj)
        assert isinstance(optimized, LogicalProjection)
        assert isinstance(optimized.input, LogicalScan)
        # Referenced price and title
        assert "price" in optimized.input.columns
        assert "title" in optimized.input.columns
        assert "product_id" not in optimized.input.columns

    def test_pretty_formatting_and_rules_info(self, optimizer: RuleBasedOptimizer) -> None:
        idx_scan = LogicalIndexScan(
            table_name="users",
            index_column="id",
            operator="=",
            value=10,
            index_name="idx_users_id",
            columns=["id", "name"],
        )
        pretty_str = idx_scan.pretty()
        assert "LogicalIndexScan" in pretty_str
        assert "idx_users_id" in pretty_str
        assert "columns: id, name" in pretty_str

        # Rule names
        for rule in optimizer.rules:
            assert isinstance(rule.name, str)
            assert len(rule.name) > 0


# ============================================================================
# 6. Edge Cases & Exception Hierarchy Tests
# ============================================================================


class TestOptimizerEdgeCasesAndExceptions:
    """Tests for exception classes, partial pushdowns, inverted ranges, and catalog edge cases."""

    def test_exception_hierarchy(self) -> None:
        assert issubclass(RuleApplicationError, OptimizerError)
        assert issubclass(OptimizerError, Exception)

        with pytest.raises(RuleApplicationError):
            raise RuleApplicationError("Rule failed")

        with pytest.raises(OptimizerError):
            raise RuleApplicationError("Rule failed")

    def test_split_and_combine_conjuncts_edge_cases(self) -> None:
        assert split_conjuncts(None) == []
        assert combine_conjuncts([]) is None
        assert combine_conjuncts([None]) is None  # type: ignore[list-item]

        single = BinaryOp(ColumnRef("a"), "=", Literal(1))
        assert split_conjuncts(single) == [single]
        assert combine_conjuncts([single]) == single

    def test_filter_above_limit_optimizes_limit_child(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        # Construct: Filter over Limit over Join
        join = LogicalJoin(
            left=LogicalScan("users", alias="u"),
            right=LogicalScan("orders", alias="o"),
            join_type=JoinType.INNER,
            condition=BinaryOp(
                BinaryOp(ColumnRef("id", "u"), "=", ColumnRef("user_id", "o")),
                BinaryOpType.AND,
                BinaryOp(ColumnRef("age", "u"), ">", Literal(18)),
            ),
        )
        limit_node = LogicalLimit(input=join, limit=10)
        filter_node = LogicalFilter(
            input=limit_node,
            condition=BinaryOp(ColumnRef("status"), "=", Literal("active")),
        )

        optimized = optimizer.pushdown_predicates(filter_node)
        # Filter must remain above Limit
        assert isinstance(optimized, LogicalFilter)
        assert isinstance(optimized.input, LogicalLimit)
        # Inside limit, join condition with age > 18 was pushed into users
        inner_join = optimized.input.input
        assert isinstance(inner_join, LogicalJoin)
        assert isinstance(inner_join.left, LogicalFilter)
        assert inner_join.left.input.table_name == "users"

    def test_projection_partial_pushdown_with_aliases(
        self, optimizer: RuleBasedOptimizer
    ) -> None:
        scan = LogicalScan(table_name="users")
        proj = LogicalProjection(
            input=scan,
            expressions=[
                SelectItem(ColumnRef("id"), alias="uid"),
                SelectItem(ColumnRef("age")),
            ],
        )
        # uid = 10 cannot push (uid is alias), but age > 18 CAN push
        filter_node = LogicalFilter(
            input=proj,
            condition=BinaryOp(
                BinaryOp(ColumnRef("uid"), "=", Literal(10)),
                BinaryOpType.AND,
                BinaryOp(ColumnRef("age"), ">", Literal(18)),
            ),
        )

        optimized = optimizer.pushdown_predicates(filter_node)
        assert isinstance(optimized, LogicalFilter)
        assert optimized.condition.left == ColumnRef("uid")
        assert isinstance(optimized.input, LogicalProjection)
        # Child scan received age > 18 filter!
        assert isinstance(optimized.input.input, LogicalFilter)
        assert optimized.input.input.condition.left == ColumnRef("age")

    def test_inverted_literal_range_operators(self, optimizer: RuleBasedOptimizer) -> None:
        # 18 < age -> age > 18
        plan1 = parse_logical_plan("SELECT name FROM users WHERE 18 < age")
        opt1 = optimizer.select_indexes(plan1)
        assert isinstance(opt1.input, LogicalIndexScan)
        assert opt1.input.operator == ">"
        assert opt1.input.value == 18
        assert opt1.input.inclusive_start is False

        # 60 >= age -> age <= 60
        plan2 = parse_logical_plan("SELECT name FROM users WHERE 60 >= age")
        opt2 = optimizer.select_indexes(plan2)
        assert isinstance(opt2.input, LogicalIndexScan)
        assert opt2.input.operator == "<="
        assert opt2.input.value == 60
        assert opt2.input.inclusive_end is True

    def test_reversed_bounded_range_between(self, optimizer: RuleBasedOptimizer) -> None:
        # Upper bound first, then lower bound: age <= 50 AND age >= 25
        plan = parse_logical_plan("SELECT name FROM users WHERE age <= 50 AND age >= 25")
        opt = optimizer.select_indexes(plan)
        assert isinstance(opt.input, LogicalIndexScan)
        assert opt.input.operator == "BETWEEN"
        assert opt.input.start_key == 25
        assert opt.input.end_key == 50

    def test_optimize_plan_alias_function(self, sample_catalog: Catalog) -> None:
        plan = parse_logical_plan("SELECT id FROM users WHERE id = 1")
        res1 = optimize(plan, catalog=sample_catalog)
        res2 = optimize_plan(plan, catalog=sample_catalog)
        assert str(res1) == str(res2)
        assert isinstance(res1.input, LogicalIndexScan)

    def test_multi_index_selection_in_join(self, optimizer: RuleBasedOptimizer) -> None:
        sql = """
        SELECT u.id, o.order_id
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.id = 10 AND o.order_id = 99
        """
        plan = parse_logical_plan(sql)
        optimized = optimizer.optimize(plan)

        assert isinstance(optimized, LogicalProjection)
        join = optimized.input
        assert isinstance(join, LogicalJoin)
        # Both sides converted to LogicalIndexScan!
        assert isinstance(join.left, LogicalIndexScan)
        assert join.left.index_column == "id"
        assert join.left.value == 10

        assert isinstance(join.right, LogicalIndexScan)
        assert join.right.index_column == "order_id"
        assert join.right.value == 99

