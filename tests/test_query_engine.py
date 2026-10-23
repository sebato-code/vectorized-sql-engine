"""Comprehensive unit test suite for vectorized SQL query engine.

Covers:
- VectorBatch: creation, slicing, mask filtering, projections, conversions, schema validation.
- Vectorized Expressions: Col, Literal, comparisons, arithmetic, logical (3VL), NULL checks, LIKE.
- Volcano Execution Lifecycle: open, next, close, context manager, iterator protocol.
- SeqScan Operator: batching, projections, various source data types, empty data.
- Filter Operator: boolean expressions, predicates, batch skipping, chained filters.
- HashJoin Operator: two-phase (build/probe), INNER, LEFT, RIGHT, FULL, SEMI, ANTI, multi-key joins.
- Aggregate Operator: GROUP BY (single and multi-key), scalar aggregations, COUNT, SUM, AVG, MIN, MAX.
- Supplementary Operators: Project, Limit, OrderBy (multi-key with NULLS LAST).
- Complex End-to-End Query Pipelines.
"""

from typing import Any, Dict, List
import pytest

from src.query.engine import (
    DEFAULT_BATCH_SIZE,
    Aggregate,
    And,
    Avg,
    Col,
    ColumnNotFoundError,
    Count,
    Div,
    Eq,
    ExecutionError,
    ExecutionOperator,
    Filter,
    Ge,
    Gt,
    HashJoin,
    InList,
    IsNotNull,
    IsNull,
    Le,
    Like,
    Limit,
    Literal,
    Lt,
    Max,
    Min,
    Mod,
    Mul,
    Ne,
    Not,
    Or,
    OrderBy,
    Project,
    QueryEngineError,
    SchemaError,
    SeqScan,
    StringAgg,
    Sub,
    Sum,
    Table,
    VectorBatch,
    col,
    lit,
)


# ============================================================================
# 1. VectorBatch Tests
# ============================================================================


class TestVectorBatch:
    """Tests for VectorBatch columnar container."""

    def test_empty_batch(self) -> None:
        batch = VectorBatch()
        assert batch.num_rows == 0
        assert len(batch) == 0
        assert not batch
        assert batch.columns == []
        assert batch.to_rows() == []

    def test_batch_from_dict(self) -> None:
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 80.0, 72.5],
        }
        batch = VectorBatch(data)
        assert batch.num_rows == 3
        assert len(batch) == 3
        assert bool(batch) is True
        assert batch.columns == ["id", "name", "score"]
        assert batch["id"] == [1, 2, 3]
        assert batch.column("name") == ["Alice", "Bob", "Charlie"]
        assert "score" in batch
        assert "missing" not in batch

    def test_mismatched_column_lengths_raise_error(self) -> None:
        with pytest.raises(SchemaError, match="length.*does not match"):
            VectorBatch({"a": [1, 2], "b": [1, 2, 3]})

    def test_invalid_column_name_type_raises_error(self) -> None:
        with pytest.raises(SchemaError, match="Column name must be str"):
            VectorBatch({123: [1, 2]})  # type: ignore[dict-item]

    def test_row_indexing_and_out_of_bounds(self) -> None:
        batch = VectorBatch({"x": [10, 20, 30], "y": ["a", "b", "c"]})
        assert batch[0] == {"x": 10, "y": "a"}
        assert batch[-1] == {"x": 30, "y": "c"}

        with pytest.raises(IndexError):
            _ = batch[3]
        with pytest.raises(IndexError):
            _ = batch[-4]

    def test_column_not_found(self) -> None:
        batch = VectorBatch({"x": [1, 2]})
        with pytest.raises(ColumnNotFoundError):
            _ = batch["non_existent"]
        with pytest.raises(ColumnNotFoundError):
            _ = batch.column("non_existent")

    def test_vectorized_filter(self) -> None:
        batch = VectorBatch({"id": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        mask = [True, False, True, False]
        filtered = batch.filter(mask)
        assert filtered.num_rows == 2
        assert filtered["id"] == [1, 3]
        assert filtered["val"] == [10, 30]

    def test_vectorized_filter_all_or_none(self) -> None:
        batch = VectorBatch({"id": [1, 2, 3]})
        assert batch.filter([True, True, True]).num_rows == 3
        assert batch.filter([False, False, False]).num_rows == 0

    def test_vectorized_filter_mismatched_mask_length(self) -> None:
        batch = VectorBatch({"id": [1, 2, 3]})
        with pytest.raises(ValueError, match="Mask length 2 does not match"):
            batch.filter([True, False])

    def test_slice(self) -> None:
        batch = VectorBatch({"id": list(range(10))})
        s = batch.slice(2, 4)
        assert s.num_rows == 4
        assert s["id"] == [2, 3, 4, 5]

        # Slicing with step in subscript
        s_step = batch[0:6:2]
        assert s_step.num_rows == 3
        assert s_step["id"] == [0, 2, 4]

        # Slicing out of bounds
        assert batch.slice(20, 5).num_rows == 0

    def test_project(self) -> None:
        batch = VectorBatch({"a": [1], "b": [2], "c": [3]})
        proj = batch.project(["c", "a"])
        assert proj.columns == ["c", "a"]
        assert proj["c"] == [3]
        assert proj["a"] == [1]

        with pytest.raises(ColumnNotFoundError):
            batch.project(["a", "unknown"])

    def test_rename(self) -> None:
        batch = VectorBatch({"a": [1, 2], "b": [3, 4]})
        renamed = batch.rename({"a": "alpha"})
        assert renamed.columns == ["alpha", "b"]
        assert renamed["alpha"] == [1, 2]

    def test_with_column(self) -> None:
        batch = VectorBatch({"a": [1, 2]})
        updated = batch.with_column("b", [10, 20])
        assert updated.columns == ["a", "b"]
        assert updated["b"] == [10, 20]

        with pytest.raises(ValueError, match="does not match batch length"):
            batch.with_column("b", [10])

    def test_from_and_to_rows(self) -> None:
        rows = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        batch = VectorBatch.from_rows(rows)
        assert batch.num_rows == 2
        assert batch.to_rows() == rows
        assert batch.to_tuples(["y", "x"]) == [("a", 1), ("b", 2)]

    def test_from_rows_empty(self) -> None:
        batch = VectorBatch.from_rows([], columns=["a", "b"])
        assert batch.num_rows == 0
        assert batch.columns == ["a", "b"]

    def test_concat_batches(self) -> None:
        b1 = VectorBatch({"a": [1, 2], "b": [3, 4]})
        b2 = VectorBatch({"a": [5, 6], "b": [7, 8]})
        combined = VectorBatch.concat([b1, b2])
        assert combined.num_rows == 4
        assert combined["a"] == [1, 2, 5, 6]
        assert combined["b"] == [3, 4, 7, 8]

        # Concatenating with schema mismatch
        b3 = VectorBatch({"a": [9], "c": [10]})
        with pytest.raises(SchemaError, match="mismatched schemas"):
            VectorBatch.concat([b1, b3])

        # Concatenate empty list
        assert VectorBatch.concat([]).num_rows == 0

    def test_equality(self) -> None:
        b1 = VectorBatch({"a": [1, 2]})
        b2 = VectorBatch({"a": [1, 2]})
        b3 = VectorBatch({"a": [1, 3]})
        assert b1 == b2
        assert b1 != b3
        assert b1 != "not_a_batch"


# ============================================================================
# 2. Vectorized Expressions Tests
# ============================================================================


class TestExpressions:
    """Tests for AST expressions and vectorized evaluation."""

    def test_col_and_literal(self) -> None:
        batch = VectorBatch({"x": [10, 20, 30]})
        c = Col("x")
        l = Literal(5)
        assert c.evaluate(batch) == [10, 20, 30]
        assert l.evaluate(batch) == [5, 5, 5]

    def test_comparison_operators(self) -> None:
        batch = VectorBatch({"a": [1, 2, 3, None], "b": [2, 2, 2, 2]})
        eq = Col("a") == Col("b")
        ne = Col("a") != Col("b")
        lt = Col("a") < Col("b")
        le = Col("a") <= Col("b")
        gt = Col("a") > Col("b")
        ge = Col("a") >= Col("b")

        assert eq.evaluate(batch) == [False, True, False, None]
        assert ne.evaluate(batch) == [True, False, True, None]
        assert lt.evaluate(batch) == [True, False, False, None]
        assert le.evaluate(batch) == [True, True, False, None]
        assert gt.evaluate(batch) == [False, False, True, None]
        assert ge.evaluate(batch) == [False, True, True, None]

    def test_arithmetic_operators(self) -> None:
        batch = VectorBatch({"x": [10, 20, 30, 40], "y": [2, 4, 0, None]})
        add = Col("x") + Col("y")
        sub = Col("x") - Col("y")
        mul = Col("x") * 2
        div = Col("x") / Col("y")
        mod = Col("x") % Col("y")

        assert add.evaluate(batch) == [12, 24, 30, None]
        assert sub.evaluate(batch) == [8, 16, 30, None]
        assert mul.evaluate(batch) == [20, 40, 60, 80]
        assert div.evaluate(batch) == [5.0, 5.0, None, None]  # 30/0 -> None
        assert mod.evaluate(batch) == [0, 0, None, None]

    def test_logical_operators_and_three_valued_logic(self) -> None:
        batch = VectorBatch({"p": [True, True, False, False, None, None], "q": [True, False, True, False, True, False]})
        and_expr = Col("p") & Col("q")
        or_expr = Col("p") | Col("q")
        not_expr = ~Col("p")

        # Three valued logic verification
        # True & True = True, True & False = False, False & True = False, False & False = False, None & True = None, None & False = False
        assert and_expr.evaluate(batch) == [True, False, False, False, None, False]
        # True | True = True, True | False = True, False | True = True, False | False = False, None | True = True, None | False = None
        assert or_expr.evaluate(batch) == [True, True, True, False, True, None]
        assert not_expr.evaluate(batch) == [False, False, True, True, None, None]

    def test_null_checks(self) -> None:
        batch = VectorBatch({"val": [1, None, 3, None]})
        is_null = Col("val").is_null()
        is_not_null = Col("val").is_not_null()

        assert is_null.evaluate(batch) == [False, True, False, True]
        assert is_not_null.evaluate(batch) == [True, False, True, False]

    def test_in_list(self) -> None:
        batch = VectorBatch({"city": ["NY", "SF", "LA", "CHI", None]})
        in_expr = Col("city").in_list(["NY", "LA"])
        assert in_expr.evaluate(batch) == [True, False, True, False, None]

    def test_like_patterns(self) -> None:
        batch = VectorBatch({"name": ["Alice", "Albert", "Bob", "Charlie", None]})
        like_al = Col("name").like("Al%")
        like_underscore = Col("name").like("B_b")

        assert like_al.evaluate(batch) == [True, True, False, False, None]
        assert like_underscore.evaluate(batch) == [False, False, True, False, None]


# ============================================================================
# 3. Volcano Execution Lifecycle & Table Tests
# ============================================================================


class TestVolcanoLifecycle:
    """Tests for Volcano execution protocol and Table operations."""

    def test_table_operations(self) -> None:
        tbl = Table("users", schema=["id", "name"])
        tbl.insert({"id": 1, "name": "Alice"})
        tbl.insert_batch(VectorBatch.from_rows([{"id": 2, "name": "Bob"}]))
        assert len(tbl) == 2
        assert tbl.schema == ["id", "name"]

    def test_operator_context_manager(self) -> None:
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        scan = SeqScan(data, batch_size=2)
        with scan as s:
            assert s.is_open is True
            b1 = s.next()
            assert b1 is not None and b1.num_rows == 2
        assert scan.is_open is False

    def test_operator_iterator_protocol(self) -> None:
        data = [{"id": i} for i in range(5)]
        scan = SeqScan(data, batch_size=2)
        batches = list(scan)
        assert len(batches) == 3
        assert [b.num_rows for b in batches] == [2, 2, 1]

    def test_operator_execute_and_all_rows(self) -> None:
        data = [{"x": 10}, {"x": 20}]
        scan = SeqScan(data)
        rows = scan.execute_all_rows()
        assert rows == data


# ============================================================================
# 4. SeqScan Operator Tests
# ============================================================================


class TestSeqScan:
    """Tests for SeqScan operator."""

    def test_seq_scan_batching(self) -> None:
        data = [{"id": i, "val": f"v{i}"} for i in range(10)]
        scan = SeqScan(data, batch_size=3)
        scan.open()

        b1 = scan.next()
        assert b1 is not None and b1.num_rows == 3 and b1["id"] == [0, 1, 2]
        b2 = scan.next()
        assert b2 is not None and b2.num_rows == 3 and b2["id"] == [3, 4, 5]
        b3 = scan.next()
        assert b3 is not None and b3.num_rows == 3 and b3["id"] == [6, 7, 8]
        b4 = scan.next()
        assert b4 is not None and b4.num_rows == 1 and b4["id"] == [9]
        b5 = scan.next()
        assert b5 is None
        scan.close()

    def test_seq_scan_with_projection(self) -> None:
        data = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
        scan = SeqScan(data, projection=["a", "c"])
        rows = scan.execute_all_rows()
        assert rows == [{"a": 1, "c": 3}, {"a": 4, "c": 6}]

    def test_seq_scan_empty_data(self) -> None:
        scan = SeqScan([])
        scan.open()
        assert scan.next() is None
        scan.close()

    def test_seq_scan_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SeqScan([{"a": 1}], batch_size=0)


# ============================================================================
# 5. Filter Operator Tests
# ============================================================================


class TestFilter:
    """Tests for Filter operator."""

    def test_filter_simple_comparison(self) -> None:
        data = [{"x": 5}, {"x": 15}, {"x": 25}, {"x": 10}]
        scan = SeqScan(data, batch_size=2)
        flt = Filter(scan, Col("x") > 12)
        rows = flt.execute_all_rows()
        assert rows == [{"x": 15}, {"x": 25}]

    def test_filter_compound_boolean_expression(self) -> None:
        data = [
            {"name": "Alice", "age": 25, "active": True},
            {"name": "Bob", "age": 17, "active": True},
            {"name": "Charlie", "age": 30, "active": False},
            {"name": "Diana", "age": 22, "active": True},
        ]
        scan = SeqScan(data, batch_size=2)
        # (age >= 21) & (active == True)
        pred = (Col("age") >= 21) & (Col("active") == True)
        flt = Filter(scan, pred)
        rows = flt.execute_all_rows()
        assert rows == [
            {"name": "Alice", "age": 25, "active": True},
            {"name": "Diana", "age": 22, "active": True},
        ]

    def test_filter_skips_batches_with_zero_matches(self) -> None:
        # First 4 rows will not match, only the 5th matches
        data = [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}, {"v": 100}]
        scan = SeqScan(data, batch_size=2)
        flt = Filter(scan, Col("v") > 50)
        flt.open()

        # The first call to next() must skip batches [1,2] and [3,4] and return [100]
        batch = flt.next()
        assert batch is not None
        assert batch["v"] == [100]
        assert flt.next() is None
        flt.close()

    def test_filter_with_callable_predicate(self) -> None:
        data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        scan = SeqScan(data)
        flt = Filter(scan, lambda r: r["text"].startswith("h"))
        rows = flt.execute_all_rows()
        assert rows == [{"id": 1, "text": "hello"}]

    def test_chained_filters(self) -> None:
        data = [{"n": i} for i in range(20)]
        scan = SeqScan(data, batch_size=5)
        f1 = Filter(scan, Col("n") >= 5)
        f2 = Filter(f1, Col("n") < 10)
        rows = f2.execute_all_rows()
        assert [r["n"] for r in rows] == [5, 6, 7, 8, 9]


# ============================================================================
# 6. HashJoin Operator Tests (Two-Phase In-Memory Build / Probe)
# ============================================================================


class TestHashJoin:
    """Tests for two-phase HashJoin operator."""

    @pytest.fixture
    def employees(self) -> List[Dict[str, Any]]:
        return [
            {"emp_id": 1, "name": "Alice", "dept_id": 10},
            {"emp_id": 2, "name": "Bob", "dept_id": 20},
            {"emp_id": 3, "name": "Charlie", "dept_id": 20},
            {"emp_id": 4, "name": "Diana", "dept_id": 99},  # Unmatched
            {"emp_id": 5, "name": "Evan", "dept_id": None},  # NULL join key
        ]

    @pytest.fixture
    def departments(self) -> List[Dict[str, Any]]:
        return [
            {"dept_id": 10, "dept_name": "Engineering"},
            {"dept_id": 20, "dept_name": "Marketing"},
            {"dept_id": 30, "dept_name": "Finance"},  # Unmatched
            {"dept_id": None, "dept_name": "Unknown"},  # NULL join key
        ]

    def test_inner_hash_join(
        self, employees: List[Dict[str, Any]], departments: List[Dict[str, Any]]
    ) -> None:
        left = SeqScan(employees, batch_size=2)
        right = SeqScan(departments, batch_size=2)
        join = HashJoin(left, right, left_on="dept_id", right_on="dept_id", join_type="INNER")
        rows = join.execute_all_rows()

        assert len(rows) == 3
        names = {r["name"] for r in rows}
        assert names == {"Alice", "Bob", "Charlie"}
        # Column collision disambiguation: right dept_id became right_dept_id
        for r in rows:
            assert r["dept_id"] == r["right_dept_id"]

    def test_left_outer_hash_join(
        self, employees: List[Dict[str, Any]], departments: List[Dict[str, Any]]
    ) -> None:
        left = SeqScan(employees, batch_size=2)
        right = SeqScan(departments, batch_size=2)
        join = HashJoin(left, right, left_on="dept_id", right_on="dept_id", join_type="LEFT")
        rows = join.execute_all_rows()

        assert len(rows) == 5
        diana = next(r for r in rows if r["name"] == "Diana")
        assert diana["dept_id"] == 99
        assert diana["dept_name"] is None

        evan = next(r for r in rows if r["name"] == "Evan")
        assert evan["dept_id"] is None
        assert evan["dept_name"] is None

    def test_right_outer_hash_join(
        self, employees: List[Dict[str, Any]], departments: List[Dict[str, Any]]
    ) -> None:
        left = SeqScan(employees, batch_size=2)
        right = SeqScan(departments, batch_size=2)
        join = HashJoin(left, right, left_on="dept_id", right_on="dept_id", join_type="RIGHT")
        rows = join.execute_all_rows()

        # Alice (10), Bob (20), Charlie (20), Finance (30, null emp), Unknown (null dept, null emp)
        assert len(rows) == 5
        finance = next(r for r in rows if r["dept_name"] == "Finance")
        assert finance["name"] is None
        assert finance["right_dept_id"] == 30

    def test_full_outer_hash_join(
        self, employees: List[Dict[str, Any]], departments: List[Dict[str, Any]]
    ) -> None:
        left = SeqScan(employees, batch_size=2)
        right = SeqScan(departments, batch_size=2)
        join = HashJoin(left, right, left_on="dept_id", right_on="dept_id", join_type="FULL")
        rows = join.execute_all_rows()

        # 3 inner + 2 unmatched left (Diana, Evan) + 2 unmatched right (Finance, Unknown) = 7
        assert len(rows) == 7

    def test_semi_hash_join(
        self, employees: List[Dict[str, Any]], departments: List[Dict[str, Any]]
    ) -> None:
        left = SeqScan(employees, batch_size=2)
        right = SeqScan(departments, batch_size=2)
        join = HashJoin(left, right, left_on="dept_id", right_on="dept_id", join_type="SEMI")
        rows = join.execute_all_rows()

        # Only employees that have a matching department
        assert len(rows) == 3
        assert {r["name"] for r in rows} == {"Alice", "Bob", "Charlie"}
        # SEMI join emits only left table columns
        assert "dept_name" not in rows[0]

    def test_anti_hash_join(
        self, employees: List[Dict[str, Any]], departments: List[Dict[str, Any]]
    ) -> None:
        left = SeqScan(employees, batch_size=2)
        right = SeqScan(departments, batch_size=2)
        join = HashJoin(left, right, left_on="dept_id", right_on="dept_id", join_type="ANTI")
        rows = join.execute_all_rows()

        # Diana (dept 99) and Evan (dept None)
        assert len(rows) == 2
        assert {r["name"] for r in rows} == {"Diana", "Evan"}

    def test_multi_column_join(self) -> None:
        t1 = [
            {"tenant": 1, "user": 10, "role": "admin"},
            {"tenant": 1, "user": 20, "role": "editor"},
            {"tenant": 2, "user": 10, "role": "viewer"},
        ]
        t2 = [
            {"t_id": 1, "u_id": 10, "email": "admin@t1.com"},
            {"t_id": 2, "u_id": 10, "email": "viewer@t2.com"},
        ]
        join = HashJoin(
            SeqScan(t1),
            SeqScan(t2),
            left_on=["tenant", "user"],
            right_on=["t_id", "u_id"],
            join_type="INNER",
        )
        rows = join.execute_all_rows()
        assert len(rows) == 2
        assert {r["email"] for r in rows} == {"admin@t1.com", "viewer@t2.com"}

    def test_custom_join_prefixes(self) -> None:
        t1 = [{"id": 1, "val": "A"}]
        t2 = [{"id": 1, "val": "B"}]
        join = HashJoin(
            SeqScan(t1),
            SeqScan(t2),
            left_on="id",
            right_on="id",
            left_prefix="l_",
            right_prefix="r_",
        )
        rows = join.execute_all_rows()
        assert rows == [{"l_id": 1, "l_val": "A", "r_id": 1, "r_val": "B"}]

    def test_hash_join_with_empty_inputs(self) -> None:
        empty = SeqScan([], batch_size=2)
        non_empty = SeqScan([{"id": 1, "name": "A"}], batch_size=2)

        # Empty right (build) side
        j1 = HashJoin(non_empty, empty, left_on="id", right_on="id", join_type="INNER")
        assert j1.execute_all_rows() == []

        # Empty left (probe) side
        j2 = HashJoin(SeqScan([]), SeqScan([{"id": 1}]), left_on="id", right_on="id", join_type="INNER")
        assert j2.execute_all_rows() == []

    def test_hash_join_invalid_parameters(self) -> None:
        s1 = SeqScan([{"a": 1}])
        s2 = SeqScan([{"b": 1}])
        with pytest.raises(ValueError, match="Number of join keys must match"):
            HashJoin(s1, s2, left_on=["a", "b"], right_on=["b"])

        with pytest.raises(ValueError, match="Unsupported join_type"):
            HashJoin(s1, s2, left_on="a", right_on="b", join_type="INVALID")


# ============================================================================
# 7. Aggregate Operator Tests (GROUP BY & Scalar Aggregations)
# ============================================================================


class TestAggregate:
    """Tests for Aggregate operator with GROUP BY and aggregate functions."""

    @pytest.fixture
    def sales_data(self) -> List[Dict[str, Any]]:
        return [
            {"region": "North", "dept": "Electronics", "sales": 100, "qty": 2},
            {"region": "North", "dept": "Electronics", "sales": 200, "qty": 4},
            {"region": "North", "dept": "Clothing", "sales": 50, "qty": 1},
            {"region": "South", "dept": "Electronics", "sales": 300, "qty": 5},
            {"region": "South", "dept": "Clothing", "sales": 150, "qty": 3},
            {"region": "South", "dept": "Clothing", "sales": None, "qty": 0},  # NULL sales
        ]

    def test_single_column_group_by(self, sales_data: List[Dict[str, Any]]) -> None:
        scan = SeqScan(sales_data, batch_size=2)
        agg = Aggregate(
            scan,
            group_by="region",
            aggregates=[
                Count("*", alias="total_rows"),
                Count("sales", alias="sales_rows"),
                Sum("sales", alias="total_sales"),
                Avg("sales", alias="avg_sales"),
                Min("sales", alias="min_sales"),
                Max("sales", alias="max_sales"),
            ],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 2

        north = next(r for r in rows if r["region"] == "North")
        assert north["total_rows"] == 3
        assert north["sales_rows"] == 3
        assert north["total_sales"] == 350
        assert north["avg_sales"] == pytest.approx(350 / 3)
        assert north["min_sales"] == 50
        assert north["max_sales"] == 200

        south = next(r for r in rows if r["region"] == "South")
        assert south["total_rows"] == 3
        assert south["sales_rows"] == 2  # NULL excluded
        assert south["total_sales"] == 450
        assert south["avg_sales"] == 225.0
        assert south["min_sales"] == 150
        assert south["max_sales"] == 300

    def test_multi_column_group_by(self, sales_data: List[Dict[str, Any]]) -> None:
        scan = SeqScan(sales_data, batch_size=2)
        agg = Aggregate(
            scan,
            group_by=["region", "dept"],
            aggregates=[
                Count("*", alias="cnt"),
                Sum("sales", alias="sum_sales"),
            ],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 4

        key_map = {(r["region"], r["dept"]): r for r in rows}
        assert key_map[("North", "Electronics")]["cnt"] == 2
        assert key_map[("North", "Electronics")]["sum_sales"] == 300
        assert key_map[("South", "Clothing")]["cnt"] == 2
        assert key_map[("South", "Clothing")]["sum_sales"] == 150

    def test_scalar_aggregate_without_group_by(self, sales_data: List[Dict[str, Any]]) -> None:
        scan = SeqScan(sales_data, batch_size=2)
        agg = Aggregate(
            scan,
            aggregates=[
                Count("*", alias="cnt"),
                Sum("sales", alias="total_sales"),
                Avg("sales", alias="avg_sales"),
            ],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 1
        assert rows[0]["cnt"] == 6
        assert rows[0]["total_sales"] == 800
        assert rows[0]["avg_sales"] == pytest.approx(800 / 5)

    def test_scalar_aggregate_on_empty_dataset(self) -> None:
        scan = SeqScan([])
        agg = Aggregate(
            scan,
            aggregates=[
                Count("*", alias="cnt"),
                Sum("x", alias="total"),
                Avg("x", alias="avg"),
                Min("x", alias="min"),
                Max("x", alias="max"),
            ],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 1
        assert rows[0]["cnt"] == 0
        assert rows[0]["total"] is None
        assert rows[0]["avg"] is None
        assert rows[0]["min"] is None
        assert rows[0]["max"] is None

    def test_group_by_on_empty_dataset(self) -> None:
        scan = SeqScan([])
        agg = Aggregate(scan, group_by="dept", aggregates=[Count("*", alias="cnt")])
        rows = agg.execute_all_rows()
        assert len(rows) == 0

    def test_distinct_aggregations(self) -> None:
        data = [
            {"grp": "A", "val": 10},
            {"grp": "A", "val": 10},
            {"grp": "A", "val": 20},
            {"grp": "A", "val": None},
        ]
        agg = Aggregate(
            SeqScan(data),
            group_by="grp",
            aggregates=[
                Count("val", alias="cnt_distinct", distinct=True),
                Sum("val", alias="sum_distinct", distinct=True),
                Avg("val", alias="avg_distinct", distinct=True),
            ],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 1
        assert rows[0]["cnt_distinct"] == 2
        assert rows[0]["sum_distinct"] == 30
        assert rows[0]["avg_distinct"] == 15.0

    def test_string_agg_function(self) -> None:
        data = [
            {"cat": "books", "title": "SQL Guide"},
            {"cat": "books", "title": "Python 101"},
            {"cat": "books", "title": None},
        ]
        agg = Aggregate(
            SeqScan(data),
            group_by="cat",
            aggregates=[StringAgg("title", delimiter="; ", alias="titles")],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 1
        assert rows[0]["titles"] == "SQL Guide; Python 101"

    def test_aggregates_passed_as_dict(self) -> None:
        data = [{"x": 10}, {"x": 20}]
        agg = Aggregate(
            SeqScan(data),
            aggregates={"c": Count("*"), "s": Sum("x")},
        )
        rows = agg.execute_all_rows()
        assert rows == [{"c": 2, "s": 30}]


# ============================================================================
# 8. Supplementary Operators Tests (Project, Limit, OrderBy)
# ============================================================================


class TestSupplementaryOperators:
    """Tests for Project, Limit, and OrderBy operators."""

    def test_project_operator(self) -> None:
        data = [{"a": 10, "b": 2}, {"a": 20, "b": 5}]
        scan = SeqScan(data)
        proj = Project(
            scan,
            projections={
                "sum": Col("a") + Col("b"),
                "product": Col("a") * Col("b"),
                "const": Literal(42),
            },
        )
        rows = proj.execute_all_rows()
        assert rows == [
            {"sum": 12, "product": 20, "const": 42},
            {"sum": 25, "product": 100, "const": 42},
        ]

    def test_limit_and_offset(self) -> None:
        data = [{"id": i} for i in range(10)]
        scan = SeqScan(data, batch_size=3)
        lim = Limit(scan, limit=4, offset=3)
        rows = lim.execute_all_rows()
        assert [r["id"] for r in rows] == [3, 4, 5, 6]

    def test_limit_zero_or_offset_beyond_data(self) -> None:
        data = [{"id": 1}, {"id": 2}]
        assert Limit(SeqScan(data), limit=0).execute_all_rows() == []
        assert Limit(SeqScan(data), limit=5, offset=10).execute_all_rows() == []

    def test_order_by_single_column_asc_desc(self) -> None:
        data = [{"val": 30}, {"val": 10}, {"val": None}, {"val": 20}]
        asc_rows = OrderBy(SeqScan(data), by="val", ascending=True).execute_all_rows()
        assert [r["val"] for r in asc_rows] == [10, 20, 30, None]

        desc_rows = OrderBy(SeqScan(data), by="val", ascending=False).execute_all_rows()
        assert [r["val"] for r in desc_rows] == [30, 20, 10, None]

    def test_order_by_multi_column(self) -> None:
        data = [
            {"dept": "Eng", "salary": 100},
            {"dept": "HR", "salary": 80},
            {"dept": "Eng", "salary": 200},
            {"dept": "HR", "salary": 120},
        ]
        # Sort dept ASC, salary DESC
        ob = OrderBy(SeqScan(data), by=[("dept", True), ("salary", False)])
        rows = ob.execute_all_rows()
        assert rows == [
            {"dept": "Eng", "salary": 200},
            {"dept": "Eng", "salary": 100},
            {"dept": "HR", "salary": 120},
            {"dept": "HR", "salary": 80},
        ]


# ============================================================================
# 9. Complex End-to-End Query Pipelines
# ============================================================================


class TestEndToEndPipelines:
    """Tests executing realistic end-to-end relational query plans."""

    def test_full_pipeline_scan_filter_join_aggregate_order_limit(self) -> None:
        # Simulate:
        # SELECT u.country, COUNT(*) as user_count, SUM(o.amount) as total_spent
        # FROM users u
        # JOIN orders o ON u.user_id = o.user_id
        # WHERE u.age >= 18 AND o.status = 'COMPLETED'
        # GROUP BY u.country
        # HAVING total_spent > 100
        # ORDER BY total_spent DESC
        # LIMIT 2
        users_table = [
            {"user_id": 1, "country": "US", "age": 25},
            {"user_id": 2, "country": "US", "age": 30},
            {"user_id": 3, "country": "UK", "age": 16},  # filtered out by age
            {"user_id": 4, "country": "UK", "age": 40},
            {"user_id": 5, "country": "CA", "age": 22},
        ]
        orders_table = [
            {"order_id": 101, "user_id": 1, "amount": 150, "status": "COMPLETED"},
            {"order_id": 102, "user_id": 1, "amount": 50, "status": "CANCELLED"},  # filtered out
            {"order_id": 103, "user_id": 2, "amount": 300, "status": "COMPLETED"},
            {"order_id": 104, "user_id": 3, "amount": 500, "status": "COMPLETED"},
            {"order_id": 105, "user_id": 4, "amount": 80, "status": "COMPLETED"},
            {"order_id": 106, "user_id": 5, "amount": 400, "status": "COMPLETED"},
        ]

        # 1. Scans with batch size 2
        scan_users = SeqScan(users_table, batch_size=2)
        scan_orders = SeqScan(orders_table, batch_size=2)

        # 2. Filters
        filtered_users = Filter(scan_users, Col("age") >= 18)
        filtered_orders = Filter(scan_orders, Col("status") == "COMPLETED")

        # 3. Two-Phase Hash Join
        joined = HashJoin(
            filtered_users,
            filtered_orders,
            left_on="user_id",
            right_on="user_id",
            join_type="INNER",
            batch_size=2,
        )

        # 4. Aggregation with GROUP BY
        aggregated = Aggregate(
            joined,
            group_by="country",
            aggregates=[
                Count("*", alias="user_count"),
                Sum("amount", alias="total_spent"),
            ],
            batch_size=2,
        )

        # 5. HAVING filter (total_spent > 100)
        having = Filter(aggregated, Col("total_spent") > 100)

        # 6. Order By total_spent DESC
        ordered = OrderBy(having, by="total_spent", ascending=False)

        # 7. Limit 2
        limited = Limit(ordered, limit=2)

        # Execute
        results = limited.execute_all_rows()

        # Expected:
        # US: user 1 (150) + user 2 (300) = 450
        # CA: user 5 (400) = 400
        # UK: user 4 (80) -> filtered out by HAVING
        assert len(results) == 2
        assert results[0] == {"country": "US", "user_count": 2, "total_spent": 450}
        assert results[1] == {"country": "CA", "user_count": 1, "total_spent": 400}

    def test_hash_join_many_to_many_matches(self) -> None:
        # Left has 3 rows with key 'A', Right has 2 rows with key 'A' -> 6 rows produced
        left_data = [{"k": "A", "lv": 1}, {"k": "A", "lv": 2}, {"k": "A", "lv": 3}]
        right_data = [{"k": "A", "rv": 10}, {"k": "A", "rv": 20}]
        join = HashJoin(
            SeqScan(left_data, batch_size=1),
            SeqScan(right_data, batch_size=1),
            left_on="k",
            right_on="k",
            join_type="INNER",
            batch_size=2,
        )
        rows = join.execute_all_rows()
        assert len(rows) == 6
        lvs = [r["lv"] for r in rows]
        rvs = [r["rv"] for r in rows]
        assert sorted(lvs) == [1, 1, 2, 2, 3, 3]
        assert sorted(rvs) == [10, 10, 10, 20, 20, 20]

    def test_volcano_repeated_next_after_eof(self) -> None:
        scan = SeqScan([{"id": 1}], batch_size=1)
        scan.open()
        assert scan.next() is not None
        assert scan.next() is None
        # Subsequent next() calls must continue to return None deterministically
        assert scan.next() is None
        assert scan.next() is None
        scan.close()

    def test_volcano_reopen_and_reuse(self) -> None:
        scan = SeqScan([{"id": 1}, {"id": 2}], batch_size=1)
        # First execution pass
        pass1 = scan.execute_all_rows()
        assert pass1 == [{"id": 1}, {"id": 2}]
        # Second execution pass (re-opening after close)
        pass2 = scan.execute_all_rows()
        assert pass2 == [{"id": 1}, {"id": 2}]

    def test_filter_complex_expression_with_arithmetic(self) -> None:
        data = [
            {"a": 10, "b": 20, "c": "yes"},  # 10*2 + 5 = 25 > 20, c not null -> True
            {"a": 5, "b": 20, "c": "yes"},   # 5*2 + 5 = 15 <= 20 -> False
            {"a": 10, "b": 20, "c": None},   # c is null -> False
        ]
        scan = SeqScan(data)
        expr = ((Col("a") * 2 + 5) > Col("b")) & Col("c").is_not_null()
        flt = Filter(scan, expr)
        rows = flt.execute_all_rows()
        assert rows == [{"a": 10, "b": 20, "c": "yes"}]

    def test_aggregate_all_nulls_group(self) -> None:
        data = [
            {"grp": "G1", "val": None},
            {"grp": "G1", "val": None},
        ]
        agg = Aggregate(
            SeqScan(data),
            group_by="grp",
            aggregates=[
                Count("val", alias="cnt"),
                Sum("val", alias="sum"),
                Avg("val", alias="avg"),
                Min("val", alias="min"),
                Max("val", alias="max"),
            ],
        )
        rows = agg.execute_all_rows()
        assert len(rows) == 1
        assert rows[0] == {
            "grp": "G1",
            "cnt": 0,
            "sum": None,
            "avg": None,
            "min": None,
            "max": None,
        }

    def test_stress_large_dataset_batch_transitions(self) -> None:
        # 1000 rows with unusual batch size (37) to test irregular boundary handling
        n_rows = 1000
        left_data = [{"id": i, "cat": f"cat_{i % 10}", "val": i * 1.5} for i in range(n_rows)]
        right_data = [{"cat_id": f"cat_{i}", "mult": i * 2} for i in range(10)]

        scan_l = SeqScan(left_data, batch_size=37)
        scan_r = SeqScan(right_data, batch_size=7)

        # Filter out odd cats
        flt = Filter(scan_l, (Col("id") % 2) == 0)

        # HashJoin on category
        join = HashJoin(
            flt,
            scan_r,
            left_on="cat",
            right_on="cat_id",
            join_type="INNER",
            batch_size=37,
        )

        # Aggregate by category: count and sum of val
        agg = Aggregate(
            join,
            group_by="cat",
            aggregates=[
                Count("*", alias="cnt"),
                Sum("val", alias="total_val"),
            ],
            batch_size=37,
        )

        results = agg.execute_all_rows()
        assert len(results) == 5

        # Only even categories (cat_0, cat_2, cat_4, cat_6, cat_8), each with 100 rows
        for r in results:
            assert r["cnt"] == 100

