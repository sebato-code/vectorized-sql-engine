"""Unit tests for SQL Parser and AST."""

import pytest

from src.query.parser import (
    BinaryOp,
    BinaryOpType,
    ColumnDef,
    ColumnRef,
    CreateTableStatement,
    Expression,
    FilterNode,
    FunctionCall,
    InsertStatement,
    JoinClause,
    JoinNode,
    JoinType,
    LimitNode,
    Literal,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalPlan,
    LogicalPlanNode,
    LogicalProjection,
    LogicalScan,
    LogicalSort,
    OrderByItem,
    Parser,
    ParserError,
    ProjectionNode,
    SQLSyntaxError,
    ScanNode,
    SelectItem,
    SelectStatement,
    SortNode,
    Star,
    Statement,
    TableRef,
    TableScanNode,
    UnaryOp,
    UnexpectedTokenError,
    build_logical_plan,
    parse,
    parse_logical_plan,
    parse_sql,
    parse_statements,
    parse_to_logical_plan,
    to_logical_plan,
)


# ===========================================================================
# CREATE TABLE Tests
# ===========================================================================

class TestCreateTable:
    def test_create_table_basic(self) -> None:
        sql = "CREATE TABLE users (id INT, name TEXT)"
        stmt = parse(sql)
        assert isinstance(stmt, CreateTableStatement)
        assert stmt.table_name == "users"
        assert stmt.if_not_exists is False
        assert len(stmt.columns) == 2

        col0 = stmt.columns[0]
        assert col0.name == "id"
        assert col0.data_type == "INT"
        assert col0.primary_key is False
        assert col0.nullable is True

        col1 = stmt.columns[1]
        assert col1.name == "name"
        assert col1.data_type == "TEXT"
        assert col1.primary_key is False
        assert col1.nullable is True

    def test_create_table_with_constraints(self) -> None:
        sql = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            sku VARCHAR(64) NOT NULL UNIQUE,
            price FLOAT DEFAULT 9.99,
            active BOOLEAN DEFAULT TRUE,
            stock INT AUTO_INCREMENT
        )
        """
        stmt = parse(sql)
        assert isinstance(stmt, CreateTableStatement)
        assert stmt.table_name == "items"
        assert len(stmt.columns) == 5

        col_id = stmt.get_column("id")
        assert col_id is not None
        assert col_id.primary_key is True
        assert col_id.is_primary_key is True
        assert col_id.nullable is False

        col_sku = stmt.get_column("sku")
        assert col_sku is not None
        assert col_sku.data_type == "VARCHAR(64)"
        assert col_sku.nullable is False
        assert col_sku.unique is True

        col_price = stmt.get_column("price")
        assert col_price is not None
        assert col_price.data_type == "FLOAT"
        assert col_price.default == 9.99

        col_active = stmt.get_column("active")
        assert col_active is not None
        assert col_active.data_type == "BOOLEAN"
        assert col_active.default is True

        col_stock = stmt.get_column("stock")
        assert col_stock is not None
        assert col_stock.auto_increment is True

    def test_create_table_if_not_exists(self) -> None:
        sql = "CREATE TABLE IF NOT EXISTS orders (order_id INT PRIMARY KEY, total DOUBLE)"
        stmt = parse(sql)
        assert isinstance(stmt, CreateTableStatement)
        assert stmt.table_name == "orders"
        assert stmt.if_not_exists is True
        assert stmt.column_names == ["order_id", "total"]

    def test_create_table_table_level_primary_key(self) -> None:
        sql = "CREATE TABLE accounts (acc_num INT, balance FLOAT, PRIMARY KEY (acc_num))"
        stmt = parse(sql)
        assert isinstance(stmt, CreateTableStatement)
        col = stmt.get_column("acc_num")
        assert col is not None
        assert col.primary_key is True
        assert col.nullable is False

    def test_create_table_case_insensitive(self) -> None:
        sql = "create table users (id integer primary key, name text not null)"
        stmt = parse(sql)
        assert isinstance(stmt, CreateTableStatement)
        assert stmt.table_name == "users"
        assert stmt.columns[0].data_type == "INTEGER"
        assert stmt.columns[0].primary_key is True
        assert stmt.columns[1].nullable is False

    def test_create_table_str_representation(self) -> None:
        sql = "CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name TEXT NOT NULL)"
        stmt = parse(sql)
        rendered = str(stmt)
        assert "CREATE TABLE IF NOT EXISTS users" in rendered
        assert "id INT PRIMARY KEY" in rendered
        assert "name TEXT" in rendered

    def test_create_table_syntax_errors(self) -> None:
        with pytest.raises(SQLSyntaxError):
            parse("CREATE TABLE")
        with pytest.raises(SQLSyntaxError):
            parse("CREATE TABLE users")
        with pytest.raises(SQLSyntaxError):
            parse("CREATE TABLE users (id INT")
        with pytest.raises(SQLSyntaxError):
            parse("CREATE TABLE IF EXISTS users (id INT)")


# ===========================================================================
# INSERT INTO Tests
# ===========================================================================

class TestInsertInto:
    def test_insert_with_columns(self) -> None:
        sql = "INSERT INTO users (id, name, score) VALUES (1, 'Alice', 95.5)"
        stmt = parse(sql)
        assert isinstance(stmt, InsertStatement)
        assert stmt.table_name == "users"
        assert stmt.columns == ["id", "name", "score"]
        assert len(stmt.values) == 1

        row = stmt.values[0]
        assert row == [Literal(1), Literal("Alice"), Literal(95.5)]
        assert stmt.raw_values == [[1, "Alice", 95.5]]

    def test_insert_without_columns(self) -> None:
        sql = "INSERT INTO users VALUES (1, 'Alice', TRUE, NULL)"
        stmt = parse(sql)
        assert isinstance(stmt, InsertStatement)
        assert stmt.table_name == "users"
        assert stmt.columns is None
        assert stmt.raw_values == [[1, "Alice", True, None]]

    def test_insert_multiple_rows(self) -> None:
        sql = "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')"
        stmt = parse(sql)
        assert isinstance(stmt, InsertStatement)
        assert len(stmt.values) == 3
        assert stmt.raw_values == [
            [1, "Alice"],
            [2, "Bob"],
            [3, "Charlie"],
        ]

    def test_insert_negative_numbers_and_escaped_strings(self) -> None:
        sql = "INSERT INTO logs VALUES (-10, -3.14, 'It''s fine')"
        stmt = parse(sql)
        assert isinstance(stmt, InsertStatement)
        assert stmt.raw_values == [[-10, -3.14, "It's fine"]]

    def test_insert_str_representation(self) -> None:
        sql = "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')"
        stmt = parse(sql)
        rendered = str(stmt)
        assert "INSERT INTO users (id, name) VALUES" in rendered
        assert "(1, 'Alice')" in rendered
        assert "(2, 'Bob')" in rendered

    def test_insert_syntax_errors(self) -> None:
        with pytest.raises(SQLSyntaxError):
            parse("INSERT users VALUES (1)")
        with pytest.raises(SQLSyntaxError):
            parse("INSERT INTO users (id) (1)")
        with pytest.raises(SQLSyntaxError):
            parse("INSERT INTO users VALUES (1,")


# ===========================================================================
# SELECT Projections Tests
# ===========================================================================

class TestSelectProjections:
    def test_select_star(self) -> None:
        sql = "SELECT * FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.projections) == 1
        assert isinstance(stmt.projections[0].expr, Star)
        assert stmt.from_table == "users"
        assert stmt.table_name == "users"

    def test_select_single_column(self) -> None:
        sql = "SELECT id FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.projections) == 1
        proj = stmt.projections[0]
        assert proj.name == "id"
        assert isinstance(proj.expr, ColumnRef)
        assert proj.expr.name == "id"
        assert proj.expr.table is None

    def test_select_multiple_columns(self) -> None:
        sql = "SELECT id, name, email FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.projections) == 3
        col_names = [p.name for p in stmt.projections]
        assert col_names == ["id", "name", "email"]

    def test_select_table_qualified_column(self) -> None:
        sql = "SELECT users.id, users.name FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        proj0 = stmt.projections[0]
        assert isinstance(proj0.expr, ColumnRef)
        assert proj0.expr.name == "id"
        assert proj0.expr.table == "users"
        assert proj0.table == "users"
        assert proj0.name == "id"

    def test_select_table_star(self) -> None:
        sql = "SELECT u.* FROM users AS u"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        proj0 = stmt.projections[0]
        assert isinstance(proj0.expr, Star)
        assert proj0.expr.table == "u"

    def test_select_alias_with_as(self) -> None:
        sql = "SELECT id AS user_id, name AS full_name FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.projections[0].alias == "user_id"
        assert stmt.projections[0].name == "user_id"
        assert stmt.projections[1].alias == "full_name"
        assert stmt.projections[1].name == "full_name"

    def test_select_alias_without_as(self) -> None:
        sql = "SELECT id user_id, name full_name FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.projections[0].alias == "user_id"
        assert stmt.projections[1].alias == "full_name"

    def test_select_distinct(self) -> None:
        sql = "SELECT DISTINCT category FROM items"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.distinct is True
        assert stmt.projections[0].name == "category"

    def test_select_literals_without_from(self) -> None:
        sql = "SELECT 1, 'hello', 3.14"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.from_table is None
        assert len(stmt.projections) == 3
        assert stmt.projections[0].expr == Literal(1)
        assert stmt.projections[1].expr == Literal("hello")
        assert stmt.projections[2].expr == Literal(3.14)

    def test_select_function_call(self) -> None:
        sql = "SELECT COUNT(*), MAX(age) AS max_age FROM users"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        f1 = stmt.projections[0].expr
        assert isinstance(f1, FunctionCall)
        assert f1.name == "COUNT"
        assert isinstance(f1.args[0], Star)

        f2 = stmt.projections[1].expr
        assert isinstance(f2, FunctionCall)
        assert f2.name == "MAX"
        assert stmt.projections[1].alias == "max_age"


# ===========================================================================
# SELECT FROM & Table Aliases Tests
# ===========================================================================

class TestSelectFrom:
    def test_from_table_alias_with_as(self) -> None:
        sql = "SELECT * FROM users AS u"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.from_table is not None
        assert stmt.from_table.name == "users"
        assert stmt.from_table.alias == "u"

    def test_from_table_alias_without_as(self) -> None:
        sql = "SELECT * FROM users u"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.from_table is not None
        assert stmt.from_table.name == "users"
        assert stmt.from_table.alias == "u"


# ===========================================================================
# SELECT WHERE Clause Tests (operators =, !=, <, >, AND, OR)
# ===========================================================================

class TestSelectWhere:
    def test_where_equals(self) -> None:
        sql = "SELECT * FROM users WHERE id = 1"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        w = stmt.where_clause
        assert isinstance(w, BinaryOp)
        assert w.op == "="
        assert w.left == ColumnRef("id")
        assert w.right == Literal(1)

    def test_where_not_equals(self) -> None:
        # != operator
        sql1 = "SELECT * FROM users WHERE status != 'banned'"
        stmt1 = parse(sql1)
        assert isinstance(stmt1.where, BinaryOp)
        assert stmt1.where.op == "!="
        assert stmt1.where.left == ColumnRef("status")
        assert stmt1.where.right == Literal("banned")

        # <> operator (SQL alternative)
        sql2 = "SELECT * FROM users WHERE status <> 'banned'"
        stmt2 = parse(sql2)
        assert isinstance(stmt2.where, BinaryOp)
        assert stmt2.where.op == "!="

    def test_where_less_than(self) -> None:
        sql = "SELECT * FROM users WHERE age < 18"
        stmt = parse(sql)
        assert isinstance(stmt.where, BinaryOp)
        assert stmt.where.op == "<"
        assert stmt.where.left == ColumnRef("age")
        assert stmt.where.right == Literal(18)

    def test_where_greater_than(self) -> None:
        sql = "SELECT * FROM users WHERE score > 50.5"
        stmt = parse(sql)
        assert isinstance(stmt.where, BinaryOp)
        assert stmt.where.op == ">"
        assert stmt.where.left == ColumnRef("score")
        assert stmt.where.right == Literal(50.5)

    def test_where_less_equals_and_greater_equals(self) -> None:
        sql = "SELECT * FROM users WHERE age <= 65 AND score >= 100"
        stmt = parse(sql)
        assert isinstance(stmt.where, BinaryOp)
        assert stmt.where.op == "AND"

        left = stmt.where.left
        assert isinstance(left, BinaryOp)
        assert left.op == "<="
        assert left.left == ColumnRef("age")
        assert left.right == Literal(65)

        right = stmt.where.right
        assert isinstance(right, BinaryOp)
        assert right.op == ">="
        assert right.left == ColumnRef("score")
        assert right.right == Literal(100)

    def test_where_and(self) -> None:
        sql = "SELECT * FROM users WHERE age > 21 AND active = TRUE"
        stmt = parse(sql)
        assert isinstance(stmt.where, BinaryOp)
        assert stmt.where.op == "AND"
        assert isinstance(stmt.where.left, BinaryOp)
        assert isinstance(stmt.where.right, BinaryOp)

    def test_where_or(self) -> None:
        sql = "SELECT * FROM users WHERE role = 'admin' OR role = 'moderator'"
        stmt = parse(sql)
        assert isinstance(stmt.where, BinaryOp)
        assert stmt.where.op == "OR"
        assert stmt.where.left.op == "="
        assert stmt.where.right.op == "="

    def test_where_operator_precedence_and_over_or(self) -> None:
        # In SQL, AND has higher precedence than OR.
        # "a = 1 OR b = 2 AND c = 3" should be parsed as:
        # a = 1 OR (b = 2 AND c = 3)
        sql = "SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3"
        stmt = parse(sql)
        w = stmt.where
        assert isinstance(w, BinaryOp)
        assert w.op == "OR"
        assert w.left == BinaryOp(ColumnRef("a"), "=", Literal(1))

        assert isinstance(w.right, BinaryOp)
        assert w.right.op == "AND"
        assert w.right.left == BinaryOp(ColumnRef("b"), "=", Literal(2))
        assert w.right.right == BinaryOp(ColumnRef("c"), "=", Literal(3))

    def test_where_parentheses_override_precedence(self) -> None:
        # "(a = 1 OR b = 2) AND c = 3"
        sql = "SELECT * FROM t WHERE (a = 1 OR b = 2) AND c = 3"
        stmt = parse(sql)
        w = stmt.where
        assert isinstance(w, BinaryOp)
        assert w.op == "AND"

        assert isinstance(w.left, BinaryOp)
        assert w.left.op == "OR"
        assert w.left.left == BinaryOp(ColumnRef("a"), "=", Literal(1))
        assert w.left.right == BinaryOp(ColumnRef("b"), "=", Literal(2))

        assert w.right == BinaryOp(ColumnRef("c"), "=", Literal(3))

    def test_where_not_operator(self) -> None:
        sql = "SELECT * FROM users WHERE NOT active = TRUE"
        stmt = parse(sql)
        w = stmt.where
        assert isinstance(w, UnaryOp)
        assert w.op == "NOT"
        assert isinstance(w.operand, BinaryOp)
        assert w.operand.op == "="

    def test_where_is_null_and_is_not_null(self) -> None:
        sql = "SELECT * FROM users WHERE deleted_at IS NULL AND email IS NOT NULL"
        stmt = parse(sql)
        w = stmt.where
        assert isinstance(w, BinaryOp)
        assert w.op == "AND"
        assert w.left.op == "IS"
        assert w.left.right == Literal(None)
        assert w.right.op == "IS NOT"
        assert w.right.right == Literal(None)

    def test_where_table_qualified_columns(self) -> None:
        sql = "SELECT * FROM users WHERE users.id = 42"
        stmt = parse(sql)
        w = stmt.where
        assert isinstance(w, BinaryOp)
        assert w.left == ColumnRef(name="id", table="users")
        assert w.right == Literal(42)


# ===========================================================================
# SELECT JOIN Clause Tests
# ===========================================================================

class TestSelectJoin:
    def test_select_join_default_inner(self) -> None:
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.joins) == 1

        join = stmt.joins[0]
        assert join.join_type == "INNER"
        assert join.join_type == JoinType.INNER
        assert join.table_name == "orders"

        cond = join.on_condition
        assert isinstance(cond, BinaryOp)
        assert cond.op == "="
        assert cond.left == ColumnRef(name="id", table="users")
        assert cond.right == ColumnRef(name="user_id", table="orders")

    def test_select_explicit_inner_join(self) -> None:
        sql = "SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id"
        stmt = parse(sql)
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == JoinType.INNER

    def test_select_left_join(self) -> None:
        sql = "SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id"
        stmt = parse(sql)
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == JoinType.LEFT
        assert stmt.joins[0].table_name == "orders"

    def test_select_left_outer_join(self) -> None:
        sql = "SELECT * FROM users LEFT OUTER JOIN orders ON users.id = orders.user_id"
        stmt = parse(sql)
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == JoinType.LEFT

    def test_select_right_join(self) -> None:
        sql = "SELECT * FROM users RIGHT JOIN orders ON users.id = orders.user_id"
        stmt = parse(sql)
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == JoinType.RIGHT

    def test_select_full_outer_join(self) -> None:
        sql = "SELECT * FROM users FULL OUTER JOIN orders ON users.id = orders.user_id"
        stmt = parse(sql)
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == JoinType.FULL

    def test_select_cross_join(self) -> None:
        sql = "SELECT * FROM users CROSS JOIN orders"
        stmt = parse(sql)
        assert len(stmt.joins) == 1
        assert stmt.joins[0].join_type == JoinType.CROSS
        assert stmt.joins[0].on is None

    def test_select_join_with_aliases(self) -> None:
        sql = "SELECT u.name, o.total FROM users AS u JOIN orders AS o ON u.id = o.user_id"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.from_table.name == "users"
        assert stmt.from_table.alias == "u"

        join = stmt.joins[0]
        assert join.table.name == "orders"
        assert join.table.alias == "o"
        assert join.on_condition == BinaryOp(
            ColumnRef(name="id", table="u"),
            "=",
            ColumnRef(name="user_id", table="o"),
        )

    def test_select_multiple_joins(self) -> None:
        sql = """
        SELECT u.name, o.total, p.title
        FROM users u
        JOIN orders o ON u.id = o.user_id
        JOIN products p ON o.product_id = p.id
        """
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.joins) == 2

        j1 = stmt.joins[0]
        assert j1.table.name == "orders"
        assert j1.table.alias == "o"
        assert j1.on == BinaryOp(ColumnRef("id", "u"), "=", ColumnRef("user_id", "o"))

        j2 = stmt.joins[1]
        assert j2.table.name == "products"
        assert j2.table.alias == "p"
        assert j2.on == BinaryOp(ColumnRef("product_id", "o"), "=", ColumnRef("id", "p"))

    def test_select_join_missing_on_error(self) -> None:
        with pytest.raises(SQLSyntaxError):
            parse("SELECT * FROM users JOIN orders WHERE id = 1")


# ===========================================================================
# SELECT ORDER BY and LIMIT Tests
# ===========================================================================

class TestSelectOrderAndLimit:
    def test_order_by_asc_desc(self) -> None:
        sql = "SELECT * FROM users ORDER BY age ASC, score DESC"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.order_by is not None
        assert len(stmt.order_by) == 2
        assert stmt.order_by[0].expr == ColumnRef("age")
        assert stmt.order_by[0].ascending is True
        assert stmt.order_by[1].expr == ColumnRef("score")
        assert stmt.order_by[1].ascending is False

    def test_limit(self) -> None:
        sql = "SELECT * FROM users LIMIT 10"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.limit == 10


# ===========================================================================
# Comprehensive Complex Query & Edge Cases Tests
# ===========================================================================

class TestComplexAndEdgeCases:
    def test_complex_query_full(self) -> None:
        sql = """
        -- Retrieve top active users and orders
        SELECT u.id AS user_id, u.name, o.amount
        FROM users AS u
        JOIN orders AS o ON u.id = o.user_id
        WHERE (u.age >= 18 AND u.status = 'active') OR u.role = 'admin'
        ORDER BY o.amount DESC
        LIMIT 25;
        """
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.projections) == 3
        assert stmt.projections[0].alias == "user_id"
        assert stmt.from_table.name == "users"
        assert stmt.from_table.alias == "u"

        assert len(stmt.joins) == 1
        assert stmt.joins[0].table.name == "orders"

        # WHERE structure
        w = stmt.where
        assert isinstance(w, BinaryOp)
        assert w.op == "OR"
        assert isinstance(w.left, BinaryOp)
        assert w.left.op == "AND"

        # ORDER BY & LIMIT
        assert stmt.order_by is not None
        assert stmt.order_by[0].ascending is False
        assert stmt.limit == 25

    def test_trailing_semicolons_and_whitespace(self) -> None:
        sql = "  \n\t SELECT id FROM users ; ; ;  \n"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.projections[0].name == "id"

    def test_parse_multiple_statements(self) -> None:
        sql = """
        CREATE TABLE t1 (id INT);
        INSERT INTO t1 VALUES (10);
        SELECT * FROM t1;
        """
        stmts = parse_statements(sql)
        assert len(stmts) == 3
        assert isinstance(stmts[0], CreateTableStatement)
        assert isinstance(stmts[1], InsertStatement)
        assert isinstance(stmts[2], SelectStatement)

    def test_block_comments_and_line_comments(self) -> None:
        sql = """
        /* This is a block comment
           spanning multiple lines */
        SELECT /* inline */ id, -- inline single
        name FROM users -- trailing
        """
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert len(stmt.projections) == 2
        assert stmt.from_table == "users"

    def test_arithmetic_expressions_in_select(self) -> None:
        sql = "SELECT a + b * c, d - 10 / 2 FROM data"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        p0 = stmt.projections[0].expr
        assert isinstance(p0, BinaryOp)
        assert p0.op == "+"
        assert p0.left == ColumnRef("a")
        assert isinstance(p0.right, BinaryOp)
        assert p0.right.op == "*"

    def test_parser_error_handling(self) -> None:
        with pytest.raises(SQLSyntaxError):
            parse("")
        with pytest.raises(SQLSyntaxError):
            parse("UNKNOWN COMMAND foo bar")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT * FROM users WHERE")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT * FROM users WHERE id =")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT (a + 1 FROM users")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT * FROM users WHERE id = 1 extra_junk")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT * FROM users extra1 extra2")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT 'unterminated string")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT /* unclosed block comment")
        with pytest.raises(SQLSyntaxError):
            parse("SELECT @invalid")

    def test_quoted_identifiers(self) -> None:
        sql = 'SELECT "id", `user_name` FROM "my_table" WHERE "id" = 1'
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.projections[0].name == "id"
        assert stmt.projections[1].name == "user_name"
        assert stmt.from_table.name == "my_table"

    def test_convenience_methods_and_properties(self) -> None:
        sql = "SELECT id, name FROM users JOIN orders ON users.id = orders.user_id WHERE id = 1"
        stmt = parse_sql(sql)
        assert isinstance(stmt, SelectStatement)
        assert stmt.join is not None
        assert stmt.join.table_name == "orders"
        assert stmt.join.condition is not None
        assert stmt.from_clause is not None
        assert stmt.from_clause.name == "users"
        assert stmt.where_clause is not None
        assert len(stmt.columns) == 2

        # Parser class methods
        stmt2 = Parser.parse_sql(sql)
        assert isinstance(stmt2, SelectStatement)
        stmt3 = Parser.parse(sql)
        assert isinstance(stmt3, SelectStatement)


# ===========================================================================
# Logical Plan Tests
# ===========================================================================

class TestLogicalPlan:
    """Tests for LogicalPlan structures and SELECT-to-LogicalPlan translation."""

    def test_logical_plan_simple_select(self) -> None:
        sql = "SELECT id, name FROM users"
        plan = parse_logical_plan(sql)

        assert isinstance(plan, LogicalProjection)
        assert isinstance(plan, LogicalPlan)
        assert isinstance(plan, LogicalPlanNode)
        assert isinstance(plan, ProjectionNode)
        assert len(plan.expressions) == 2
        assert plan.projections == plan.expressions
        assert plan.columns == plan.expressions
        assert plan.child is plan.input
        assert len(plan.children) == 1

        scan = plan.input
        assert isinstance(scan, LogicalScan)
        assert isinstance(scan, ScanNode)
        assert isinstance(scan, TableScanNode)
        assert scan.table_name == "users"
        assert scan.table == "users"
        assert scan.alias is None
        assert scan.children == []
        assert scan.root is scan

    def test_logical_plan_select_with_where(self) -> None:
        sql = "SELECT id FROM users WHERE age > 18 AND status = 'active'"
        plan = parse_logical_plan(sql)

        assert isinstance(plan, LogicalProjection)
        filter_node = plan.input
        assert isinstance(filter_node, LogicalFilter)
        assert isinstance(filter_node, FilterNode)
        assert filter_node.child is filter_node.input
        assert filter_node.predicate is filter_node.condition
        assert len(filter_node.children) == 1

        # Check filter condition
        cond = filter_node.condition
        assert isinstance(cond, BinaryOp)
        assert cond.op == "AND"
        assert isinstance(cond.left, BinaryOp)
        assert cond.left.op == ">"
        assert cond.left.left == ColumnRef("age")
        assert cond.left.right == Literal(18)
        assert isinstance(cond.right, BinaryOp)
        assert cond.right.op == "="
        assert cond.right.left == ColumnRef("status")
        assert cond.right.right == Literal("active")

        # Child of filter is scan
        scan = filter_node.input
        assert isinstance(scan, LogicalScan)
        assert scan.table_name == "users"

    def test_logical_plan_select_with_join_and_on(self) -> None:
        sql = "SELECT u.id, o.amount FROM users AS u JOIN orders AS o ON u.id = o.user_id"
        plan = parse_logical_plan(sql)

        assert isinstance(plan, LogicalProjection)
        join_node = plan.input
        assert isinstance(join_node, LogicalJoin)
        assert isinstance(join_node, JoinNode)
        assert join_node.join_type == JoinType.INNER
        assert len(join_node.children) == 2
        assert join_node.on is join_node.condition
        assert join_node.on_condition is join_node.condition

        # Condition
        assert isinstance(join_node.condition, BinaryOp)
        assert join_node.condition.op == "="
        assert join_node.condition.left == ColumnRef("id", "u")
        assert join_node.condition.right == ColumnRef("user_id", "o")

        # Left scan
        left_scan = join_node.left
        assert isinstance(left_scan, LogicalScan)
        assert left_scan.table_name == "users"
        assert left_scan.alias == "u"

        # Right scan
        right_scan = join_node.right
        assert isinstance(right_scan, LogicalScan)
        assert right_scan.table_name == "orders"
        assert right_scan.alias == "o"

    def test_logical_plan_select_with_join_where_and_or_not_equals(self) -> None:
        sql = """
        SELECT u.name, o.total
        FROM users AS u
        LEFT JOIN orders AS o ON u.id = o.user_id
        WHERE (u.age >= 21 AND o.total > 100) OR (u.status != 'inactive' AND o.total < 500)
        """
        plan = parse_logical_plan(sql)

        assert isinstance(plan, LogicalProjection)
        filter_node = plan.input
        assert isinstance(filter_node, LogicalFilter)

        # Condition tree has OR, AND, >=, >, !=, <
        cond = filter_node.condition
        assert isinstance(cond, BinaryOp)
        assert cond.op == "OR"
        assert isinstance(cond.left, BinaryOp)
        assert cond.left.op == "AND"
        assert isinstance(cond.right, BinaryOp)
        assert cond.right.op == "AND"

        # Child of filter is the LEFT JOIN
        join_node = filter_node.input
        assert isinstance(join_node, LogicalJoin)
        assert join_node.join_type == JoinType.LEFT
        assert join_node.left.table_name == "users"
        assert join_node.right.table_name == "orders"

    def test_logical_plan_select_multiple_joins(self) -> None:
        sql = """
        SELECT a.x, b.y, c.z
        FROM a
        INNER JOIN b ON a.id = b.a_id
        RIGHT JOIN c ON b.id = c.b_id
        WHERE a.x > 0
        """
        plan = parse_logical_plan(sql)

        assert isinstance(plan, LogicalProjection)
        filter_node = plan.input
        assert isinstance(filter_node, LogicalFilter)

        # Outer join (join with c)
        join_c = filter_node.input
        assert isinstance(join_c, LogicalJoin)
        assert join_c.join_type == JoinType.RIGHT
        assert isinstance(join_c.right, LogicalScan)
        assert join_c.right.table_name == "c"

        # Inner join (join with b)
        join_b = join_c.left
        assert isinstance(join_b, LogicalJoin)
        assert join_b.join_type == JoinType.INNER
        assert join_b.left.table_name == "a"
        assert join_b.right.table_name == "b"

    def test_logical_plan_with_order_by_and_limit(self) -> None:
        sql = "SELECT id, name FROM users WHERE id > 10 ORDER BY id DESC LIMIT 5"
        plan = parse_logical_plan(sql)

        assert isinstance(plan, LogicalLimit)
        assert isinstance(plan, LimitNode)
        assert plan.limit == 5

        proj = plan.input
        assert isinstance(proj, LogicalProjection)

        sort_node = proj.input
        assert isinstance(sort_node, LogicalSort)
        assert isinstance(sort_node, SortNode)
        assert len(sort_node.order_by) == 1
        assert sort_node.order_by[0].ascending is False

        filter_node = sort_node.input
        assert isinstance(filter_node, LogicalFilter)

        scan = filter_node.input
        assert isinstance(scan, LogicalScan)
        assert scan.table_name == "users"

    def test_logical_plan_statement_methods(self) -> None:
        sql = "SELECT id FROM users WHERE id = 1"
        stmt = parse(sql)
        assert isinstance(stmt, SelectStatement)

        plan1 = stmt.to_logical_plan()
        plan2 = stmt.logical_plan
        plan3 = stmt.plan
        plan4 = build_logical_plan(stmt)

        assert isinstance(plan1, LogicalProjection)
        assert isinstance(plan2, LogicalProjection)
        assert isinstance(plan3, LogicalProjection)
        assert isinstance(plan4, LogicalProjection)
        assert plan1.input.condition == plan2.input.condition

    def test_logical_plan_convenience_functions_and_flag(self) -> None:
        sql = "SELECT id FROM users"
        # parse with logical_plan=True
        plan1 = parse(sql, logical_plan=True)
        assert isinstance(plan1, LogicalPlan)
        assert isinstance(plan1, LogicalProjection)

        # parse_sql with logical_plan=True
        plan2 = parse_sql(sql, logical_plan=True)
        assert isinstance(plan2, LogicalProjection)

        # parse_to_logical_plan
        plan3 = parse_to_logical_plan(sql)
        assert isinstance(plan3, LogicalProjection)

        # to_logical_plan with SQL string and Statement
        plan4 = to_logical_plan(sql)
        assert isinstance(plan4, LogicalProjection)
        stmt = parse(sql)
        plan5 = to_logical_plan(stmt)
        assert isinstance(plan5, LogicalProjection)

        # Parser class methods
        plan6 = Parser.parse_logical_plan(sql)
        assert isinstance(plan6, LogicalProjection)
        plan7 = Parser(sql).to_logical_plan()
        assert isinstance(plan7, LogicalProjection)

    def test_logical_plan_pretty_and_str(self) -> None:
        sql = "SELECT u.id FROM users AS u JOIN orders AS o ON u.id = o.user_id WHERE u.id = 1"
        plan = parse_logical_plan(sql)

        pretty_text = plan.pretty()
        assert "LogicalProjection" in pretty_text
        assert "LogicalFilter" in pretty_text
        assert "LogicalJoin" in pretty_text
        assert "LogicalScan" in pretty_text
        assert "users AS u" in pretty_text
        assert "orders AS o" in pretty_text

        assert "LogicalProjection" in str(plan)
        assert "LogicalFilter" in str(plan.input)
        assert "LogicalJoin" in str(plan.input.input)

    def test_logical_plan_select_without_from(self) -> None:
        sql = "SELECT 1 + 2"
        plan = parse_logical_plan(sql)
        assert isinstance(plan, LogicalProjection)
        assert isinstance(plan.input, LogicalScan)
        assert plan.input.table_name == ""

    def test_logical_plan_wrapper(self) -> None:
        scan = LogicalScan(table_name="t")
        plan_wrapper = LogicalPlan(root=scan)
        assert plan_wrapper.root is scan
        assert plan_wrapper.children == [scan]
        assert scan.root is scan

    def test_logical_plan_non_select_error(self) -> None:
        create_stmt = parse("CREATE TABLE t (id INT)")
        with pytest.raises(TypeError, match="Logical plan generation only supported for SelectStatement"):
            build_logical_plan(create_stmt)

        insert_stmt = parse("INSERT INTO t VALUES (1)")
        with pytest.raises(TypeError, match="Logical plan generation only supported for SelectStatement"):
            build_logical_plan(insert_stmt)

        with pytest.raises(TypeError, match="Logical plan generation only supported for SelectStatement"):
            parse_logical_plan("CREATE TABLE t (id INT)")
