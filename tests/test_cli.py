"""Comprehensive unit and integration test suite for the SQL REPL and CLI.

Verifies end-to-end integration across all engine layers:
- DiskManager (4KB page file management)
- BufferPoolManager (frame caching and eviction)
- BPlusTree (on-disk index search and range scans)
- SQL Parser (lexer, typed AST, multi-statement parsing)
- Rule-Based Optimizer (RBO: predicate pushdown, index selection, projection pruning)
- Vectorized Volcano Engine (VectorBatch, TableScan, IndexScan, HashJoin, Aggregate, OrderBy, Limit)
- ASCII Table Formatting and Execution Statistics (execution time, pages read)
- REPL interactive console, prompt, special dot commands (.tables, .schema, .exit), and error handling.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.cli import (
    DatabaseEngine,
    SQLRepl,
    TableMetadata,
    TableStorage,
    format_cell_value,
    main,
    render_ascii_table,
)
from src.index.bplus_tree import BPlusTree
from src.query.engine import ExecutionError, VectorBatch
from src.query.optimizer import LogicalIndexScan
from src.query.parser import SQLSyntaxError, parse_sql


# ============================================================================
# 1. ASCII Table Formatter Tests
# ============================================================================


class TestASCIIFormatter:
    """Tests for ASCII table rendering and cell value formatting."""

    def test_format_cell_values(self) -> None:
        assert format_cell_value(None) == "NULL"
        assert format_cell_value(True) == "TRUE"
        assert format_cell_value(False) == "FALSE"
        assert format_cell_value(123) == "123"
        assert format_cell_value(12.5) == "12.5"
        assert format_cell_value("hello") == "hello"

    def test_render_ascii_table_basic(self) -> None:
        columns = ["id", "name"]
        rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        table_str = render_ascii_table(columns, rows)
        lines = table_str.strip().split("\n")

        assert len(lines) == 6
        assert lines[0] == "+----+-------+"
        assert lines[1] == "| id | name  |"
        assert lines[2] == "+----+-------+"
        assert lines[3] == "| 1  | Alice |"
        assert lines[4] == "| 2  | Bob   |"
        assert lines[5] == "+----+-------+"

    def test_render_ascii_table_empty_rows(self) -> None:
        columns = ["id", "val"]
        rows: List[Dict[str, Any]] = []
        table_str = render_ascii_table(columns, rows)
        lines = table_str.strip().split("\n")

        assert len(lines) == 4
        assert lines[0] == "+----+-----+"
        assert lines[1] == "| id | val |"
        assert lines[2] == "+----+-----+"
        assert lines[3] == "+----+-----+"

    def test_render_ascii_table_inferred_columns(self) -> None:
        rows = [{"x": 10, "y": 20}]
        table_str = render_ascii_table([], rows)
        assert "| x  | y  |" in table_str
        assert "| 10 | 20 |" in table_str

    def test_render_empty_columns_and_rows(self) -> None:
        assert render_ascii_table([], []) == ""


# ============================================================================
# 2. Database Engine Integration Tests (Storage -> Parser -> RBO -> Engine)
# ============================================================================


class TestDatabaseEngineIntegration:
    """Tests verifying full end-to-end integration across all engine layers."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> DatabaseEngine:
        db_file = tmp_path / "test_cli_db.db"
        eng = DatabaseEngine(db_file=db_file, pool_size=10)
        yield eng
        eng.close()

    def test_create_table_and_insert(self, engine: DatabaseEngine) -> None:
        # 1. CREATE TABLE
        cols, rows, elapsed, pages, msg = engine.execute_sql(
            "CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT);"
        )
        assert msg == "Table 'users' created successfully."
        assert "users" in engine.tables
        assert elapsed >= 0.0
        assert pages >= 1

        # 2. INSERT INTO
        cols, rows, elapsed, pages, msg = engine.execute_sql(
            "INSERT INTO users VALUES (1, 'Alice', 25), (2, 'Bob', 30);"
        )
        assert msg == "Inserted 2 row(s)."
        assert pages >= 1

        # 3. SELECT *
        cols, rows, elapsed, pages, msg = engine.execute_sql("SELECT * FROM users;")
        assert cols == ["id", "name", "age"]
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "Alice", "age": 25}
        assert rows[1] == {"id": 2, "name": "Bob", "age": 30}
        assert pages >= 1
        assert elapsed >= 0.0

    def test_primary_key_duplicate_error(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, name TEXT);")
        engine.execute_sql("INSERT INTO items VALUES (1, 'Apple');")

        with pytest.raises(ExecutionError, match="Duplicate entry '1' for primary key"):
            engine.execute_sql("INSERT INTO items VALUES (1, 'Orange');")

    def test_select_where_filtering(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE nums (val INT, tag TEXT);")
        engine.execute_sql(
            "INSERT INTO nums VALUES (10, 'low'), (50, 'mid'), (90, 'high');"
        )

        cols, rows, elapsed, pages, msg = engine.execute_sql(
            "SELECT val, tag FROM nums WHERE val >= 50;"
        )
        assert len(rows) == 2
        vals = [r["val"] for r in rows]
        assert sorted(vals) == [50, 90]

    def test_bplus_tree_index_selection_in_where(self, engine: DatabaseEngine) -> None:
        # Table with primary key creates an on-disk BPlusTree index
        engine.execute_sql("CREATE TABLE accounts (acc_id INT PRIMARY KEY, balance FLOAT);")
        engine.execute_sql(
            "INSERT INTO accounts VALUES (101, 1500.0), (102, 2300.5), (103, 50.0);"
        )

        # Verify optimizer selects LogicalIndexScan
        plan = parse_sql("SELECT balance FROM accounts WHERE acc_id = 102", logical_plan=True)
        optimized = engine.optimizer.optimize(plan)
        # Projection -> IndexScan
        assert isinstance(optimized.input, LogicalIndexScan)
        assert optimized.input.index_column == "acc_id"
        assert optimized.input.value == 102

        # Execute query end-to-end
        cols, rows, elapsed, pages, msg = engine.execute_sql(
            "SELECT balance FROM accounts WHERE acc_id = 102;"
        )
        assert len(rows) == 1
        assert rows[0]["balance"] == 2300.5
        assert pages >= 1

    def test_join_execution(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE users (id INT PRIMARY KEY, username TEXT);")
        engine.execute_sql("CREATE TABLE orders (order_id INT, user_id INT, total FLOAT);")

        engine.execute_sql("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');")
        engine.execute_sql(
            "INSERT INTO orders VALUES (101, 1, 99.9), (102, 1, 14.5), (103, 2, 200.0);"
        )

        sql = """
        SELECT users.username, orders.order_id, orders.total
        FROM users
        JOIN orders ON users.id = orders.user_id;
        """
        cols, rows, elapsed, pages, msg = engine.execute_sql(sql)
        assert len(rows) == 3
        usernames = [r["username"] for r in rows]
        assert sorted(usernames) == ["Alice", "Alice", "Bob"]
        assert "Charlie" not in usernames

    def test_group_by_and_aggregates(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE sales (region TEXT, amount INT);")
        engine.execute_sql(
            "INSERT INTO sales VALUES ('North', 100), ('South', 200), ('North', 300), ('West', 50);"
        )

        sql = "SELECT region, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales GROUP BY region;"
        cols, rows, elapsed, pages, msg = engine.execute_sql(sql)

        assert len(rows) == 3
        north_row = next(r for r in rows if r["region"] == "North")
        assert north_row["COUNT(*)"] == 2
        assert north_row["SUM(amount)"] == 400
        assert north_row["AVG(amount)"] == 200.0
        assert north_row["MIN(amount)"] == 100
        assert north_row["MAX(amount)"] == 300

    def test_order_by_and_limit(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE scores (player TEXT, score INT);")
        engine.execute_sql(
            "INSERT INTO scores VALUES ('P1', 10), ('P2', 90), ('P3', 50), ('P4', 80);"
        )

        sql = "SELECT player, score FROM scores ORDER BY score DESC LIMIT 2;"
        cols, rows, elapsed, pages, msg = engine.execute_sql(sql)
        assert len(rows) == 2
        assert rows[0] == {"player": "P2", "score": 90}
        assert rows[1] == {"player": "P4", "score": 80}

    def test_select_constant_expression(self, engine: DatabaseEngine) -> None:
        cols, rows, elapsed, pages, msg = engine.execute_sql("SELECT 10 + 20 AS total, 'ok' AS status;")
        assert cols == ["total", "status"]
        assert len(rows) == 1
        assert rows[0] == {"total": 30, "status": "ok"}
        assert pages == 0

    def test_select_scalar_aggregation(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE metrics (val INT);")
        engine.execute_sql("INSERT INTO metrics VALUES (10), (20), (30);")
        cols, rows, elapsed, pages, msg = engine.execute_sql(
            "SELECT COUNT(*), SUM(val), AVG(val), MIN(val), MAX(val) FROM metrics;"
        )
        assert len(rows) == 1
        assert rows[0]["COUNT(*)"] == 3
        assert rows[0]["SUM(val)"] == 60
        assert rows[0]["AVG(val)"] == 20.0
        assert rows[0]["MIN(val)"] == 10
        assert rows[0]["MAX(val)"] == 30

    def test_bplus_tree_range_scan_operators(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE inventory (sku INT PRIMARY KEY, stock INT);")
        engine.execute_sql(
            "INSERT INTO inventory VALUES (10, 5), (20, 15), (30, 25), (40, 35);"
        )

        # > operator
        cols, rows, _, _, _ = engine.execute_sql("SELECT sku, stock FROM inventory WHERE sku > 20;")
        assert [r["sku"] for r in rows] == [30, 40]

        # <= operator
        cols, rows, _, _, _ = engine.execute_sql("SELECT sku, stock FROM inventory WHERE sku <= 20;")
        assert [r["sku"] for r in rows] == [10, 20]

    def test_null_value_handling_and_formatting(self, engine: DatabaseEngine) -> None:
        engine.execute_sql("CREATE TABLE nullable_data (id INT, note TEXT);")
        engine.execute_sql("INSERT INTO nullable_data VALUES (1, NULL), (2, 'Present');")
        cols, rows, _, _, _ = engine.execute_sql("SELECT * FROM nullable_data;")
        assert len(rows) == 2
        assert rows[0]["note"] is None
        assert rows[1]["note"] == "Present"

        rendered = render_ascii_table(cols, rows)
        assert "| NULL    |" in rendered
        assert "| Present |" in rendered

    def test_comments_and_whitespace(self, engine: DatabaseEngine) -> None:
        sql = """
        -- Setup table
        CREATE TABLE /* inline comment */ commented (
            id INT PRIMARY KEY, -- Primary key
            name TEXT
        );
        INSERT INTO commented VALUES (1, 'CommentedRow');
        """
        engine.execute_sql(sql)
        cols, rows, _, _, _ = engine.execute_sql("SELECT * FROM commented; -- fetch all")
        assert len(rows) == 1
        assert rows[0]["name"] == "CommentedRow"

    def test_on_disk_persistence_reload(self, tmp_path: Path) -> None:
        db_file = tmp_path / "persistence_engine.db"

        # Session 1: Create, insert, close
        eng1 = DatabaseEngine(db_file=db_file, pool_size=5)
        eng1.execute_sql("CREATE TABLE notes (id INT PRIMARY KEY, content TEXT);")
        eng1.execute_sql("INSERT INTO notes VALUES (1, 'Persisted Note 1'), (2, 'Note 2');")
        eng1.close()

        # Session 2: Reopen from disk
        eng2 = DatabaseEngine(db_file=db_file, pool_size=5)
        assert "notes" in eng2.tables

        cols, rows, elapsed, pages, msg = eng2.execute_sql("SELECT * FROM notes WHERE id = 1;")
        assert len(rows) == 1
        assert rows[0]["content"] == "Persisted Note 1"
        eng2.close()


# ============================================================================
# 3. SQL REPL Interactive Console Tests
# ============================================================================


class TestSQLReplConsole:
    """Tests verifying the REPL interactive loop, dot commands, and error handling."""

    @pytest.fixture
    def setup_repl(self) -> tuple[SQLRepl, io.StringIO, io.StringIO]:
        stdin_stream = io.StringIO()
        stdout_stream = io.StringIO()
        engine = DatabaseEngine()
        repl = SQLRepl(engine=engine, stdin=stdin_stream, stdout=stdout_stream)
        return repl, stdin_stream, stdout_stream

    def test_repl_banner_and_exit(self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        stdin_stream.write(".exit\n")
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        assert "Vectorized SQL Engine REPL" in output
        assert "sql> " in output
        assert "Goodbye!" in output

    def test_repl_special_commands(self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        input_commands = (
            ".tables\n"
            "CREATE TABLE clients (id INT PRIMARY KEY, company TEXT NOT NULL);\n"
            ".tables\n"
            ".schema clients\n"
            ".schema nonexistent\n"
            ".help\n"
            ".quit\n"
        )
        stdin_stream.write(input_commands)
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        assert "No tables found." in output
        assert "Table 'clients' created successfully." in output
        assert "clients" in output
        assert "CREATE TABLE clients" in output
        assert "company TEXT NOT NULL" in output
        assert "Error: Table 'nonexistent' not found." in output
        assert "Available special commands:" in output
        assert "Goodbye!" in output

    def test_repl_syntax_error_friendly_handling(
        self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]
    ) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        commands = (
            "SELECT * FROM;\n"
            "FOOBAR BAZ;\n"
            "CREATE TABLE t (x INT);\n"
            "INSERT INTO t VALUES (777);\n"
            "SELECT * FROM t;\n"
            ".exit\n"
        )
        stdin_stream.write(commands)
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        # Syntax errors reported friendly without Python tracebacks
        assert "Syntax Error:" in output
        assert "Traceback" not in output

        # Ensure subsequent valid statements still executed
        assert "Table 't' created successfully." in output
        assert "| 777 |" in output
        assert "Execution time:" in output
        assert "Pages read:" in output

    def test_repl_runtime_error_friendly_handling(
        self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]
    ) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        commands = (
            "SELECT * FROM missing_table;\n"
            ".exit\n"
        )
        stdin_stream.write(commands)
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        assert "Error: Table 'missing_table' does not exist." in output
        assert "Traceback" not in output

    def test_repl_empty_table_select(
        self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]
    ) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        commands = (
            "CREATE TABLE empty_tbl (x INT, y TEXT);\n"
            "SELECT * FROM empty_tbl;\n"
            ".exit\n"
        )
        stdin_stream.write(commands)
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        assert "| x | y |" in output
        assert "(0 rows in" in output
        assert "Execution time:" in output
        assert "Pages read:" in output

    def test_repl_unknown_dot_command(
        self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]
    ) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        commands = (
            ".foobar\n"
            ".schema\n"
            ".exit\n"
        )
        stdin_stream.write(commands)
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        assert "Unknown special command '.foobar'" in output
        assert "No tables defined." in output

    def test_repl_multiline_sql_execution(
        self, setup_repl: tuple[SQLRepl, io.StringIO, io.StringIO]
    ) -> None:
        repl, stdin_stream, stdout_stream = setup_repl
        multiline = (
            "CREATE TABLE multi (\n"
            "    id INT,\n"
            "    val TEXT\n"
            ");\n"
            "INSERT INTO multi\n"
            "VALUES (1, 'multiline_row');\n"
            "SELECT\n"
            "    val\n"
            "FROM multi\n"
            "WHERE id = 1;\n"
            ".exit\n"
        )
        stdin_stream.write(multiline)
        stdin_stream.seek(0)

        repl.run()
        output = stdout_stream.getvalue()

        assert "   -> " in output
        assert "Table 'multi' created successfully." in output
        assert "| multiline_row |" in output


# ============================================================================
# 4. CLI Process & Subprocess Verification ('python -m src.cli')
# ============================================================================


class TestCLIProcess:
    """Verifies executable invocation via 'python -m src.cli'."""

    def test_python_m_src_cli_command_flag(self, tmp_path: Path) -> None:
        db_file = str(tmp_path / "proc_test.db")
        sql = (
            "CREATE TABLE t (id INT PRIMARY KEY, name TEXT); "
            "INSERT INTO t VALUES (10, 'SubprocAlice'), (20, 'SubprocBob'); "
            "SELECT * FROM t WHERE id = 10;"
        )
        cmd = [sys.executable, "-m", "src.cli", db_file, "-c", sql]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "Table 't' created successfully." in result.stdout
        assert "Inserted 2 row(s)." in result.stdout
        assert "| 10 | SubprocAlice |" in result.stdout
        assert "Execution time:" in result.stdout
        assert "Pages read:" in result.stdout

    def test_python_m_src_cli_interactive_pipe(self) -> None:
        cmd = [sys.executable, "-m", "src.cli"]
        input_data = (
            "CREATE TABLE test_pipe (msg TEXT);\n"
            "INSERT INTO test_pipe VALUES ('pipe_success');\n"
            "SELECT * FROM test_pipe;\n"
            ".exit\n"
        )
        proc = subprocess.run(cmd, input=input_data, capture_output=True, text=True)

        assert proc.returncode == 0
        assert "sql> " in proc.stdout
        assert "| pipe_success |" in proc.stdout
        assert "Goodbye!" in proc.stdout

    def test_cli_help_flag(self) -> None:
        cmd = [sys.executable, "-m", "src.cli", "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "usage: python -m src.cli" in proc.stdout.lower()
