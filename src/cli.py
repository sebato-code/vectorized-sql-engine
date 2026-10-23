"""Interactive SQL REPL and End-to-End Database Execution Engine.

Connects all layers of the Vectorized SQL Engine:
- DiskManager (fixed 4KB pages on disk)
- BufferPoolManager (in-memory frame caching & eviction)
- BPlusTree (on-disk B+Tree indexes)
- SQL Parser (lexer, recursive descent parser, typed AST)
- Rule-Based Optimizer (RBO: predicate pushdown, index selection, projection pruning)
- Vectorized Execution Engine (columnar VectorBatch, Volcano operators)
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Engine imports
from src.index.bplus_tree import BPlusTree
from src.query.engine import (
    DEFAULT_BATCH_SIZE,
    Aggregate,
    AggregateFunction,
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
    Literal as EngineLiteral,
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
    Table as EngineTable,
    VectorBatch,
    col,
    lit,
)
from src.query.optimizer import (
    Catalog,
    IndexInfo,
    LogicalIndexScan,
    RuleBasedOptimizer,
)
from src.query.parser import (
    BinaryOp,
    BinaryOpType,
    ColumnDef,
    ColumnRef,
    CreateTableStatement,
    Expression as ASTExpression,
    FunctionCall,
    InsertStatement,
    JoinClause,
    JoinType,
    Literal as ASTLiteral,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalPlan,
    LogicalProjection,
    LogicalScan,
    LogicalSort,
    OrderByItem,
    Parser,
    ParserError,
    SQLSyntaxError,
    SelectItem,
    SelectStatement,
    Star,
    Statement,
    TableRef,
    UnaryOp,
    UnexpectedTokenError,
    build_logical_plan,
    parse_sql,
    parse_statements,
)
from src.storage.buffer_pool import BufferPoolManager, Page
from src.storage.disk_manager import PAGE_SIZE, DiskManager


# ============================================================================
# ASCII Table Formatter
# ============================================================================


def format_cell_value(val: Any) -> str:
    """Format an arbitrary Python value as a readable string for table display."""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, float):
        # Avoid scientific notation for simple floats, strip trailing zeroes if integer
        if val.is_integer():
            return f"{val:.1f}"
        return f"{val:.4g}"
    return str(val)


def render_ascii_table(columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    """Render a sequence of row dictionaries into a formatted ASCII table string.

    Example output:
    +----+-------+
    | id | name  |
    +----+-------+
    | 1  | Alice |
    | 2  | Bob   |
    +----+-------+
    """
    if not columns:
        if not rows:
            return ""
        columns = list(rows[0].keys())

    # Stringify all cells
    string_rows: List[List[str]] = []
    for r in rows:
        string_rows.append([format_cell_value(r.get(c)) for c in columns])

    # Calculate column widths (minimum width is column name length)
    widths: List[int] = [len(str(c)) for c in columns]
    for row_strings in string_rows:
        for idx, val in enumerate(row_strings):
            if len(val) > widths[idx]:
                widths[idx] = len(val)

    # Build horizontal border
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    # Build header line
    header_cells = [f" {str(col).ljust(widths[i])} " for i, col in enumerate(columns)]
    header_line = "|" + "|".join(header_cells) + "|"

    lines: List[str] = [border, header_line, border]

    # Build row lines
    for row_strings in string_rows:
        cells = [f" {val.ljust(widths[i])} " for i, val in enumerate(row_strings)]
        lines.append("|" + "|".join(cells) + "|")

    lines.append(border)
    return "\n".join(lines)


# ============================================================================
# Table Storage over BufferPoolManager
# ============================================================================


@dataclass
class TableMetadata:
    """Metadata and storage management for a table backed by 4KB pages."""

    name: str
    columns: List[ColumnDef]
    primary_key: Optional[str] = None
    data_page_ids: List[int] = field(default_factory=list)
    indexes: Dict[str, BPlusTree] = field(default_factory=dict)
    index_root_pages: Dict[str, int] = field(default_factory=dict)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["indexes"] = {}  # Exclude BPlusTree instances from pickle to avoid locking objects
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "indexes" not in self.__dict__:
            self.indexes = {}

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnDef]:
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None


class TableStorage:
    """Handles row serialization and page allocations for a table."""

    def __init__(self, metadata: TableMetadata, bpm: BufferPoolManager) -> None:
        self.meta = metadata
        self.bpm = bpm
        self._lock = bpm._lock

    def get_all_rows(self) -> List[Dict[str, Any]]:
        """Read all rows across all table data pages."""
        rows: List[Dict[str, Any]] = []
        for page_id in self.meta.data_page_ids:
            page = self.bpm.fetch_page(page_id)
            try:
                page_rows = self._deserialize_page_rows(page)
                rows.extend(page_rows)
            finally:
                self.bpm.unpin_page(page_id, is_dirty=False)
        return rows

    def insert_rows(self, new_rows: Sequence[Dict[str, Any]]) -> int:
        """Insert rows into 4KB data pages and update B+Tree indexes."""
        if not new_rows:
            return 0

        inserted_count = 0
        for row in new_rows:
            # Check unique/primary key constraints
            if self.meta.primary_key:
                pk_val = row.get(self.meta.primary_key)
                pk_tree = self.meta.indexes.get(self.meta.primary_key)
                if pk_tree is not None and pk_val is not None:
                    if pk_tree.search(pk_val) is not None:
                        raise ExecutionError(
                            f"Duplicate entry '{pk_val}' for primary key in table '{self.meta.name}'"
                        )

            # Insert into data pages
            self._append_row_to_pages(row)

            # Insert into all indexes
            for col_name, tree in self.meta.indexes.items():
                if col_name in row:
                    val = row[col_name]
                    tree.insert(val, row)
                    self.meta.index_root_pages[col_name] = tree.root_page_id

            inserted_count += 1

        return inserted_count

    def _append_row_to_pages(self, row: Dict[str, Any]) -> None:
        """Append a single row dictionary into the current page or allocate a new page."""
        # Clean dictionary matching declared column names
        cleaned_row = {c: row.get(c) for c in self.meta.column_names}

        if not self.meta.data_page_ids:
            new_page = self.bpm.new_page()
            page_id = new_page.page_id
            self.meta.data_page_ids.append(page_id)
            self._serialize_page_rows(new_page, [cleaned_row])
            self.bpm.unpin_page(page_id, is_dirty=True)
            return

        last_page_id = self.meta.data_page_ids[-1]
        page = self.bpm.fetch_page(last_page_id)
        current_rows = self._deserialize_page_rows(page)
        tentative_rows = current_rows + [cleaned_row]

        payload = pickle.dumps(tentative_rows, protocol=5)
        # 4 bytes for payload length
        if len(payload) + 4 <= PAGE_SIZE:
            # Fits in current page
            self._serialize_page_rows(page, tentative_rows)
            self.bpm.unpin_page(last_page_id, is_dirty=True)
        else:
            # Must allocate a new page
            self.bpm.unpin_page(last_page_id, is_dirty=False)
            new_page = self.bpm.new_page()
            new_page_id = new_page.page_id
            self.meta.data_page_ids.append(new_page_id)
            self._serialize_page_rows(new_page, [cleaned_row])
            self.bpm.unpin_page(new_page_id, is_dirty=True)

    @staticmethod
    def _deserialize_page_rows(page: Page) -> List[Dict[str, Any]]:
        """Read and unpickle records stored in a 4096-byte page."""
        payload_len = struct.unpack(">I", page.data[:4])[0]
        if payload_len == 0 or payload_len > PAGE_SIZE - 4:
            return []
        try:
            return pickle.loads(page.data[4 : 4 + payload_len])
        except Exception:
            return []

    @staticmethod
    def _serialize_page_rows(page: Page, rows: List[Dict[str, Any]]) -> None:
        """Pickle rows into the 4096-byte page."""
        payload = pickle.dumps(rows, protocol=5)
        total_len = len(payload)
        page.data[:4] = struct.pack(">I", total_len)
        page.data[4 : 4 + total_len] = payload
        page.data[4 + total_len :] = b"\x00" * (PAGE_SIZE - (4 + total_len))
        page.is_dirty = True


# ============================================================================
# Physical Execution Operators
# ============================================================================


class PhysicalTableScan(ExecutionOperator):
    """Vectorized scan operator reading 4KB pages from BufferPoolManager."""

    def __init__(
        self,
        table_storage: TableStorage,
        batch_size: int = DEFAULT_BATCH_SIZE,
        projection: Optional[Sequence[str]] = None,
        alias: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.table_storage = table_storage
        self.bpm = table_storage.bpm
        self.batch_size = batch_size
        self.projection = list(projection) if projection is not None else None
        self.alias = alias
        self.table_name = table_storage.meta.name

        self._page_idx: int = 0
        self._buffered_rows: List[Dict[str, Any]] = []

    def open(self) -> None:
        super().open()
        self._page_idx = 0
        self._buffered_rows = []

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        page_ids = self.table_storage.meta.data_page_ids

        while len(self._buffered_rows) < self.batch_size and self._page_idx < len(page_ids):
            page_id = page_ids[self._page_idx]
            page = self.bpm.fetch_page(page_id)
            try:
                page_rows = TableStorage._deserialize_page_rows(page)
                for r in page_rows:
                    row_dict = dict(r)
                    # Add alias-qualified and table-qualified column references
                    for col_name, val in r.items():
                        row_dict[f"{self.table_name}.{col_name}"] = val
                        if self.alias:
                            row_dict[f"{self.alias}.{col_name}"] = val
                    self._buffered_rows.append(row_dict)
            finally:
                self.bpm.unpin_page(page_id, is_dirty=False)
            self._page_idx += 1

        if not self._buffered_rows:
            return None

        chunk = self._buffered_rows[: self.batch_size]
        self._buffered_rows = self._buffered_rows[self.batch_size :]

        # Determine output columns
        if self.projection:
            target_cols = self.projection
        else:
            # Emit base table columns
            target_cols = list(self.table_storage.meta.column_names)

        # Build columnar dictionary
        data: Dict[str, List[Any]] = {}
        for c in target_cols:
            data[c] = [r.get(c) for r in chunk]

        # Also preserve all qualified columns so downstream operators (Join/Filter) can find them
        for k in chunk[0].keys():
            if k not in data:
                data[k] = [r.get(k) for r in chunk]

        return VectorBatch(data=data, num_rows=len(chunk))

    def close(self) -> None:
        self._buffered_rows = []
        super().close()


class PhysicalIndexScan(ExecutionOperator):
    """Vectorized index scan operator executing exact match or range scan on BPlusTree."""

    def __init__(
        self,
        table_storage: TableStorage,
        btree: BPlusTree,
        operator: str = "=",
        value: Any = None,
        projection: Optional[Sequence[str]] = None,
        alias: Optional[str] = None,
        start_key: Any = None,
        end_key: Any = None,
        inclusive_start: bool = True,
        inclusive_end: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.table_storage = table_storage
        self.btree = btree
        self.operator = operator
        self.value = value
        self.projection = list(projection) if projection is not None else None
        self.alias = alias
        self.start_key = start_key
        self.end_key = end_key
        self.inclusive_start = inclusive_start
        self.inclusive_end = inclusive_end
        self.batch_size = batch_size
        self.table_name = table_storage.meta.name

        self._matched_rows: List[Dict[str, Any]] = []
        self._offset: int = 0

    def open(self) -> None:
        super().open()
        self._matched_rows = []
        self._offset = 0

        # Execute search / range scan on B+Tree
        if self.operator == "=":
            res = self.btree.search(self.value)
            if res is not None:
                if isinstance(res, list):
                    self._matched_rows.extend(res)
                else:
                    self._matched_rows.append(res)
        elif self.operator in (">", ">=", "<", "<=", "BETWEEN"):
            start = self.start_key
            end = self.end_key
            inc_s = self.inclusive_start
            inc_e = self.inclusive_end

            if self.operator == ">":
                start, end = self.value, None
                inc_s, inc_e = False, True
            elif self.operator == ">=":
                start, end = self.value, None
                inc_s, inc_e = True, True
            elif self.operator == "<":
                start, end = None, self.value
                inc_s, inc_e = True, False
            elif self.operator == "<=":
                start, end = None, self.value
                inc_s, inc_e = True, True

            entries = self.btree.range_scan(
                start_key=start,
                end_key=end,
                inclusive_start=inc_s,
                inclusive_end=inc_e,
            )
            for _k, row_val in entries:
                if isinstance(row_val, list):
                    self._matched_rows.extend(row_val)
                else:
                    self._matched_rows.append(row_val)

        # Enrich with qualified aliases
        enriched: List[Dict[str, Any]] = []
        for r in self._matched_rows:
            row_dict = dict(r)
            for col_name, val in r.items():
                row_dict[f"{self.table_name}.{col_name}"] = val
                if self.alias:
                    row_dict[f"{self.alias}.{col_name}"] = val
            enriched.append(row_dict)
        self._matched_rows = enriched

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        if self._offset >= len(self._matched_rows):
            return None

        chunk = self._matched_rows[self._offset : self._offset + self.batch_size]
        self._offset += len(chunk)

        target_cols = self.projection or list(self.table_storage.meta.column_names)
        data: Dict[str, List[Any]] = {}
        for c in target_cols:
            data[c] = [r.get(c) for r in chunk]

        for k in chunk[0].keys():
            if k not in data:
                data[k] = [r.get(k) for r in chunk]

        return VectorBatch(data=data, num_rows=len(chunk))

    def close(self) -> None:
        self._matched_rows = []
        self._offset = 0
        super().close()


# ============================================================================
# AST to Physical Expression & Plan Compiler
# ============================================================================


def compile_ast_expression(expr: ASTExpression) -> Any:
    """Compile an AST Expression into a vectorized Engine Expression."""
    if isinstance(expr, ASTLiteral):
        return EngineLiteral(expr.value)

    if isinstance(expr, ColumnRef):
        if expr.table:
            # Table-qualified column reference
            return Col(f"{expr.table}.{expr.name}")
        return Col(expr.name)

    if isinstance(expr, BinaryOp):
        left = compile_ast_expression(expr.left)
        right = compile_ast_expression(expr.right)
        op_str = str(expr.op).upper()

        if op_str in ("=", "=="):
            return Eq(left, right)
        if op_str in ("!=", "<>"):
            return Ne(left, right)
        if op_str == "<":
            return Lt(left, right)
        if op_str == "<=":
            return Le(left, right)
        if op_str == ">":
            return Gt(left, right)
        if op_str == ">=":
            return Ge(left, right)
        if op_str == "AND" or op_str == BinaryOpType.AND:
            return And(left, right)
        if op_str == "OR" or op_str == BinaryOpType.OR:
            return Or(left, right)
        if op_str == "+" or op_str == BinaryOpType.PLUS:
            return left + right
        if op_str == "-" or op_str == BinaryOpType.MINUS:
            return left - right
        if op_str == "*" or op_str == BinaryOpType.STAR:
            return left * right
        if op_str == "/" or op_str == BinaryOpType.SLASH:
            return left / right
        if op_str == "%":
            return left % right
        if op_str == "IS":
            return IsNull(left)
        if op_str == "IS NOT":
            return IsNotNull(left)

        raise ExecutionError(f"Unsupported binary operator in expression: '{expr.op}'")

    if isinstance(expr, UnaryOp):
        operand = compile_ast_expression(expr.operand)
        if expr.op.upper() == "NOT":
            return Not(operand)
        if expr.op == "-":
            return EngineLiteral(0) - operand
        raise ExecutionError(f"Unsupported unary operator: '{expr.op}'")

    if isinstance(expr, Star):
        return Col("*")

    raise ExecutionError(f"Cannot compile expression of type '{type(expr).__name__}'")


# ============================================================================
# Database Engine (Full Integration of All Layers)
# ============================================================================


class DatabaseEngine:
    """Embedded Vectorized Database Engine.

    Manages DiskManager, BufferPoolManager, BPlusTree indexes, Catalog,
    Parser, Rule-Based Optimizer, and Vectorized Execution.
    """

    def __init__(
        self,
        db_file: Union[str, Path, None] = None,
        pool_size: int = 20,
    ) -> None:
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        if db_file is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            db_file = Path(self._temp_dir.name) / "engine.db"

        self.db_path = Path(db_file)
        self.disk_manager = DiskManager(self.db_path)
        self.bpm = BufferPoolManager(self.disk_manager, pool_size=pool_size)

        # Track pages read across the entire system
        self.total_pages_read: int = 0
        self._hook_bpm_fetch()

        self.catalog = Catalog()
        self.tables: Dict[str, TableStorage] = {}
        self.optimizer = RuleBasedOptimizer(catalog=self.catalog)

        # Load existing schema/catalog from disk if database file has metadata
        self._load_system_catalog()

    def _hook_bpm_fetch(self) -> None:
        """Instrument BufferPoolManager to track every page fetched from buffer pool / disk."""
        orig_fetch = self.bpm.fetch_page

        def tracked_fetch(page_id: int) -> Page:
            self.total_pages_read += 1
            return orig_fetch(page_id)

        self.bpm.fetch_page = tracked_fetch

    def _load_system_catalog(self) -> None:
        """Load catalog metadata page (page 0) if it exists."""
        if self.disk_manager.num_pages == 0:
            # Allocate page 0 for system catalog header
            cat_page = self.bpm.new_page()
            cat_page.data[:4] = struct.pack(">I", 0)
            self.bpm.unpin_page(cat_page.page_id, is_dirty=True)
            return

        cat_page = self.bpm.fetch_page(0)
        try:
            length = struct.unpack(">I", cat_page.data[:4])[0]
            if length > 0 and length <= PAGE_SIZE - 4:
                loaded_meta: Dict[str, TableMetadata] = pickle.loads(
                    cat_page.data[4 : 4 + length]
                )
                for t_name, meta in loaded_meta.items():
                    storage = TableStorage(meta, self.bpm)
                    # Restore BPlusTree instances
                    for col_name, root_id in meta.index_root_pages.items():
                        tree = BPlusTree(self.bpm, root_page_id=root_id, max_keys=32)
                        meta.indexes[col_name] = tree
                        self.catalog.add_index(
                            t_name, col_name, index=tree, index_name=f"idx_{t_name}_{col_name}"
                        )
                    self.tables[t_name] = storage
                    self.catalog.add_table(t_name, meta.column_names)
        except Exception:
            pass
        finally:
            self.bpm.unpin_page(0, is_dirty=False)

    def _persist_system_catalog(self) -> None:
        """Persist table definitions and index root page IDs into page 0."""
        cat_page = self.bpm.fetch_page(0)
        try:
            meta_dict = {t_name: storage.meta for t_name, storage in self.tables.items()}
            payload = pickle.dumps(meta_dict, protocol=5)
            if len(payload) + 4 <= PAGE_SIZE:
                cat_page.data[:4] = struct.pack(">I", len(payload))
                cat_page.data[4 : 4 + len(payload)] = payload
                cat_page.data[4 + len(payload) :] = b"\x00" * (PAGE_SIZE - (4 + len(payload)))
                cat_page.is_dirty = True
        finally:
            self.bpm.unpin_page(0, is_dirty=True)

    def close(self) -> None:
        """Flush buffer pool pages and close disk manager."""
        self._persist_system_catalog()
        self.bpm.flush_all_pages()
        self.disk_manager.close()
        if self._temp_dir:
            self._temp_dir.cleanup()

    # ------------------------------------------------------------------------
    # Execution Engine API
    # ------------------------------------------------------------------------

    def execute_statement(
        self, stmt: Statement
    ) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], float, int, str]:
        """Execute a parsed Statement AST end-to-end.

        Returns:
            Tuple: (columns, rows, execution_time_ms, pages_read, message)
        """
        start_pages = self.total_pages_read
        start_time = time.perf_counter()

        if isinstance(stmt, CreateTableStatement):
            self._execute_create_table(stmt)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            pages_read = self.total_pages_read - start_pages
            msg = f"Table '{stmt.table_name}' created successfully."
            return None, None, elapsed_ms, pages_read, msg

        if isinstance(stmt, InsertStatement):
            inserted = self._execute_insert(stmt)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            pages_read = self.total_pages_read - start_pages
            msg = f"Inserted {inserted} row(s)."
            return None, None, elapsed_ms, pages_read, msg

        if isinstance(stmt, SelectStatement):
            cols, rows = self._execute_select(stmt)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            pages_read = self.total_pages_read - start_pages
            return cols, rows, elapsed_ms, pages_read, ""

        raise ExecutionError(f"Unsupported SQL statement type: {type(stmt).__name__}")

    def execute_sql(
        self, sql: str
    ) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], float, int, str]:
        """Execute one or more SQL statements end-to-end.

        Returns:
            Tuple: (columns, rows, execution_time_ms, pages_read, message) of the last executed statement.
        """
        sql_stripped = sql.strip()
        if not sql_stripped:
            return None, None, 0.0, 0, ""

        stmts = parse_statements(sql_stripped)
        if not stmts:
            return None, None, 0.0, 0, ""

        last_result: Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], float, int, str] = (
            None,
            None,
            0.0,
            0,
            "",
        )
        total_time = 0.0
        total_pages = 0
        for s in stmts:
            cols, rows, elapsed, pages, msg = self.execute_statement(s)
            total_time += elapsed
            total_pages += pages
            last_result = (cols, rows, total_time, total_pages, msg)

        return last_result

    # ------------------------------------------------------------------------
    # DDL: CREATE TABLE
    # ------------------------------------------------------------------------

    def _execute_create_table(self, stmt: CreateTableStatement) -> None:
        t_name = stmt.table_name
        if t_name in self.tables:
            if stmt.if_not_exists:
                return
            raise ExecutionError(f"Table '{t_name}' already exists.")

        pk_col: Optional[str] = None
        for col_def in stmt.columns:
            if col_def.primary_key:
                pk_col = col_def.name
                break

        meta = TableMetadata(
            name=t_name,
            columns=list(stmt.columns),
            primary_key=pk_col,
        )

        # Create BPlusTree index on primary key or unique columns
        indexes: Dict[str, BPlusTree] = {}
        index_roots: Dict[str, int] = {}
        for col_def in stmt.columns:
            if col_def.primary_key or col_def.unique:
                btree = BPlusTree(self.bpm, max_keys=32)
                indexes[col_def.name] = btree
                index_roots[col_def.name] = btree.root_page_id
                self.catalog.add_index(
                    t_name,
                    col_def.name,
                    index=btree,
                    index_name=f"idx_{t_name}_{col_def.name}",
                    is_unique=True,
                )

        meta.indexes = indexes
        meta.index_root_pages = index_roots

        storage = TableStorage(meta, self.bpm)
        self.tables[t_name] = storage
        self.catalog.add_table(t_name, meta.column_names)
        self._persist_system_catalog()

    # ------------------------------------------------------------------------
    # DML: INSERT INTO
    # ------------------------------------------------------------------------

    def _execute_insert(self, stmt: InsertStatement) -> int:
        t_name = stmt.table_name
        if t_name not in self.tables:
            raise ExecutionError(f"Table '{t_name}' does not exist.")

        storage = self.tables[t_name]
        col_names = stmt.columns or storage.meta.column_names

        rows_to_insert: List[Dict[str, Any]] = []
        for raw_row in stmt.raw_values:
            if len(raw_row) != len(col_names):
                raise ExecutionError(
                    f"Column count mismatch: expected {len(col_names)} columns, got {len(raw_row)}"
                )
            row_dict: Dict[str, Any] = {}
            for c_name, val in zip(col_names, raw_row):
                row_dict[c_name] = val
            rows_to_insert.append(row_dict)

        count = storage.insert_rows(rows_to_insert)
        self._persist_system_catalog()
        return count

    # ------------------------------------------------------------------------
    # DQL: SELECT with RBO & Vectorized Execution
    # ------------------------------------------------------------------------

    def _execute_select(self, stmt: SelectStatement) -> Tuple[List[str], List[Dict[str, Any]]]:
        # Handle SELECT without FROM (e.g., SELECT 1 + 2)
        if stmt.from_table is None:
            return self._execute_select_constant(stmt)

        # 1. Translate AST to relational LogicalPlan
        logical_plan = build_logical_plan(stmt)

        # 2. Optimize plan using Rule-Based Optimizer (RBO)
        # Applies Predicate Pushdown, Index Selection, and Projection Pruning
        optimized_plan = self.optimizer.optimize(logical_plan)

        # 3. Compile optimized LogicalPlan into physical Vectorized Volcano pipeline
        physical_op = self._compile_plan(optimized_plan, stmt)

        # 4. Pull all rows through the vectorized Volcano iterator
        raw_rows = physical_op.execute_all_rows()

        # 5. Apply final projections, aliases, and formatting
        return self._format_final_projections(stmt, raw_rows)

    def _execute_select_constant(
        self, stmt: SelectStatement
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Execute constant SELECT queries without FROM."""
        row: Dict[str, Any] = {}
        cols: List[str] = []
        for idx, p in enumerate(stmt.projections):
            col_name = p.alias or f"col_{idx + 1}"
            cols.append(col_name)
            expr = compile_ast_expression(p.expr)
            single_batch = VectorBatch(num_rows=1)
            val = expr.evaluate(single_batch)[0]
            row[col_name] = val
        return cols, [row]

    def _compile_plan(
        self, node: LogicalPlan, stmt: SelectStatement
    ) -> ExecutionOperator:
        """Recursively compile a LogicalPlan node into physical ExecutionOperators."""
        if isinstance(node, LogicalLimit):
            child_op = self._compile_plan(node.input, stmt)
            return Limit(child_op, limit=node.limit)

        if isinstance(node, LogicalProjection):
            # Check if this query involves aggregation / GROUP BY
            has_aggregates = any(
                isinstance(p.expr, FunctionCall) for p in stmt.projections
            )
            has_group_by = bool(stmt.group_by)

            child_op = self._compile_plan(node.input, stmt)

            if has_aggregates or has_group_by:
                agg_op = self._compile_aggregation(child_op, stmt)
                if stmt.having:
                    having_expr = compile_ast_expression(stmt.having)
                    return Filter(agg_op, having_expr)
                return agg_op

            return child_op

        if isinstance(node, LogicalSort):
            child_op = self._compile_plan(node.input, stmt)
            sort_items: List[Tuple[str, bool]] = []
            for item in node.order_by:
                if isinstance(item.expr, ColumnRef):
                    col_name = (
                        f"{item.expr.table}.{item.expr.name}"
                        if item.expr.table
                        else item.expr.name
                    )
                else:
                    col_name = str(item.expr)
                sort_items.append((col_name, item.ascending))

            return OrderBy(child_op, by=sort_items)

        if isinstance(node, LogicalFilter):
            child_op = self._compile_plan(node.input, stmt)
            predicate = compile_ast_expression(node.condition)
            return Filter(child_op, predicate)

        if isinstance(node, LogicalJoin):
            left_op = self._compile_plan(node.left, stmt)
            right_op = self._compile_plan(node.right, stmt)

            left_key, right_key = self._extract_join_keys(node.condition, node.left, node.right)
            join_type_str = str(node.join_type.value).upper()

            return HashJoin(
                left=left_op,
                right=right_op,
                left_on=left_key,
                right_on=right_key,
                join_type=join_type_str,
                batch_size=DEFAULT_BATCH_SIZE,
            )

        if isinstance(node, LogicalIndexScan):
            t_name = node.table_name
            if t_name not in self.tables:
                raise ExecutionError(f"Table '{t_name}' not found during index scan")
            storage = self.tables[t_name]
            btree = node.index or storage.meta.indexes.get(node.index_column)
            if btree is None:
                # Fallback to physical table scan if index is missing
                return PhysicalTableScan(
                    table_storage=storage,
                    projection=node.columns,
                    alias=node.alias,
                )

            return PhysicalIndexScan(
                table_storage=storage,
                btree=btree,
                operator=node.operator,
                value=node.value,
                projection=node.columns,
                alias=node.alias,
                start_key=node.start_key,
                end_key=node.end_key,
                inclusive_start=node.inclusive_start,
                inclusive_end=node.inclusive_end,
                batch_size=DEFAULT_BATCH_SIZE,
            )

        if isinstance(node, LogicalScan):
            t_name = node.table_name
            if t_name not in self.tables:
                raise ExecutionError(f"Table '{t_name}' does not exist.")
            storage = self.tables[t_name]
            return PhysicalTableScan(
                table_storage=storage,
                projection=node.columns,
                alias=node.alias,
            )

        raise ExecutionError(f"Cannot compile unsupported logical plan node: {type(node).__name__}")

    def _compile_aggregation(
        self, child_op: ExecutionOperator, stmt: SelectStatement
    ) -> Aggregate:
        """Compile GROUP BY and aggregate functions into a physical Aggregate operator."""
        group_by_cols: List[str] = []
        if stmt.group_by:
            for g_expr in stmt.group_by:
                if isinstance(g_expr, ColumnRef):
                    name = f"{g_expr.table}.{g_expr.name}" if g_expr.table else g_expr.name
                    group_by_cols.append(name)
                elif isinstance(g_expr, ASTLiteral):
                    group_by_cols.append(str(g_expr.value))

        aggregates: List[AggregateFunction] = []
        for p in stmt.projections:
            if isinstance(p.expr, FunctionCall):
                func_name = p.expr.name.upper()
                arg = p.expr.args[0] if p.expr.args else None
                arg_name = "*"
                if isinstance(arg, ColumnRef):
                    arg_name = arg.name
                elif isinstance(arg, Star):
                    arg_name = "*"
                elif arg is not None:
                    arg_name = str(arg)

                alias = p.alias or p.name or str(p.expr)

                if func_name == "COUNT":
                    aggregates.append(Count(arg_name, alias=alias))
                elif func_name == "SUM":
                    aggregates.append(Sum(arg_name, alias=alias))
                elif func_name == "AVG":
                    aggregates.append(Avg(arg_name, alias=alias))
                elif func_name == "MIN":
                    aggregates.append(Min(arg_name, alias=alias))
                elif func_name == "MAX":
                    aggregates.append(Max(arg_name, alias=alias))
                elif func_name == "STRING_AGG":
                    aggregates.append(StringAgg(arg_name, alias=alias))
                else:
                    raise ExecutionError(f"Unsupported aggregate function: '{func_name}'")

        return Aggregate(
            child=child_op,
            group_by=group_by_cols,
            aggregates=aggregates,
            batch_size=DEFAULT_BATCH_SIZE,
        )

    def _extract_join_keys(
        self,
        condition: Optional[ASTExpression],
        left_plan: LogicalPlan,
        right_plan: LogicalPlan,
    ) -> Tuple[str, str]:
        """Extract left and right join column names from an equi-join condition."""
        if not isinstance(condition, BinaryOp) or condition.op not in ("=", "=="):
            raise ExecutionError(f"Vectorized engine only supports equi-joins, got '{condition}'")

        def _col_str(expr: ASTExpression) -> str:
            if isinstance(expr, ColumnRef):
                return expr.name
            raise ExecutionError(f"Join key must be a ColumnRef, got {type(expr).__name__}")

        left_name = _col_str(condition.left)
        right_name = _col_str(condition.right)

        # Check which table owns left_name vs right_name
        left_tables = [t.lower() for t in self._get_tables(left_plan)]
        right_tables = [t.lower() for t in self._get_tables(right_plan)]

        if isinstance(condition.left, ColumnRef) and condition.left.table:
            t = condition.left.table.lower()
            if t in right_tables:
                # Swapped
                return right_name, left_name

        return left_name, right_name

    def _get_tables(self, plan: LogicalPlan) -> List[str]:
        """Get all table names / aliases under a plan subtree."""
        tbls: List[str] = []
        if isinstance(plan, (LogicalScan, LogicalIndexScan)):
            tbls.append(plan.table_name)
            if plan.alias:
                tbls.append(plan.alias)
        for c in plan.children:
            tbls.extend(self._get_tables(c))
        return tbls

    def _format_final_projections(
        self, stmt: SelectStatement, rows: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Format final rows according to SELECT projections (aliases, star, ordering)."""
        if not rows:
            # Build column headers from projections
            cols = []
            for p in stmt.projections:
                if isinstance(p.expr, Star):
                    if stmt.from_table:
                        tbl = self.tables.get(stmt.from_table.name)
                        if tbl:
                            cols.extend(tbl.meta.column_names)
                else:
                    cols.append(p.name)
            return cols, []

        has_star = any(isinstance(p.expr, Star) for p in stmt.projections)

        if has_star and len(stmt.projections) == 1:
            # Full wildcard SELECT *
            # Keep only unqualified column names (or first occurrence)
            cols: List[str] = []
            sample = rows[0]
            # Prioritize table column names
            if stmt.from_table and stmt.from_table.name in self.tables:
                t_cols = self.tables[stmt.from_table.name].meta.column_names
                for c in t_cols:
                    if c in sample:
                        cols.append(c)
            # Add remaining keys that don't have dots
            for k in sample.keys():
                if "." not in k and k not in cols:
                    cols.append(k)
            if not cols:
                cols = list(sample.keys())

            formatted_rows = [{c: r.get(c) for c in cols} for r in rows]
            return cols, formatted_rows

        # Specific projected expressions or functions
        output_cols: List[str] = []
        for p in stmt.projections:
            col_label = p.alias or p.name or str(p.expr)
            output_cols.append(col_label)

        formatted_rows = []
        for r in rows:
            new_r: Dict[str, Any] = {}
            for p, label in zip(stmt.projections, output_cols):
                if isinstance(p.expr, ColumnRef):
                    # Try qualified name then unqualified name
                    val = None
                    if p.expr.table:
                        val = r.get(f"{p.expr.table}.{p.expr.name}")
                    if val is None:
                        val = r.get(p.expr.name)
                    new_r[label] = val
                elif isinstance(p.expr, FunctionCall):
                    val = r.get(label)
                    if val is None:
                        val = r.get(str(p.expr))
                    new_r[label] = val
                else:
                    new_r[label] = r.get(label, r.get(p.name, r.get(str(p.expr))))
            formatted_rows.append(new_r)

        return output_cols, formatted_rows


# ============================================================================
# SQL REPL Console
# ============================================================================


class SQLRepl:
    """Interactive SQL Read-Eval-Print-Loop (REPL)."""

    PROMPT_PRIMARY: str = "sql> "
    PROMPT_CONTINUATION: str = "   -> "

    def __init__(
        self,
        engine: Optional[DatabaseEngine] = None,
        stdin: Optional[Any] = None,
        stdout: Optional[Any] = None,
    ) -> None:
        self.engine: DatabaseEngine = engine if engine is not None else DatabaseEngine()
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._running: bool = False

    def println(self, text: str = "") -> None:
        """Print a line to the configured stdout."""
        print(text, file=self.stdout)
        if hasattr(self.stdout, "flush"):
            self.stdout.flush()

    def run(self) -> None:
        """Run the interactive REPL loop."""
        self._running = True
        self.println("Vectorized SQL Engine REPL")
        self.println("Type SQL statements or .help for commands. Type .exit to quit.")

        buffer: List[str] = []

        while self._running:
            prompt = self.PROMPT_CONTINUATION if buffer else self.PROMPT_PRIMARY
            try:
                line = self._read_line(prompt)
            except EOFError:
                self.println("\nGoodbye!")
                break
            except KeyboardInterrupt:
                self.println("\nKeyboardInterrupt")
                buffer = []
                continue

            if line is None:
                break

            stripped = line.strip()

            # Handle dot commands immediately if not currently in a multi-line buffer
            if not buffer and stripped.startswith("."):
                self._handle_dot_command(stripped)
                continue

            if not stripped and not buffer:
                continue

            buffer.append(line)
            combined_input = "\n".join(buffer).strip()

            # Check if command is complete (ends with semicolon or single-line complete query)
            if combined_input.endswith(";") or self._is_complete_single_line(combined_input):
                # Strip trailing semicolon for execution
                clean_sql = combined_input.rstrip(";").strip()
                buffer = []
                if clean_sql:
                    self.execute_sql(clean_sql)

        self.engine.close()

    def _read_line(self, prompt: str) -> Optional[str]:
        """Read a line of text, compatible with standard input or custom stream."""
        if hasattr(self.stdin, "readline"):
            if sys.stdin == self.stdin:
                try:
                    return input(prompt)
                except EOFError:
                    return None
            else:
                self.stdout.write(prompt)
                if hasattr(self.stdout, "flush"):
                    self.stdout.flush()
                line = self.stdin.readline()
                if not line:
                    return None
                return line.rstrip("\r\n")
        return None

    def _is_complete_single_line(self, text: str) -> bool:
        """Heuristic to determine if single-line input is a self-contained SQL statement."""
        if "\n" in text:
            return False
        # If it parses without error, execute immediately
        try:
            parse_sql(text)
            return True
        except Exception:
            return False

    def _handle_dot_command(self, cmd: str) -> None:
        """Handle special system commands: .tables, .schema, .exit, .quit, .help."""
        parts = cmd.split()
        keyword = parts[0].lower()

        if keyword in (".exit", ".quit"):
            self.println("Goodbye!")
            self._running = False
            return

        if keyword == ".tables":
            tables = sorted(self.engine.tables.keys())
            if not tables:
                self.println("No tables found.")
            else:
                for t in tables:
                    self.println(t)
            return

        if keyword == ".schema":
            if len(parts) == 1:
                # Print schema for all tables
                if not self.engine.tables:
                    self.println("No tables defined.")
                else:
                    for t_name in sorted(self.engine.tables.keys()):
                        self._print_table_schema(t_name)
            else:
                target_table = parts[1]
                if target_table not in self.engine.tables:
                    self.println(f"Error: Table '{target_table}' not found.")
                else:
                    self._print_table_schema(target_table)
            return

        if keyword == ".help":
            self.println("Available special commands:")
            self.println("  .tables             List all existing tables")
            self.println("  .schema [<table>]   Show CREATE TABLE schema for tables")
            self.println("  .exit / .quit       Exit the REPL")
            self.println("  .help               Display this help text")
            return

        self.println(f"Unknown special command '{cmd}'. Type .help for available commands.")

    def _print_table_schema(self, table_name: str) -> None:
        """Print the formatted CREATE TABLE DDL statement for a table."""
        storage = self.engine.tables[table_name]
        col_defs = []
        for col in storage.meta.columns:
            pk_str = " PRIMARY KEY" if col.primary_key else ""
            null_str = " NOT NULL" if not col.nullable else ""
            uniq_str = " UNIQUE" if col.unique else ""
            col_defs.append(f"  {col.name} {col.data_type}{pk_str}{null_str}{uniq_str}")

        ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
        self.println(ddl)

    def execute_sql(self, sql: str) -> bool:
        """Execute one or more SQL commands separated by semicolons with friendly error handling.

        Returns True on success, False on error.
        """
        try:
            stmts = parse_statements(sql)
            if not stmts:
                return True

            for stmt in stmts:
                cols, rows, elapsed_ms, pages_read, msg = self.engine.execute_statement(stmt)

                if msg:
                    self.println(msg)
                    self.println(f"Execution time: {elapsed_ms:.2f} ms | Pages read: {pages_read}")
                elif cols is not None and rows is not None:
                    # Print ASCII table
                    ascii_table = render_ascii_table(cols, rows)
                    if ascii_table:
                        self.println(ascii_table)

                    row_count = len(rows)
                    row_label = "1 row" if row_count == 1 else f"{row_count} rows"
                    page_label = "1 page" if pages_read == 1 else f"{pages_read} pages"

                    self.println(f"({row_label} in {elapsed_ms:.2f} ms, {page_label} read)")
                    self.println(f"Execution time: {elapsed_ms:.2f} ms | Pages read: {pages_read}")

            return True

        except (SQLSyntaxError, ParserError, UnexpectedTokenError) as err:
            self.println(f"Syntax Error: {err}")
            return False
        except (QueryEngineError, ExecutionError, SchemaError, ColumnNotFoundError) as err:
            self.println(f"Error: {err}")
            return False
        except Exception as err:
            self.println(f"Error: {err}")
            return False


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Command-line entry point for 'python -m src.cli'."""
    parser = argparse.ArgumentParser(
        description="Vectorized SQL Engine Interactive REPL",
        prog="python -m src.cli",
    )
    parser.add_argument(
        "database",
        nargs="?",
        default=None,
        help="Path to database file (optional; defaults to in-memory/temporary)",
    )
    parser.add_argument(
        "-c",
        "--command",
        dest="command",
        default=None,
        help="Execute single SQL command and exit",
    )
    args = parser.parse_args()

    engine = DatabaseEngine(db_file=args.database)
    repl = SQLRepl(engine=engine)

    if args.command:
        # One-shot command execution
        repl.execute_sql(args.command)
        engine.close()
    else:
        # Start interactive REPL
        repl.run()


if __name__ == "__main__":
    main()
