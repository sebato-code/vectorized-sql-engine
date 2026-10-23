"""Vectorized query execution engine implementing the Volcano iterator model.

Provides batch-oriented relational query execution:
- VectorBatch: In-memory columnar data container supporting vectorized filtering,
  slicing, and projections.
- Volcano Execution Operators:
  * ExecutionOperator: Abstract base class implementing the Volcano lifecycle
    (open, next, close) with Python iterator and context manager support.
  * SeqScan: Sequential scan over tabular data producing VectorBatches.
  * Filter: Vectorized predicate evaluation and filtering.
  * HashJoin: Two-phase in-memory Hash Join (Build and Probe).
  * Aggregate: Hash-based grouping and aggregation (GROUP BY, scalar aggregations).
  * Project, Limit, OrderBy: Supplementary operators for complete SQL execution.
- Vectorized Expressions:
  * AST expression nodes with operator overloading for boolean, arithmetic,
    and null-checking expressions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import fnmatch
from functools import cmp_to_key
from math import isnan
import operator
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

DEFAULT_BATCH_SIZE: int = 1024


# ============================================================================
# Exceptions
# ============================================================================


class QueryEngineError(Exception):
    """Base exception for all query engine errors."""


class SchemaError(QueryEngineError, ValueError):
    """Raised for schema mismatches or invalid column definitions."""


class ColumnNotFoundError(QueryEngineError, KeyError):
    """Raised when a requested column does not exist in a batch or table."""


class ExecutionError(QueryEngineError, RuntimeError):
    """Raised when an error occurs during operator execution."""


# ============================================================================
# VectorBatch
# ============================================================================


class VectorBatch:
    """In-memory columnar batch of tuples for vectorized Volcano execution.

    Internal storage maps column names to homogeneous Python lists:
        _columns: Dict[str, List[Any]]

    Key properties:
    - Fixed length across all column vectors in the batch (num_rows).
    - True columnar vector operations (filter by boolean mask, project, slice).
    - Seamless conversions to/from row dictionaries and tuples.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Sequence[Any]]] = None,
        num_rows: Optional[int] = None,
    ) -> None:
        self._columns: Dict[str, List[Any]] = {}

        if data is None or len(data) == 0:
            self._num_rows: int = num_rows if num_rows is not None else 0
            if data is not None:
                for k in data:
                    self._columns[k] = []
            return

        # Validate and convert columns
        row_count: Optional[int] = None
        for col_name, col_data in data.items():
            if not isinstance(col_name, str):
                raise SchemaError(f"Column name must be str, got {type(col_name).__name__}")
            converted = list(col_data)
            if row_count is None:
                row_count = len(converted)
            elif len(converted) != row_count:
                raise SchemaError(
                    f"Column '{col_name}' length {len(converted)} does not match "
                    f"batch row count {row_count}"
                )
            self._columns[col_name] = converted

        if num_rows is not None and row_count is not None and num_rows != row_count:
            raise SchemaError(f"Explicit num_rows {num_rows} does not match column length {row_count}")

        self._num_rows = row_count if row_count is not None else (num_rows or 0)

    @property
    def columns(self) -> List[str]:
        """Return list of column names in the batch."""
        return list(self._columns.keys())

    @property
    def num_rows(self) -> int:
        """Return number of rows in the batch."""
        return self._num_rows

    def __len__(self) -> int:
        """Return row count."""
        return self._num_rows

    def __bool__(self) -> bool:
        """True if batch has at least one row."""
        return self._num_rows > 0

    def __contains__(self, col_name: str) -> bool:
        """Check if column exists in batch."""
        return col_name in self._columns

    def __getitem__(self, key: Union[str, int, slice]) -> Any:
        """Access columns by name, rows by index, or slice rows."""
        if isinstance(key, str):
            if key not in self._columns:
                raise ColumnNotFoundError(f"Column '{key}' not found in VectorBatch")
            return self._columns[key]
        if isinstance(key, int):
            return self.get_row(key)
        if isinstance(key, slice):
            start, stop, step = key.indices(self._num_rows)
            if step != 1:
                indices = list(range(start, stop, step))
                return VectorBatch(
                    data={c: [self._columns[c][i] for i in indices] for c in self._columns},
                    num_rows=len(indices),
                )
            length = max(0, stop - start)
            return self.slice(offset=start, length=length)
        raise TypeError(f"Invalid subscript key type: {type(key).__name__}")

    def column(self, name: str) -> List[Any]:
        """Retrieve list of values for a given column name."""
        if name not in self._columns:
            raise ColumnNotFoundError(f"Column '{name}' not found in VectorBatch")
        return self._columns[name]

    def get_row(self, index: int) -> Dict[str, Any]:
        """Retrieve a single row dictionary at the specified row index."""
        if index < 0:
            index += self._num_rows
        if index < 0 or index >= self._num_rows:
            raise IndexError(f"Row index {index} out of range [0, {self._num_rows})")
        return {col: vals[index] for col, vals in self._columns.items()}

    def filter(self, mask: Sequence[Any]) -> VectorBatch:
        """Filter batch rows using a boolean mask vector.

        Returns a new VectorBatch containing only rows where mask[i] evaluates to True.
        """
        if len(mask) != self._num_rows:
            raise ValueError(
                f"Mask length {len(mask)} does not match VectorBatch row count {self._num_rows}"
            )
        indices = [i for i, m in enumerate(mask) if bool(m) is True]
        filtered_cols = {
            col: [vals[i] for i in indices]
            for col, vals in self._columns.items()
        }
        return VectorBatch(data=filtered_cols, num_rows=len(indices))

    def slice(self, offset: int, length: Optional[int] = None) -> VectorBatch:
        """Return a slice of the VectorBatch across all columns."""
        if offset < 0:
            offset = max(0, self._num_rows + offset)
        end = self._num_rows if length is None else min(self._num_rows, offset + length)
        if offset >= self._num_rows or offset >= end:
            return VectorBatch(data={col: [] for col in self._columns}, num_rows=0)
        sliced_data = {
            col: self._columns[col][offset:end]
            for col in self._columns
        }
        return VectorBatch(data=sliced_data, num_rows=end - offset)

    def project(self, columns: Sequence[str]) -> VectorBatch:
        """Create a new VectorBatch containing only the requested columns in order."""
        for col in columns:
            if col not in self._columns:
                raise ColumnNotFoundError(
                    f"Cannot project non-existent column '{col}' (available: {list(self._columns.keys())})"
                )
        projected = {col: list(self._columns[col]) for col in columns}
        return VectorBatch(data=projected, num_rows=self._num_rows)

    def rename(self, mapping: Dict[str, str]) -> VectorBatch:
        """Rename columns according to mapping dict (old_name -> new_name)."""
        new_cols: Dict[str, List[Any]] = {}
        for old_name, vals in self._columns.items():
            new_name = mapping.get(old_name, old_name)
            new_cols[new_name] = list(vals)
        return VectorBatch(data=new_cols, num_rows=self._num_rows)

    def with_column(self, col_name: str, values: Sequence[Any]) -> VectorBatch:
        """Return a new VectorBatch with an added or updated column."""
        vals_list = list(values)
        if len(vals_list) != self._num_rows and self._num_rows != 0:
            raise ValueError(
                f"Column '{col_name}' length {len(vals_list)} does not match batch length {self._num_rows}"
            )
        new_data = {k: list(v) for k, v in self._columns.items()}
        new_data[col_name] = vals_list
        row_count = len(vals_list) if not self._columns else self._num_rows
        return VectorBatch(data=new_data, num_rows=row_count)

    def to_rows(self) -> List[Dict[str, Any]]:
        """Convert columnar batch to a list of row dictionaries."""
        cols = list(self._columns.items())
        return [
            {name: vals[i] for name, vals in cols}
            for i in range(self._num_rows)
        ]

    def to_tuples(self, columns: Optional[Sequence[str]] = None) -> List[Tuple[Any, ...]]:
        """Convert columnar batch to a list of row tuples."""
        target_cols = list(columns) if columns is not None else self.columns
        for c in target_cols:
            if c not in self._columns:
                raise ColumnNotFoundError(f"Column '{c}' not found in VectorBatch")
        vectors = [self._columns[c] for c in target_cols]
        return [tuple(vec[i] for vec in vectors) for i in range(self._num_rows)]

    def to_dict(self) -> Dict[str, List[Any]]:
        """Return a copy of the underlying columnar dictionary."""
        return {k: list(v) for k, v in self._columns.items()}

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Dict[str, Any]],
        columns: Optional[Sequence[str]] = None,
    ) -> VectorBatch:
        """Construct a VectorBatch from a sequence of row dictionaries."""
        if not rows:
            if columns is not None:
                return cls(data={col: [] for col in columns}, num_rows=0)
            return cls(data={}, num_rows=0)

        if columns is not None:
            col_names = list(columns)
        else:
            # Preserve insertion order from first row dict
            col_names = list(rows[0].keys())

        data: Dict[str, List[Any]] = {col: [] for col in col_names}
        for row in rows:
            for col in col_names:
                data[col].append(row.get(col))

        return cls(data=data, num_rows=len(rows))

    @classmethod
    def from_dict(cls, data: Dict[str, Sequence[Any]]) -> VectorBatch:
        """Construct a VectorBatch from a dictionary mapping column names to sequences."""
        return cls(data=data)

    @classmethod
    def concat(cls, batches: Sequence[VectorBatch]) -> VectorBatch:
        """Concatenate multiple VectorBatches with identical column schemas."""
        if not batches:
            return cls(data={}, num_rows=0)
        first = batches[0]
        col_names = first.columns
        combined: Dict[str, List[Any]] = {c: [] for c in col_names}
        total_rows = 0
        for b in batches:
            if b.columns != col_names:
                raise SchemaError(
                    f"Cannot concatenate batches with mismatched schemas: {b.columns} vs {col_names}"
                )
            for c in col_names:
                combined[c].extend(b._columns[c])
            total_rows += b.num_rows
        return cls(data=combined, num_rows=total_rows)

    def __repr__(self) -> str:
        return f"VectorBatch(num_rows={self._num_rows}, columns={self.columns})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VectorBatch):
            return False
        if self._num_rows != other._num_rows or self.columns != other.columns:
            return False
        for c in self.columns:
            if self._columns[c] != other._columns[c]:
                return False
        return True


# ============================================================================
# Vectorized Expressions (AST and Vectorized Evaluation)
# ============================================================================


class Expression(ABC):
    """Abstract base class for all expressions evaluated on a VectorBatch."""

    @abstractmethod
    def evaluate(self, batch: VectorBatch) -> List[Any]:
        """Evaluate expression on the given VectorBatch returning a list of values."""

    def __eq__(self, other: Any) -> Expression:  # type: ignore[override]
        return Eq(self, _to_expr(other))

    def __ne__(self, other: Any) -> Expression:  # type: ignore[override]
        return Ne(self, _to_expr(other))

    def __lt__(self, other: Any) -> Expression:
        return Lt(self, _to_expr(other))

    def __le__(self, other: Any) -> Expression:
        return Le(self, _to_expr(other))

    def __gt__(self, other: Any) -> Expression:
        return Gt(self, _to_expr(other))

    def __ge__(self, other: Any) -> Expression:
        return Ge(self, _to_expr(other))

    def __add__(self, other: Any) -> Expression:
        return Add(self, _to_expr(other))

    def __sub__(self, other: Any) -> Expression:
        return Sub(self, _to_expr(other))

    def __mul__(self, other: Any) -> Expression:
        return Mul(self, _to_expr(other))

    def __truediv__(self, other: Any) -> Expression:
        return Div(self, _to_expr(other))

    def __mod__(self, other: Any) -> Expression:
        return Mod(self, _to_expr(other))

    def __and__(self, other: Any) -> Expression:
        return And(self, _to_expr(other))

    def __or__(self, other: Any) -> Expression:
        return Or(self, _to_expr(other))

    def __invert__(self) -> Expression:
        return Not(self)

    def is_null(self) -> Expression:
        """Create an IS NULL expression."""
        return IsNull(self)

    def is_not_null(self) -> Expression:
        """Create an IS NOT NULL expression."""
        return IsNotNull(self)

    def in_list(self, values: Sequence[Any]) -> Expression:
        """Create an IN expression."""
        return InList(self, values)

    def like(self, pattern: str) -> Expression:
        """Create a SQL LIKE expression."""
        return Like(self, pattern)


def _to_expr(value: Any) -> Expression:
    """Helper to wrap literals into Expression instances."""
    if isinstance(value, Expression):
        return value
    return Literal(value)


class Col(Expression):
    """Reference to a column by name in a VectorBatch."""

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Column name must be str, got {type(name).__name__}")
        self.name: str = name

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        return list(batch.column(self.name))

    def __repr__(self) -> str:
        return f"Col('{self.name}')"


def col(name: str) -> Col:
    """Convenience factory function for Col."""
    return Col(name)


class Literal(Expression):
    """Constant scalar literal expression."""

    def __init__(self, value: Any) -> None:
        self.value: Any = value

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        return [self.value] * batch.num_rows

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


def lit(value: Any) -> Literal:
    """Convenience factory function for Literal."""
    return Literal(value)


class BinaryOp(Expression):
    """Base class for binary operations on expressions."""

    def __init__(self, left: Expression, right: Expression) -> None:
        self.left: Expression = _to_expr(left)
        self.right: Expression = _to_expr(right)

    @abstractmethod
    def _op(self, a: Any, b: Any) -> Any:
        """Scalar operation implementation."""

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        left_vals = self.left.evaluate(batch)
        right_vals = self.right.evaluate(batch)
        return [self._safe_op(a, b) for a, b in zip(left_vals, right_vals)]

    def _safe_op(self, a: Any, b: Any) -> Any:
        if a is None or b is None:
            return None
        try:
            return self._op(a, b)
        except Exception:
            return None


class Eq(BinaryOp):
    """Vectorized equality comparison (==). Follows SQL NULL semantics."""

    def _op(self, a: Any, b: Any) -> bool:
        return bool(a == b)

    def __repr__(self) -> str:
        return f"({self.left} == {self.right})"


class Ne(BinaryOp):
    """Vectorized inequality comparison (!=)."""

    def _op(self, a: Any, b: Any) -> bool:
        return bool(a != b)

    def __repr__(self) -> str:
        return f"({self.left} != {self.right})"


class Lt(BinaryOp):
    """Vectorized less than comparison (<)."""

    def _op(self, a: Any, b: Any) -> bool:
        return bool(a < b)

    def __repr__(self) -> str:
        return f"({self.left} < {self.right})"


class Le(BinaryOp):
    """Vectorized less than or equal comparison (<=)."""

    def _op(self, a: Any, b: Any) -> bool:
        return bool(a <= b)

    def __repr__(self) -> str:
        return f"({self.left} <= {self.right})"


class Gt(BinaryOp):
    """Vectorized greater than comparison (>)."""

    def _op(self, a: Any, b: Any) -> bool:
        return bool(a > b)

    def __repr__(self) -> str:
        return f"({self.left} > {self.right})"


class Ge(BinaryOp):
    """Vectorized greater than or equal comparison (>=)."""

    def _op(self, a: Any, b: Any) -> bool:
        return bool(a >= b)

    def __repr__(self) -> str:
        return f"({self.left} >= {self.right})"


class Add(BinaryOp):
    """Vectorized addition (+)."""

    def _op(self, a: Any, b: Any) -> Any:
        return a + b

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


class Sub(BinaryOp):
    """Vectorized subtraction (-)."""

    def _op(self, a: Any, b: Any) -> Any:
        return a - b

    def __repr__(self) -> str:
        return f"({self.left} - {self.right})"


class Mul(BinaryOp):
    """Vectorized multiplication (*)."""

    def _op(self, a: Any, b: Any) -> Any:
        return a * b

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"


class Div(BinaryOp):
    """Vectorized true division (/), handles division by zero returning None."""

    def _op(self, a: Any, b: Any) -> Any:
        if b == 0:
            return None
        return a / b

    def __repr__(self) -> str:
        return f"({self.left} / {self.right})"


class Mod(BinaryOp):
    """Vectorized modulo (%)."""

    def _op(self, a: Any, b: Any) -> Any:
        if b == 0:
            return None
        return a % b

    def __repr__(self) -> str:
        return f"({self.left} % {self.right})"


class And(BinaryOp):
    """Vectorized logical AND with SQL three-valued logic."""

    def _op(self, a: Any, b: Any) -> Any:
        return bool(a and b)

    def _safe_op(self, a: Any, b: Any) -> Any:
        # SQL Three-valued logic:
        # False AND NULL -> False
        # NULL AND False -> False
        # True AND NULL -> NULL
        # NULL AND True -> NULL
        # NULL AND NULL -> NULL
        if a is False or b is False:
            return False
        if a is None or b is None:
            return None
        return bool(a and b)

    def __repr__(self) -> str:
        return f"({self.left} & {self.right})"


class Or(BinaryOp):
    """Vectorized logical OR with SQL three-valued logic."""

    def _op(self, a: Any, b: Any) -> Any:
        return bool(a or b)

    def _safe_op(self, a: Any, b: Any) -> Any:
        # SQL Three-valued logic:
        # True OR NULL -> True
        # NULL OR True -> True
        # False OR NULL -> NULL
        # NULL OR False -> NULL
        # NULL OR NULL -> NULL
        if a is True or b is True:
            return True
        if a is None or b is None:
            return None
        return bool(a or b)

    def __repr__(self) -> str:
        return f"({self.left} | {self.right})"


class Not(Expression):
    """Vectorized logical NOT."""

    def __init__(self, expr: Expression) -> None:
        self.expr: Expression = _to_expr(expr)

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        vals = self.expr.evaluate(batch)
        return [None if v is None else not bool(v) for v in vals]

    def __repr__(self) -> str:
        return f"(~{self.expr})"


class IsNull(Expression):
    """Vectorized IS NULL predicate."""

    def __init__(self, expr: Expression) -> None:
        self.expr: Expression = _to_expr(expr)

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        vals = self.expr.evaluate(batch)
        return [v is None for v in vals]

    def __repr__(self) -> str:
        return f"({self.expr}.is_null())"


class IsNotNull(Expression):
    """Vectorized IS NOT NULL predicate."""

    def __init__(self, expr: Expression) -> None:
        self.expr: Expression = _to_expr(expr)

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        vals = self.expr.evaluate(batch)
        return [v is not None for v in vals]

    def __repr__(self) -> str:
        return f"({self.expr}.is_not_null())"


class InList(Expression):
    """Vectorized IN (values...) predicate."""

    def __init__(self, expr: Expression, values: Sequence[Any]) -> None:
        self.expr: Expression = _to_expr(expr)
        self.values_set: Set[Any] = set(values)

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        vals = self.expr.evaluate(batch)
        return [None if v is None else (v in self.values_set) for v in vals]

    def __repr__(self) -> str:
        return f"({self.expr} IN {list(self.values_set)})"


def _sql_like_to_regex(pattern: str) -> str:
    """Convert SQL LIKE pattern (% and _) into a standard regex."""
    result = ["^"]
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "%":
            result.append(".*")
        elif c == "_":
            result.append(".")
        elif c == "\\":
            i += 1
            if i < n:
                result.append(re.escape(pattern[i]))
        else:
            result.append(re.escape(c))
        i += 1
    result.append("$")
    return "".join(result)


class Like(Expression):
    """Vectorized SQL LIKE pattern matching (supports % and _)."""

    def __init__(self, expr: Expression, pattern: str) -> None:
        self.expr: Expression = _to_expr(expr)
        self.pattern: str = pattern
        self._compiled_regex = re.compile(_sql_like_to_regex(pattern))

    def evaluate(self, batch: VectorBatch) -> List[Any]:
        vals = self.expr.evaluate(batch)
        result: List[Any] = []
        for v in vals:
            if v is None:
                result.append(None)
            else:
                result.append(bool(self._compiled_regex.match(str(v))))
        return result

    def __repr__(self) -> str:
        return f"({self.expr} LIKE '{self.pattern}')"


# ============================================================================
# Volcano Execution Operator Base Class
# ============================================================================


class ExecutionOperator(ABC):
    """Base class for all Volcano iterator execution operators.

    Contract:
    - open(): Prepares operator state and recursively opens child operators.
    - next(): Produces the next VectorBatch or returns None when EOF is reached.
    - close(): Releases resources and recursively closes child operators.
    """

    def __init__(self) -> None:
        self._is_open: bool = False

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the operator."""
        self._is_open = True

    @abstractmethod
    def next(self) -> Optional[VectorBatch]:
        """Fetch the next VectorBatch, or None if exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Close operator and cleanup resources."""
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if operator is currently open."""
        return self._is_open

    @property
    def columns(self) -> Optional[List[str]]:
        """Return output column schema names if statically known, else None."""
        return None

    def __iter__(self) -> Iterator[VectorBatch]:
        """Support standard Python iteration over VectorBatches."""
        if not self._is_open:
            self.open()
        try:
            while True:
                batch = self.next()
                if batch is None:
                    break
                yield batch
        finally:
            self.close()

    def __next__(self) -> VectorBatch:
        """Support next() on operator instance."""
        if not self._is_open:
            self.open()
        batch = self.next()
        if batch is None:
            raise StopIteration
        return batch

    def __enter__(self) -> ExecutionOperator:
        """Context manager entry opens the operator."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit ensures operator is closed."""
        self.close()

    def execute(self) -> List[VectorBatch]:
        """Execute query pipeline to completion and collect all VectorBatches."""
        self.open()
        batches: List[VectorBatch] = []
        try:
            while True:
                batch = self.next()
                if batch is None:
                    break
                batches.append(batch)
            return batches
        finally:
            self.close()

    def execute_all_rows(self) -> List[Dict[str, Any]]:
        """Execute pipeline and return combined rows as a list of dicts."""
        batches = self.execute()
        rows: List[Dict[str, Any]] = []
        for b in batches:
            rows.extend(b.to_rows())
        return rows


# ============================================================================
# Table Container
# ============================================================================


class Table:
    """In-memory relational table for data storage and scan queries."""

    def __init__(
        self,
        name: str,
        schema: Optional[Sequence[str]] = None,
        data: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.name: str = name
        self._schema: List[str] = list(schema) if schema is not None else []
        self._rows: List[Dict[str, Any]] = []
        if data:
            for row in data:
                self.insert(row)

    @property
    def schema(self) -> List[str]:
        return list(self._schema)

    def insert(self, row: Dict[str, Any]) -> None:
        """Insert a row dictionary into the table."""
        if not self._schema:
            self._schema = list(row.keys())
        self._rows.append(dict(row))

    def insert_batch(self, batch: VectorBatch) -> None:
        """Insert all rows from a VectorBatch."""
        if not self._schema and batch.columns:
            self._schema = list(batch.columns)
        self._rows.extend(batch.to_rows())

    def scan(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        projection: Optional[Sequence[str]] = None,
    ) -> SeqScan:
        """Return a SeqScan operator over this table."""
        return SeqScan(self, batch_size=batch_size, projection=projection)

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"Table('{self.name}', schema={self._schema}, rows={len(self._rows)})"


# ============================================================================
# SeqScan Operator
# ============================================================================


class SeqScan(ExecutionOperator):
    """Sequential scan operator emitting VectorBatches.

    Accepts:
    - Table instance
    - List of row dictionaries
    - Dict of column sequences
    - List of existing VectorBatches
    - Single VectorBatch
    """

    def __init__(
        self,
        data: Union[
            Table,
            Sequence[Dict[str, Any]],
            Dict[str, Sequence[Any]],
            Sequence[VectorBatch],
            VectorBatch,
        ],
        batch_size: int = DEFAULT_BATCH_SIZE,
        projection: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size: int = batch_size
        self.projection: Optional[List[str]] = list(projection) if projection is not None else None

        # Normalized storage: list of VectorBatches or row dicts
        self._source_batches: List[VectorBatch] = []
        self._current_batch_idx: int = 0
        self._current_row_idx: int = 0

        if isinstance(data, Table):
            vb = VectorBatch.from_rows(data._rows, columns=data._schema if data._schema else None)
            self._source_batches = [vb] if vb.num_rows > 0 else []
        elif isinstance(data, VectorBatch):
            self._source_batches = [data] if data.num_rows > 0 else []
        elif isinstance(data, dict):
            vb = VectorBatch.from_dict(data)
            self._source_batches = [vb] if vb.num_rows > 0 else []
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                self._source_batches = []
            elif isinstance(data[0], VectorBatch):
                self._source_batches = [b for b in data if b.num_rows > 0]  # type: ignore[arg-type]
            elif isinstance(data[0], dict):
                vb = VectorBatch.from_rows(data)  # type: ignore[arg-type]
                self._source_batches = [vb] if vb.num_rows > 0 else []
            else:
                raise TypeError(f"Unsupported sequence item type: {type(data[0]).__name__}")
        else:
            raise TypeError(f"Unsupported data source type for SeqScan: {type(data).__name__}")

        # Total rows across all source batches
        self._total_rows = sum(b.num_rows for b in self._source_batches)
        self._cursor: int = 0

        # Cached flat rows or concatenated batch if multiple
        if len(self._source_batches) == 1:
            self._flat_batch: Optional[VectorBatch] = self._source_batches[0]
        elif len(self._source_batches) > 1:
            self._flat_batch = VectorBatch.concat(self._source_batches)
        else:
            self._flat_batch = None

        # Schema resolution
        if self.projection is not None:
            self._schema = list(self.projection)
        elif isinstance(data, Table) and data.schema:
            self._schema = list(data.schema)
        elif self._flat_batch is not None and self._flat_batch.columns:
            self._schema = list(self._flat_batch.columns)
        elif isinstance(data, dict):
            self._schema = list(data.keys())
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
            self._schema = list(data[0].keys())
        else:
            self._schema = []

    @property
    def columns(self) -> Optional[List[str]]:
        return list(self._schema) if self._schema else None

    def open(self) -> None:
        super().open()
        self._cursor = 0

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        if self._flat_batch is None or self._cursor >= self._total_rows:
            return None

        length = min(self.batch_size, self._total_rows - self._cursor)
        batch = self._flat_batch.slice(self._cursor, length)
        self._cursor += length

        if self.projection is not None:
            batch = batch.project(self.projection)

        return batch

    def close(self) -> None:
        super().close()
        self._cursor = 0


# ============================================================================
# Filter Operator
# ============================================================================


class Filter(ExecutionOperator):
    """Vectorized Filter operator applying boolean predicates.

    Filters input batches from a child operator using an Expression or callable predicate.
    Pulls batches from child until a non-empty batch of filtered tuples is found or EOF.
    """

    def __init__(
        self,
        child: ExecutionOperator,
        predicate: Union[
            Expression,
            Callable[[VectorBatch], Sequence[bool]],
            Callable[[Dict[str, Any]], bool],
        ],
    ) -> None:
        super().__init__()
        self.child: ExecutionOperator = child
        self.predicate: Union[
            Expression,
            Callable[[VectorBatch], Sequence[bool]],
            Callable[[Dict[str, Any]], bool],
        ] = predicate

    def open(self) -> None:
        super().open()
        self.child.open()

    def _evaluate_mask(self, batch: VectorBatch) -> List[bool]:
        """Vectorized evaluation of predicate producing a boolean list."""
        if isinstance(self.predicate, Expression):
            vals = self.predicate.evaluate(batch)
            return [bool(v) is True for v in vals]

        if callable(self.predicate):
            try:
                # Try passing batch directly
                res = self.predicate(batch)  # type: ignore[call-arg]
                if isinstance(res, (list, tuple)) and len(res) == batch.num_rows:
                    return [bool(m) is True for m in res]
            except Exception:
                pass

            # Fallback to row-level evaluation
            return [bool(self.predicate(row)) is True for row in batch.to_rows()]  # type: ignore[call-arg]

        raise TypeError(f"Unsupported predicate type: {type(self.predicate).__name__}")

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        while True:
            batch = self.child.next()
            if batch is None:
                return None
            if len(batch) == 0:
                continue

            mask = self._evaluate_mask(batch)
            filtered = batch.filter(mask)
            if len(filtered) > 0:
                return filtered
            # If all rows were filtered out, continue pulling from child

    def close(self) -> None:
        super().close()
        self.child.close()

    @property
    def columns(self) -> Optional[List[str]]:
        return self.child.columns


# ============================================================================
# HashJoin Operator (Two-Phase In-Memory Build / Probe)
# ============================================================================


class HashJoin(ExecutionOperator):
    """Two-phase in-memory Hash Join operator (Build and Probe).

    Execution phases:
    1. Build Phase (in open()):
       Consumes the entire build child (right child) into an in-memory hash table
       keyed by the hash of right_on key columns.
    2. Probe Phase (in next()):
       Streams the probe child (left child) batch by batch, looks up matches in
       the hash table, and outputs batches of joined rows.

    Supports join types:
    - INNER
    - LEFT (LEFT_OUTER)
    - RIGHT (RIGHT_OUTER)
    - FULL (FULL_OUTER)
    - SEMI
    - ANTI
    """

    def __init__(
        self,
        left: ExecutionOperator,
        right: ExecutionOperator,
        left_on: Union[str, Sequence[str]],
        right_on: Union[str, Sequence[str]],
        join_type: str = "INNER",
        batch_size: int = DEFAULT_BATCH_SIZE,
        left_prefix: str = "",
        right_prefix: str = "",
    ) -> None:
        super().__init__()
        self.left: ExecutionOperator = left
        self.right: ExecutionOperator = right

        self.left_on: List[str] = [left_on] if isinstance(left_on, str) else list(left_on)
        self.right_on: List[str] = [right_on] if isinstance(right_on, str) else list(right_on)

        if len(self.left_on) != len(self.right_on):
            raise ValueError(
                f"Number of join keys must match: left has {len(self.left_on)}, right has {len(self.right_on)}"
            )

        self.join_type: str = join_type.upper().replace("_OUTER", "")
        valid_types = {"INNER", "LEFT", "RIGHT", "FULL", "SEMI", "ANTI"}
        if self.join_type not in valid_types:
            raise ValueError(f"Unsupported join_type '{join_type}'. Must be one of {valid_types}")

        self.batch_size: int = batch_size
        self.left_prefix: str = left_prefix
        self.right_prefix: str = right_prefix

        # Build phase state
        self._hash_table: Dict[Tuple[Any, ...], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        self._right_rows: List[Dict[str, Any]] = []
        self._right_columns: List[str] = []
        self._left_columns: List[str] = []
        self._matched_right_indices: Set[int] = set()

        # Probe phase state
        self._left_eof: bool = False
        self._unmatched_right_emitted: bool = False
        self._output_buffer: List[Dict[str, Any]] = []

    def open(self) -> None:
        super().open()
        self.left.open()
        self.right.open()

        self._hash_table.clear()
        self._right_rows.clear()
        self._right_columns.clear()
        self._left_columns.clear()
        self._matched_right_indices.clear()
        self._output_buffer.clear()
        self._left_eof = False
        self._unmatched_right_emitted = False

        if getattr(self.right, "columns", None):
            self._right_columns = list(self.right.columns)  # type: ignore[arg-type]
        if getattr(self.left, "columns", None):
            self._left_columns = list(self.left.columns)  # type: ignore[arg-type]

        # Phase 1: BUILD PHASE
        # Fully consume the right child to build the hash table
        while True:
            batch = self.right.next()
            if batch is None:
                break
            if not self._right_columns and batch.columns:
                self._right_columns = list(batch.columns)

            for row in batch.to_rows():
                right_idx = len(self._right_rows)
                self._right_rows.append(row)

                # Extract join key tuple
                key = tuple(row.get(k) for k in self.right_on)
                # In SQL equi-joins, NULL never matches NULL
                if any(v is None for v in key):
                    continue
                self._hash_table[key].append((right_idx, row))

    def _merge_row(
        self,
        left_row: Optional[Dict[str, Any]],
        right_row: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine left and right row dicts handling prefix and column collisions."""
        merged: Dict[str, Any] = {}

        if left_row is not None:
            for k, v in left_row.items():
                col_name = f"{self.left_prefix}{k}" if self.left_prefix else k
                merged[col_name] = v
        else:
            for k in self._left_columns:
                col_name = f"{self.left_prefix}{k}" if self.left_prefix else k
                merged[col_name] = None

        if right_row is not None:
            for k, v in right_row.items():
                if self.right_prefix:
                    col_name = f"{self.right_prefix}{k}"
                elif k in merged:
                    col_name = f"right_{k}"
                else:
                    col_name = k
                merged[col_name] = v
        else:
            for k in self._right_columns:
                if self.right_prefix:
                    col_name = f"{self.right_prefix}{k}"
                elif k in merged:
                    col_name = f"right_{k}"
                else:
                    col_name = k
                merged[col_name] = None

        return merged

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        # Phase 2: PROBE PHASE
        while len(self._output_buffer) < self.batch_size and not self._left_eof:
            batch = self.left.next()
            if batch is None:
                self._left_eof = True
                break
            if len(batch) == 0:
                continue

            if not self._left_columns and batch.columns:
                self._left_columns = list(batch.columns)

            # Probe batch against hash table
            for left_row in batch.to_rows():
                key = tuple(left_row.get(k) for k in self.left_on)
                has_null = any(v is None for v in key)

                if has_null:
                    matches: List[Tuple[int, Dict[str, Any]]] = []
                else:
                    matches = self._hash_table.get(key, [])

                if self.join_type == "INNER":
                    for right_idx, right_row in matches:
                        self._output_buffer.append(self._merge_row(left_row, right_row))

                elif self.join_type == "LEFT":
                    if matches:
                        for right_idx, right_row in matches:
                            self._output_buffer.append(self._merge_row(left_row, right_row))
                    else:
                        self._output_buffer.append(self._merge_row(left_row, None))

                elif self.join_type == "SEMI":
                    if matches:
                        self._output_buffer.append(dict(left_row))

                elif self.join_type == "ANTI":
                    if not matches:
                        self._output_buffer.append(dict(left_row))

                elif self.join_type == "RIGHT":
                    if matches:
                        for right_idx, right_row in matches:
                            self._matched_right_indices.add(right_idx)
                            self._output_buffer.append(self._merge_row(left_row, right_row))

                elif self.join_type == "FULL":
                    if matches:
                        for right_idx, right_row in matches:
                            self._matched_right_indices.add(right_idx)
                            self._output_buffer.append(self._merge_row(left_row, right_row))
                    else:
                        self._output_buffer.append(self._merge_row(left_row, None))

        # Check if left stream ended and we need to emit unmatched right rows (RIGHT / FULL outer join)
        if self._left_eof and self.join_type in ("RIGHT", "FULL") and not self._unmatched_right_emitted:
            for idx, right_row in enumerate(self._right_rows):
                if idx not in self._matched_right_indices:
                    self._output_buffer.append(self._merge_row(None, right_row))
            self._unmatched_right_emitted = True

        if self._output_buffer:
            chunk = self._output_buffer[: self.batch_size]
            self._output_buffer = self._output_buffer[self.batch_size :]
            return VectorBatch.from_rows(chunk)

        return None

    def close(self) -> None:
        super().close()
        self.left.close()
        self.right.close()
        self._hash_table.clear()
        self._right_rows.clear()
        self._matched_right_indices.clear()
        self._output_buffer.clear()


# ============================================================================
# Aggregate Functions and Accumulators
# ============================================================================


class Accumulator(ABC):
    """Abstract accumulator maintaining intermediate state for an aggregation."""

    @abstractmethod
    def update(self, value: Any) -> None:
        """Update accumulator state with a value from a row."""

    @abstractmethod
    def finalize(self) -> Any:
        """Return the finalized aggregate value."""


class CountAccumulator(Accumulator):
    """Accumulator for COUNT aggregate."""

    def __init__(self, count_nulls: bool = False, distinct: bool = False) -> None:
        self.count_nulls: bool = count_nulls
        self.distinct: bool = distinct
        self._count: int = 0
        self._seen: Set[Any] = set()

    def update(self, value: Any) -> None:
        if value is None and not self.count_nulls:
            return
        if self.distinct:
            if value not in self._seen:
                self._seen.add(value)
                self._count += 1
        else:
            self._count += 1

    def finalize(self) -> int:
        return self._count


class SumAccumulator(Accumulator):
    """Accumulator for SUM aggregate."""

    def __init__(self, distinct: bool = False) -> None:
        self.distinct: bool = distinct
        self._sum: Optional[Union[int, float]] = None
        self._seen: Set[Any] = set()

    def update(self, value: Any) -> None:
        if value is None:
            return
        if self.distinct:
            if value in self._seen:
                return
            self._seen.add(value)
        if self._sum is None:
            self._sum = value
        else:
            self._sum += value

    def finalize(self) -> Optional[Union[int, float]]:
        return self._sum


class AvgAccumulator(Accumulator):
    """Accumulator for AVG aggregate."""

    def __init__(self, distinct: bool = False) -> None:
        self.distinct: bool = distinct
        self._sum: float = 0.0
        self._count: int = 0
        self._seen: Set[Any] = set()

    def update(self, value: Any) -> None:
        if value is None:
            return
        if self.distinct:
            if value in self._seen:
                return
            self._seen.add(value)
        self._sum += float(value)
        self._count += 1

    def finalize(self) -> Optional[float]:
        if self._count == 0:
            return None
        return self._sum / self._count


class MinAccumulator(Accumulator):
    """Accumulator for MIN aggregate."""

    def __init__(self) -> None:
        self._min: Any = None

    def update(self, value: Any) -> None:
        if value is None:
            return
        if self._min is None or value < self._min:
            self._min = value

    def finalize(self) -> Any:
        return self._min


class MaxAccumulator(Accumulator):
    """Accumulator for MAX aggregate."""

    def __init__(self) -> None:
        self._max: Any = None

    def update(self, value: Any) -> None:
        if value is None:
            return
        if self._max is None or value > self._max:
            self._max = value

    def finalize(self) -> Any:
        return self._max


class StringAggAccumulator(Accumulator):
    """Accumulator for string concatenation aggregate."""

    def __init__(self, delimiter: str = ",", distinct: bool = False) -> None:
        self.delimiter: str = delimiter
        self.distinct: bool = distinct
        self._parts: List[str] = []
        self._seen: Set[str] = set()

    def update(self, value: Any) -> None:
        if value is None:
            return
        s = str(value)
        if self.distinct:
            if s in self._seen:
                return
            self._seen.add(s)
        self._parts.append(s)

    def finalize(self) -> Optional[str]:
        if not self._parts:
            return None
        return self.delimiter.join(self._parts)


class AggregateFunction(ABC):
    """Base class for aggregation function specifications."""

    def __init__(
        self,
        col_name: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> None:
        self.col_name: Optional[str] = col_name
        self.alias: str = alias if alias is not None else self._default_alias()

    @abstractmethod
    def _default_alias(self) -> str:
        """Default column alias if not explicitly provided."""

    @abstractmethod
    def create_accumulator(self) -> Accumulator:
        """Instantiate a fresh accumulator for a group."""


class Count(AggregateFunction):
    """COUNT aggregate function.

    - Count() or Count("*"): counts all rows in group (including NULLs).
    - Count("col"): counts non-NULL values of column.
    - Count("col", distinct=True): counts distinct non-NULL values.
    """

    def __init__(
        self,
        col_name: Optional[str] = None,
        alias: Optional[str] = None,
        distinct: bool = False,
    ) -> None:
        self.distinct: bool = distinct
        self.count_nulls: bool = col_name is None or col_name == "*"
        super().__init__(col_name=col_name, alias=alias)

    def _default_alias(self) -> str:
        name = self.col_name if self.col_name and self.col_name != "*" else "all"
        dist = "distinct_" if self.distinct else ""
        return f"count_{dist}{name}"

    def create_accumulator(self) -> Accumulator:
        return CountAccumulator(count_nulls=self.count_nulls, distinct=self.distinct)


class Sum(AggregateFunction):
    """SUM aggregate function."""

    def __init__(
        self,
        col_name: str,
        alias: Optional[str] = None,
        distinct: bool = False,
    ) -> None:
        self.distinct: bool = distinct
        super().__init__(col_name=col_name, alias=alias)

    def _default_alias(self) -> str:
        dist = "distinct_" if self.distinct else ""
        return f"sum_{dist}{self.col_name}"

    def create_accumulator(self) -> Accumulator:
        return SumAccumulator(distinct=self.distinct)


class Avg(AggregateFunction):
    """AVG aggregate function."""

    def __init__(
        self,
        col_name: str,
        alias: Optional[str] = None,
        distinct: bool = False,
    ) -> None:
        self.distinct: bool = distinct
        super().__init__(col_name=col_name, alias=alias)

    def _default_alias(self) -> str:
        dist = "distinct_" if self.distinct else ""
        return f"avg_{dist}{self.col_name}"

    def create_accumulator(self) -> Accumulator:
        return AvgAccumulator(distinct=self.distinct)


class Min(AggregateFunction):
    """MIN aggregate function."""

    def __init__(self, col_name: str, alias: Optional[str] = None) -> None:
        super().__init__(col_name=col_name, alias=alias)

    def _default_alias(self) -> str:
        return f"min_{self.col_name}"

    def create_accumulator(self) -> Accumulator:
        return MinAccumulator()


class Max(AggregateFunction):
    """MAX aggregate function."""

    def __init__(self, col_name: str, alias: Optional[str] = None) -> None:
        super().__init__(col_name=col_name, alias=alias)

    def _default_alias(self) -> str:
        return f"max_{self.col_name}"

    def create_accumulator(self) -> Accumulator:
        return MaxAccumulator()


class StringAgg(AggregateFunction):
    """STRING_AGG aggregate function."""

    def __init__(
        self,
        col_name: str,
        delimiter: str = ",",
        alias: Optional[str] = None,
        distinct: bool = False,
    ) -> None:
        self.delimiter: str = delimiter
        self.distinct: bool = distinct
        super().__init__(col_name=col_name, alias=alias)

    def _default_alias(self) -> str:
        return f"string_agg_{self.col_name}"

    def create_accumulator(self) -> Accumulator:
        return StringAggAccumulator(delimiter=self.delimiter, distinct=self.distinct)


# ============================================================================
# Aggregate Operator (with GROUP BY support)
# ============================================================================


class Aggregate(ExecutionOperator):
    """Aggregate execution operator supporting GROUP BY and scalar aggregates.

    Pipeline Breaker:
    - In open(): Consumes all input batches from child.
    - Groups rows into hash buckets by group_by columns.
    - Computes aggregate accumulators per group.
    - If group_by is empty (scalar aggregate) and input is empty:
      emits 1 row with default aggregate values (COUNT=0, SUM/AVG/MIN/MAX=None).
    - If group_by is specified and input is empty:
      emits 0 rows.
    - In next(): Returns computed results in VectorBatches of batch_size.
    """

    def __init__(
        self,
        child: ExecutionOperator,
        group_by: Optional[Union[str, Sequence[str]]] = None,
        aggregates: Optional[
            Union[Sequence[AggregateFunction], Dict[str, AggregateFunction]]
        ] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.child: ExecutionOperator = child
        if group_by is None:
            self.group_by: List[str] = []
        elif isinstance(group_by, str):
            self.group_by = [group_by]
        else:
            self.group_by = list(group_by)

        self.batch_size: int = batch_size

        # Normalize aggregates list
        self.aggregates: List[AggregateFunction] = []
        if aggregates is None:
            self.aggregates = [Count(alias="count")]
        elif isinstance(aggregates, dict):
            for alias, agg_fn in aggregates.items():
                agg_fn.alias = alias
                self.aggregates.append(agg_fn)
        else:
            self.aggregates = list(aggregates)

        self._result_rows: List[Dict[str, Any]] = []
        self._cursor: int = 0

    def open(self) -> None:
        super().open()
        self.child.open()
        self._result_rows.clear()
        self._cursor = 0

        # Mapping: group_key -> list of Accumulators corresponding to self.aggregates
        groups: Dict[Tuple[Any, ...], List[Accumulator]] = {}
        rows_seen = 0

        while True:
            batch = self.child.next()
            if batch is None:
                break
            if len(batch) == 0:
                continue

            rows_seen += len(batch)
            rows = batch.to_rows()

            for row in rows:
                if self.group_by:
                    group_key = tuple(row.get(col) for col in self.group_by)
                else:
                    group_key = ()

                if group_key not in groups:
                    groups[group_key] = [agg.create_accumulator() for agg in self.aggregates]

                accs = groups[group_key]
                for agg, acc in zip(self.aggregates, accs):
                    if agg.col_name is None or agg.col_name == "*":
                        val = tuple(row.items()) if getattr(agg, "distinct", False) else 1
                    else:
                        val = row.get(agg.col_name)
                    acc.update(val)

        # Handle empty input case
        if rows_seen == 0 and not self.group_by:
            # SQL scalar aggregate on empty relation emits 1 row with empty accumulators
            empty_accs = [agg.create_accumulator() for agg in self.aggregates]
            groups[()] = empty_accs

        # Materialize result rows
        for group_key, accs in groups.items():
            result_row: Dict[str, Any] = {}
            for col_name, val in zip(self.group_by, group_key):
                result_row[col_name] = val
            for agg, acc in zip(self.aggregates, accs):
                result_row[agg.alias] = acc.finalize()
            self._result_rows.append(result_row)

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        if self._cursor >= len(self._result_rows):
            return None

        chunk = self._result_rows[self._cursor : self._cursor + self.batch_size]
        self._cursor += len(chunk)
        return VectorBatch.from_rows(chunk)

    def close(self) -> None:
        super().close()
        self.child.close()
        self._result_rows.clear()
        self._cursor = 0


# ============================================================================
# Supplementary Operators (Project, Limit, OrderBy)
# ============================================================================


class Project(ExecutionOperator):
    """Project operator evaluating expressions or column selections on VectorBatches."""

    def __init__(
        self,
        child: ExecutionOperator,
        projections: Union[
            Sequence[str],
            Dict[str, Union[str, Expression, Callable[[VectorBatch], Sequence[Any]]]],
        ],
    ) -> None:
        super().__init__()
        self.child: ExecutionOperator = child

        # Map: output_col_name -> Expression / callable
        self._expr_map: Dict[str, Union[Expression, Callable[[VectorBatch], Sequence[Any]]]] = {}
        if isinstance(projections, (list, tuple)):
            for col_name in projections:
                self._expr_map[col_name] = Col(col_name)
        elif isinstance(projections, dict):
            for out_name, expr in projections.items():
                if isinstance(expr, str):
                    self._expr_map[out_name] = Col(expr)
                elif isinstance(expr, Expression) or callable(expr):
                    self._expr_map[out_name] = expr
                else:
                    self._expr_map[out_name] = Literal(expr)
        else:
            raise TypeError(f"Invalid projections type: {type(projections).__name__}")

    def open(self) -> None:
        super().open()
        self.child.open()

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        batch = self.child.next()
        if batch is None:
            return None
        if len(batch) == 0:
            return VectorBatch(data={k: [] for k in self._expr_map}, num_rows=0)

        out_data: Dict[str, List[Any]] = {}
        for out_name, expr in self._expr_map.items():
            if isinstance(expr, Expression):
                out_data[out_name] = expr.evaluate(batch)
            elif callable(expr):
                out_data[out_name] = list(expr(batch))

        return VectorBatch(data=out_data, num_rows=batch.num_rows)

    def close(self) -> None:
        super().close()
        self.child.close()

    @property
    def columns(self) -> Optional[List[str]]:
        return list(self._expr_map.keys())


class Limit(ExecutionOperator):
    """Limit operator restricting the maximum number of tuples produced."""

    def __init__(
        self,
        child: ExecutionOperator,
        limit: int,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.child: ExecutionOperator = child
        self.limit: int = max(0, limit)
        self.offset: int = max(0, offset)
        self._skipped: int = 0
        self._produced: int = 0

    def open(self) -> None:
        super().open()
        self.child.open()
        self._skipped = 0
        self._produced = 0

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        if self._produced >= self.limit:
            return None

        while True:
            batch = self.child.next()
            if batch is None:
                return None
            if len(batch) == 0:
                continue

            batch_len = len(batch)

            # Skip offset rows
            if self._skipped < self.offset:
                to_skip = min(self.offset - self._skipped, batch_len)
                self._skipped += to_skip
                if to_skip == batch_len:
                    continue
                batch = batch.slice(to_skip)
                batch_len = len(batch)

            # Take up to remaining limit
            remaining = self.limit - self._produced
            if batch_len > remaining:
                batch = batch.slice(0, remaining)

            self._produced += len(batch)
            return batch

    def close(self) -> None:
        super().close()
        self.child.close()


class OrderBy(ExecutionOperator):
    """OrderBy operator sorting tuples across batches.

    Pipeline Breaker:
    - Consumes all batches into memory.
    - Sorts all rows by sort keys.
    - Emits sorted VectorBatches of batch_size.
    """

    def __init__(
        self,
        child: ExecutionOperator,
        by: Union[str, Sequence[str], Sequence[Tuple[str, bool]]],
        ascending: bool = True,
        nulls_last: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.child: ExecutionOperator = child
        self.nulls_last: bool = nulls_last
        self.batch_size: int = batch_size

        # Normalize sort keys: list of (column_name, ascending_bool)
        self.sort_keys: List[Tuple[str, bool]] = []
        if isinstance(by, str):
            self.sort_keys = [(by, ascending)]
        elif isinstance(by, (list, tuple)):
            for item in by:
                if isinstance(item, str):
                    self.sort_keys.append((item, ascending))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    self.sort_keys.append((item[0], bool(item[1])))
                else:
                    raise ValueError(f"Invalid sort key item: {item}")
        else:
            raise TypeError(f"Invalid sort key specification: {type(by).__name__}")

        self._sorted_rows: List[Dict[str, Any]] = []
        self._cursor: int = 0

    def open(self) -> None:
        super().open()
        self.child.open()
        self._sorted_rows.clear()
        self._cursor = 0

        # Collect all rows
        rows: List[Dict[str, Any]] = []
        while True:
            batch = self.child.next()
            if batch is None:
                break
            rows.extend(batch.to_rows())

        # Multi-key comparator with NULLS LAST support
        def compare_rows(r1: Dict[str, Any], r2: Dict[str, Any]) -> int:
            for col_name, asc in self.sort_keys:
                v1 = r1.get(col_name)
                v2 = r2.get(col_name)
                if v1 is None and v2 is None:
                    continue
                if v1 is None:
                    return 1 if self.nulls_last else -1
                if v2 is None:
                    return -1 if self.nulls_last else 1
                if v1 != v2:
                    try:
                        if v1 < v2:
                            return -1 if asc else 1
                        else:
                            return 1 if asc else -1
                    except TypeError:
                        s1, s2 = str(v1), str(v2)
                        if s1 < s2:
                            return -1 if asc else 1
                        else:
                            return 1 if asc else -1
            return 0

        rows.sort(key=cmp_to_key(compare_rows))
        self._sorted_rows = rows

    def next(self) -> Optional[VectorBatch]:
        if not self._is_open:
            self.open()

        if self._cursor >= len(self._sorted_rows):
            return None

        chunk = self._sorted_rows[self._cursor : self._cursor + self.batch_size]
        self._cursor += len(chunk)
        return VectorBatch.from_rows(chunk)

    def close(self) -> None:
        super().close()
        self.child.close()
        self._sorted_rows.clear()
        self._cursor = 0
