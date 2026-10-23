"""Rule-Based Query Optimizer (RBO) for Vectorized SQL Engine.

Transforms LogicalPlan trees produced by the SQL parser by applying
algebraic query rewrite optimizations:
1. Predicate Pushdown: Pushes selection predicates down towards source scans
   and before joins to minimize intermediate tuple cardinality.
2. Index Selection: Replaces table scans and filter predicates with fast
   LogicalIndexScan (backed by B+Tree indexes) when available.
3. Projection Pruning: Eliminates unused column reads across the query tree,
   restricting leaf scans to only the columns required downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.query.parser import (
    ASTNode,
    BinaryOp,
    BinaryOpType,
    ColumnRef,
    Expression,
    FunctionCall,
    JoinType,
    Literal,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalPlan,
    LogicalProjection,
    LogicalScan,
    LogicalSort,
    SelectItem,
    Star,
    UnaryOp,
)


# ============================================================================
# Exceptions
# ============================================================================


class OptimizerError(Exception):
    """Base exception for all query optimizer errors."""


class RuleApplicationError(OptimizerError):
    """Raised when an optimization rule fails during transformation."""


# ============================================================================
# Logical Plan Node: LogicalIndexScan
# ============================================================================


@dataclass
class LogicalIndexScan(LogicalPlan):
    """Logical index scan node reading records via a B+Tree index.

    Replaces a LogicalFilter over a LogicalScan when a compatible index
    exists on the filtered column.
    """

    table_name: str
    index_column: str
    operator: str = "="
    value: Any = None
    condition: Optional[Expression] = None
    alias: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    index_name: Optional[str] = None
    index: Optional[Any] = None
    start_key: Any = None
    end_key: Any = None
    inclusive_start: bool = True
    inclusive_end: bool = True

    @property
    def table(self) -> str:
        """Alias for table_name."""
        return self.table_name

    @property
    def column_name(self) -> str:
        """Alias for index_column."""
        return self.index_column

    @property
    def key_column(self) -> str:
        """Alias for index_column."""
        return self.index_column

    @property
    def key(self) -> Any:
        """Alias for value."""
        return self.value

    @property
    def keys(self) -> Any:
        """Alias for value."""
        return self.value

    @property
    def values(self) -> Any:
        """Alias for value."""
        return self.value

    @property
    def op(self) -> str:
        """Alias for operator."""
        return self.operator

    @property
    def predicate(self) -> Optional[Expression]:
        """Alias for condition."""
        return self.condition

    @property
    def children(self) -> list[LogicalPlan]:
        """Leaf nodes have no child plans."""
        return []

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        alias_str = f" AS {self.alias}" if self.alias else ""
        idx_str = f" USING INDEX {self.index_name}" if self.index_name else ""
        cols_str = f" (columns: {', '.join(self.columns)})" if self.columns else ""
        cond_str = f" ON {self.index_column} {self.operator} {self.value}"
        return f"{prefix}LogicalIndexScan: {self.table_name}{alias_str}{idx_str}{cond_str}{cols_str}"

    def __str__(self) -> str:
        alias_str = f" AS {self.alias}" if self.alias else ""
        return f"LogicalIndexScan({self.table_name}{alias_str}, {self.index_column} {self.operator} {self.value})"


# Aliases for LogicalIndexScan
IndexScan = LogicalIndexScan
IndexScanNode = LogicalIndexScan


# ============================================================================
# Catalog & Index Metadata
# ============================================================================


@dataclass
class IndexInfo:
    """Metadata describing a secondary or primary B+Tree index."""

    table_name: str
    column_name: str
    index_name: str
    index_instance: Optional[Any] = None
    is_unique: bool = False


class Catalog:
    """Database catalog storing table schemas and registered B+Tree indexes."""

    def __init__(
        self,
        tables: Optional[Dict[str, Sequence[str]]] = None,
        indexes: Optional[Any] = None,
    ) -> None:
        self._tables: Dict[str, List[str]] = {}
        self._indexes: Dict[Tuple[str, str], IndexInfo] = {}

        if tables:
            for t_name, cols in tables.items():
                self.add_table(t_name, cols)

        if indexes:
            self._load_indexes(indexes)

    def _load_indexes(self, indexes: Any) -> None:
        """Load index definitions from diverse dictionary/sequence structures."""
        if isinstance(indexes, Catalog):
            for k, info in indexes._indexes.items():
                self._indexes[k] = info
            return

        if isinstance(indexes, dict):
            for key, val in indexes.items():
                if isinstance(key, tuple) and len(key) == 2:
                    t_name, col_name = key
                    self.add_index(t_name, col_name, index=val)
                elif isinstance(key, str) and "." in key:
                    t_name, col_name = key.split(".", 1)
                    self.add_index(t_name, col_name, index=val)
                elif isinstance(key, str):
                    t_name = key
                    if isinstance(val, dict):
                        for col_name, idx_obj in val.items():
                            self.add_index(t_name, col_name, index=idx_obj)
                    elif isinstance(val, (list, tuple, set)):
                        for col_name in val:
                            self.add_index(t_name, str(col_name))
                    elif isinstance(val, str):
                        self.add_index(t_name, val)
        elif isinstance(indexes, (list, tuple, set)):
            for item in indexes:
                if isinstance(item, tuple) and len(item) == 2:
                    self.add_index(item[0], item[1])
                elif isinstance(item, str) and "." in item:
                    t_name, col_name = item.split(".", 1)
                    self.add_index(t_name, col_name)

    def add_table(self, table_name: str, columns: Sequence[str]) -> None:
        """Register a table and its column schema."""
        self._tables[table_name] = list(columns)

    def register_table(self, table_name: str, columns: Sequence[str]) -> None:
        """Alias for add_table."""
        self.add_table(table_name, columns)

    def has_table(self, table_name: str) -> bool:
        """Check if a table exists in the catalog."""
        return table_name in self._tables

    def get_table_columns(self, table_name: str) -> Optional[List[str]]:
        """Return the list of columns for a table, or None if unknown."""
        return self._tables.get(table_name)

    def add_index(
        self,
        table_name: str,
        column_name: str,
        index: Optional[Any] = None,
        index_name: Optional[str] = None,
        is_unique: bool = False,
    ) -> None:
        """Register an index on a table column."""
        name = index_name or f"idx_{table_name}_{column_name}"
        info = IndexInfo(
            table_name=table_name,
            column_name=column_name,
            index_name=name,
            index_instance=index,
            is_unique=is_unique,
        )
        self._indexes[(table_name, column_name)] = info

    def register_index(
        self,
        table_name: str,
        column_name: str,
        index: Optional[Any] = None,
        index_name: Optional[str] = None,
        is_unique: bool = False,
    ) -> None:
        """Alias for add_index."""
        self.add_index(table_name, column_name, index, index_name, is_unique)

    def has_index(self, table_name: str, column_name: str) -> bool:
        """Check whether an index exists on the given table and column."""
        return (table_name, column_name) in self._indexes

    def get_index_info(self, table_name: str, column_name: str) -> Optional[IndexInfo]:
        """Return IndexInfo metadata for a table column, or None."""
        return self._indexes.get((table_name, column_name))

    def get_index(self, table_name: str, column_name: str) -> Optional[Any]:
        """Return the B+Tree index instance if present, or None."""
        info = self.get_index_info(table_name, column_name)
        return info.index_instance if info else None

    def get_index_name(self, table_name: str, column_name: str) -> Optional[str]:
        """Return the index name if present, or None."""
        info = self.get_index_info(table_name, column_name)
        return info.index_name if info else None


# ============================================================================
# AST Expression Utilities
# ============================================================================


def split_conjuncts(expr: Expression) -> List[Expression]:
    """Split an expression into a list of AND-connected conjuncts.

    For example: (A AND B) AND C -> [A, B, C].
    """
    if isinstance(expr, BinaryOp):
        op_str = str(expr.op).upper()
        if op_str == "AND" or op_str == BinaryOpType.AND:
            return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def combine_conjuncts(exprs: Sequence[Expression]) -> Optional[Expression]:
    """Combine a sequence of conjuncts using AND into a single expression.

    Returns None if the sequence is empty.
    """
    if not exprs:
        return None
    result = exprs[0]
    for next_expr in exprs[1:]:
        result = BinaryOp(left=result, op=BinaryOpType.AND, right=next_expr)
    return result


def get_referenced_columns(expr: Any) -> Set[Tuple[Optional[str], str]]:
    """Extract all (table_qualifier, column_name) references from an AST expression."""
    cols: Set[Tuple[Optional[str], str]] = set()

    def _traverse(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, ColumnRef):
            cols.add((node.table, node.name))
        elif isinstance(node, SelectItem):
            _traverse(node.expr)
        elif isinstance(node, BinaryOp):
            _traverse(node.left)
            _traverse(node.right)
        elif isinstance(node, UnaryOp):
            _traverse(node.operand)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                _traverse(arg)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                _traverse(item)

    _traverse(expr)
    return cols


def get_plan_tables(plan: LogicalPlan) -> Dict[str, str]:
    """Extract all table names and aliases from a LogicalPlan subtree.

    Returns a mapping: alias_or_name -> canonical_table_name.
    """
    tables: Dict[str, str] = {}

    def _visit(node: LogicalPlan) -> None:
        if isinstance(node, (LogicalScan, LogicalIndexScan)):
            tables[node.table_name] = node.table_name
            if node.alias:
                tables[node.alias] = node.table_name
        for child in node.children:
            _visit(child)

    _visit(plan)
    return tables


def _is_operator_str(op: Any, expected: str) -> bool:
    """Compare an operator value against an expected string representation."""
    if isinstance(op, BinaryOpType):
        return op.value.upper() == expected.upper()
    return str(op).upper() == expected.upper()


# ============================================================================
# Optimizer Rules
# ============================================================================


class OptimizerRule(ABC):
    """Abstract base class for all rule-based algebraic plan transformations."""

    @property
    def name(self) -> str:
        """Name of the optimization rule."""
        return self.__class__.__name__

    @abstractmethod
    def apply(self, plan: LogicalPlan, optimizer: RuleBasedOptimizer) -> LogicalPlan:
        """Apply the optimization rule to the logical plan tree."""


class PredicatePushdownRule(OptimizerRule):
    """Predicate Pushdown Rule.

    Pushes selection filters down through projections, sorts, and joins
    as close as possible to the base relation scans. This reduces intermediate
    row counts early in query evaluation.
    """

    def apply(self, plan: LogicalPlan, optimizer: RuleBasedOptimizer) -> LogicalPlan:
        return self._pushdown(plan, optimizer)

    def _pushdown(
        self,
        node: LogicalPlan,
        optimizer: RuleBasedOptimizer,
    ) -> LogicalPlan:
        if isinstance(node, LogicalFilter):
            # Decompose condition into individual AND-connected conjuncts
            conjuncts = split_conjuncts(node.condition)
            return self._pushdown_filter(node.input, conjuncts, optimizer)

        if isinstance(node, LogicalJoin):
            return self._pushdown_join(node, predicates=[], optimizer=optimizer)

        if isinstance(node, LogicalProjection):
            return LogicalProjection(
                input=self._pushdown(node.input, optimizer),
                expressions=node.expressions,
            )

        if isinstance(node, LogicalSort):
            return LogicalSort(
                input=self._pushdown(node.input, optimizer),
                order_by=node.order_by,
            )

        if isinstance(node, LogicalLimit):
            return LogicalLimit(
                input=self._pushdown(node.input, optimizer),
                limit=node.limit,
            )

        # Scans and leaf nodes remain unchanged
        return node

    def _pushdown_filter(
        self,
        child: LogicalPlan,
        predicates: List[Expression],
        optimizer: RuleBasedOptimizer,
    ) -> LogicalPlan:
        """Push a list of filter predicates down into child."""
        if not predicates:
            return self._pushdown(child, optimizer)

        if isinstance(child, LogicalJoin):
            return self._pushdown_join(child, predicates=predicates, optimizer=optimizer)

        if isinstance(child, LogicalSort):
            # Filter can be pushed through Sort: filter first, then sort!
            new_child = self._pushdown_filter(child.input, predicates, optimizer)
            return LogicalSort(input=new_child, order_by=child.order_by)

        if isinstance(child, LogicalFilter):
            # Combine filters and continue pushdown
            combined = predicates + split_conjuncts(child.condition)
            return self._pushdown_filter(child.input, combined, optimizer)

        if isinstance(child, LogicalProjection):
            # Check if all predicates reference only columns present in projection child
            can_push = True
            for pred in predicates:
                refs = get_referenced_columns(pred)
                # If references an alias created by projection, cannot push naively
                for tbl, col in refs:
                    for item in child.expressions:
                        if item.alias and item.alias == col:
                            can_push = False
                            break
                    if not can_push:
                        break
                if not can_push:
                    break

            if can_push:
                new_input = self._pushdown_filter(child.input, predicates, optimizer)
                return LogicalProjection(input=new_input, expressions=child.expressions)
            else:
                # Keep filter above projection, but optimize under projection
                comb = combine_conjuncts(predicates)
                new_input = self._pushdown(child, optimizer)
                return LogicalFilter(input=new_input, condition=comb) if comb else new_input

        # Child is LogicalScan or LogicalIndexScan or other leaf:
        comb = combine_conjuncts(predicates)
        if comb is None:
            return child
        return LogicalFilter(input=child, condition=comb)

    def _pushdown_join(
        self,
        join: LogicalJoin,
        predicates: List[Expression],
        optimizer: RuleBasedOptimizer,
    ) -> LogicalPlan:
        """Push WHERE predicates and ON condition conjuncts down into join children."""
        left_tables = get_plan_tables(join.left)
        right_tables = get_plan_tables(join.right)

        join_type_str = (
            join.join_type.value
            if isinstance(join.join_type, JoinType)
            else str(join.join_type).upper()
        )

        left_push: List[Expression] = []
        right_push: List[Expression] = []
        remaining_filter: List[Expression] = []
        remaining_on: List[Expression] = []

        # 1. Classify WHERE predicates
        for pred in predicates:
            side = self._classify_predicate(pred, left_tables, right_tables, optimizer)
            if side == "left":
                if join_type_str in ("INNER", "CROSS", "LEFT"):
                    left_push.append(pred)
                else:
                    remaining_filter.append(pred)
            elif side == "right":
                if join_type_str in ("INNER", "CROSS", "RIGHT"):
                    right_push.append(pred)
                else:
                    remaining_filter.append(pred)
            else:
                remaining_filter.append(pred)

        # 2. Classify ON condition conjuncts
        if join.condition is not None:
            on_conjuncts = split_conjuncts(join.condition)
            for c in on_conjuncts:
                side = self._classify_predicate(c, left_tables, right_tables, optimizer)
                if join_type_str in ("INNER", "CROSS"):
                    if side == "left":
                        left_push.append(c)
                    elif side == "right":
                        right_push.append(c)
                    else:
                        remaining_on.append(c)
                elif join_type_str == "LEFT":
                    if side == "right":
                        right_push.append(c)
                    else:
                        remaining_on.append(c)
                elif join_type_str == "RIGHT":
                    if side == "left":
                        left_push.append(c)
                    else:
                        remaining_on.append(c)
                else:
                    remaining_on.append(c)

        new_left = self._pushdown_filter(join.left, left_push, optimizer)
        new_right = self._pushdown_filter(join.right, right_push, optimizer)

        new_on = combine_conjuncts(remaining_on) if remaining_on else None
        new_join = LogicalJoin(
            left=new_left,
            right=new_right,
            join_type=join.join_type,
            condition=new_on,
        )

        if remaining_filter:
            comb = combine_conjuncts(remaining_filter)
            if comb is not None:
                return LogicalFilter(input=new_join, condition=comb)
        return new_join

    def _classify_predicate(
        self,
        pred: Expression,
        left_tables: Dict[str, str],
        right_tables: Dict[str, str],
        optimizer: RuleBasedOptimizer,
    ) -> str:
        """Classify whether a predicate belongs to 'left', 'right', or 'both'/'unknown'."""
        refs = get_referenced_columns(pred)
        if not refs:
            return "left"

        refers_left = False
        refers_right = False

        for table_ref, col_name in refs:
            if table_ref is not None:
                if table_ref in left_tables:
                    refers_left = True
                elif table_ref in right_tables:
                    refers_right = True
                else:
                    # Unknown table qualifier
                    return "both"
            else:
                # Unqualified column name: try resolving via catalog
                left_match = False
                right_match = False

                for t in left_tables.values():
                    cols = optimizer.get_table_columns(t)
                    if cols and col_name in cols:
                        left_match = True
                        break

                for t in right_tables.values():
                    cols = optimizer.get_table_columns(t)
                    if cols and col_name in cols:
                        right_match = True
                        break

                if left_match and not right_match:
                    refers_left = True
                elif right_match and not left_match:
                    refers_right = True
                else:
                    # If single table on one side and column is unique, or ambiguous
                    if len(left_tables) == 1 and len(right_tables) == 1:
                        # Cannot disambiguate without catalog
                        return "both"
                    return "both"

        if refers_left and not refers_right:
            return "left"
        if refers_right and not refers_left:
            return "right"
        return "both"


class IndexSelectionRule(OptimizerRule):
    """Index Selection Rule.

    Transforms a LogicalFilter operating on a LogicalScan into a LogicalIndexScan
    when a B+Tree index is registered on the filtered column. Supports equality (=),
    range (<, <=, >, >=), two-sided range (BETWEEN / AND), and IN predicates.
    """

    def apply(self, plan: LogicalPlan, optimizer: RuleBasedOptimizer) -> LogicalPlan:
        return self._select_indexes(plan, optimizer)

    def _select_indexes(
        self,
        node: LogicalPlan,
        optimizer: RuleBasedOptimizer,
    ) -> LogicalPlan:
        if isinstance(node, LogicalFilter):
            # Recursively process child first
            optimized_child = self._select_indexes(node.input, optimizer)

            if isinstance(optimized_child, LogicalScan):
                table_name = optimized_child.table_name
                alias = optimized_child.alias

                # Try finding matching index for filter conjuncts
                conjuncts = split_conjuncts(node.condition)
                match_result = self._find_index_match(
                    conjuncts, table_name, alias, optimizer
                )

                if match_result is not None:
                    matched_exprs, idx_scan_node = match_result
                    idx_scan_node.columns = list(optimized_child.columns)
                    idx_scan_node.alias = optimized_child.alias

                    remaining = [c for c in conjuncts if c not in matched_exprs]
                    if remaining:
                        comb = combine_conjuncts(remaining)
                        if comb is not None:
                            return LogicalFilter(input=idx_scan_node, condition=comb)
                    return idx_scan_node

            return LogicalFilter(input=optimized_child, condition=node.condition)

        if isinstance(node, LogicalJoin):
            return LogicalJoin(
                left=self._select_indexes(node.left, optimizer),
                right=self._select_indexes(node.right, optimizer),
                join_type=node.join_type,
                condition=node.condition,
            )

        if isinstance(node, LogicalProjection):
            return LogicalProjection(
                input=self._select_indexes(node.input, optimizer),
                expressions=node.expressions,
            )

        if isinstance(node, LogicalSort):
            return LogicalSort(
                input=self._select_indexes(node.input, optimizer),
                order_by=node.order_by,
            )

        if isinstance(node, LogicalLimit):
            return LogicalLimit(
                input=self._select_indexes(node.input, optimizer),
                limit=node.limit,
            )

        return node

    def _find_index_match(
        self,
        conjuncts: List[Expression],
        table_name: str,
        alias: Optional[str],
        optimizer: RuleBasedOptimizer,
    ) -> Optional[Tuple[List[Expression], LogicalIndexScan]]:
        """Search conjuncts for an indexed column match."""
        valid_tables = {table_name}
        if alias:
            valid_tables.add(alias)

        # 1. First priority: check for two-sided range on the same indexed column (BETWEEN / AND)
        for i, c1 in enumerate(conjuncts):
            for j, c2 in enumerate(conjuncts):
                if i >= j:
                    continue
                range_match = self._match_bounded_range(
                    c1, c2, valid_tables, table_name, optimizer
                )
                if range_match is not None:
                    return [c1, c2], range_match

        # 2. Second priority: check for equality predicates (=)
        for c in conjuncts:
            eq_match = self._match_equality(c, valid_tables, table_name, optimizer)
            if eq_match is not None:
                return [c], eq_match

        # 3. Third priority: check for single-sided range comparisons (<, <=, >, >=)
        for c in conjuncts:
            range_match = self._match_range(c, valid_tables, table_name, optimizer)
            if range_match is not None:
                return [c], range_match

        # 4. Fourth priority: check for IN predicates
        for c in conjuncts:
            in_match = self._match_in(c, valid_tables, table_name, optimizer)
            if in_match is not None:
                return [c], in_match

        return None

    def _col_valid(
        self, col_node: Expression, valid_tables: Set[str]
    ) -> Optional[str]:
        """Check if expression is a ColumnRef matching valid table names/aliases."""
        if isinstance(col_node, ColumnRef):
            if col_node.table is None or col_node.table in valid_tables:
                return col_node.name
        return None

    def _match_equality(
        self,
        expr: Expression,
        valid_tables: Set[str],
        table_name: str,
        optimizer: RuleBasedOptimizer,
    ) -> Optional[LogicalIndexScan]:
        if not isinstance(expr, BinaryOp) or not _is_operator_str(expr.op, "="):
            return None

        col_name = None
        val = None

        col_left = self._col_valid(expr.left, valid_tables)
        col_right = self._col_valid(expr.right, valid_tables)

        if col_left and isinstance(expr.right, Literal):
            col_name = col_left
            val = expr.right.value
        elif col_right and isinstance(expr.left, Literal):
            col_name = col_right
            val = expr.left.value

        if col_name and optimizer.has_index(table_name, col_name):
            return LogicalIndexScan(
                table_name=table_name,
                index_column=col_name,
                operator="=",
                value=val,
                condition=expr,
                index_name=optimizer.get_index_name(table_name, col_name),
                index=optimizer.get_index(table_name, col_name),
                start_key=val,
                end_key=val,
                inclusive_start=True,
                inclusive_end=True,
            )
        return None

    def _match_range(
        self,
        expr: Expression,
        valid_tables: Set[str],
        table_name: str,
        optimizer: RuleBasedOptimizer,
    ) -> Optional[LogicalIndexScan]:
        if not isinstance(expr, BinaryOp):
            return None

        op_str = str(expr.op).upper()
        if op_str not in ("<", "<=", ">", ">="):
            return None

        col_name = None
        val = None
        norm_op = op_str

        col_left = self._col_valid(expr.left, valid_tables)
        col_right = self._col_valid(expr.right, valid_tables)

        if col_left and isinstance(expr.right, Literal):
            col_name = col_left
            val = expr.right.value
            norm_op = op_str
        elif col_right and isinstance(expr.left, Literal):
            col_name = col_right
            val = expr.left.value
            # Invert operator when literal is on left
            invert = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
            norm_op = invert.get(op_str, op_str)

        if col_name and optimizer.has_index(table_name, col_name):
            start_k = val if norm_op in (">", ">=") else None
            end_k = val if norm_op in ("<", "<=") else None
            inc_s = norm_op == ">="
            inc_e = norm_op == "<="

            return LogicalIndexScan(
                table_name=table_name,
                index_column=col_name,
                operator=norm_op,
                value=val,
                condition=expr,
                index_name=optimizer.get_index_name(table_name, col_name),
                index=optimizer.get_index(table_name, col_name),
                start_key=start_k,
                end_key=end_k,
                inclusive_start=inc_s,
                inclusive_end=inc_e,
            )
        return None

    def _match_bounded_range(
        self,
        c1: Expression,
        c2: Expression,
        valid_tables: Set[str],
        table_name: str,
        optimizer: RuleBasedOptimizer,
    ) -> Optional[LogicalIndexScan]:
        """Detect col >= low AND col <= high (or permutations)."""
        scan1 = self._match_range(c1, valid_tables, table_name, optimizer)
        scan2 = self._match_range(c2, valid_tables, table_name, optimizer)

        if scan1 is None or scan2 is None:
            return None
        if scan1.index_column != scan2.index_column:
            return None

        # One must be a lower bound (> or >=) and one must be an upper bound (< or <=)
        lower_scan = scan1 if scan1.operator in (">", ">=") else scan2 if scan2.operator in (">", ">=") else None
        upper_scan = scan1 if scan1.operator in ("<", "<=") else scan2 if scan2.operator in ("<", "<=") else None

        if lower_scan and upper_scan and lower_scan != upper_scan:
            comb = BinaryOp(left=c1, op=BinaryOpType.AND, right=c2)
            col = lower_scan.index_column
            return LogicalIndexScan(
                table_name=table_name,
                index_column=col,
                operator="BETWEEN",
                value=(lower_scan.start_key, upper_scan.end_key),
                condition=comb,
                index_name=optimizer.get_index_name(table_name, col),
                index=optimizer.get_index(table_name, col),
                start_key=lower_scan.start_key,
                end_key=upper_scan.end_key,
                inclusive_start=lower_scan.inclusive_start,
                inclusive_end=upper_scan.inclusive_end,
            )
        return None

    def _match_in(
        self,
        expr: Expression,
        valid_tables: Set[str],
        table_name: str,
        optimizer: RuleBasedOptimizer,
    ) -> Optional[LogicalIndexScan]:
        if isinstance(expr, BinaryOp) and _is_operator_str(expr.op, "IN"):
            col_name = self._col_valid(expr.left, valid_tables)
            if col_name and optimizer.has_index(table_name, col_name):
                # Right side could be Literal (tuple/list) or list of literals
                values: Any = []
                if isinstance(expr.right, Literal) and isinstance(expr.right.value, (list, tuple, set)):
                    values = list(expr.right.value)
                elif isinstance(expr.right, (list, tuple)):
                    values = [item.value if isinstance(item, Literal) else item for item in expr.right]
                return LogicalIndexScan(
                    table_name=table_name,
                    index_column=col_name,
                    operator="IN",
                    value=values,
                    condition=expr,
                    index_name=optimizer.get_index_name(table_name, col_name),
                    index=optimizer.get_index(table_name, col_name),
                )
        return None


class ProjectionPruningRule(OptimizerRule):
    """Projection Pruning Rule (Column Pruning).

    Collects all columns referenced anywhere in the query (projections,
    filters, join conditions, sort criteria) and prunes leaf scans
    (LogicalScan, LogicalIndexScan) so they read only required columns.
    Also simplifies redundant/nested projections.
    """

    def apply(self, plan: LogicalPlan, optimizer: RuleBasedOptimizer) -> LogicalPlan:
        # Pass 1: Collapse redundant nested projections
        simplified = self._simplify_projections(plan)

        # Pass 2: Collect required columns per table
        required_cols, full_tables = self._collect_required_columns(simplified, optimizer)

        # Pass 3: Prune leaf scans and simplify projections
        return self._prune_plan(simplified, required_cols, full_tables, optimizer)

    def _simplify_projections(self, node: LogicalPlan) -> LogicalPlan:
        """Collapse redundant stacked projections, e.g. Projection(Projection(child))."""
        if isinstance(node, LogicalProjection):
            child = self._simplify_projections(node.input)
            if isinstance(child, LogicalProjection):
                return LogicalProjection(
                    input=child.input,
                    expressions=node.expressions,
                )
            return LogicalProjection(
                input=child,
                expressions=node.expressions,
            )
        if isinstance(node, LogicalFilter):
            return LogicalFilter(
                input=self._simplify_projections(node.input),
                condition=node.condition,
            )
        if isinstance(node, LogicalJoin):
            return LogicalJoin(
                left=self._simplify_projections(node.left),
                right=self._simplify_projections(node.right),
                join_type=node.join_type,
                condition=node.condition,
            )
        if isinstance(node, LogicalSort):
            return LogicalSort(
                input=self._simplify_projections(node.input),
                order_by=node.order_by,
            )
        if isinstance(node, LogicalLimit):
            return LogicalLimit(
                input=self._simplify_projections(node.input),
                limit=node.limit,
            )
        return node

    def _collect_required_columns(
        self,
        plan: LogicalPlan,
        optimizer: RuleBasedOptimizer,
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """Collect {table_name: set_of_columns} and set of tables that need all columns (*)."""
        required: Dict[str, Set[str]] = {}
        full_tables: Set[str] = set()

        all_tables_map = get_plan_tables(plan)

        def _record_col(table_ref: Optional[str], col_name: str) -> None:
            if table_ref:
                canonical = all_tables_map.get(table_ref, table_ref)
                required.setdefault(canonical, set()).add(col_name)
                required.setdefault(table_ref, set()).add(col_name)
            else:
                # Unqualified: check catalog
                resolved = False
                for t_name in all_tables_map.values():
                    cols = optimizer.get_table_columns(t_name)
                    if cols and col_name in cols:
                        required.setdefault(t_name, set()).add(col_name)
                        resolved = True
                if not resolved:
                    # Add to all active tables
                    for t_name in all_tables_map.values():
                        required.setdefault(t_name, set()).add(col_name)

        def _visit_node(node: LogicalPlan) -> None:
            if isinstance(node, LogicalProjection):
                for item in node.expressions:
                    if isinstance(item.expr, Star):
                        if item.expr.table:
                            canonical = all_tables_map.get(item.expr.table, item.expr.table)
                            full_tables.add(canonical)
                            full_tables.add(item.expr.table)
                        else:
                            for t in all_tables_map.values():
                                full_tables.add(t)
                    else:
                        for tbl, c in get_referenced_columns(item.expr):
                            _record_col(tbl, c)

            elif isinstance(node, LogicalFilter):
                for tbl, c in get_referenced_columns(node.condition):
                    _record_col(tbl, c)

            elif isinstance(node, LogicalJoin):
                if node.condition:
                    for tbl, c in get_referenced_columns(node.condition):
                        _record_col(tbl, c)

            elif isinstance(node, LogicalSort):
                for order_item in node.order_by:
                    for tbl, c in get_referenced_columns(order_item.expr):
                        _record_col(tbl, c)

            elif isinstance(node, LogicalIndexScan):
                _record_col(node.alias or node.table_name, node.index_column)
                if node.condition:
                    for tbl, c in get_referenced_columns(node.condition):
                        _record_col(tbl, c)

            for child in node.children:
                _visit_node(child)

        _visit_node(plan)
        return required, full_tables

    def _prune_plan(
        self,
        node: LogicalPlan,
        required_cols: Dict[str, Set[str]],
        full_tables: Set[str],
        optimizer: RuleBasedOptimizer,
    ) -> LogicalPlan:
        if isinstance(node, LogicalScan):
            t_name = node.table_name
            alias = node.alias

            is_full = t_name in full_tables or (alias in full_tables if alias else False)

            if is_full:
                catalog_cols = optimizer.get_table_columns(t_name)
                cols = list(catalog_cols) if catalog_cols else list(node.columns)
            else:
                needed = set()
                if t_name in required_cols:
                    needed.update(required_cols[t_name])
                if alias and alias in required_cols:
                    needed.update(required_cols[alias])

                catalog_cols = optimizer.get_table_columns(t_name)
                if catalog_cols:
                    cols = [c for c in catalog_cols if c in needed]
                elif needed:
                    cols = sorted(list(needed))
                else:
                    cols = list(node.columns)

            return LogicalScan(
                table_name=node.table_name,
                alias=node.alias,
                columns=cols,
            )

        if isinstance(node, LogicalIndexScan):
            t_name = node.table_name
            alias = node.alias

            is_full = t_name in full_tables or (alias in full_tables if alias else False)

            if is_full:
                catalog_cols = optimizer.get_table_columns(t_name)
                cols = list(catalog_cols) if catalog_cols else list(node.columns)
            else:
                needed = set()
                if t_name in required_cols:
                    needed.update(required_cols[t_name])
                if alias and alias in required_cols:
                    needed.update(required_cols[alias])

                # The index column must always be retained
                needed.add(node.index_column)

                catalog_cols = optimizer.get_table_columns(t_name)
                if catalog_cols:
                    cols = [c for c in catalog_cols if c in needed]
                elif needed:
                    cols = sorted(list(needed))
                else:
                    cols = list(node.columns)

            return LogicalIndexScan(
                table_name=node.table_name,
                index_column=node.index_column,
                operator=node.operator,
                value=node.value,
                condition=node.condition,
                alias=node.alias,
                columns=cols,
                index_name=node.index_name,
                index=node.index,
                start_key=node.start_key,
                end_key=node.end_key,
                inclusive_start=node.inclusive_start,
                inclusive_end=node.inclusive_end,
            )

        if isinstance(node, LogicalProjection):
            pruned_child = self._prune_plan(node.input, required_cols, full_tables, optimizer)

            # Eliminate nested redundant projections: Projection(Projection(child))
            if isinstance(pruned_child, LogicalProjection):
                return LogicalProjection(
                    input=pruned_child.input,
                    expressions=node.expressions,
                )

            return LogicalProjection(
                input=pruned_child,
                expressions=node.expressions,
            )

        if isinstance(node, LogicalFilter):
            return LogicalFilter(
                input=self._prune_plan(node.input, required_cols, full_tables, optimizer),
                condition=node.condition,
            )

        if isinstance(node, LogicalJoin):
            return LogicalJoin(
                left=self._prune_plan(node.left, required_cols, full_tables, optimizer),
                right=self._prune_plan(node.right, required_cols, full_tables, optimizer),
                join_type=node.join_type,
                condition=node.condition,
            )

        if isinstance(node, LogicalSort):
            return LogicalSort(
                input=self._prune_plan(node.input, required_cols, full_tables, optimizer),
                order_by=node.order_by,
            )

        if isinstance(node, LogicalLimit):
            return LogicalLimit(
                input=self._prune_plan(node.input, required_cols, full_tables, optimizer),
                limit=node.limit,
            )

        return node


# ============================================================================
# Rule-Based Optimizer (RBO) Engine
# ============================================================================


class RuleBasedOptimizer:
    """Rule-Based Optimizer (RBO) for relational LogicalPlan trees.

    Applies algebraic rewrite rules in fixed canonical order:
    1. Predicate Pushdown: push selection conditions close to scans.
    2. Index Selection: convert filtered scans to B+Tree index scans.
    3. Projection Pruning: eliminate unused columns from relation scans.
    """

    def __init__(
        self,
        catalog: Optional[Catalog] = None,
        indexes: Optional[Any] = None,
        rules: Optional[Sequence[OptimizerRule]] = None,
        predicate_pushdown: bool = True,
        index_selection: bool = True,
        projection_pruning: bool = True,
        max_iterations: int = 10,
    ) -> None:
        self.catalog: Catalog = catalog if catalog is not None else Catalog()
        if indexes is not None:
            self.catalog._load_indexes(indexes)

        self.max_iterations: int = max_iterations
        self._predicate_pushdown_rule = PredicatePushdownRule()
        self._index_selection_rule = IndexSelectionRule()
        self._projection_pruning_rule = ProjectionPruningRule()

        if rules is not None:
            self.rules: List[OptimizerRule] = list(rules)
        else:
            self.rules = []
            if predicate_pushdown:
                self.rules.append(self._predicate_pushdown_rule)
            if index_selection:
                self.rules.append(self._index_selection_rule)
            if projection_pruning:
                self.rules.append(self._projection_pruning_rule)

    def register_table(
        self, table_name: str, columns: Sequence[str]
    ) -> RuleBasedOptimizer:
        """Register a table and its columns in the catalog."""
        self.catalog.add_table(table_name, columns)
        return self

    def add_table(
        self, table_name: str, columns: Sequence[str]
    ) -> RuleBasedOptimizer:
        """Alias for register_table."""
        return self.register_table(table_name, columns)

    def register_index(
        self,
        table_name: str,
        column_name: str,
        index: Optional[Any] = None,
        index_name: Optional[str] = None,
        is_unique: bool = False,
    ) -> RuleBasedOptimizer:
        """Register a B+Tree index in the catalog."""
        self.catalog.add_index(table_name, column_name, index, index_name, is_unique)
        return self

    def add_index(
        self,
        table_name: str,
        column_name: str,
        index: Optional[Any] = None,
        index_name: Optional[str] = None,
        is_unique: bool = False,
    ) -> RuleBasedOptimizer:
        """Alias for register_index."""
        return self.register_index(table_name, column_name, index, index_name, is_unique)

    def has_index(self, table_name: str, column_name: str) -> bool:
        """Check if an index is registered on table_name.column_name."""
        return self.catalog.has_index(table_name, column_name)

    def get_index(self, table_name: str, column_name: str) -> Optional[Any]:
        """Retrieve the B+Tree index instance if registered."""
        return self.catalog.get_index(table_name, column_name)

    def get_index_name(self, table_name: str, column_name: str) -> Optional[str]:
        """Retrieve the registered index name."""
        return self.catalog.get_index_name(table_name, column_name)

    def get_table_columns(self, table_name: str) -> Optional[List[str]]:
        """Retrieve table columns from catalog."""
        return self.catalog.get_table_columns(table_name)

    def add_rule(self, rule: OptimizerRule) -> None:
        """Append a custom transformation rule to the pipeline."""
        self.rules.append(rule)

    # ------------------------------------------------------------------------
    # Individual Rule Invocations
    # ------------------------------------------------------------------------

    def pushdown_predicates(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply only the Predicate Pushdown transformation."""
        return self._predicate_pushdown_rule.apply(plan, self)

    def select_indexes(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply only the Index Selection transformation."""
        return self._index_selection_rule.apply(plan, self)

    def prune_projections(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply only the Projection Pruning transformation."""
        return self._projection_pruning_rule.apply(plan, self)

    # ------------------------------------------------------------------------
    # Complete Optimization Pipeline
    # ------------------------------------------------------------------------

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Optimize a LogicalPlan by sequentially applying all active rules."""
        current = plan
        for rule in self.rules:
            current = rule.apply(current, self)
        return current


def optimize(
    plan: LogicalPlan,
    catalog: Optional[Catalog] = None,
    indexes: Optional[Any] = None,
    tables: Optional[Dict[str, Sequence[str]]] = None,
    predicate_pushdown: bool = True,
    index_selection: bool = True,
    projection_pruning: bool = True,
) -> LogicalPlan:
    """Convenience helper to optimize a LogicalPlan with a fresh optimizer instance."""
    cat = catalog or Catalog(tables=tables, indexes=indexes)
    if tables and not catalog:
        for t, c in tables.items():
            cat.add_table(t, c)
    if indexes and not catalog:
        cat._load_indexes(indexes)

    optimizer = RuleBasedOptimizer(
        catalog=cat,
        predicate_pushdown=predicate_pushdown,
        index_selection=index_selection,
        projection_pruning=projection_pruning,
    )
    return optimizer.optimize(plan)


def optimize_plan(
    plan: LogicalPlan,
    catalog: Optional[Catalog] = None,
    indexes: Optional[Any] = None,
    **kwargs: Any,
) -> LogicalPlan:
    """Alias for optimize()."""
    return optimize(plan, catalog=catalog, indexes=indexes, **kwargs)
