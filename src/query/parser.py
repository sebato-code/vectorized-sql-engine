"""SQL Parser and Typed AST for the Vectorized SQL Engine.

Supports parsing SQL statements into typed AST representations:
- CREATE TABLE [IF NOT EXISTS] with typed columns, constraints (PRIMARY KEY, NOT NULL, DEFAULT, UNIQUE)
- INSERT INTO with optional column list and multiple rows of values
- SELECT with projections, column/table aliases, JOIN with ON condition,
  WHERE clause with comparison (=, !=, <>, <, >, <=, >=) and logical operators (AND, OR, NOT),
  ORDER BY, and LIMIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, List, Optional, Sequence, Union

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ParserError(Exception):
    """Base exception for all query parser errors."""


class SQLSyntaxError(ParserError):
    """Raised when SQL syntax is invalid or an unexpected token is encountered."""


class UnexpectedTokenError(SQLSyntaxError):
    """Raised when a specific unexpected token is encountered during parsing."""


# ---------------------------------------------------------------------------
# Token Types & Lexer
# ---------------------------------------------------------------------------

class TokenType(StrEnum):
    # Keywords
    SELECT = "SELECT"
    FROM = "FROM"
    WHERE = "WHERE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    JOIN = "JOIN"
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    OUTER = "OUTER"
    ON = "ON"
    CREATE = "CREATE"
    TABLE = "TABLE"
    IF = "IF"
    EXISTS = "EXISTS"
    INSERT = "INSERT"
    INTO = "INTO"
    VALUES = "VALUES"
    PRIMARY = "PRIMARY"
    KEY = "KEY"
    AS = "AS"
    ORDER = "ORDER"
    BY = "BY"
    ASC = "ASC"
    DESC = "DESC"
    LIMIT = "LIMIT"
    DISTINCT = "DISTINCT"
    NULL = "NULL"
    TRUE = "TRUE"
    FALSE = "FALSE"
    DEFAULT = "DEFAULT"
    UNIQUE = "UNIQUE"
    AUTO_INCREMENT = "AUTO_INCREMENT"
    IS = "IS"
    GROUP = "GROUP"
    HAVING = "HAVING"

    # Literals and Identifiers
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"

    # Operators and Punctuation
    EQUALS = "="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    GREATER_THAN = ">"
    LESS_EQUALS = "<="
    GREATER_EQUALS = ">="
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    COMMA = ","
    DOT = "."
    LPAREN = "("
    RPAREN = ")"
    SEMICOLON = ";"
    EOF = "EOF"


KEYWORDS: dict[str, TokenType] = {
    "SELECT": TokenType.SELECT,
    "FROM": TokenType.FROM,
    "WHERE": TokenType.WHERE,
    "AND": TokenType.AND,
    "OR": TokenType.OR,
    "NOT": TokenType.NOT,
    "JOIN": TokenType.JOIN,
    "INNER": TokenType.INNER,
    "LEFT": TokenType.LEFT,
    "RIGHT": TokenType.RIGHT,
    "FULL": TokenType.FULL,
    "CROSS": TokenType.CROSS,
    "OUTER": TokenType.OUTER,
    "ON": TokenType.ON,
    "CREATE": TokenType.CREATE,
    "TABLE": TokenType.TABLE,
    "IF": TokenType.IF,
    "EXISTS": TokenType.EXISTS,
    "INSERT": TokenType.INSERT,
    "INTO": TokenType.INTO,
    "VALUES": TokenType.VALUES,
    "PRIMARY": TokenType.PRIMARY,
    "KEY": TokenType.KEY,
    "AS": TokenType.AS,
    "ORDER": TokenType.ORDER,
    "BY": TokenType.BY,
    "ASC": TokenType.ASC,
    "DESC": TokenType.DESC,
    "LIMIT": TokenType.LIMIT,
    "DISTINCT": TokenType.DISTINCT,
    "DEFAULT": TokenType.DEFAULT,
    "UNIQUE": TokenType.UNIQUE,
    "AUTO_INCREMENT": TokenType.AUTO_INCREMENT,
    "AUTOINCREMENT": TokenType.AUTO_INCREMENT,
    "IS": TokenType.IS,
    "GROUP": TokenType.GROUP,
    "HAVING": TokenType.HAVING,
}


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

    def __str__(self) -> str:
        return f"Token({self.type.value}, {self.value!r}, line={self.line}, col={self.column})"


class Lexer:
    """SQL Tokenizer supporting keywords, identifiers, strings, numbers, and operators."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.length = len(text)

    def is_at_end(self) -> bool:
        return self.pos >= self.length

    def peek_char(self, offset: int = 0) -> str:
        p = self.pos + offset
        if p >= self.length:
            return ""
        return self.text[p]

    def advance_char(self) -> str:
        if self.is_at_end():
            return ""
        c = self.text[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def skip_whitespace_and_comments(self) -> None:
        while not self.is_at_end():
            c = self.peek_char()
            if c in " \t\r\n":
                self.advance_char()
            elif c == "-" and self.peek_char(1) == "-":
                # Single-line comment
                while not self.is_at_end() and self.peek_char() != "\n":
                    self.advance_char()
            elif c == "/" and self.peek_char(1) == "*":
                # Multi-line comment
                start_line = self.line
                start_col = self.col
                self.advance_char()  # /
                self.advance_char()  # *
                closed = False
                while not self.is_at_end():
                    if self.peek_char() == "*" and self.peek_char(1) == "/":
                        self.advance_char()
                        self.advance_char()
                        closed = True
                        break
                    self.advance_char()
                if not closed:
                    raise SQLSyntaxError(
                        f"Unterminated block comment starting at line {start_line}, col {start_col}"
                    )
            else:
                break

    def scan_string(self) -> Token:
        start_line = self.line
        start_col = self.col
        quote_char = self.advance_char()  # '
        chars: list[str] = []
        while not self.is_at_end():
            c = self.peek_char()
            if c == quote_char:
                if self.peek_char(1) == quote_char:
                    # Escaped quote ''
                    chars.append(quote_char)
                    self.advance_char()
                    self.advance_char()
                else:
                    self.advance_char()
                    return Token(TokenType.STRING, "".join(chars), start_line, start_col)
            elif c == "\\" and self.peek_char(1) == quote_char:
                chars.append(quote_char)
                self.advance_char()
                self.advance_char()
            else:
                chars.append(self.advance_char())
        raise SQLSyntaxError(
            f"Unterminated string literal starting at line {start_line}, col {start_col}"
        )

    def scan_quoted_identifier(self, quote_char: str) -> Token:
        start_line = self.line
        start_col = self.col
        self.advance_char()  # consume ` or "
        chars: list[str] = []
        while not self.is_at_end():
            c = self.peek_char()
            if c == quote_char:
                self.advance_char()
                return Token(TokenType.IDENTIFIER, "".join(chars), start_line, start_col)
            chars.append(self.advance_char())
        raise SQLSyntaxError(
            f"Unterminated identifier starting at line {start_line}, col {start_col}"
        )

    def scan_number(self) -> Token:
        start_line = self.line
        start_col = self.col
        num_str: list[str] = []
        is_float = False

        while not self.is_at_end() and self.peek_char().isdigit():
            num_str.append(self.advance_char())

        if not self.is_at_end() and self.peek_char() == "." and self.peek_char(1).isdigit():
            is_float = True
            num_str.append(self.advance_char())  # consume .
            while not self.is_at_end() and self.peek_char().isdigit():
                num_str.append(self.advance_char())

        if not self.is_at_end() and self.peek_char() in "eE":
            p1 = self.peek_char(1)
            p2 = self.peek_char(2)
            if p1.isdigit() or (p1 in "+-" and p2.isdigit()):
                is_float = True
                num_str.append(self.advance_char())
                if not self.is_at_end() and self.peek_char() in "+-":
                    num_str.append(self.advance_char())
                while not self.is_at_end() and self.peek_char().isdigit():
                    num_str.append(self.advance_char())

        s = "".join(num_str)
        if is_float:
            return Token(TokenType.FLOAT, float(s), start_line, start_col)
        return Token(TokenType.INTEGER, int(s), start_line, start_col)

    def scan_word(self) -> Token:
        start_line = self.line
        start_col = self.col
        chars: list[str] = []
        while not self.is_at_end() and (self.peek_char().isalnum() or self.peek_char() == "_"):
            chars.append(self.advance_char())
        word = "".join(chars)
        upper = word.upper()

        if upper == "TRUE":
            return Token(TokenType.BOOLEAN, True, start_line, start_col)
        if upper == "FALSE":
            return Token(TokenType.BOOLEAN, False, start_line, start_col)
        if upper == "NULL":
            return Token(TokenType.NULL, None, start_line, start_col)
        if upper in KEYWORDS:
            return Token(KEYWORDS[upper], upper, start_line, start_col)
        return Token(TokenType.IDENTIFIER, word, start_line, start_col)

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while not self.is_at_end():
            self.skip_whitespace_and_comments()
            if self.is_at_end():
                break

            c = self.peek_char()
            line = self.line
            col = self.col

            if c == "'":
                tokens.append(self.scan_string())
            elif c in ('"', '`'):
                tokens.append(self.scan_quoted_identifier(c))
            elif c.isdigit():
                tokens.append(self.scan_number())
            elif c == "." and self.peek_char(1).isdigit():
                tokens.append(self.scan_number())
            elif c.isalpha() or c == "_":
                tokens.append(self.scan_word())
            elif c == "!":
                if self.peek_char(1) == "=":
                    self.advance_char()
                    self.advance_char()
                    tokens.append(Token(TokenType.NOT_EQUALS, "!=", line, col))
                else:
                    raise SQLSyntaxError(f"Unexpected character '!' at line {line}, col {col}")
            elif c == "<":
                if self.peek_char(1) == "=":
                    self.advance_char()
                    self.advance_char()
                    tokens.append(Token(TokenType.LESS_EQUALS, "<=", line, col))
                elif self.peek_char(1) == ">":
                    self.advance_char()
                    self.advance_char()
                    tokens.append(Token(TokenType.NOT_EQUALS, "!=", line, col))
                else:
                    self.advance_char()
                    tokens.append(Token(TokenType.LESS_THAN, "<", line, col))
            elif c == ">":
                if self.peek_char(1) == "=":
                    self.advance_char()
                    self.advance_char()
                    tokens.append(Token(TokenType.GREATER_EQUALS, ">=", line, col))
                else:
                    self.advance_char()
                    tokens.append(Token(TokenType.GREATER_THAN, ">", line, col))
            elif c == "=":
                self.advance_char()
                tokens.append(Token(TokenType.EQUALS, "=", line, col))
            elif c == "+":
                self.advance_char()
                tokens.append(Token(TokenType.PLUS, "+", line, col))
            elif c == "-":
                self.advance_char()
                tokens.append(Token(TokenType.MINUS, "-", line, col))
            elif c == "*":
                self.advance_char()
                tokens.append(Token(TokenType.STAR, "*", line, col))
            elif c == "/":
                self.advance_char()
                tokens.append(Token(TokenType.SLASH, "/", line, col))
            elif c == ",":
                self.advance_char()
                tokens.append(Token(TokenType.COMMA, ",", line, col))
            elif c == ".":
                self.advance_char()
                tokens.append(Token(TokenType.DOT, ".", line, col))
            elif c == "(":
                self.advance_char()
                tokens.append(Token(TokenType.LPAREN, "(", line, col))
            elif c == ")":
                self.advance_char()
                tokens.append(Token(TokenType.RPAREN, ")", line, col))
            elif c == ";":
                self.advance_char()
                tokens.append(Token(TokenType.SEMICOLON, ";", line, col))
            else:
                raise SQLSyntaxError(f"Unexpected character '{c}' at line {line}, col {col}")

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens


# ---------------------------------------------------------------------------
# AST Hierarchy
# ---------------------------------------------------------------------------

class ASTNode:
    """Base class for all AST nodes."""


class Expression(ASTNode):
    """Base class for all expression nodes."""


class BinaryOpType(StrEnum):
    EQ = "="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    AND = "AND"
    OR = "OR"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    IS = "IS"
    IS_NOT = "IS NOT"


@dataclass(frozen=True)
class Literal(Expression):
    """Literal constant value (int, float, str, bool, None)."""
    value: Any

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Literal):
            return self.value == other.value and type(self.value) is type(other.value)
        return self.value == other and type(self.value) is type(other)

    def __str__(self) -> str:
        if self.value is None:
            return "NULL"
        if isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        if isinstance(self.value, str):
            escaped = self.value.replace("'", "''")
            return f"'{escaped}'"
        return str(self.value)


@dataclass(frozen=True)
class ColumnRef(Expression):
    """Column reference with optional table qualification, e.g. users.id or id."""
    name: str
    table: Optional[str] = None

    @property
    def column_name(self) -> str:
        return self.name

    @property
    def table_name(self) -> Optional[str]:
        return self.table

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ColumnRef):
            return self.name == other.name and self.table == other.table
        if isinstance(other, str) and self.table is None:
            return self.name == other
        return False

    def __str__(self) -> str:
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name


@dataclass(frozen=True)
class Star(Expression):
    """Wildcard projection `*` or `table.*`."""
    table: Optional[str] = None

    def __str__(self) -> str:
        if self.table:
            return f"{self.table}.*"
        return "*"


@dataclass(frozen=True)
class BinaryOp(Expression):
    """Binary expression: left OP right."""
    left: Expression
    op: Union[BinaryOpType, str]
    right: Expression

    @property
    def operator(self) -> str:
        return str(self.op)

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnaryOp(Expression):
    """Unary expression: OP operand (e.g. NOT cond or -value)."""
    op: str
    operand: Expression

    @property
    def operator(self) -> str:
        return self.op

    def __str__(self) -> str:
        return f"({self.op} {self.operand})"


@dataclass(frozen=True)
class FunctionCall(Expression):
    """Function call expression, e.g. COUNT(*), SUM(val)."""
    name: str
    args: list[Expression] = field(default_factory=list)

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass
class SelectItem(ASTNode):
    """Projection item in SELECT statement: expression with optional alias."""
    expr: Expression
    alias: Optional[str] = None

    @property
    def name(self) -> Optional[str]:
        if self.alias:
            return self.alias
        if isinstance(self.expr, ColumnRef):
            return self.expr.name
        return None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SelectItem):
            return self.expr == other.expr and self.alias == other.alias
        if isinstance(other, Expression) and self.alias is None:
            return self.expr == other
        return False

    def __getattr__(self, name: str) -> Any:
        # Transparently forward attributes from inner expression
        if hasattr(self.expr, name):
            return getattr(self.expr, name)
        raise AttributeError(f"'SelectItem' object has no attribute '{name}'")

    def __str__(self) -> str:
        if self.alias:
            return f"{self.expr} AS {self.alias}"
        return str(self.expr)


@dataclass
class TableRef(ASTNode):
    """Table reference in FROM or JOIN with optional alias."""
    name: str
    alias: Optional[str] = None

    @property
    def table_name(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TableRef):
            return self.name == other.name and self.alias == other.alias
        if isinstance(other, str) and self.alias is None:
            return self.name == other
        return False

    def __str__(self) -> str:
        if self.alias:
            return f"{self.name} AS {self.alias}"
        return self.name


class JoinType(StrEnum):
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"


@dataclass
class JoinClause(ASTNode):
    """JOIN clause with join type, joined table, and ON condition."""
    join_type: Union[JoinType, str]
    table: TableRef
    on: Optional[Expression] = None

    @property
    def table_name(self) -> str:
        return self.table.name

    @property
    def alias(self) -> Optional[str]:
        return self.table.alias

    @property
    def on_condition(self) -> Optional[Expression]:
        return self.on

    @property
    def condition(self) -> Optional[Expression]:
        return self.on

    def __str__(self) -> str:
        s = f"{self.join_type} JOIN {self.table}"
        if self.on is not None:
            s += f" ON {self.on}"
        return s


@dataclass
class OrderByItem(ASTNode):
    """ORDER BY expression and sort direction."""
    expr: Expression
    ascending: bool = True

    def __str__(self) -> str:
        direction = "ASC" if self.ascending else "DESC"
        return f"{self.expr} {direction}"


@dataclass
class ColumnDef(ASTNode):
    """Column definition in CREATE TABLE."""
    name: str
    data_type: str
    primary_key: bool = False
    nullable: bool = True
    default: Optional[Any] = None
    unique: bool = False
    auto_increment: bool = False

    @property
    def is_primary_key(self) -> bool:
        return self.primary_key

    @property
    def is_nullable(self) -> bool:
        return self.nullable

    @property
    def is_unique(self) -> bool:
        return self.unique

    @property
    def column_name(self) -> str:
        return self.name

    @property
    def type_name(self) -> str:
        return self.data_type

    @property
    def default_value(self) -> Optional[Any]:
        return self.default

    def __str__(self) -> str:
        parts = [self.name, self.data_type]
        if self.primary_key:
            parts.append("PRIMARY KEY")
        if not self.nullable and not self.primary_key:
            parts.append("NOT NULL")
        if self.unique:
            parts.append("UNIQUE")
        if self.default is not None:
            parts.append(f"DEFAULT {Literal(self.default)}")
        if self.auto_increment:
            parts.append("AUTO_INCREMENT")
        return " ".join(parts)


class Statement(ASTNode):
    """Base class for all SQL statements."""


@dataclass
class CreateTableStatement(Statement):
    """CREATE TABLE statement."""
    table_name: str
    columns: list[ColumnDef]
    if_not_exists: bool = False

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnDef]:
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None

    def __str__(self) -> str:
        if_clause = "IF NOT EXISTS " if self.if_not_exists else ""
        cols = ", ".join(str(c) for c in self.columns)
        return f"CREATE TABLE {if_clause}{self.table_name} ({cols})"


@dataclass
class InsertStatement(Statement):
    """INSERT INTO statement."""
    table_name: str
    columns: Optional[list[str]] = None
    values: list[list[Expression]] = field(default_factory=list)

    @property
    def raw_values(self) -> list[list[Any]]:
        result: list[list[Any]] = []
        for row in self.values:
            row_vals: list[Any] = []
            for val in row:
                if isinstance(val, Literal):
                    row_vals.append(val.value)
                else:
                    row_vals.append(val)
            result.append(row_vals)
        return result

    @property
    def rows(self) -> list[list[Expression]]:
        return self.values

    @property
    def table(self) -> str:
        return self.table_name

    def __str__(self) -> str:
        cols_part = f" ({', '.join(self.columns)})" if self.columns else ""
        row_strs: list[str] = []
        for r in self.values:
            row_strs.append(f"({', '.join(str(v) for v in r)})")
        return f"INSERT INTO {self.table_name}{cols_part} VALUES {', '.join(row_strs)}"


@dataclass
class SelectStatement(Statement):
    """SELECT statement with projections, table, joins, where, group by, having, order by, limit."""
    projections: list[SelectItem]
    from_table: Optional[TableRef] = None
    joins: list[JoinClause] = field(default_factory=list)
    where: Optional[Expression] = None
    group_by: Optional[list[Expression]] = None
    having: Optional[Expression] = None
    order_by: Optional[list[OrderByItem]] = None
    limit: Optional[int] = None
    distinct: bool = False

    @property
    def table_name(self) -> Optional[str]:
        return self.from_table.name if self.from_table else None

    @property
    def from_clause(self) -> Optional[TableRef]:
        return self.from_table

    @property
    def where_clause(self) -> Optional[Expression]:
        return self.where

    @property
    def join(self) -> Optional[JoinClause]:
        return self.joins[0] if self.joins else None

    @property
    def columns(self) -> list[SelectItem]:
        return self.projections

    def to_logical_plan(self) -> LogicalPlan:
        """Convert this SELECT statement AST into a typed LogicalPlan tree."""
        return build_logical_plan(self)

    @property
    def logical_plan(self) -> LogicalPlan:
        """Relational logical query plan representation of this SELECT statement."""
        return self.to_logical_plan()

    @property
    def plan(self) -> LogicalPlan:
        """Alias for logical_plan."""
        return self.to_logical_plan()

    def __str__(self) -> str:
        distinct_str = "DISTINCT " if self.distinct else ""
        proj_str = ", ".join(str(p) for p in self.projections)
        s = f"SELECT {distinct_str}{proj_str}"
        if self.from_table:
            s += f" FROM {self.from_table}"
        for j in self.joins:
            s += f" {j}"
        if self.where:
            s += f" WHERE {self.where}"
        if self.group_by:
            s += f" GROUP BY {', '.join(str(g) for g in self.group_by)}"
        if self.having:
            s += f" HAVING {self.having}"
        if self.order_by:
            s += f" ORDER BY {', '.join(str(o) for o in self.order_by)}"
        if self.limit is not None:
            s += f" LIMIT {self.limit}"
        return s


# ---------------------------------------------------------------------------
# Logical Query Plan AST Nodes
# ---------------------------------------------------------------------------

class LogicalPlan(ASTNode):
    """Base class for all logical query plan nodes."""

    def __init__(self, root: Optional[LogicalPlan] = None) -> None:
        self._root = root

    @property
    def root(self) -> LogicalPlan:
        return getattr(self, "_root", None) or self

    @property
    def children(self) -> list[LogicalPlan]:
        if hasattr(self, "_root") and self._root is not None:
            return [self._root]
        return []

    def pretty(self, indent: int = 0) -> str:
        """Return a pretty-printed tree representation of the logical plan."""
        prefix = "  " * indent
        lines = [f"{prefix}{self.__class__.__name__}"]
        for child in self.children:
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)


LogicalPlanNode = LogicalPlan


@dataclass
class LogicalScan(LogicalPlan):
    """Logical table scan node representing the scan of a table/relation."""
    table_name: str
    alias: Optional[str] = None
    columns: list[str] = field(default_factory=list)

    @property
    def table(self) -> str:
        return self.table_name

    @property
    def children(self) -> list[LogicalPlan]:
        return []

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        alias_str = f" AS {self.alias}" if self.alias else ""
        cols_str = f" (columns: {', '.join(self.columns)})" if self.columns else ""
        return f"{prefix}LogicalScan: {self.table_name}{alias_str}{cols_str}"

    def __str__(self) -> str:
        alias_str = f" AS {self.alias}" if self.alias else ""
        return f"LogicalScan({self.table_name}{alias_str})"


@dataclass
class LogicalJoin(LogicalPlan):
    """Logical join node combining two input relations."""
    left: LogicalPlan
    right: LogicalPlan
    join_type: JoinType = JoinType.INNER
    condition: Optional[Expression] = None

    @property
    def on(self) -> Optional[Expression]:
        return self.condition

    @property
    def on_condition(self) -> Optional[Expression]:
        return self.condition

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.left, self.right]

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        cond_str = f" ON {self.condition}" if self.condition else ""
        lines = [f"{prefix}LogicalJoin: {self.join_type.value}{cond_str}"]
        lines.append(self.left.pretty(indent + 1))
        lines.append(self.right.pretty(indent + 1))
        return "\n".join(lines)

    def __str__(self) -> str:
        cond_str = f" ON {self.condition}" if self.condition else ""
        return f"LogicalJoin({self.join_type.value}{cond_str})"


@dataclass
class LogicalFilter(LogicalPlan):
    """Logical filter node applying a predicate condition to its child."""
    input: LogicalPlan
    condition: Expression

    @property
    def child(self) -> LogicalPlan:
        return self.input

    @property
    def predicate(self) -> Expression:
        return self.condition

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}LogicalFilter: {self.condition}"]
        lines.append(self.input.pretty(indent + 1))
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"LogicalFilter({self.condition})"


@dataclass
class LogicalProjection(LogicalPlan):
    """Logical projection node computing projected expressions/columns."""
    input: LogicalPlan
    expressions: list[SelectItem]

    @property
    def child(self) -> LogicalPlan:
        return self.input

    @property
    def projections(self) -> list[SelectItem]:
        return self.expressions

    @property
    def columns(self) -> list[SelectItem]:
        return self.expressions

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        expr_strs = ", ".join(str(e) for e in self.expressions)
        lines = [f"{prefix}LogicalProjection: [{expr_strs}]"]
        lines.append(self.input.pretty(indent + 1))
        return "\n".join(lines)

    def __str__(self) -> str:
        expr_strs = ", ".join(str(e) for e in self.expressions)
        return f"LogicalProjection([{expr_strs}])"


@dataclass
class LogicalSort(LogicalPlan):
    """Logical sort / order-by node."""
    input: LogicalPlan
    order_by: list[OrderByItem]

    @property
    def child(self) -> LogicalPlan:
        return self.input

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        order_str = ", ".join(str(o) for o in self.order_by)
        lines = [f"{prefix}LogicalSort: [{order_str}]"]
        lines.append(self.input.pretty(indent + 1))
        return "\n".join(lines)

    def __str__(self) -> str:
        order_str = ", ".join(str(o) for o in self.order_by)
        return f"LogicalSort([{order_str}])"


@dataclass
class LogicalLimit(LogicalPlan):
    """Logical limit node restricting row count."""
    input: LogicalPlan
    limit: int

    @property
    def child(self) -> LogicalPlan:
        return self.input

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def pretty(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}LogicalLimit: {self.limit}"]
        lines.append(self.input.pretty(indent + 1))
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"LogicalLimit({self.limit})"


# Aliases for logical plan nodes
ScanNode = LogicalScan
TableScanNode = LogicalScan
JoinNode = LogicalJoin
FilterNode = LogicalFilter
ProjectionNode = LogicalProjection
SortNode = LogicalSort
LimitNode = LogicalLimit


def build_logical_plan(stmt: Statement) -> LogicalPlan:
    """Translate a parsed SELECT statement AST into a typed LogicalPlan tree.

    The resulting plan tree follows canonical relational algebra ordering:
    - LogicalLimit (if LIMIT present)
      - LogicalProjection (SELECT projections)
        - LogicalSort (if ORDER BY present)
          - LogicalFilter (if WHERE present)
            - LogicalJoin (if JOIN clauses present)
              - LogicalScan (FROM table and JOIN tables)
    """
    if not isinstance(stmt, SelectStatement):
        raise TypeError(
            f"Logical plan generation only supported for SelectStatement, got {type(stmt).__name__}"
        )

    # 1. Base Scan from FROM table
    if stmt.from_table is not None:
        current: LogicalPlan = LogicalScan(
            table_name=stmt.from_table.name,
            alias=stmt.from_table.alias
        )
    else:
        # SELECT without FROM, e.g. SELECT 1 + 1
        current = LogicalScan(table_name="")

    # 2. Sequential / chained joins
    for join_clause in stmt.joins:
        right_scan = LogicalScan(
            table_name=join_clause.table.name,
            alias=join_clause.table.alias
        )
        current = LogicalJoin(
            left=current,
            right=right_scan,
            join_type=join_clause.join_type,
            condition=join_clause.condition
        )

    # 3. Filter from WHERE condition
    if stmt.where is not None:
        current = LogicalFilter(
            input=current,
            condition=stmt.where
        )

    # 4. Sort from ORDER BY
    if stmt.order_by:
        current = LogicalSort(
            input=current,
            order_by=stmt.order_by
        )

    # 5. Projection from SELECT expressions
    current = LogicalProjection(
        input=current,
        expressions=stmt.projections
    )

    # 6. Limit from LIMIT
    if stmt.limit is not None:
        current = LogicalLimit(
            input=current,
            limit=stmt.limit
        )

    return current


# ---------------------------------------------------------------------------
# Parser Implementation
# ---------------------------------------------------------------------------

CLAUSE_START_KEYWORDS = {
    TokenType.FROM,
    TokenType.WHERE,
    TokenType.JOIN,
    TokenType.INNER,
    TokenType.LEFT,
    TokenType.RIGHT,
    TokenType.FULL,
    TokenType.CROSS,
    TokenType.ORDER,
    TokenType.LIMIT,
    TokenType.GROUP,
    TokenType.HAVING,
    TokenType.SEMICOLON,
    TokenType.EOF,
    TokenType.COMMA,
    TokenType.ON,
}


class Parser:
    """Recursive descent SQL Parser for Vectorized Engine statements."""

    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.lexer = Lexer(sql)
        self.tokens: list[Token] = self.lexer.tokenize()
        self.pos = 0

    @classmethod
    def parse_sql(cls, sql: str) -> Statement:
        return cls(sql).parse()

    @classmethod
    def parse(cls, sql: str) -> Statement:
        return cls(sql).parse()

    @classmethod
    def parse_logical_plan(cls, sql: str) -> LogicalPlan:
        return build_logical_plan(cls(sql).parse())

    def to_logical_plan(self) -> LogicalPlan:
        return build_logical_plan(self.parse())

    # -- Token navigation helpers --

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def previous(self) -> Token:
        return self.tokens[self.pos - 1]

    def advance(self) -> Token:
        if not self.is_at_end():
            self.pos += 1
        return self.previous()

    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return token_type == TokenType.EOF
        return self.peek().type == token_type

    def match(self, *token_types: TokenType) -> bool:
        for t in token_types:
            if self.check(t):
                self.advance()
                return True
        return False

    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.check(token_type):
            return self.advance()
        current = self.peek()
        raise UnexpectedTokenError(
            f"{message}. Found '{current.value}' (type={current.type.value}) at line {current.line}, col {current.column}"
        )

    # -- Top-level Parsing --

    def parse_statement(self) -> Statement:
        while self.match(TokenType.SEMICOLON):
            pass

        if self.is_at_end():
            raise SQLSyntaxError("Empty SQL query or missing statement")

        if self.check(TokenType.CREATE):
            return self._parse_create_table()
        elif self.check(TokenType.INSERT):
            return self._parse_insert_into()
        elif self.check(TokenType.SELECT):
            return self._parse_select()
        else:
            tok = self.peek()
            raise SQLSyntaxError(
                f"Unsupported or invalid statement starting with '{tok.value}' at line {tok.line}, col {tok.column}"
            )

    def parse(self_or_sql: Union[Parser, str]) -> Statement:
        if isinstance(self_or_sql, str):
            return Parser(self_or_sql).parse()
        self = self_or_sql
        stmt = self.parse_statement()
        self.match(TokenType.SEMICOLON)
        while self.match(TokenType.SEMICOLON):
            pass
        if not self.is_at_end():
            tok = self.peek()
            raise SQLSyntaxError(
                f"Unexpected token '{tok.value}' after statement at line {tok.line}, col {tok.column}"
            )
        return stmt

    def parse_all(self) -> list[Statement]:
        statements: list[Statement] = []
        while not self.is_at_end():
            while self.match(TokenType.SEMICOLON):
                pass
            if self.is_at_end():
                break
            statements.append(self.parse_statement())
            self.match(TokenType.SEMICOLON)
        return statements

    # -- CREATE TABLE Parser --

    def _parse_create_table(self) -> CreateTableStatement:
        self.consume(TokenType.CREATE, "Expected 'CREATE'")
        self.consume(TokenType.TABLE, "Expected 'TABLE' after 'CREATE'")

        if_not_exists = False
        if self.match(TokenType.IF):
            self.consume(TokenType.NOT, "Expected 'NOT' after 'IF'")
            self.consume(TokenType.EXISTS, "Expected 'EXISTS' after 'IF NOT'")
            if_not_exists = True

        tbl_tok = self.consume(TokenType.IDENTIFIER, "Expected table name in CREATE TABLE")
        table_name = tbl_tok.value

        self.consume(TokenType.LPAREN, "Expected '(' after table name")

        columns: list[ColumnDef] = []
        while not self.check(TokenType.RPAREN) and not self.is_at_end():
            # Check for table-level PRIMARY KEY (col1, col2)
            if self.match(TokenType.PRIMARY):
                self.consume(TokenType.KEY, "Expected 'KEY' after 'PRIMARY'")
                self.consume(TokenType.LPAREN, "Expected '(' after PRIMARY KEY")
                pk_cols: list[str] = []
                while True:
                    pk_col_tok = self.consume(TokenType.IDENTIFIER, "Expected column name in PRIMARY KEY")
                    pk_cols.append(pk_col_tok.value.lower())
                    if not self.match(TokenType.COMMA):
                        break
                self.consume(TokenType.RPAREN, "Expected ')' after PRIMARY KEY column list")
                for col in columns:
                    if col.name.lower() in pk_cols:
                        col.primary_key = True
                        col.nullable = False
            else:
                col_name_tok = self.consume(TokenType.IDENTIFIER, "Expected column name")
                col_name = col_name_tok.value
                data_type = self._parse_data_type()

                primary_key = False
                nullable = True
                default_val: Optional[Any] = None
                unique = False
                auto_increment = False

                while True:
                    if self.match(TokenType.PRIMARY):
                        self.consume(TokenType.KEY, "Expected 'KEY' after 'PRIMARY'")
                        primary_key = True
                        nullable = False
                    elif self.match(TokenType.NOT):
                        self.consume(TokenType.NULL, "Expected 'NULL' after 'NOT'")
                        nullable = False
                    elif self.match(TokenType.NULL):
                        nullable = True
                    elif self.match(TokenType.DEFAULT):
                        default_val = self._parse_literal_value()
                    elif self.match(TokenType.UNIQUE):
                        unique = True
                    elif self.match(TokenType.AUTO_INCREMENT):
                        auto_increment = True
                    else:
                        break

                columns.append(
                    ColumnDef(
                        name=col_name,
                        data_type=data_type,
                        primary_key=primary_key,
                        nullable=nullable,
                        default=default_val,
                        unique=unique,
                        auto_increment=auto_increment,
                    )
                )

            if not self.match(TokenType.COMMA):
                break

        self.consume(TokenType.RPAREN, "Expected ')' after column definitions")
        return CreateTableStatement(
            table_name=table_name,
            columns=columns,
            if_not_exists=if_not_exists,
        )

    def _parse_data_type(self) -> str:
        tok = self.consume(TokenType.IDENTIFIER, "Expected data type name")
        type_name = tok.value.upper()
        if self.match(TokenType.LPAREN):
            # Parametric type e.g. VARCHAR(255), DECIMAL(10, 2)
            params: list[str] = []
            while not self.check(TokenType.RPAREN) and not self.is_at_end():
                if self.match(TokenType.INTEGER):
                    params.append(str(self.previous().value))
                elif self.match(TokenType.IDENTIFIER):
                    params.append(str(self.previous().value))
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.RPAREN, "Expected ')' after type arguments")
            type_name = f"{type_name}({', '.join(params)})"
        return type_name

    def _parse_literal_value(self) -> Any:
        if self.match(TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING, TokenType.BOOLEAN):
            return self.previous().value
        if self.match(TokenType.NULL):
            return None
        if self.match(TokenType.MINUS):
            num = self.consume(TokenType.INTEGER, "Expected number after '-'")
            return -num.value
        tok = self.peek()
        raise SQLSyntaxError(
            f"Expected literal value at line {tok.line}, col {tok.column}, found '{tok.value}'"
        )

    # -- INSERT INTO Parser --

    def _parse_insert_into(self) -> InsertStatement:
        self.consume(TokenType.INSERT, "Expected 'INSERT'")
        self.consume(TokenType.INTO, "Expected 'INTO' after 'INSERT'")

        tbl_tok = self.consume(TokenType.IDENTIFIER, "Expected table name in INSERT INTO")
        table_name = tbl_tok.value

        columns: Optional[list[str]] = None
        if self.match(TokenType.LPAREN):
            columns = []
            while True:
                col_tok = self.consume(TokenType.IDENTIFIER, "Expected column name")
                columns.append(col_tok.value)
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.RPAREN, "Expected ')' after column list")

        self.consume(TokenType.VALUES, "Expected 'VALUES' in INSERT INTO")

        values: list[list[Expression]] = []
        while True:
            self.consume(TokenType.LPAREN, "Expected '(' starting row values")
            row: list[Expression] = []
            while True:
                expr = self.parse_expression()
                row.append(expr)
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.RPAREN, "Expected ')' ending row values")
            values.append(row)
            if not self.match(TokenType.COMMA):
                break

        return InsertStatement(
            table_name=table_name,
            columns=columns,
            values=values,
        )

    # -- SELECT Parser --

    def _parse_select(self) -> SelectStatement:
        self.consume(TokenType.SELECT, "Expected 'SELECT'")

        distinct = False
        if self.match(TokenType.DISTINCT):
            distinct = True

        projections: list[SelectItem] = []
        while True:
            if self.match(TokenType.STAR):
                expr: Expression = Star(table=None)
            else:
                expr = self.parse_expression()

            alias: Optional[str] = None
            if self.match(TokenType.AS):
                alias_tok = self.consume(TokenType.IDENTIFIER, "Expected alias name after AS")
                alias = alias_tok.value
            elif self.check(TokenType.IDENTIFIER) and not self._is_clause_start(self.peek()):
                alias = self.advance().value

            projections.append(SelectItem(expr=expr, alias=alias))
            if not self.match(TokenType.COMMA):
                break

        from_table: Optional[TableRef] = None
        if self.match(TokenType.FROM):
            tbl_tok = self.consume(TokenType.IDENTIFIER, "Expected table name after FROM")
            tbl_alias: Optional[str] = None
            if self.match(TokenType.AS):
                tbl_alias = self.consume(TokenType.IDENTIFIER, "Expected table alias after AS").value
            elif self.check(TokenType.IDENTIFIER) and not self._is_clause_start(self.peek()):
                tbl_alias = self.advance().value
            from_table = TableRef(name=tbl_tok.value, alias=tbl_alias)

        joins: list[JoinClause] = []
        while self._is_join_start(self.peek()):
            join_type = self._parse_join_type()
            join_tbl = self.consume(TokenType.IDENTIFIER, "Expected table name after JOIN")
            join_alias: Optional[str] = None
            if self.match(TokenType.AS):
                join_alias = self.consume(TokenType.IDENTIFIER, "Expected table alias after AS").value
            elif self.check(TokenType.IDENTIFIER) and not self._is_clause_start(self.peek()) and not self.check(TokenType.ON):
                join_alias = self.advance().value

            table_ref = TableRef(name=join_tbl.value, alias=join_alias)

            on_condition: Optional[Expression] = None
            if self.match(TokenType.ON):
                on_condition = self.parse_expression()
            elif join_type != JoinType.CROSS:
                raise SQLSyntaxError(
                    f"Expected ON condition after JOIN {join_tbl.value} at line {join_tbl.line}, col {join_tbl.column}"
                )

            joins.append(JoinClause(join_type=join_type, table=table_ref, on=on_condition))

        where_clause: Optional[Expression] = None
        if self.match(TokenType.WHERE):
            where_clause = self.parse_expression()

        group_by: Optional[list[Expression]] = None
        if self.match(TokenType.GROUP):
            self.consume(TokenType.BY, "Expected 'BY' after 'GROUP'")
            group_by = []
            while True:
                group_by.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break

        having_clause: Optional[Expression] = None
        if self.match(TokenType.HAVING):
            having_clause = self.parse_expression()

        order_by: Optional[list[OrderByItem]] = None
        if self.match(TokenType.ORDER):
            self.consume(TokenType.BY, "Expected 'BY' after 'ORDER'")
            order_by = []
            while True:
                ord_expr = self.parse_expression()
                ascending = True
                if self.match(TokenType.ASC):
                    ascending = True
                elif self.match(TokenType.DESC):
                    ascending = False
                order_by.append(OrderByItem(expr=ord_expr, ascending=ascending))
                if not self.match(TokenType.COMMA):
                    break

        limit: Optional[int] = None
        if self.match(TokenType.LIMIT):
            lim_tok = self.consume(TokenType.INTEGER, "Expected integer count after LIMIT")
            limit = int(lim_tok.value)

        return SelectStatement(
            projections=projections,
            from_table=from_table,
            joins=joins,
            where=where_clause,
            group_by=group_by,
            having=having_clause,
            order_by=order_by,
            limit=limit,
            distinct=distinct,
        )

    def _is_clause_start(self, tok: Token) -> bool:
        return tok.type in CLAUSE_START_KEYWORDS

    def _is_join_start(self, tok: Token) -> bool:
        return tok.type in (
            TokenType.JOIN,
            TokenType.INNER,
            TokenType.LEFT,
            TokenType.RIGHT,
            TokenType.FULL,
            TokenType.CROSS,
        )

    def _parse_join_type(self) -> JoinType:
        if self.match(TokenType.CROSS):
            self.consume(TokenType.JOIN, "Expected 'JOIN' after 'CROSS'")
            return JoinType.CROSS
        elif self.match(TokenType.INNER):
            self.consume(TokenType.JOIN, "Expected 'JOIN' after 'INNER'")
            return JoinType.INNER
        elif self.match(TokenType.LEFT):
            self.match(TokenType.OUTER)
            self.consume(TokenType.JOIN, "Expected 'JOIN' after 'LEFT'")
            return JoinType.LEFT
        elif self.match(TokenType.RIGHT):
            self.match(TokenType.OUTER)
            self.consume(TokenType.JOIN, "Expected 'JOIN' after 'RIGHT'")
            return JoinType.RIGHT
        elif self.match(TokenType.FULL):
            self.match(TokenType.OUTER)
            self.consume(TokenType.JOIN, "Expected 'JOIN' after 'FULL'")
            return JoinType.FULL
        elif self.match(TokenType.JOIN):
            return JoinType.INNER
        else:
            tok = self.peek()
            raise SQLSyntaxError(f"Unexpected token '{tok.value}' when parsing JOIN at line {tok.line}, col {tok.column}")

    # -- Expression Parsing with Operator Precedence --

    def parse_expression(self) -> Expression:
        return self._parse_or()

    def _parse_or(self) -> Expression:
        expr = self._parse_and()
        while self.match(TokenType.OR):
            op = BinaryOpType.OR
            right = self._parse_and()
            expr = BinaryOp(left=expr, op=op, right=right)
        return expr

    def _parse_and(self) -> Expression:
        expr = self._parse_not()
        while self.match(TokenType.AND):
            op = BinaryOpType.AND
            right = self._parse_not()
            expr = BinaryOp(left=expr, op=op, right=right)
        return expr

    def _parse_not(self) -> Expression:
        if self.match(TokenType.NOT):
            operand = self._parse_not()
            return UnaryOp(op="NOT", operand=operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        expr = self._parse_additive()
        if self.match(
            TokenType.EQUALS,
            TokenType.NOT_EQUALS,
            TokenType.LESS_THAN,
            TokenType.GREATER_THAN,
            TokenType.LESS_EQUALS,
            TokenType.GREATER_EQUALS,
        ):
            op_tok = self.previous()
            op_str = op_tok.value
            if op_str == "<>":
                op_str = "!="
            op = BinaryOpType(op_str)
            right = self._parse_additive()
            expr = BinaryOp(left=expr, op=op, right=right)
        elif self.match(TokenType.IS):
            if self.match(TokenType.NOT):
                self.consume(TokenType.NULL, "Expected NULL after IS NOT")
                expr = BinaryOp(left=expr, op=BinaryOpType.IS_NOT, right=Literal(None))
            else:
                self.consume(TokenType.NULL, "Expected NULL after IS")
                expr = BinaryOp(left=expr, op=BinaryOpType.IS, right=Literal(None))
        return expr

    def _parse_additive(self) -> Expression:
        expr = self._parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_tok = self.previous()
            op = BinaryOpType(op_tok.value)
            right = self._parse_multiplicative()
            expr = BinaryOp(left=expr, op=op, right=right)
        return expr

    def _parse_multiplicative(self) -> Expression:
        expr = self._parse_unary()
        while self.match(TokenType.STAR, TokenType.SLASH):
            op_tok = self.previous()
            op = BinaryOpType(op_tok.value)
            right = self._parse_unary()
            expr = BinaryOp(left=expr, op=op, right=right)
        return expr

    def _parse_unary(self) -> Expression:
        if self.match(TokenType.MINUS):
            operand = self._parse_unary()
            if isinstance(operand, Literal) and isinstance(operand.value, (int, float)):
                return Literal(-operand.value)
            return UnaryOp(op="-", operand=operand)
        elif self.match(TokenType.PLUS):
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        # Literal values
        if self.match(TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING, TokenType.BOOLEAN, TokenType.NULL):
            return Literal(self.previous().value)

        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN, "Expected ')' after parenthesized expression")
            return expr

        # Wildcard *
        if self.match(TokenType.STAR):
            return Star(table=None)

        # Identifiers (Column reference, function call, table.column, or table.*)
        if self.match(TokenType.IDENTIFIER):
            ident = self.previous().value

            # Function call: ident(args)
            if self.match(TokenType.LPAREN):
                args: list[Expression] = []
                if not self.check(TokenType.RPAREN):
                    while True:
                        if self.match(TokenType.STAR):
                            args.append(Star(table=None))
                        else:
                            args.append(self.parse_expression())
                        if not self.match(TokenType.COMMA):
                            break
                self.consume(TokenType.RPAREN, "Expected ')' after function arguments")
                return FunctionCall(name=ident, args=args)

            # Table qualification: ident.column or ident.*
            if self.match(TokenType.DOT):
                if self.match(TokenType.STAR):
                    return Star(table=ident)
                col_tok = self.consume(TokenType.IDENTIFIER, f"Expected column name after '{ident}.'")
                return ColumnRef(name=col_tok.value, table=ident)

            return ColumnRef(name=ident, table=None)

        tok = self.peek()
        raise SQLSyntaxError(
            f"Unexpected token '{tok.value}' (type={tok.type.value}) at line {tok.line}, col {tok.column}"
        )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def parse_sql(sql: str, logical_plan: bool = False) -> Union[Statement, LogicalPlan]:
    """Parse a single SQL statement string into its typed AST representation or LogicalPlan."""
    stmt = Parser(sql).parse()
    if logical_plan:
        return build_logical_plan(stmt)
    return stmt


def parse(sql: str, logical_plan: bool = False) -> Union[Statement, LogicalPlan]:
    """Alias for parse_sql."""
    return parse_sql(sql, logical_plan=logical_plan)


def parse_to_logical_plan(sql: str) -> LogicalPlan:
    """Parse a SQL SELECT query directly into a LogicalPlan."""
    return to_logical_plan(sql)


def parse_logical_plan(sql: str) -> LogicalPlan:
    """Alias for parse_to_logical_plan."""
    return to_logical_plan(sql)


def to_logical_plan(stmt_or_sql: Union[Statement, str]) -> LogicalPlan:
    """Convert a Statement or SQL string into a LogicalPlan."""
    if isinstance(stmt_or_sql, str):
        stmt = Parser(stmt_or_sql).parse()
    else:
        stmt = stmt_or_sql
    return build_logical_plan(stmt)


def parse_statements(sql: str) -> list[Statement]:
    """Parse multiple SQL statements separated by semicolons."""
    return Parser(sql).parse_all()
