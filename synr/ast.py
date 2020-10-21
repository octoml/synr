import ast as py_ast

import attr
from enum import Enum, auto
import typing
from typing import Optional, Any, List, Dict, Union, Sequence

NoneType = type(None)


@attr.s(auto_attribs=True, frozen=True)
class Span:
    """An contiguous interval in a source file from a starting line and column
    to an ending line and column.

    Notes
    -----
    Line and column numbers are one indexed. The interval spanned is inclusive
    of the start and exclusive of the end.
    """

    filename: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    @staticmethod
    def from_ast(filename: str, node: py_ast.AST) -> "Span":
        """Extract the span of a python AST node"""
        if hasattr(node, "end_lineno") and node.end_lineno is not None:  # type: ignore
            end_lineno = node.end_lineno  # type: ignore
        else:
            end_lineno = node.lineno
        if hasattr(node, "end_col_offset") and node.end_col_offset is not None:  # type: ignore
            end_col_offset = node.end_col_offset + 1  # type: ignore
        else:
            end_col_offset = node.col_offset + 2
        return Span(
            filename, node.lineno, node.col_offset + 1, end_lineno, end_col_offset
        )

    def merge(self, span: "Span") -> "Span":
        """Return the span starting from the beginning of the first span and
        ending at the end of the second span.

        Notes
        -----
        Gaps between the end of the first span and the start of the second span
        are contained in the merged span. The spans also do not need to be in
        order.

        Example
        -------

            >>> Span("", 1, 2, 1, 4).merge(Span("", 2, 1, 2, 3))
            Span(filename="", start_line=1, start_column=2, end_line=2, end_column=3)
            >>> Span("", 2, 3, 2, 1).merge(Span("", 1, 2, 1, 4))
            Span(filename="", start_line=1, start_column=2, end_line=2, end_column=3)

        """
        assert self.filename == span.filename, "Spans must be from the same file"
        self_start = (self.start_line, self.start_column)
        span_start = (span.start_line, span.start_column)
        start_line, start_col = min(self_start, span_start)

        self_end = (self.end_line, self.end_column)
        span_end = (span.end_line, span.end_column)
        end_line, end_col = max(self_end, span_end)

        return Span(self.filename, start_line, start_col, end_line, end_col)

    @staticmethod
    def union(spans: Sequence["Span"]) -> "Span":
        """A span containing all the given spans with no gaps.

        This function is the equivalent to merge with more than two spans.

        Example
        -------

            >>> Span.union(Span("", 1, 2, 1, 4), Span("", 2, 1, 2, 3))
            Span(filename="", start_line=1, start_column=2, end_line=2, end_column=3)

        """
        if len(spans) == 0:
            return Span.invalid()
        span = spans[0]
        for s in spans[1:]:
            span = span.merge(s)
        return span

    def between(self, span: "Span") -> "Span":
        """The span starting after the end of the first span and ending before
        the start of the second span.

        Example
        --------

            >>> Span("", 1, 2, 1, 4).between(Span("", 2, 1, 2, 3))
            Span(filename="", start_line=1, start_column=4, end_line=2, end_column=1)

        """
        assert self.filename == span.filename, "Spans must be from the same file"
        return Span(
            self.filename,
            self.end_line,
            self.end_column,
            span.start_line,
            span.start_column,
        )

    def subtract(self, span: "Span") -> "Span":
        """The span from the start of the first span to the start of the second span.

        Example
        --------

            >>> Span("", 1, 2, 2, 4).subtract(Span("", 2, 1, 2, 3))
            Span(filename="", start_line=1, start_column=2, end_line=2, end_column=1)

        """
        assert self.filename == span.filename, "Spans must be from the same file"
        return Span(
            self.filename,
            self.start_line,
            self.start_column,
            span.start_line,
            span.start_column,
        )

    @staticmethod
    def invalid() -> "Span":
        """An invalid span"""
        return Span("", -1, -1, -1, -1)


@attr.s(auto_attribs=True, frozen=True)
class Id:
    name: str

    @staticmethod
    def invalid() -> "Id":
        return Id("")


Name = str  # TODO: add span to this


@attr.s(auto_attribs=True, frozen=True)
class Node:
    """Base class of any AST node.

    All AST nodes must have a span.
    """

    span: Span


class Type(Node):
    """Base class of a type in the AST."""

    pass


class Stmt(Node):
    """Base class of a statement in the AST."""

    pass


class Expr(Node):
    """Base class of a expression in the AST."""

    pass


@attr.s(auto_attribs=True, frozen=True)
class Parameter(Node):
    """Parameter in a function declaration.

    Example
    -------
    `x: str` in `def my_function(x: str)`.
    """

    name: Name
    ty: Optional[Type]


@attr.s(auto_attribs=True, frozen=True)
class Var(Expr):
    """A variable in an expression.

    Examples
    --------
    `x` and `y` in `x = y`.
    `x` in `x + 2`.
    """

    id: Id

    @staticmethod
    def invalid() -> "Var":
        return Var(Span.invalid(), Id.invalid())


@attr.s(auto_attribs=True, frozen=True)
class Attr(Expr):
    """Field access on variable or structure or module.

    Examples
    --------
    In `x.y`, `x` is the `object` and `y` is the `field`.
    For multiple field accesses in a row, they are grouped left to right. For
    example, `x.y.z` becomes `Attr(Attr(object='x', field='y'), field='z')`.
    """

    object: Expr
    field: Id


@attr.s(auto_attribs=True, frozen=True)
class TypeVar(Type):
    """Type variable in a type expression.

    This is equivalent to `Var`, but for types.

    Example
    -------
    `X` in `y: X = 2`.
    """

    id: Id


@attr.s(auto_attribs=True, frozen=True)
class TypeAttr(Type):
    """Field access in a type expression.

    This is equivalent to `Attr`, but for types.

    Example
    -------
    In `y: X.Z = 2`, `object` is `X` and `field` is `Z`.
    """

    object: Type
    field: Id


class BuiltinOp(Enum):
    """Various built in operators including binary operators, unary operators,
    and array access.

    Examples
    --------
    `+` in `2 + 3`.
    `[]` in `x[2]`.
    `or` in `true or false`.
    `>` in `3 > 2`.
    """

    Add = auto()
    Sub = auto()
    Mul = auto()
    Div = auto()
    FloorDiv = auto()
    Mod = auto()
    Subscript = auto()
    SubscriptAssign = auto()
    And = auto()
    Or = auto()
    Eq = auto()
    GT = auto()
    GE = auto()
    LT = auto()
    LE = auto()
    Not = auto()
    Invalid = auto()  # placeholder op if op failed to parse


@attr.s(auto_attribs=True, frozen=True)
class TypeCall(Type):
    """Function call in type expression.

    This is equivalent to `Call`, but for type expressions.

    Example
    -------
    In `x : List[type(None)]`, `type(None)` is a `TypeCall`. `type` is the
    `func_name` and `None` is `params[0]`.
    """

    func_name: Union[Expr, BuiltinOp]
    params: List[Type]


@attr.s(auto_attribs=True, frozen=True)
class TypeApply(Type):
    """Type application.

    Example
    -------
    In `x: List[str]`, `List[str]` is a `TypeCall`. In this case, `List` is the
    `id`, and `str` is `params[0]`.
    """

    id: Id
    params: Sequence[Type]


@attr.s(auto_attribs=True, frozen=True)
class TypeTuple(Type):
    """Tuple in a type expression.

    Example
    -------
    `(str, int)` in `x: (str, int)`.
    """

    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class TypeConstant(Type):
    """Literal value in a type expression.

    This is equivalent to `Constant`, but for type expressions.

    Examples
    --------
    `1` in `x: Array[1]`.
    `None` in `def my_function() -> None:`.
    """

    value: Union[str, NoneType, bool, complex, float, int]


@attr.s(auto_attribs=True, frozen=True)
class TypeSlice(Type):
    """Slice in a type expression.

    This is equivalent to `Slice`, but for type expressions.

    Example
    -------
    `1:2` in `x: Array[1:2]`.
    """

    start: Type
    step: Type
    end: Type


@attr.s(auto_attribs=True, frozen=True)
class Tuple(Expr):
    """Tuple in an expression.

    Example
    -------
    `(1, 2)` in `x = (1, 2)`.
    """

    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class DictLiteral(Expr):
    """Dictionary literal in an expression.

    Example
    -------
    `{"x": 2}` in `x = {"x": 2}`.
    """

    keys: Sequence[Expr]
    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class ArrayLiteral(Expr):
    """Array literal in an expression.

    Example
    -------
    `[1, 2]` in `x = [1, 2]`.
    """

    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class Op(Node):
    """Builtin operator.

    See `BuiltinOp` for supported operators.

    Example
    -------
    `+` in `x = 1 + 2`.
    """

    name: BuiltinOp


@attr.s(auto_attribs=True, frozen=True)
class Constant(Expr):
    """Literal value in an expression.

    Example
    -------
    `1` in `x = 1`.
    """

    value: Union[str, NoneType, bool, complex, float, int]


@attr.s(auto_attribs=True, frozen=True)
class Call(Expr):
    """A function call.

    Function calls can be:
    - A regular function call of the form :python:`x.y(1, 2, z=3)`. In this
      case, `func_name` will contain the function name (`x.y`), `params` will
      contain the arguments (`1, 2`), and `keywords_params` will contain any
      keyword arguments (`z=3`).
    - A binary operation like `x + 2`. In this case, `func_name` will be the
      binary operation from `BuiltinOp`, `params[0]` will be the left hand
      side (`x`), and `params[1]` will be the right hand side `2`.
    - A unary operation like `not x`. In this case, `func_name` will be the
      unary operation from `BuiltinOp`, and `params[0]` will be the operand
      (`x`).
    - An array access like `x[2, y]`. In this case, `func_name` will be
      `BuiltinOp.Subscript`, `params[0]` will be the operand (`x`), `params[1]`
      will be a `Tuple` containing the indices (`2, y`).
    - An array assignment like `x[2, 3] = y`. In this case, `func_name` will be
      `BuiltinOp.SubscriptAssign`, `params[0]` will be the operand (`x`),
      `params[1]` will be a `Tuple` containing the indices (`2, 3`), and
      `params[2]` will contain the right hand side of the assignment (`y`).
    """

    func_name: Union[Expr, Op]
    params: List[Expr]
    keyword_params: Dict[Expr, Expr]


@attr.s(auto_attribs=True, frozen=True)
class Slice(Expr):
    """Slice in an expression.

    A slice in a range from [start,end) with step size step.

    Notes
    -----
    If not defined, start is 0, end is -1 and step is 1.

    Example
    -------
    `1:2` in `x = y[1:2]`.
    """

    start: Expr
    step: Expr
    end: Expr


@attr.s(auto_attribs=True, frozen=True)
class Return(Stmt):
    """Return statement.

    Example
    -------
    `return x`.
    """

    value: Optional[Expr]


@attr.s(auto_attribs=True, frozen=True)
class Assign(Stmt):
    """Assignment statement.

    Example
    -------
    In `x: int = 2`, `x` is `lhs`, `int` is `ty`, and `2` is `rhs.
    """

    lhs: Var
    ty: Optional[Type]
    rhs: Expr


@attr.s(auto_attribs=True, frozen=True)
class UnassignedCall(Stmt):
    """A standalone function call.

    Example
    -------
    In ::
    .. code-block:: python

        def my_function():
            test_call()
            return 2

    `test_call()` is an `UnassignedCall`.
    """

    call: Call


@attr.s(auto_attribs=True, frozen=True)
class Block(Node):
    """A sequence of statements.

    Examples
    --------
    In ::
    .. code-block:: python

        def my_function():
            test_call()
            return 2

    `test_call()\nreturn 2` is an `Block`.

    In ::
    .. code-block:: python

        if x > 2:
            y = 2

    `y = 2` is a `Block`.
    """

    stmts: List[Stmt]


@attr.s(auto_attribs=True, frozen=True)
class Function(Node):
    """Function declaration.

    Example
    -------
    In ::
    .. code-block:: python

        def my_function(x: int):
            return x + 2

    `name` is `my_function`, `x: int` is `params[0]`, and `body` is `return x + 2`.
    """

    name: Name
    params: List[Parameter]
    ret_type: Optional[Type]
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class For(Stmt):
    """A for statement.

    Example
    -------
    In ::
    .. code-block:: python

        for x in range(2):
            pass

    `lhs` will be `x`, `rhs` will be `range(2)`, and `body` will be `pass`.
    """

    lhs: Var
    rhs: Expr
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class With(Stmt):
    """A with statement.

    Example
    -------
    In ::
    .. code-block:: python

        with open(x) as f:
            pass

    `lhs` will be `f`, `rhs` will be `open(x)`, and `body` will be `pass`.
    """

    lhs: Optional[Var]
    rhs: Expr
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class Assert(Stmt):
    """An assert statement.

    Example
    -------
    In `assert 2 == 2, "Oh no!"`, `condition` is `2 == 2`, and `msg` is `"Oh no!"`.
    """

    condition: Expr
    msg: Optional[Expr]


@attr.s(auto_attribs=True, frozen=True)
class If(Stmt):
    """An if statement.

    Notes
    -----
    An if statement with `elif` branches becomes multiple nested if statements.

    Examples
    --------
    In ::
    .. code-block:: python

        if x == 2:
            return x
        else:
            return 3

    `condition` is `x == 2`, `true` is `return x`, and `false` is `return 3`.

    Multiple `elif` statements become nested ifs. For example, ::
    .. code-block:: python

        if x == 2:
            return x
        elif x == 3:
            return "hi"
        else:
            return 3

    becomes `If('x == 2', 'return x', If('x == 3', 'return "hi"', 'return 3'))`.
    """

    condition: Expr
    true: Block
    false: Block


@attr.s(auto_attribs=True, frozen=True)
class Class(Node):
    """A class definition.

    Example
    -------
    In ::
    .. code-block:: python

        class MyClass:
            x = 2
            def return_x():
                return x

    `name` is `MyClass`, `funcs["return_x"]` is `def return_x():\n    return x`,
    and `assignments[0]` is `x = 2`.
    """

    name: Name
    funcs: Dict[Name, Function]
    assignments: List[Assign]


@attr.s(auto_attribs=True, frozen=True)
class Module(Node):
    """A collection of classes and functions."""

    funcs: Dict[Name, Union[Class, Function]]
