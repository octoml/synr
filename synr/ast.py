"""
`synr.ast` contains definitions of all nodes in the synr AST. This AST is
independent of python version: different versions of python will give the same
AST. In addition, all nodes contain span information.
"""
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


Name = str  # TODO: add span to this


@attr.s(auto_attribs=True, frozen=True)
class Node:
    """Base class of any AST node.

    All AST nodes must have a span.
    """

    span: Span


@attr.s(auto_attribs=True, frozen=True)
class Id(Node):
    name: str

    @staticmethod
    def invalid() -> "Id":
        return Id(Span.invalid(), "")


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
    :code:`x: str` in :code:`def my_function(x: str)`.
    """

    name: Name
    ty: Optional[Type]


@attr.s(auto_attribs=True, frozen=True)
class Var(Expr):
    """A variable in an expression.

    Examples
    --------
    :code:`x` and :code:`y` in :code:`x = y`.
    :code:`x` in :code:`x + 2`.
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
    In :code:`x.y`, :code:`x` is the :code:`object` and :code:`y` is the
    :code:`field`.  For multiple field accesses in a row, they are grouped left
    to right. For example, :code:`x.y.z` becomes
    :code:`Attr(Attr(object='x', field='y'), field='z')`.
    """

    object: Expr
    field: Id


@attr.s(auto_attribs=True, frozen=True)
class TypeVar(Type):
    """Type variable in a type expression.

    This is equivalent to :code:`Var`, but for types.

    Example
    -------
    :code:`X` in :code:`y: X = 2`.
    """

    id: Id


@attr.s(auto_attribs=True, frozen=True)
class TypeAttr(Type):
    """Field access in a type expression.

    This is equivalent to :code:`Attr`, but for types.

    Example
    -------
    In :code:`y: X.Z = 2`, :code:`object` is :code:`X` and :code:`field` is
    :code:`Z`.
    """

    object: Type
    field: Id


class BuiltinOp(Enum):
    """Various built in operators including binary operators, unary operators,
    and array access.

    Examples
    --------
    :code:`+` in :code:`2 + 3`.
    :code:`[]` in :code:`x[2]`.
    :code:`or` in :code:`true or false`.
    :code:`>` in :code:`3 > 2`.
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
    NotEq = auto()
    GT = auto()
    GE = auto()
    LT = auto()
    LE = auto()
    Not = auto()
    BitOr = auto()
    BitAnd = auto()
    BitXor = auto()
    USub = auto()
    UAdd = auto()
    Invert = auto()
    Invalid = auto()  # placeholder op if op failed to parse


@attr.s(auto_attribs=True, frozen=True)
class TypeCall(Type):
    """A function call in type expression.

    This is equivalent to :code:`Call`, but for type expressions.

    Example
    -------
    In :code:`x : List[type(None)]`, :code:`type(None)` is a :code:`TypeCall`.
    :code:`type` is the :code:`func_name` and :code:`None` is
    :code:`params[0]`.
    """

    func_name: Union[Expr, BuiltinOp]
    params: List[Type]


@attr.s(auto_attribs=True, frozen=True)
class TypeApply(Type):
    """Type application.

    Example
    -------
    In :code:`x: List[str]`, :code:`List[str]` is a :code:`TypeCall`. In this
    case, :code:`List` is the :code:`id`, and :code:`str` is :code:`params[0]`.
    """

    id: Id
    params: Sequence[Type]


@attr.s(auto_attribs=True, frozen=True)
class TypeTuple(Type):
    """A tuple in a type expression.

    Example
    -------
    :code:`(str, int)` in :code:`x: (str, int)`.
    """

    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class TypeConstant(Type):
    """A literal value in a type expression.

    This is equivalent to :code:`Constant`, but for type expressions.

    Examples
    --------
    :code:`1` in :code:`x: Array[1]`.
    :code:`None` in :code:`def my_function() -> None:`.
    """

    value: Union[str, NoneType, bool, complex, float, int]


@attr.s(auto_attribs=True, frozen=True)
class TypeSlice(Type):
    """A slice in a type expression.

    This is equivalent to :code:`Slice`, but for type expressions.

    Example
    -------
    :code:`1:2` in :code:`x: Array[1:2]`.
    """

    start: Type
    step: Type
    end: Type


@attr.s(auto_attribs=True, frozen=True)
class Tuple(Expr):
    """A tuple in an expression.

    Example
    -------
    :code:`(1, 2)` in :code:`x = (1, 2)`.
    """

    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class DictLiteral(Expr):
    """A sictionary literal in an expression.

    Example
    -------
    :code:`{"x": 2}` in :code:`x = {"x": 2}`.
    """

    keys: Sequence[Expr]
    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class ArrayLiteral(Expr):
    """An array literal in an expression.

    Example
    -------
    :code:`[1, 2]` in :code:`x = [1, 2]`.
    """

    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class Op(Node):
    """A builtin operator.

    See :code:`BuiltinOp` for supported operators.

    Example
    -------
    :code:`+` in :code:`x = 1 + 2`.
    """

    name: BuiltinOp


@attr.s(auto_attribs=True, frozen=True)
class Constant(Expr):
    """A literal value in an expression.

    Example
    -------
    :code:`1` in :code:`x = 1`.
    """

    value: Union[str, NoneType, bool, complex, float, int]


@attr.s(auto_attribs=True, frozen=True)
class Call(Expr):
    """A function call.

    Function calls can be:

    - A regular function call of the form :code:`x.y(1, 2, z=3)`. In this case,
      :code:`func_name` will contain the function name (`x.y`), :code:`params`
      will contain the arguments (`1, 2`), and :code:`keywords_params` will
      contain any keyword arguments (`z=3`).
    - A binary operation like :code:`x + 2`. In this case, :code:`func_name`
      will be the binary operation from :code:`BuiltinOp`, :code:`params[0]`
      will be the left hand side (`x`), and :code:`params[1]` will be the right
      hand side :code:`2`.
    - A unary operation like :code:`not x`. In this case, :code:`func_name`
      will be the unary operation from :code:`BuiltinOp`, and :code:`params[0]`
      will be the operand (`x`).
    - An array access like :code:`x[2, y]`. In this case, :code:`func_name`
      will be :code:`BuiltinOp.Subscript`, :code:`params[0]` will be the
      operand (`x`), :code:`params[1]` will be a :code:`Tuple` containing the
      indices (`2, y`).
    - An array assignment like :code:`x[2, 3] = y`. In this case,
      :code:`func_name` will be :code:`BuiltinOp.SubscriptAssign`,
      :code:`params[0]` will be the operand (`x`), :code:`params[1]` will be a
      :code:`Tuple` containing the indices (`2, 3`), and :code:`params[2]` will
      contain the right hand side of the assignment (`y`)."""

    func_name: Union[Expr, Op]
    params: List[Expr]
    keyword_params: Dict[Expr, Expr]


@attr.s(auto_attribs=True, frozen=True)
class Slice(Expr):
    """A slice in an expression.

    A slice in a range from [start,end) with step size step.

    Notes
    -----
    If not defined, start is 0, end is -1 and step is 1.

    Example
    -------
    :code:`1:2` in :code:`x = y[1:2]`.
    """

    start: Expr
    step: Expr
    end: Expr


@attr.s(auto_attribs=True, frozen=True)
class Return(Stmt):
    """A return statement.

    Example
    -------
    :code:`return x`.
    """

    value: Optional[Expr]


@attr.s(auto_attribs=True, frozen=True)
class Assign(Stmt):
    """An assignment statement.

    Notes
    -----
    Augmented assignment statements like :code:`x += 2` are translated into an
    operator call and an assignment (i.e. `x = x + 2`).

    Example
    -------
    In :code:`x: int = 2`, :code:`x` is :code:`lhs`, :code:`int` is :code:`ty`,
    and :code:`2` is :code:`rhs`.
    """

    lhs: Var
    ty: Optional[Type]
    rhs: Expr


@attr.s(auto_attribs=True, frozen=True)
class UnassignedCall(Stmt):
    """A standalone function call.

    Example
    -------
    .. code-block:: python

        def my_function():
            test_call()
            return 2

    Here :code:`test_call()` is an :code:`UnassignedCall`.
    """

    call: Call


@attr.s(auto_attribs=True, frozen=True)
class Block(Node):
    """A sequence of statements.

    Examples
    --------
    .. code-block:: python

        def my_function():
            test_call()
            return 2


    Here :code:`test_call()` and :code:`return 2` forms a :code:`Block`.

    .. code-block:: python

        if x > 2:
            y = 2

    Here :code:`y = 2` is a :code:`Block`.
    """

    stmts: List[Stmt]


@attr.s(auto_attribs=True, frozen=True)
class Function(Node):
    """A function declaration.

    Example
    -------
    .. code-block:: python

        def my_function(x: int):
            return x + 2

    Here :code:`name` is :code:`my_function`, :code:`x: int` is
    :code:`params[0]`, and :code:`body` is :code:`return x + 2`.
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
    .. code-block:: python

        for x in range(2):
            pass

    Here :code:`lhs` will be :code:`x`, :code:`rhs` will be :code:`range(2)`,
    and :code:`body` will be :code:`pass`.
    """

    lhs: Var
    rhs: Expr
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class With(Stmt):
    """A with statement.

    Example
    -------
    .. code-block:: python

        with open(x) as f:
            pass

    Here :code:`lhs` will be :code:`f`, :code:`rhs` will be :code:`open(x)`,
    and :code:`body` will be :code:`pass`.
    """

    lhs: Optional[Var]
    rhs: Expr
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class Assert(Stmt):
    """An assert statement.

    Example
    -------
    In :code:`assert 2 == 2, "Oh no!"`, :code:`condition` is :code:`2 == 2`,
    and :code:`msg` is :code:`"Oh no!"`.
    """

    condition: Expr
    msg: Optional[Expr]


@attr.s(auto_attribs=True, frozen=True)
class If(Stmt):
    """An if statement.

    Notes
    -----
    An if statement with :code:`elif` branches becomes multiple nested if
    statements.

    Examples
    --------
    .. code-block:: python

        if x == 2:
            return x
        else:
            return 3

    Here :code:`condition` is :code:`x == 2`, :code:`true` is :code:`return x`,
    and :code:`false` is :code:`return 3`.

    Multiple :code:`elif` statements become nested ifs. For example,

    .. code-block:: python

        if x == 2:
            return x
        elif x == 3:
            return "hi"
        else:
            return 3

    becomes :code:`If('x == 2', 'return x', If('x == 3', 'return "hi"', 'return 3'))`.
    """

    condition: Expr
    true: Block
    false: Block


@attr.s(auto_attribs=True, frozen=True)
class Class(Node):
    """A class definition.

    Example
    -------
    .. code-block:: python

        class MyClass:
            x = 2
            def return_x():
                return x

    Here :code:`name` is :code:`MyClass`, :code:`funcs["return_x"]` is
    :code:`def return_x():\n    return x`, and :code:`assignments[0]` is
    :code:`x = 2`.
    """

    name: Name
    funcs: Dict[Name, Function]
    assignments: List[Assign]


@attr.s(auto_attribs=True, frozen=True)
class Module(Node):
    """A collection of classes and functions."""

    funcs: Dict[Name, Union[Class, Function]]
