import ast as py_ast

import attr
from enum import Enum, auto
import typing
from typing import Optional, Any, List, Dict, Union, Sequence

NoneType = type(None)


@attr.s(auto_attribs=True, frozen=True)
class Span:
    filename: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    @staticmethod
    def from_ast(filename: str, node: py_ast.AST) -> "Span":
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
        if len(spans) == 0:
            return Span.invalid()
        span = spans[0]
        for s in spans[1:]:
            span = span.merge(s)
        return span

    def between(self, span: "Span") -> "Span":
        """The span between two spans"""
        assert self.filename == span.filename, "Spans must be from the same file"
        return Span(
            self.filename,
            self.end_line,
            self.end_column,
            span.start_line,
            span.start_column,
        )

    def subtract(self, span: "Span") -> "Span":
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
    span: Span


class Type(Node):
    pass


class Stmt(Node):
    pass


class Expr(Node):
    pass


@attr.s(auto_attribs=True, frozen=True)
class Parameter(Node):
    name: Name
    ty: Optional[Type]


@attr.s(auto_attribs=True, frozen=True)
class Var(Expr):
    id: Id

    @staticmethod
    def invalid() -> "Var":
        return Var(Span.invalid(), Id.invalid())


@attr.s(auto_attribs=True, frozen=True)
class Attr(Expr):
    object: Expr
    field: Id


@attr.s(auto_attribs=True, frozen=True)
class TypeVar(Type):
    id: Id


@attr.s(auto_attribs=True, frozen=True)
class TypeAttr(Type):
    object: Type
    field: Id


class BuiltinOp(Enum):
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
    id: Union[Expr, BuiltinOp]
    params: List[Type]


@attr.s(auto_attribs=True, frozen=True)
class TypeApply(Type):
    id: Id
    params: Sequence[Type]


@attr.s(auto_attribs=True, frozen=True)
class TypeTuple(Type):
    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class TypeConstant(Type):
    value: Union[str, NoneType, bool, complex, float, int]


@attr.s(auto_attribs=True, frozen=True)
class TypeSlice(Type):
    start: Type
    step: Type
    end: Type


@attr.s(auto_attribs=True, frozen=True)
class Tuple(Expr):
    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class DictLiteral(Expr):
    keys: Sequence[Expr]
    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class ArrayLiteral(Expr):
    values: Sequence[Expr]


@attr.s(auto_attribs=True, frozen=True)
class Op(Node):
    name: BuiltinOp


@attr.s(auto_attribs=True, frozen=True)
class Constant(Expr):
    value: Union[str, NoneType, bool, complex, float, int]


@attr.s(auto_attribs=True, frozen=True)
class Call(Expr):
    func_name: Union[Expr, Op]
    params: List[Expr]
    keyword_params: Dict[Expr, Expr]


@attr.s(auto_attribs=True, frozen=True)
class Slice(Expr):
    start: Expr
    step: Expr
    end: Expr


@attr.s(auto_attribs=True, frozen=True)
class Return(Stmt):
    value: Optional[Expr]


@attr.s(auto_attribs=True, frozen=True)
class Assign(Stmt):
    lhs: Var
    ty: Optional[Type]
    rhs: Expr


@attr.s(auto_attribs=True, frozen=True)
class UnassignedCall(Stmt):
    call: Call


@attr.s(auto_attribs=True, frozen=True)
class Block(Node):
    stmts: List[Stmt]


@attr.s(auto_attribs=True, frozen=True)
class Function(Node):
    name: Name
    params: List[Parameter]
    ret_type: Optional[Type]
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class For(Stmt):
    lhs: Var
    rhs: Expr
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class With(Stmt):
    lhs: Optional[Var]
    rhs: Expr
    body: Block


@attr.s(auto_attribs=True, frozen=True)
class Assert(Stmt):
    condition: Expr
    msg: Optional[Expr]


@attr.s(auto_attribs=True, frozen=True)
class If(Stmt):
    condition: Expr
    true: Block
    false: Block


@attr.s(auto_attribs=True, frozen=True)
class Class(Node):
    funcs: Dict[Name, Function]


@attr.s(auto_attribs=True, frozen=True)
class Module(Node):
    funcs: Dict[Name, Union[Class, Function]]
