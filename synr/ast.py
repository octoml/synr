from __future__ import annotations

import ast as py_ast

import attr
from typing import Optional, Any, List, Dict

@attr.s(auto_attribs=True)
class Span:
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    @staticmethod
    def from_ast(node: py_ast.AST) -> Span:
        return Span(node.lineno, node.col_offset + 1, node.end_lineno, node.end_col_offset + 1)

    def merge(self, span: Span) -> Span:
        self_start = (self.start_line, self.start_column)
        span_start = (span.start_line, span.start_column)
        if self_start < span_start:
            start_line, start_col = self_start
        else:
            start_line, start_col = span_start
        self_end = (self.end_line, self.end_column)
        span_end = (span.end_line, span.end_column)
        if self_end < span_end:
            end_line, end_col = self_end
        else:
            end_line, end_col = span_end
        return Span(start_line, start_col, end_line, end_col)

Id = str

@attr.s(auto_attribs=True)
class Node:
    span: Span

@attr.s(auto_attribs=True)
class Parameter(Node):
    name: Id
    ty: Optional[Type]

class Type(Node):
    pass

class Stmt(Node):
    pass

class Expr(Node):
    pass

@attr.s(auto_attribs=True)
class Module(Node):
    funcs: Dict[Id, Function]

@attr.s(auto_attribs=True)
class Var(Expr):
    name: Id

@attr.s(auto_attribs=True)
class Constant(Expr):
    value: Union[float, int]

@attr.s(auto_attribs=True)
class Call(Expr):
    name: Var
    params: List[Expr]

@attr.s(auto_attribs=True)
class Return(Stmt):
    value: Optional[Expr]

@attr.s(auto_attribs=True)
class Assign(Stmt):
    lhs: Id
    rhs: Expr

@attr.s(auto_attribs=True)
class Function(Stmt):
    name: Id
    params: List[Parameter]
    ret_type: Type
    body: Stmt

@attr.s(auto_attribs=True)
class Class(Node):
    functions: List[Function]
