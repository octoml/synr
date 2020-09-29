from __future__ import annotations

import attr
from typing import Optional, Any, List

@attr.s(auto_attribs=True)
class Span:
    start_line: int
    start_column: int
    end_line: Optional[int]
    end_column: Optional[int]

    @staticmethod
    def from_ast(node: ast.AST) -> Span:
        return Span(node.lineno, node.col_offset + 1, node.end_lineno, node.end_col_offset + 1)

@attr.s(auto_attribs=True)
class Node:
    span: Span

class Parameters(Node):
    pass

class Type(Node):
    pass

class Stmt(Node):
    pass

class Expr(Node):
    pass

class Module(Node):
    pass

@attr.s(auto_attribs=True)
class Var(Expr):
    name: str

@attr.s(auto_attribs=True)
class Return(Stmt):
    value: Optional[Expr]

@attr.s(auto_attribs=True)
class Function(Stmt):
    name: str
    params: Parameters
    ret_type: Type
    body: Stmt
