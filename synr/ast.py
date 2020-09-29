from __future__ import annotations

import ast as py_ast

import attr
from typing import Optional, Any, List, Dict

@attr.s(auto_attribs=True)
class Span:
    start_line: int
    start_column: int
    end_line: Optional[int]
    end_column: Optional[int]

    @staticmethod
    def from_ast(node: py_ast.AST) -> Span:
        return Span(node.lineno, node.col_offset + 1, node.end_lineno, node.end_col_offset + 1)

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
