"""This module handles converting a synr AST into the users desired
representation. We provide a visitor class that the user can inherit from to
write their conversion.
"""
from typing import TypeVar, Generic, Union

from . import ast
from .diagnostic_context import DiagnosticContext

M = TypeVar("M")
F = TypeVar("F")
S = TypeVar("S")
E = TypeVar("E")
T = TypeVar("T")
B = TypeVar("B")
P = TypeVar("P")


class Transformer(Generic[M, F, S, E, B, T]):
    """A visitor to handle user specified transformations on the AST."""

    def do_transform(
        self, node: ast.Node, diag: "DiagnosticContext"
    ) -> Union[M, F, S, E, B, P, T, None]:
        """Entry point for the transformation.

        This is called with the synr AST and the diagnostic context used in parsing the python AST.
        """
        self._diagnostic_context = diag
        return self.transform(node)

    def error(self, message, span):
        """Report an error on a given span."""
        self._diagnostic_context.emit("error", message, span)

    def transform(self, node: ast.Node) -> Union[M, F, S, E, B, P, T, None]:
        """Visitor function.

        Call this to recurse into child nodes.
        """
        if isinstance(node, ast.Module):
            return self.transform_module(node)
        if isinstance(node, ast.Function):
            return self.transform_function(node)
        if isinstance(node, ast.Stmt):
            return self.transform_stmt(node)
        if isinstance(node, ast.Expr):
            return self.transform_expr(node)
        if isinstance(node, ast.Type):
            return self.transform_type(node)
        if isinstance(node, ast.Block):
            return self.transform_block(node)
        if isinstance(node, ast.Parameter):
            return self.transform_parameter(node)
        self.error(f"Unexpected synr ast type {type(node)}", node.span)
        return None

    def transform_module(self, mod: ast.Module) -> M:
        pass

    def transform_function(self, func: ast.Function) -> F:
        pass

    def transform_stmt(self, stmt: ast.Stmt) -> S:
        pass

    def transform_expr(self, expr: ast.Expr) -> E:
        pass

    def transform_block(self, expr: ast.Block) -> B:
        pass

    def transform_parameter(self, expr: ast.Parameter) -> P:
        pass

    def transform_type(self, ty: ast.Type) -> T:
        pass
