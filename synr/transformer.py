from typing import TypeVar, Generic, Union

from . import ast

M = TypeVar("M")
F = TypeVar("F")
S = TypeVar("S")
E = TypeVar("E")
T = TypeVar("T")
B = TypeVar("B")
P = TypeVar("P")


class Transformer(Generic[M, F, S, E, B, T]):
    def do_transform(
        self, node: ast.Node, diag: "DiagnosticContext"
    ) -> Union[M, F, S, E, B, P, T]:
        self._diagnostic_context = diag
        return self.transform(node)

    def error(self, message, span):
        self._diagnostic_context.emit("error", message, span)

    def transform(self, node: ast.Node) -> Union[M, F, S, E, B, P, T]:
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
        raise RuntimeError(f"Unexpected synr ast type {type(node)}")

    def transform_module(self, mod: ast.Module) -> M:
        pass

    def transform_function(self, func: ast.Function) -> F:
        pass

    def transform_stmt(self, stmt: ast.Stmt) -> S:
        pass

    def transform_expr(self, expr: ast.Expr) -> E:
        pass

    def transform_block(self, expr: ast.Block) -> B:
        return [self.transform(x) for x in expr.stmts]

    def transform_parameter(self, expr: ast.Parameter) -> P:
        pass

    def transform_type(self, ty: ast.Type) -> T:
        pass
