from typing import TypeVar, Generic

from . import ast

M = TypeVar('M')
S = TypeVar('S')
E = TypeVar('E')

class Transformer(Generic[M, S, E]):
    def transform(self):
        pass

    def transform_module(self, mod: ast.Module) -> M:
        pass

    def tranform_stmt(self, stmt: ast.Stmt) -> S:
        pass

    def transform_expr(self, expr: ast.Expr) -> E:
        pass


# new file my_mod
class MyMod:
    pass

class MyExpr:
    pass

class MyStmt:
    pass

class MyTransformer(Transformer[MyMod, MyStmt, MyExpr]):
    def transform_module(self, mod: ast.Module) -> M:
        pass
