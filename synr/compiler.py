from __future__ import annotations

import attr
import ast as py_ast
import inspect
from typing import Optional, Any, List

from .ast import *
from .diagnostic_context import DiagnosticContext
from .transformer import Transformer

class Compiler:
    start_line: int
    transformer: Optional[Transformer]
    diagnostic_ctx: DiagnosticContext

    def __init__(self, start_line: int, transformer: Optional[Transformer], diagnostic_ctx: DiagnosticContext):
        self.start_line = 0 # start_line
        self.transformer = transformer
        self.diagnostic_ctx = diagnostic_ctx

    def error(self, message, span):
        self.diagnostic_ctx.emit("error", message, span)

    def span_from_ast(self, node: py_ast.AST):
        span = Span.from_ast(node)
        span.start_line += self.start_line
        span.end_line += self.start_line
        return span

    def compile_module(self, program: py_ast.Module) -> Any:
        funcs: Dict[str, Function] = {}
        # Merge later
        span = self.span_from_ast(program.body[0])
        for stmt in program.body:
            # need to build module
            if isinstance(stmt, py_ast.FunctionDef):
                new_func = self.compile_stmt(stmt)
                assert isinstance(new_func, Function)
                funcs[stmt.name] = new_func
            else:
                self.error(
                    "can only transform top-level functions, other statements are invalid",
                    span)
        return Module(span, funcs)

    def compile_func_body(self, stmts: List[py_ast.stmt]) -> Stmt:
        if len(stmts) != 1:
            span = Span.from_ast(stmts[0])
            self.error(
                "currently synr only supports single statement function bodies",
                span)

        return self.compile_stmt(stmts[0])

    def compile_stmt(self, stmt: py_ast.stmt) -> Stmt:
        stmt_span = self.span_from_ast(stmt)
        if isinstance(stmt, py_ast.FunctionDef):
            name = stmt.name
            args = stmt.args
            body = self.compile_func_body(stmt.body)
            return Function(stmt_span, name, [], Type(None), body)
        elif isinstance(stmt, py_ast.Return):
            if stmt.value:
                value = self.compile_expr(stmt.value)
                return Return(stmt_span, value)
            else:
                return Return(stmt_span, None)
        else:
            self.error(f"found {type(stmt)}", stmt_span)

    def compile_expr(self, expr: py_ast.expr) -> Expr:
        expr_span = Span.from_ast(expr)
        if isinstance(expr, py_ast.Name):
            return Var(expr_span, expr.id)
        else:
            raise Exception(f"found {type(expr)}")


def to_ast(program: Any, diagnostic_ctx: DiagnosticContext, transformer: Optional[Transformer] = None):
    source_name = inspect.getsourcefile(program)
    assert source_name, "source name must be valid"
    lines, start_line = inspect.getsourcelines(program)
    source = "".join(lines)
    diagnostic_ctx.add_source(source_name, source)
    program_ast = py_ast.parse(source)
    compiler = Compiler(start_line, transformer, diagnostic_ctx)
    assert isinstance(program_ast, py_ast.Module), "support module"
    prog = compiler.compile_module(program_ast)
    diagnostic_ctx.diag_ctx.render()
    return prog
