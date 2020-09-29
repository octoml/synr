import tvm
from typing import Dict

from . import ast
from .diagnostic_context import DiagnosticContext
from tvm.ir.diagnostics import DiagnosticContext as TVMCtx
from tvm.ir.diagnostics import get_default_renderer, DiagnosticLevel, Diagnostic

def to_tvm_span(src_name, ast_span: ast.Span) -> tvm.ir.Span:
    return tvm.ir.Span(
        src_name,
        ast_span.start_line,
        ast_span.end_line,
        ast_span.start_column,
        ast_span.end_column)

class TVMDiagnosticCtx(DiagnosticContext):
    diag_ctx: TVMCtx

    def __init__(self, module: tvm.IRModule) -> None:
        self.diag_ctx = TVMCtx(module, get_default_renderer()())
        self.source_name = None

    def add_source(self, name: str, source: str) -> None:
        src_name = self.diag_ctx.module.source_map.add(name, source)
        self.source_name = src_name

    def emit(self, level, message, span):
        span = to_tvm_span(self.source_name, span)
        return self.diag_ctx.emit(
            Diagnostic(DiagnosticLevel.ERROR, span, message))
