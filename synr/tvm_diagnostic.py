import tvm
from typing import Dict
from .diagnostic_context import DiagnosticContext

class TVMDiagnosticCtx(DiagnosticContext):
    module: tvm.IRModule

    def __init__(self, module: tvm.IRModule) -> None:
        self.module = module

    def add_source(self, name: str, source: str) -> None:
        return self.module.source_map.add(name, source)

    def emit(self, level, message, span):
        raise NotImplemented()
