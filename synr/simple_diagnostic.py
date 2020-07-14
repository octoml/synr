from typing import Dict
from .diagnostic_context import DiagnosticContext

class SimpleDiagnosticCtx(DiagnosticContext):
    source_map: Dict[str, str]

    def __init__(self) -> None:
        self.source_map = {}

    def add_source(self, name: str, source: str) -> None:
        self.source_map[name] = source
