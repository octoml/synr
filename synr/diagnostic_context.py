from .ast import Span

class DiagnosticContext:
    def add_source(self, name: str, source: str) -> None:
        raise NotImplemented()

    def emit(self, level, message, span):
        raise NotImplemented()
