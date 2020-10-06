from .ast import Span
from typing import Optional, Sequence, List, Tuple, Any, Dict
import attr


class DiagnosticContext:
    def add_source(self, name: str, source: str) -> None:
        raise NotImplemented()

    def emit(self, level: str, message: str, span: Span) -> None:
        raise NotImplemented()

    def render(self) -> Optional[Any]:
        raise NotImplemented()


@attr.s(auto_attribs=True)
class PrinterDiagnosticContext:
    sources: Dict[str, Sequence[str]] = attr.ib(default=attr.Factory(dict))
    errors: List[Tuple[str, str, Span]] = attr.ib(default=attr.Factory(list))

    def add_source(self, name: str, source: str) -> None:
        self.sources[name] = source.split("\n")

    def emit(self, level: str, message: str, span: Span):
        self.errors.append((level, message, span))

    def render(self) -> Optional[str]:
        msgs = []
        for level, message, span in self.errors:
            msg = f"Parse error on line {span.filename}:{span.start_line}:{span.start_column}:\n"
            msg += "\n".join(
                self.sources[span.filename][span.start_line - 1 : span.end_line]
            )
            msg += "\n"
            msg += (
                " " * (span.start_column - 1)
                + "^" * (span.end_column - span.start_column)
                + "\n"
            )
            msg += message
            msgs.append(msg)
        if len(msgs) == 0:
            return None
        else:
            return "\n\n".join(msgs)
