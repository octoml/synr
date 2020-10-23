"""Error handling in Synr is done through a `DiagnosticContext`. This context
accumulates errors and renders them together after parsing has finished.
"""
from .ast import Span
from typing import Optional, Sequence, List, Tuple, Any, Dict
import attr


class DiagnosticContext:
    def add_source(self, name: str, source: str) -> None:
        """Add a file with source code to the context. This will be called
        before any call to :py:func:`emit` that contains a span in this
        file.
        """
        raise NotImplemented()

    def emit(self, level: str, message: str, span: Span) -> None:
        """Called when an error has occured."""
        raise NotImplemented()

    def render(self) -> Optional[Any]:
        """Render out all error messages. Can either return a value or raise
        and execption.
        """
        raise NotImplemented()


@attr.s(auto_attribs=True)
class PrinterDiagnosticContext(DiagnosticContext):
    """Simple diagnostic context that prints the error and underlines its
    location in the source. Raises a RuntimeError on the first error hit.
    """

    sources: Dict[str, Sequence[str]] = attr.ib(default=attr.Factory(dict))
    errors: List[Tuple[str, str, Span]] = attr.ib(default=attr.Factory(list))

    def add_source(self, name: str, source: str) -> None:
        self.sources[name] = source.split("\n")

    def emit(self, level: str, message: str, span: Span):
        self.errors.append((level, message, span))
        raise RuntimeError(self.render())

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
