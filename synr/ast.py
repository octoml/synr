import attr
import ast
import inspect
from typing import Optional, Any

from .diagnostic_context import DiagnosticContext

class Node:
    pass

@attr.s(auto_attribs=True)
class Span:
    start_line: int
    start_column: int
    end_line: int
    end_column: int

class Parameters:
    pass

class Type:
    pass

class Stmt:
    pass

@attr.s(auto_attribs=True)
class Function:
    name: str
    params: Parameters
    ret_type: Type
    body: Stmt

class Transformer:
    pass

class Compiler:
    transformer: Optional[Transformer]
    diagnostic_ctx: DiagnosticContext

    def __init__(self, transformer: Optional[Transformer], diagnostic_ctx: DiagnosticContext)

    def compile(self, program: ast.AST) -> Any:
        if isinstance(program, ast.FunctionDef):
            name = program.name
            args = program.args
            body = program.body

        import pdb; pdb.set_trace()


def span_for(node: ast.AST) -> Span:
    import pdb; pdb.set_trace()

def to_ast(program_src, diagnostic_ctx, transformer):

    import pdb; pdb.set_trace()
