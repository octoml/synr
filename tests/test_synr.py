import synr
from synr import __version__
from typing import Any

def test_version():
    assert __version__ == '0.1.0'

def to_ast(program: Any) -> Any:
    diag_ctx = synr.simple_diagnostic.SimpleDiagnosticCtx()
    transformer = None
    return synr.to_ast(program, diag_ctx, transformer)

def identity(x):
    return x

def test_id_function():
    fn_ast = to_ast(identity)
    import pdb; pdb.set_trace()
