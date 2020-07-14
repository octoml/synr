import synr
from synr import __version__


def test_version():
    assert __version__ == '0.1.0'

def to_ast(program):
    diag_ctx = synr.simple_diagnostic.SimpleDiagnosticCtx()
    transformer = None
    synr.ast.to_ast(program, diag_ctx, transformer)

def id(x):
    return x

def test_id_function():
    fn_ast = synr.ast.to_ast(id)
