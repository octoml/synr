import tvm
import synr
from synr import __version__
from typing import Any

def test_version():
    assert __version__ == '0.1.0'

def to_ast(program: Any) -> Any:
    diag_ctx = synr.tvm_diagnostic.TVMDiagnosticCtx(tvm.IRModule({}))
    transformer = None
    return synr.to_ast(program, diag_ctx, transformer)

def assert_one_fn(module, name, no_params=None):
    func = module.funcs.get(name)
    assert func, "the function was not found"
    if no_params:
        assert len(func.params) == no_params, "the parameters do not match"
    return func

def identity(x):
    return x

def test_id_function():
    module = to_ast(identity)
    ast_fn = assert_one_fn(module, "identity", no_params=1)
    assert isinstance(ast_fn.body, synr.ast.Return)
    return_var = ast_fn.body.value
    assert isinstance(return_var, synr.ast.Var)
    assert return_var.name == "x"

class Class:
    def func():
        return func(3)

def test_class():
    module = to_ast(Class)
    print(module)

if __name__ == "__main__":
    test_id_function()
    test_class()
