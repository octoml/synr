import tvm
import synr
from synr import __version__
from typing import Any


def test_version():
    assert __version__ == "0.1.0"


def to_ast(program: Any) -> Any:
    diag_ctx = synr.tvm_diagnostic.TVMDiagnosticCtx(tvm.IRModule({}))
    transformer = None
    return synr.to_ast(program, diag_ctx, transformer)


def assert_one_fn(module, name, no_params=None):
    func = module.funcs.get(name)
    assert func, "the function `%s` was not found" % name
    if no_params:
        assert len(func.params) == no_params, "the parameters do not match"
    assert isinstance(func.body, synr.ast.Block)
    return func


def identity(x):
    return x


def test_id_function():
    module = to_ast(identity)
    ast_fn = assert_one_fn(module, "identity", no_params=1)
    return_var = ast_fn.body.stmts[-1].value
    assert isinstance(return_var, synr.ast.Var)
    assert return_var.name == "x"


class TestClass:
    def test_func():
        return 3


def test_class():
    module = to_ast(TestClass)
    cls = module.funcs.get("TestClass")
    assert cls, "TestClass not found"
    assert isinstance(cls, synr.ast.Class), "Class was not parsed as a Class"
    assert len(cls.funcs) == 1, "test_func not found"
    fn = cls.funcs[0]
    assert isinstance(fn, synr.ast.Function), "test_func not found"
    assert fn.name == "test_func", "test_func not found"
    return_var = fn.body.stmts[-1].value
    assert isinstance(return_var, synr.ast.Constant)
    assert return_var.value == 3

def func_for():
    for x in range(3):
        return x

def test_for():
    module = to_ast(func_for)
    fn = assert_one_fn(module, "func_for", no_params=0)
    fr = fn.body.stmts[0]
    assert isinstance(fr, synr.ast.For), "Did not find for loop"
    assert fr.lhs.name == "x", "For lhs is incorrect"
    assert isinstance(fr.rhs, synr.ast.Call)
    assert fr.rhs.name.name == "range"
    assert fr.rhs.params[0].value == 3
    assert isinstance(fr.body.stmts[0], synr.ast.Return)
    assert fr.body.stmts[0].value.name == "x"

def func_with():
    with x as y:
        return x

def test_with():
    module = to_ast(func_with)
    fn = assert_one_fn(module, "func_with", no_params=0)
    wth = fn.body.stmts[0]
    assert isinstance(wth, synr.ast.With), "Did not find With statement, found %s" % type(wth)
    assert wth.lhs.name == "x"
    assert wth.rhs.name == "y"
    assert isinstance(wth.body.stmts[0], synr.ast.Return)
    assert wth.body.stmts[0].value.name == "x"


if __name__ == "__main__":
    test_id_function()
    test_class()
    test_for()
    test_with()
