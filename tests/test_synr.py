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


class ExampleClass:
    def func():
        return 3


def test_class():
    module = to_ast(ExampleClass)
    cls = module.funcs.get("ExampleClass")
    assert cls, "ExampleClass not found"
    assert isinstance(cls, synr.ast.Class), "ExampleClass was not parsed as a Class"
    assert len(cls.funcs) == 1, "func not found"
    fn = cls.funcs[0]
    assert isinstance(fn, synr.ast.Function), "func not found"
    assert fn.name == "func", "func not found"
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

def func_block():
    y = x
    z = y
    return z

def test_block():
    module = to_ast(func_block)
    fn = assert_one_fn(module, "func_block", no_params=0)
    block = fn.body
    assert isinstance(block, synr.ast.Block)
    assert len(block.stmts) == 3
    assert isinstance(block.stmts[0], synr.ast.Assign)
    assert isinstance(block.stmts[1], synr.ast.Assign)
    assert isinstance(block.stmts[2], synr.ast.Return)

def func_assign():
    y = 2

def test_assign():
    module = to_ast(func_assign)
    fn = assert_one_fn(module, "func_assign", no_params=0)
    assign = fn.body.stmts[0]
    assert isinstance(assign, synr.ast.Assign)
    assert isinstance(assign.lhs, synr.ast.Var)
    assert assign.lhs.name == "y"
    assert isinstance(assign.rhs, synr.ast.Constant)
    assert assign.rhs.value == 2


if __name__ == "__main__":
    test_id_function()
    test_class()
    test_for()
    test_with()
    test_block()
    test_assign()
