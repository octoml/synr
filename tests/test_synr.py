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
    assert return_var.name.full_name == "x"


class ExampleClass:
    def func():
        return 3


def test_class():
    module = to_ast(ExampleClass)
    cls = module.funcs.get("ExampleClass")
    assert cls, "ExampleClass not found"
    assert isinstance(cls, synr.ast.Class), "ExampleClass was not parsed as a Class"
    assert len(cls.funcs) == 1, "func not found"
    fn = cls.funcs["func"]
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
    assert fr.lhs.name.full_name == "x", "For lhs is incorrect"
    assert isinstance(fr.rhs, synr.ast.Call)
    assert fr.rhs.name.name.full_name == "range"
    assert fr.rhs.params[0].value == 3
    assert isinstance(fr.body.stmts[0], synr.ast.Return)
    assert fr.body.stmts[0].value.name.full_name == "x"


def func_with():
    with x as y:
        return x


def test_with():
    module = to_ast(func_with)
    fn = assert_one_fn(module, "func_with", no_params=0)
    wth = fn.body.stmts[0]
    assert isinstance(
        wth, synr.ast.With
    ), "Did not find With statement, found %s" % type(wth)
    assert wth.lhs.name.full_name == "x"
    assert wth.rhs.name.full_name == "y"
    assert isinstance(wth.body.stmts[0], synr.ast.Return)
    assert wth.body.stmts[0].value.name.full_name == "x"


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
    assert assign.lhs.name.full_name == "y"
    assert isinstance(assign.rhs, synr.ast.Constant)
    assert assign.rhs.value == 2


def func_var():
    return x.y.z


def test_var():
    module = to_ast(func_var)
    fn = assert_one_fn(module, "func_var", no_params=0)
    ret = fn.body.stmts[0]
    assert ret.value.name.names == ["x", "y", "z"]


def func_binop():
    x = 1 + 2
    x = 1 - 2
    x = 1 * 2
    x = 1 / 2
    x = 1 // 2
    x = 1 % 2
    x = 1 == 2
    x = 1 != 2
    x = 1 >= 2
    x = 1 <= 2
    x = 1 < 2
    x = 1 > 2
    x = not True
    x = True and False
    x = True or False


def test_binop():
    module = to_ast(func_binop)
    fn = assert_one_fn(module, "func_binop", no_params=0)
    stmts = fn.body.stmts

    def verify(stmt, op, vals):
        assert isinstance(stmt, synr.ast.Call)
        assert stmt.name.name == op, f"Expect {op.name}, got {stmt.name.name}"
        assert len(vals) == len(stmt.params)
        for i in range(len(vals)):
            assert stmt.params[i].value == vals[i]

    verify(stmts[0].rhs, synr.ast.BuiltinOp.Add, [1, 2])
    verify(stmts[1].rhs, synr.ast.BuiltinOp.Sub, [1, 2])
    verify(stmts[2].rhs, synr.ast.BuiltinOp.Mul, [1, 2])
    verify(stmts[3].rhs, synr.ast.BuiltinOp.Div, [1, 2])
    verify(stmts[4].rhs, synr.ast.BuiltinOp.FloorDiv, [1, 2])
    verify(stmts[5].rhs, synr.ast.BuiltinOp.Mod, [1, 2])
    verify(stmts[6].rhs, synr.ast.BuiltinOp.Eq, [1, 2])

    assert isinstance(stmts[7].rhs, synr.ast.Call)
    assert stmts[7].rhs.name.name == synr.ast.BuiltinOp.Not
    assert len(stmts[7].rhs.params) == 1
    call = stmts[7].rhs.params[0]
    assert call.name.name == synr.ast.BuiltinOp.Eq
    assert call.params[0].value == 1
    assert call.params[1].value == 2

    verify(stmts[8].rhs, synr.ast.BuiltinOp.GE, [1, 2])
    verify(stmts[9].rhs, synr.ast.BuiltinOp.LE, [1, 2])
    verify(stmts[10].rhs, synr.ast.BuiltinOp.LT, [1, 2])
    verify(stmts[11].rhs, synr.ast.BuiltinOp.GT, [1, 2])
    verify(stmts[12].rhs, synr.ast.BuiltinOp.Not, [True])
    verify(stmts[13].rhs, synr.ast.BuiltinOp.And, [True, False])
    verify(stmts[14].rhs, synr.ast.BuiltinOp.Or, [True, False])


if __name__ == "__main__":
    test_id_function()
    test_class()
    test_for()
    test_with()
    test_block()
    test_assign()
    test_var()
    test_binop()
