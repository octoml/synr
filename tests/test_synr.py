import synr
from synr import __version__
from typing import Any
import inspect


def test_version():
    assert __version__ == "0.1.0"


def to_ast(program: Any) -> Any:
    diag_ctx = synr.PrinterDiagnosticContext()
    transformer = None
    res = synr.to_ast(program, diag_ctx, transformer)
    if isinstance(res, str):
        raise (RuntimeError(res))
    return res


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
    assert return_var.id.name == "x"


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

    for x, y in grid(5, 6):
        return x


def test_for():
    module = to_ast(func_for)
    fn = assert_one_fn(module, "func_for", no_params=0)

    fr = fn.body.stmts[0]
    assert isinstance(fr, synr.ast.For), "Did not find for loop"
    assert fr.lhs.id.name == "x", "For lhs is incorrect"
    assert isinstance(fr.rhs, synr.ast.Call)
    assert fr.rhs.func_name.id.name == "range"
    assert fr.rhs.params[0].value == 3
    assert isinstance(fr.body.stmts[0], synr.ast.Return)
    assert fr.body.stmts[0].value.id.name == "x"

    fr = fn.body.stmts[1]
    assert isinstance(fr, synr.ast.For), "Did not find for loop"
    assert isinstance(fr.lhs, synr.ast.Tuple)
    assert fr.lhs.values[0].id.name == "x", "For lhs is incorrect"
    assert fr.lhs.values[1].id.name == "y", "For lhs is incorrect"
    assert isinstance(fr.rhs, synr.ast.Call)
    assert fr.rhs.func_name.id.name == "grid"
    assert fr.rhs.params[0].value == 5
    assert fr.rhs.params[1].value == 6
    assert isinstance(fr.body.stmts[0], synr.ast.Return)
    assert fr.body.stmts[0].value.id.name == "x"


def func_with():
    with x as y:
        return x

    with block() as [x, y]:
        return x


def test_with():
    module = to_ast(func_with)
    fn = assert_one_fn(module, "func_with", no_params=0)
    wth = fn.body.stmts[0]
    assert isinstance(
        wth, synr.ast.With
    ), "Did not find With statement, found %s" % type(wth)
    assert wth.rhs.id.name == "x"
    assert wth.lhs.id.name == "y"
    assert isinstance(wth.body.stmts[0], synr.ast.Return)
    assert wth.body.stmts[0].value.id.name == "x"

    wth = fn.body.stmts[1]
    assert isinstance(
        wth, synr.ast.With
    ), "Did not find With statement, found %s" % type(wth)
    assert isinstance(wth.rhs, synr.ast.Call)
    assert wth.rhs.func_name.id.name == "block"
    assert isinstance(wth.lhs, synr.ast.ArrayLiteral)
    assert wth.lhs.values[0].id.name == "x"
    assert wth.lhs.values[1].id.name == "y"
    assert isinstance(wth.body.stmts[0], synr.ast.Return)
    assert wth.body.stmts[0].value.id.name == "x"


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
    assert assign.lhs.id.name == "y"
    assert isinstance(assign.rhs, synr.ast.Constant)
    assert assign.rhs.value == 2


def func_var():
    return x.y.z


def test_var():
    module = to_ast(func_var)
    fn = assert_one_fn(module, "func_var", no_params=0)
    ret = fn.body.stmts[0]
    assert ret.value.field.name == "z"
    assert ret.value.object.field.name == "y"
    assert ret.value.object.object.id.name == "x"


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
    x += 1
    x -= 1
    x /= 1
    x *= 1
    x //= 1
    x %= 1
    x = (1 + 3) / (4 % 2)
    x = -1
    x = +1
    x = ~1


def test_binop():
    module = to_ast(func_binop)
    fn = assert_one_fn(module, "func_binop", no_params=0)
    stmts = fn.body.stmts

    def verify(stmt, op, vals):
        assert isinstance(stmt, synr.ast.Call)
        assert stmt.func_name.name == op, f"Expect {op.name}, got {stmt.func_name}"
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
    verify(stmts[7].rhs, synr.ast.BuiltinOp.NotEq, [1, 2])
    verify(stmts[8].rhs, synr.ast.BuiltinOp.GE, [1, 2])
    verify(stmts[9].rhs, synr.ast.BuiltinOp.LE, [1, 2])
    verify(stmts[10].rhs, synr.ast.BuiltinOp.LT, [1, 2])
    verify(stmts[11].rhs, synr.ast.BuiltinOp.GT, [1, 2])
    verify(stmts[12].rhs, synr.ast.BuiltinOp.Not, [True])
    verify(stmts[13].rhs, synr.ast.BuiltinOp.And, [True, False])
    verify(stmts[14].rhs, synr.ast.BuiltinOp.Or, [True, False])

    def verify_assign(stmt, op, vals):
        assert isinstance(stmt.rhs, synr.ast.Call)
        assert stmt.rhs.func_name.name == op, f"Expect {op.name}, got {stmt.id.name}"
        assert len(vals) + 1 == len(stmt.rhs.params)
        assert stmt.lhs.id.name == stmt.rhs.params[0].id.name
        for i in range(len(vals)):
            assert stmt.rhs.params[i + 1].value == vals[i]

    verify_assign(stmts[15], synr.ast.BuiltinOp.Add, [1])
    verify_assign(stmts[16], synr.ast.BuiltinOp.Sub, [1])
    verify_assign(stmts[17], synr.ast.BuiltinOp.Div, [1])
    verify_assign(stmts[18], synr.ast.BuiltinOp.Mul, [1])
    verify_assign(stmts[19], synr.ast.BuiltinOp.FloorDiv, [1])
    verify_assign(stmts[20], synr.ast.BuiltinOp.Mod, [1])
    verify(stmts[22].rhs, synr.ast.BuiltinOp.USub, [1])
    verify(stmts[23].rhs, synr.ast.BuiltinOp.UAdd, [1])
    verify(stmts[24].rhs, synr.ast.BuiltinOp.Invert, [1])


def func_if():
    if 1 and 2 and 3 or 4:
        return 1
    elif 1:
        return 2
    else:
        return 3


def test_if():
    module = to_ast(func_if)
    fn = assert_one_fn(module, "func_if", no_params=0)

    if_stmt = fn.body.stmts[0]
    assert isinstance(if_stmt, synr.ast.If)
    assert isinstance(if_stmt.true.stmts[0], synr.ast.Return)
    assert if_stmt.true.stmts[0].value.value == 1
    cond = if_stmt.condition
    assert isinstance(cond, synr.ast.Call)
    assert cond.func_name.name == synr.ast.BuiltinOp.Or
    assert cond.params[1].value == 4
    elif_stmt = if_stmt.false.stmts[0]
    assert isinstance(elif_stmt.true.stmts[0], synr.ast.Return)
    assert elif_stmt.true.stmts[0].value.value == 2
    assert elif_stmt.condition.value == 1
    assert isinstance(elif_stmt.false.stmts[0], synr.ast.Return)
    assert elif_stmt.false.stmts[0].value.value == 3


def func_subscript():
    z = x[1:2, y]
    z = x[1.0:3.0:2]
    x[1:2] = 3
    z = x[y, z]
    return x[:1]


def test_subscript():
    module = to_ast(func_subscript)
    fn = assert_one_fn(module, "func_subscript", no_params=0)

    sub = fn.body.stmts[0].rhs
    assert isinstance(sub, synr.ast.Call)
    assert sub.func_name.name == synr.ast.BuiltinOp.Subscript
    assert sub.params[0].id.name == "x"
    assert sub.params[1].values[0].start.value == 1
    assert sub.params[1].values[0].step.value == 1
    assert sub.params[1].values[0].end.value == 2
    assert sub.params[1].values[1].id.name == "y"

    sub2 = fn.body.stmts[1].rhs
    assert sub2.params[1].values[0].step.value == 2

    sub3 = fn.body.stmts[2]
    assert isinstance(sub3, synr.ast.UnassignedCall)
    assert isinstance(sub3.call, synr.ast.Call)
    assert sub3.call.func_name.name == synr.ast.BuiltinOp.SubscriptAssign
    assert sub3.call.params[0].id.name == "x"
    assert isinstance(sub3.call.params[1], synr.ast.Tuple)
    assert isinstance(sub3.call.params[1].values[0], synr.ast.Slice)
    assert sub3.call.params[1].values[0].start.value == 1
    assert sub3.call.params[1].values[0].end.value == 2
    assert sub3.call.params[2].value == 3

    sub4 = fn.body.stmts[3].rhs
    assert sub4.params[1].values[0].id.name == "y"
    assert sub4.params[1].values[1].id.name == "z"


def func_literals():
    x = 1
    x = 2.0
    x = (1, 2.0)


def test_literals():
    module = to_ast(func_literals)
    fn = assert_one_fn(module, "func_literals", no_params=0)

    assert fn.body.stmts[0].rhs.value == 1
    assert isinstance(fn.body.stmts[0].rhs.value, int)

    assert fn.body.stmts[1].rhs.value == 2.0
    assert isinstance(fn.body.stmts[1].rhs.value, float)

    assert fn.body.stmts[2].rhs.values[0].value == 1
    assert fn.body.stmts[2].rhs.values[1].value == 2.0
    assert isinstance(fn.body.stmts[2].rhs, synr.ast.Tuple)


class X:
    pass


class Y:
    pass


def func_type(x: X) -> Y:
    x: test.X = 1
    x: X[Y] = 1
    x: X[X, Y] = 1
    x: X[X:Y] = 1
    x: X[1] = 1


def test_type():
    module = to_ast(func_type)
    fn = assert_one_fn(module, "func_type", no_params=0)

    assert isinstance(fn.ret_type, synr.ast.TypeVar)
    assert isinstance(fn.params[0].ty, synr.ast.TypeVar), fn.params[0].ty
    assert fn.params[0].ty.id.name == "X"

    stmts = fn.body.stmts
    assert stmts[0].ty.object.id.name == "test"
    assert stmts[0].ty.field.name == "X"

    assert isinstance(stmts[1].ty, synr.ast.TypeApply)
    assert stmts[1].ty.id.name == "X"
    assert stmts[1].ty.params[0].id.name == "Y"

    assert isinstance(stmts[2].ty, synr.ast.TypeApply)
    assert stmts[2].ty.id.name == "X"
    assert stmts[2].ty.params[0].id.name == "X"
    assert stmts[2].ty.params[1].id.name == "Y"


def func_call():
    test()


def test_call():
    module = to_ast(func_call)
    fn = assert_one_fn(module, "func_call", no_params=0)

    assert isinstance(fn.body.stmts[0], synr.ast.UnassignedCall)
    assert fn.body.stmts[0].call.func_name.id.name == "test"


def func_constants():
    x = {"test": 1, "another": 3j}
    y = ["an", "array", 2.0, None, True, False]
    z = ("hi",)


def test_constants():
    module = to_ast(func_constants)
    fn = assert_one_fn(module, "func_constants", no_params=0)

    d = fn.body.stmts[0].rhs
    assert isinstance(d, synr.ast.DictLiteral)
    k = [x.value for x in d.keys]
    v = [x.value for x in d.values]
    assert dict(zip(k, v)) == {"test": 1, "another": 3j}

    ary = fn.body.stmts[1].rhs
    assert isinstance(ary, synr.ast.ArrayLiteral)
    assert [x.value for x in ary.values] == ["an", "array", 2.0, None, True, False]

    t = fn.body.stmts[2].rhs
    assert isinstance(t, synr.ast.Tuple)
    assert [x.value for x in t.values] == ["hi"]


class ErrorAccumulator:
    def __init__(self):
        self.errors = {}
        self.sources = {}

    def add_source(self, name, source):
        self.sources[name] = source

    def emit(self, level, message, span):
        if span.start_line in self.errors:
            self.errors[span.start_line].append((level, message, span))
        else:
            self.errors[span.start_line] = [(level, message, span)]

    def render(self):
        return self.errors


def to_ast_err(program: Any) -> Any:
    diag_ctx = ErrorAccumulator()
    transformer = None
    return synr.to_ast(program, diag_ctx, transformer)


def func_err(x=2, *args, **kwargs):
    x, y = 2
    x: X


def test_err_msg():
    _, start = inspect.getsourcelines(func_err)
    errs = to_ast_err(func_err)
    def_errs = sorted(
        [(x[1], x[2]) for x in errs[start]], key=lambda x: x[1].start_column
    )

    def check_err(err, msg, filename, start_line, start_column):
        assert (
            err[0] == msg
        ), f"Error message `{err[0]}` does not match expected message `{msg}`"
        span = err[1]
        assert span.filename.endswith(
            filename
        ), f"File name `{span.filename}` does not end with `{filename}`"
        assert (
            span.start_line == start_line
        ), f"Starting line of error does not match expected: {span.start_line} vs {start_line}"
        assert (
            span.start_column == start_column
        ), f"Starting column of error does not match expected: {span.start_column} vs {start_column}"

    check_err(
        def_errs[0],
        "currently synr does not support defaults",
        "test_synr.py",
        start,
        16,
    )
    check_err(
        def_errs[1],
        "currently synr does not support varargs",
        "test_synr.py",
        start,
        20,
    )
    check_err(
        def_errs[2],
        "currently synr does not support kwarg",
        "test_synr.py",
        start,
        28,
    )

    assert errs[start + 1][0][1] == "Left hand side of assignment must be a variable"
    assert errs[start + 2][0][1] == "Empty type assignment not supported"


if __name__ == "__main__":
    test_id_function()
    test_class()
    test_for()
    test_with()
    test_block()
    test_assign()
    test_var()
    test_binop()
    test_if()
    test_subscript()
    test_literals()
    test_type()
    test_call()
    test_constants()
    test_err_msg()
