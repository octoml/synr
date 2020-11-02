"""This module contains the main compiler from the python AST to the synr AST."""
import ast as py_ast
import inspect
from typing import Optional, Any, List, Union, Sequence

from .ast import *
from .diagnostic_context import DiagnosticContext
from .transformer import Transformer


class Compiler:
    filename: str
    start_line: int
    transformer: Optional[Transformer]
    diagnostic_ctx: DiagnosticContext

    _builtin_ops = {
        py_ast.Add: BuiltinOp.Add,
        py_ast.Sub: BuiltinOp.Sub,
        py_ast.Mult: BuiltinOp.Mul,
        py_ast.Div: BuiltinOp.Div,
        py_ast.FloorDiv: BuiltinOp.FloorDiv,
        py_ast.Mod: BuiltinOp.Mod,
        py_ast.Eq: BuiltinOp.Eq,
        py_ast.NotEq: BuiltinOp.NotEq,
        py_ast.GtE: BuiltinOp.GE,
        py_ast.LtE: BuiltinOp.LE,
        py_ast.Gt: BuiltinOp.GT,
        py_ast.Lt: BuiltinOp.LT,
        py_ast.Not: BuiltinOp.Not,
        py_ast.Or: BuiltinOp.Or,
        py_ast.And: BuiltinOp.And,
        py_ast.BitOr: BuiltinOp.BitOr,
        py_ast.BitAnd: BuiltinOp.BitAnd,
        py_ast.BitXor: BuiltinOp.BitXor,
        py_ast.USub: BuiltinOp.USub,
        py_ast.UAdd: BuiltinOp.UAdd,
        py_ast.Invert: BuiltinOp.Invert,
    }

    def __init__(
        self,
        filename: str,
        start_line: int,
        transformer: Optional[Transformer],
        diagnostic_ctx: DiagnosticContext,
    ):
        self.filename = filename
        self.start_line = start_line - 1
        self.transformer = transformer
        self.diagnostic_ctx = diagnostic_ctx

    def error(self, message, span):
        self.diagnostic_ctx.emit("error", message, span)

    def span_from_ast(self, node: py_ast.AST) -> Span:
        if isinstance(node, py_ast.withitem):
            span = Span.from_ast(self.filename, node.context_expr)
            if node.optional_vars:
                end_span = Span.from_ast(self.filename, node.optional_vars)
                span = span.merge(end_span)
        elif isinstance(node, py_ast.Slice):
            spans = []
            if node.lower is not None:
                spans.append(Span.from_ast(self.filename, node.lower))
            if node.upper is not None:
                spans.append(Span.from_ast(self.filename, node.upper))
            if node.step is not None:
                spans.append(Span.from_ast(self.filename, node.step))
            span = Span.union(spans)
        elif isinstance(node, py_ast.Index):
            span = Span.from_ast(self.filename, node.value)
        elif isinstance(node, py_ast.keyword):
            # unfortunately this span does not include the keyword name itself
            span = Span.from_ast(self.filename, node.value)
        else:
            span = Span.from_ast(self.filename, node)
        return Span(
            span.filename,
            span.start_line + self.start_line,
            span.start_column,
            span.end_line + self.start_line,
            span.end_column,
        )

    def span_from_asts(self, nodes: Sequence[py_ast.AST]) -> Span:
        return Span.union([self.span_from_ast(node) for node in nodes])

    def compile_module(self, program: py_ast.Module) -> Module:
        funcs: Dict[Name, Union[Function, Class]] = {}
        # Merge later
        span = self.span_from_ast(program.body[0])
        for stmt in program.body:
            # need to build module
            if isinstance(stmt, py_ast.FunctionDef):
                new_func = self.compile_def(stmt)
                assert isinstance(new_func, Function)
                funcs[stmt.name] = new_func
            elif isinstance(stmt, py_ast.ClassDef):
                new_class = self.compile_class(stmt)
                assert isinstance(new_class, Class)
                funcs[stmt.name] = new_class
            else:
                self.error(
                    "can only transform top-level functions, other statements are invalid",
                    span,
                )
        return Module(span, funcs)

    def compile_block(self, stmts: List[py_ast.stmt]) -> Block:
        assert (
            len(stmts) != 0
        ), "Internal Error: python ast forbids empty function bodies."
        return Block(
            self.span_from_asts(stmts),
            [self.compile_stmt(st) for st in stmts],
        )

    def compile_args_to_params(self, args: py_ast.arguments) -> List[Parameter]:
        # TODO: arguments object doesn't have line column so its either the
        # functions span needs to be passed or we need to reconstruct span
        # information from arg objects.
        #
        # The below solution is temporary hack.

        if hasattr(args, "posonlyargs") and len(args.posonlyargs):  # type: ignore
            self.error(
                "currently synr only supports non-position only arguments",
                self.span_from_asts(args.posonlyargs),  # type: ignore
            )

        if args.vararg:
            self.error(
                "currently synr does not support varargs",
                self.span_from_ast(args.vararg),
            )

        if len(args.kw_defaults):
            self.error(
                "currently synr does not support kw_defaults",
                self.span_from_asts(args.kw_defaults),
            )

        if args.kwarg:
            self.error(
                "currently synr does not support kwarg", self.span_from_ast(args.kwarg)
            )

        if len(args.defaults) > 0:
            self.error(
                "currently synr does not support defaults",
                self.span_from_asts(args.defaults),
            )

        params = []
        for arg in args.args:
            span = self.span_from_ast(arg)
            ty: Optional[Type] = None
            if arg.annotation is not None:
                ty = self.compile_type(arg.annotation)
            params.append(Parameter(span, arg.arg, ty))

        return params

    def compile_def(self, stmt: py_ast.stmt) -> Function:
        stmt_span = self.span_from_ast(stmt)
        if isinstance(stmt, py_ast.FunctionDef):
            name = stmt.name
            args = stmt.args
            params = self.compile_args_to_params(args)
            body = self.compile_block(stmt.body)
            ty: Optional[Type] = None
            if stmt.returns is not None:
                ty = self.compile_type(stmt.returns)
            return Function(stmt_span, name, params, ty, body)
        else:
            self.error(
                f"Unexpected {type(stmt)} when looking for a function definition",
                stmt_span,
            )
            return Function(
                Span.invalid(), "", [], Type(Span.invalid()), Block(Span.invalid(), [])
            )

    def compile_stmt(self, stmt: py_ast.stmt) -> Stmt:
        stmt_span = self.span_from_ast(stmt)
        if isinstance(stmt, py_ast.Return):
            if stmt.value:
                value = self.compile_expr(stmt.value)
                return Return(stmt_span, value)
            else:
                return Return(stmt_span, None)

        elif (
            isinstance(stmt, py_ast.Assign)
            or isinstance(stmt, py_ast.AugAssign)
            or isinstance(stmt, py_ast.AnnAssign)
        ):
            if isinstance(stmt, py_ast.Assign):
                if len(stmt.targets) > 1:
                    self.error("Multiple assignment is not supported", stmt_span)
                lhs = self.compile_expr(stmt.targets[0])
            else:
                lhs = self.compile_expr(stmt.target)

            if stmt.value is None:
                self.error("Empty type assignment not supported", stmt_span)
                rhs = Expr(stmt_span)
            else:
                rhs = self.compile_expr(stmt.value)

            if isinstance(stmt, py_ast.AugAssign):
                op = self._builtin_ops.get(type(stmt.op))
                op_span = lhs.span.between(rhs.span)
                if op is None:
                    self.error(
                        "Unsupported op {type(op)} in assignment",
                        op_span,
                    )
                    op = BuiltinOp.Invalid
                rhs = Call(stmt_span, Op(op_span, op), [lhs, rhs], {})

            # if the lhs is a subscript, we replace the whole expression with a SubscriptAssign
            if (
                isinstance(lhs, Call)
                and isinstance(lhs.func_name, Op)
                and lhs.func_name.name == BuiltinOp.Subscript
            ):
                return UnassignedCall(
                    stmt_span,
                    Call(
                        stmt_span,
                        Op(lhs.func_name.span, BuiltinOp.SubscriptAssign),
                        [
                            lhs.params[0],
                            Tuple(
                                Span.union([x.span for x in lhs.params[1].values]),  # type: ignore
                                lhs.params[1].values,  # type: ignore
                            ),
                            rhs,
                        ],
                        {},
                    ),
                )
            elif not isinstance(lhs, Var):
                self.error(
                    "Left hand side of assignment must be a variable",
                    lhs.span,
                )
                lhs = Var(Span.invalid(), Id.invalid())
            ty: Optional[Type] = None
            if isinstance(stmt, py_ast.AnnAssign):
                ty = self.compile_type(stmt.annotation)
            return Assign(stmt_span, lhs, ty, rhs)

        elif isinstance(stmt, py_ast.For):
            lhs = self.compile_expr(stmt.target)
            if not isinstance(lhs, Var):
                self.error(
                    "Left hand side of for loop (the x in `for x in range(...)`) must be a single variable",
                    self.span_from_ast(stmt.target),
                )
                lhs = Var(Span.invalid(), Id.invalid())
            rhs = self.compile_expr(stmt.iter)
            body = self.compile_block(stmt.body)
            return For(self.span_from_ast(stmt), lhs, rhs, body)

        elif isinstance(stmt, py_ast.With):
            if len(stmt.items) != 1:
                self.error(
                    "Only one `x as y` statement allowed in with statements",
                    self.span_from_asts(stmt.items),
                )
            wth = stmt.items[0]
            lhs_var: Optional[Var]
            if wth.optional_vars:
                l = self.compile_expr(wth.optional_vars)
                if not isinstance(l, Var):
                    self.error(
                        "Right hand side of with statement (y in `with x as y:`) must be a variable",
                        self.span_from_ast(wth.optional_vars),
                    )
                    lhs_var = Var.invalid()
                else:
                    lhs_var = l
            else:
                lhs_var = None
            rhs = self.compile_expr(wth.context_expr)
            body = self.compile_block(stmt.body)
            return With(self.span_from_ast(stmt), lhs_var, rhs, body)

        elif isinstance(stmt, py_ast.If):
            true = self.compile_block(stmt.body)
            if len(stmt.orelse) > 0:
                false = self.compile_block(stmt.orelse)
            else:
                false = Block(stmt_span, [])  # TODO: this span isn't correct
            condition = self.compile_expr(stmt.test)
            return If(stmt_span, condition, true, false)

        elif isinstance(stmt, py_ast.Expr):
            expr = self.compile_expr(stmt.value)
            if isinstance(expr, Call):
                return UnassignedCall(expr.span, expr)
            else:
                self.error(
                    f"Found unexpected expression of type {type(expr)} when looking for a statement",
                    expr.span,
                )
                return Stmt(Span.invalid())

        elif isinstance(stmt, py_ast.Assert):
            return Assert(
                stmt_span,
                self.compile_expr(stmt.test),
                None if stmt.msg is None else self.compile_expr(stmt.msg),
            )

        else:
            self.error(f"Found unexpected {type(stmt)} when compiling stmt", stmt_span)
            return Stmt(Span.invalid())

    def compile_var(self, expr: py_ast.expr) -> Union[Var, Attr]:
        expr_span = self.span_from_ast(expr)
        if isinstance(expr, py_ast.Name):
            return Var(expr_span, Id(expr_span, expr.id))
        if isinstance(expr, py_ast.Attribute):
            sub_var = self.compile_expr(expr.value)
            # The name of the field we are accessing comes at the end of the
            # span, we can just infer the span for it from counting from the
            # end.
            attr_span = Span(
                expr_span.filename,
                expr_span.end_line,
                expr_span.end_column - len(expr.attr),
                expr_span.end_line,
                expr_span.end_column,
            )
            return Attr(expr_span, sub_var, Id(attr_span, expr.attr))
        self.error("Expected a variable name of the form a.b.c", expr_span)
        return Var.invalid()

    def compile_expr(self, expr: py_ast.expr) -> Expr:
        expr_span = self.span_from_ast(expr)
        if isinstance(expr, py_ast.Name) or isinstance(expr, py_ast.Attribute):
            return self.compile_var(expr)
        if isinstance(expr, py_ast.Constant):
            if (
                isinstance(expr.value, float)
                or isinstance(expr.value, complex)
                or isinstance(expr.value, int)
                or isinstance(expr.value, str)
                or isinstance(expr.value, type(None))
                or isinstance(expr.value, bool)
            ):
                return Constant(expr_span, expr.value)
            self.error(
                "Only float, complex, int, str, bool, and None constants are allowed. "
                f"{type(expr.value)} was provided.",
                expr_span,
            )
            return Constant(expr_span, float("nan"))
        if isinstance(expr, py_ast.Num):
            return Constant(expr_span, expr.n)
        if isinstance(expr, py_ast.Str):
            return Constant(expr_span, expr.s)
        if isinstance(expr, py_ast.NameConstant):
            return Constant(expr_span, expr.value)
        if isinstance(expr, py_ast.Call):
            return self.compile_call(expr)
        if isinstance(expr, py_ast.BinOp):
            lhs = self.compile_expr(expr.left)
            rhs = self.compile_expr(expr.right)
            op_span = lhs.span.between(rhs.span)
            ty = type(expr.op)
            op = self._builtin_ops.get(ty)
            if op is None:
                self.error(
                    f"Binary operator {ty} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            return Call(expr_span, Op(op_span, op), [lhs, rhs], {})
        if isinstance(expr, py_ast.Compare):
            lhs = self.compile_expr(expr.left)
            if (
                len(expr.ops) != 1
                or len(expr.comparators) != 1
                or len(expr.comparators) != len(expr.ops)
            ):
                self.error(
                    "Only one comparison operator is allowed",
                    self.span_from_asts(expr.comparators),
                )
            rhs = self.compile_expr(expr.comparators[0])
            op_span = lhs.span.between(rhs.span)
            ty2 = type(expr.ops[0])
            op = self._builtin_ops.get(ty2)
            if op is None:
                self.error(
                    f"Comparison operator {ty2} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            return Call(expr_span, Op(op_span, op), [lhs, rhs], {})
        if isinstance(expr, py_ast.UnaryOp):
            lhs = self.compile_expr(expr.operand)
            op_span = expr_span.subtract(lhs.span)
            ty3 = type(expr.op)
            op = self._builtin_ops.get(ty3)
            if op is None:
                self.error(
                    f"Unary operator {ty3} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            return Call(expr_span, Op(op_span, op), [lhs], {})
        if isinstance(expr, py_ast.BoolOp):
            lhs = self.compile_expr(expr.values[0])
            rhs = self.compile_expr(expr.values[1])
            op_span = lhs.span.between(rhs.span)
            ty4 = type(expr.op)
            op = self._builtin_ops.get(ty4)
            if op is None:
                self.error(
                    f"Binary operator {ty4} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            call = Call(lhs.span.merge(rhs.span), Op(op_span, op), [lhs, rhs], {})
            for arg in expr.values[2:]:
                rhs = self.compile_expr(arg)
                call = Call(
                    call.span.merge(rhs.span),
                    Op(call.span.between(rhs.span), op),
                    [call, rhs],
                    {},
                )
            return call
        if isinstance(expr, py_ast.Subscript):
            lhs = self.compile_expr(expr.value)
            rhs = self.compile_slice(expr.slice)
            return Call(expr_span, Op(rhs.span, BuiltinOp.Subscript), [lhs, rhs], {})
        if isinstance(expr, py_ast.Tuple):
            return Tuple(expr_span, [self.compile_expr(x) for x in expr.elts])
        if isinstance(expr, py_ast.Dict):
            return DictLiteral(
                expr_span,
                [self.compile_expr(x) for x in expr.keys],  # type: ignore
                [self.compile_expr(x) for x in expr.values],
            )
        if isinstance(expr, py_ast.List):
            return ArrayLiteral(expr_span, [self.compile_expr(x) for x in expr.elts])

        self.error(f"Unexpected expression {type(expr)}", expr_span)
        return Expr(Span.invalid())

    def _compile_slice(self, slice: py_ast.slice) -> Expr:
        if isinstance(slice, py_ast.Slice):
            # TODO: handle spans for slice without start and end
            if slice.lower is None:
                start: Expr = Constant(Span.invalid(), 0)
            else:
                start = self.compile_expr(slice.lower)
            if slice.upper is None:
                end: Expr = Constant(Span.invalid(), -1)
            else:
                end = self.compile_expr(slice.upper)
            if slice.step is None:
                step: Expr = Constant(Span.invalid(), 1)
            else:
                step = self.compile_expr(slice.step)
            span = self.span_from_ast(slice)
            return Slice(span, start, step, end)
        # x[1:2, z] is an ExtSlice in the python ast
        if isinstance(slice, py_ast.ExtSlice):
            slices = [self._compile_slice(d) for d in slice.dims]
            return Tuple(Span.union([s.span for s in slices]), slices)
        if isinstance(slice, py_ast.Index):
            return self.compile_expr(slice.value)
        self.error(f"Unexpected slice type {type(slice)}", self.span_from_ast(slice))
        return Expr(Span.invalid())

    def compile_slice(self, slice: py_ast.slice) -> Tuple:
        # We ensure that slices are always a tuple
        s = self._compile_slice(slice)
        if isinstance(s, Tuple):
            return s
        return Tuple(s.span, [s])

    def compile_call(self, call: py_ast.Call) -> Call:
        kws: Dict[Expr, Expr] = dict()
        for x in call.keywords:
            if x.arg in kws:
                self.error(
                    f"Duplicate keyword argument {x.arg}", self.span_from_ast(x.value)
                )
            kws[Constant(self.span_from_ast(x), x.arg)] = self.compile_expr(x.value)

        func = self.compile_expr(call.func)

        args = []
        for arg in call.args:
            args.append(self.compile_expr(arg))

        return Call(self.span_from_ast(call), func, args, kws)

    def _expr2type(self, expr: Expr) -> Type:
        if isinstance(expr, Var):
            return TypeVar(expr.span, expr.id)
        if isinstance(expr, Constant):
            return TypeConstant(expr.span, expr.value)
        if isinstance(expr, Tuple):
            return TypeTuple(expr.span, expr.values)
        if isinstance(expr, Slice):
            return TypeSlice(
                expr.span,
                self._expr2type(expr.start),
                self._expr2type(expr.step),
                self._expr2type(expr.end),
            )
        if isinstance(expr, Call):
            if expr.func_name.name == BuiltinOp.Subscript:  # type: ignore
                assert isinstance(
                    expr.params[0], Var
                ), f"Expected subscript call to have lhs of Var, but it is {type(expr.params[0])}"
                assert isinstance(
                    expr.params[1], Tuple
                ), f"Expected subscript call to have rhs of Tuple, but it is {type(expr.params[1])}"
                return TypeApply(
                    expr.span,
                    expr.params[0].id,
                    [self._expr2type(x) for x in expr.params[1].values],
                )
        if isinstance(expr, Attr):
            return TypeAttr(expr.span, self._expr2type(expr.object), expr.field)
        self.error(f"Found unknown kind (type of type) {type(expr)}", expr.span)
        return Type(Span.invalid())

    def compile_type(self, ty: py_ast.expr) -> Type:
        return self._expr2type(self.compile_expr(ty))

    def compile_class(self, cls: py_ast.ClassDef) -> Class:
        span = self.span_from_ast(cls)
        funcs: Dict[Name, Function] = {}
        stmts: List[Assign] = []
        for stmt in cls.body:
            stmt_span = self.span_from_ast(stmt)
            if isinstance(stmt, py_ast.FunctionDef):
                f = self.compile_def(stmt)
                funcs[f.name] = f
            elif isinstance(stmt, (py_ast.AnnAssign, py_ast.Assign)):
                assign = self.compile_stmt(stmt)
                if isinstance(assign, Assign):
                    stmts.append(assign)
                else:
                    self.error(
                        f"Unexpected {type(assign)} in class definition. Only function definitions and assignments are allowed",
                        stmt_span,
                    )
            else:
                self.error(
                    "Only functions definitions and assignments are allowed within a class",
                    stmt_span,
                )
        return Class(span, cls.name, funcs, stmts)


def to_ast(
    program: Union[Any, str],
    diagnostic_ctx: DiagnosticContext,
    transformer: Optional[Transformer] = None,
) -> Any:
    """Parse an abstract syntax tree from a Python program.

    Examples
    --------

    :py:func:`to_ast` can be used with a given python function or class:

    .. code-block:: python

        import synr

        def my_function(x):
            return x + 2

        synr.to_ast(my_function, synr.PrinterDiagnosticContext())


    :py:func:`to_ast` can also be used with a string containing python code:

    .. code-block:: python

        import synr
        f = "def my_function(x):\\\\n    return x + 2"
        synr.to_ast(f, synr.PrinterDiagnosticContext())



    Parameters
    ----------
    program : Union[Any, str]
        A string containing a python program or a python function or class. If
        a python function or class is used, then line numbers will be localized
        to the file that the function or class came from.
    diagnostic_ctx : DiagnosticContext
        A diagnostic context to handle reporting of errors.
    transformer : Optional[Transformer]
        An optional transformer to apply to the AST after parsing. The
        transformer allows for converting from synr's AST to a user specified
        AST using the same diagnostic context and error handling.

    Returns
    -------
    Union[synr.ast.Module, Any]
        A synr AST if no transformer was specified and not errors occured. Or
        an error if one occured. Or the result of applying the transformer to
        the synr AST.
    """
    if isinstance(program, str):
        source_name = "<string input>"
        source = program
        full_source = source
        start_line = 1
    else:
        source_name = inspect.getsourcefile(program)  # type: ignore
        assert source_name, "source name must be valid"
        lines, start_line = inspect.getsourcelines(program)
        source = "".join(lines)
        mod = inspect.getmodule(program)
        if mod is not None:
            full_source = inspect.getsource(mod)
        else:
            full_source = source
    diagnostic_ctx.add_source(source_name, full_source)
    program_ast = py_ast.parse(source)
    compiler = Compiler(source_name, start_line, transformer, diagnostic_ctx)
    assert isinstance(program_ast, py_ast.Module), "Synr only supports module inputs"
    prog = compiler.compile_module(program_ast)
    err = diagnostic_ctx.render()
    if err is not None:
        return err
    if transformer is not None:
        transformed = transformer.do_transform(prog, diagnostic_ctx)
        err = diagnostic_ctx.render()
        if err is not None:
            return err
        return transformed
    else:
        return prog
