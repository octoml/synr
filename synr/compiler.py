from __future__ import annotations

import attr
import ast as py_ast
import inspect
from typing import Optional, Any, List, Union, Sequence

from .ast import *
from .diagnostic_context import DiagnosticContext
from .transformer import Transformer


class Compiler:
    start_line: int
    transformer: Optional[Transformer]
    diagnostic_ctx: DiagnosticContext

    builtin_ops = {
        py_ast.Add: BuiltinOp.Add,
        py_ast.Sub: BuiltinOp.Sub,
        py_ast.Mult: BuiltinOp.Mul,
        py_ast.Div: BuiltinOp.Div,
        py_ast.FloorDiv: BuiltinOp.FloorDiv,
        py_ast.Mod: BuiltinOp.Mod,
        py_ast.Eq: BuiltinOp.Eq,
        py_ast.GtE: BuiltinOp.GE,
        py_ast.LtE: BuiltinOp.LE,
        py_ast.Gt: BuiltinOp.GT,
        py_ast.Lt: BuiltinOp.LT,
        py_ast.Not: BuiltinOp.Not,
        py_ast.Or: BuiltinOp.Or,
        py_ast.And: BuiltinOp.And,
    }

    def __init__(
        self,
        start_line: int,
        transformer: Optional[Transformer],
        diagnostic_ctx: DiagnosticContext,
    ):
        self.start_line = 0  # FIXME: start_line
        self.transformer = transformer
        self.diagnostic_ctx = diagnostic_ctx

    def error(self, message, span):
        self.diagnostic_ctx.emit("error", message, span)

    def span_from_ast(self, node: py_ast.AST) -> Span:
        if isinstance(node, py_ast.withitem):
            span = Span.from_ast(node.context_expr)
            if node.optional_vars:
                end_span = Span.from_ast(node.optional_vars)
                span = span.merge(end_span)
        else:
            span = Span.from_ast(node)
        span.start_line += self.start_line
        span.end_line += self.start_line
        return span

    def span_from_asts(self, nodes: Sequence[py_ast.AST]) -> Span:
        assert len(nodes) > 0
        span = self.span_from_ast(nodes[0])
        for node in nodes[1:]:
            span = span.merge(self.span_from_ast(node))
        return span

    def compile_module(self, program: py_ast.Module) -> Any:
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

        if len(args.posonlyargs):
            self.error(
                "currently synr only supports non-position only arguments",
                self.span_from_asts(args.posonlyargs),
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
            params.append(Parameter(span, arg.arg, None))

        return params

    def compile_def(self, stmt: py_ast.stmt) -> Function:
        stmt_span = self.span_from_ast(stmt)
        if isinstance(stmt, py_ast.FunctionDef):
            name = stmt.name
            args = stmt.args
            params = self.compile_args_to_params(args)
            body = self.compile_block(stmt.body)
            return Function(stmt_span, name, params, Type(Span.invalid()), body)
        else:
            self.error(f"found {type(stmt)}", stmt_span)
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
        elif isinstance(stmt, py_ast.Assign):
            if len(stmt.targets) > 1:
                self.error("Multiple assignment is not supported", stmt_span)
            lhs = self.compile_expr(stmt.targets[0])
            if not isinstance(lhs, Var):
                self.error(
                    "Left hand side of assignment must be a variable",
                    self.span_from_ast(stmt.targets[0]),
                )
                lhs = Var(Span.invalid(), Id.invalid())
            rhs = self.compile_expr(stmt.value)
            return Assign(stmt_span, lhs, rhs)
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
            if wth.optional_vars:
                rhs = self.compile_expr(wth.optional_vars)
                if not isinstance(rhs, Var):
                    self.error(
                        "Right hand side of with statement (y in `with x as y:`) must be a variable",
                        self.span_from_ast(wth.optional_vars),
                    )
                    rhs = Var.invalid()
            else:
                self.error(
                    "With statement must contain an `as _`", self.span_from_ast(wth)
                )
                rhs = Var.invalid()
            lhs = self.compile_expr(wth.context_expr)
            body = self.compile_block(stmt.body)
            return With(self.span_from_ast(stmt), lhs, rhs, body)
        elif isinstance(stmt, py_ast.If):
            true = self.compile_block(stmt.body)
            if len(stmt.orelse) > 0:
                false = self.compile_block(stmt.orelse)
            else:
                false = Block(stmt_span, [])  # TODO: this span isn't correct
            condition = self.compile_expr(stmt.test)
            return If(stmt_span, condition, true, false)
        else:
            self.error(f"Found unexpected {type(stmt)} when compiling stmt", stmt_span)
            return Stmt(Span.invalid())

    def compile_var(self, expr: py_ast.expr) -> Var:
        expr_span = self.span_from_ast(expr)
        if isinstance(expr, py_ast.Name):
            return Var(expr_span, Id([expr.id]))
        if isinstance(expr, py_ast.Attribute):
            sub_var = self.compile_var(expr.value)
            names = sub_var.name.names
            names.append(expr.attr)
            return Var(expr_span, Id(names))
        self.error("Expected a variable name of the form a.b.c", expr_span)
        return Var.invalid()

    def compile_expr(self, expr: py_ast.expr) -> Expr:
        expr_span = self.span_from_ast(expr)
        if isinstance(expr, py_ast.Name) or isinstance(expr, py_ast.Attribute):
            return self.compile_var(expr)
        if isinstance(expr, py_ast.Constant):
            if isinstance(expr.value, float) or isinstance(expr.value, int):
                return Constant(expr_span, expr.value)
            self.error("Only float and int constants are allowed", expr_span)
        if isinstance(expr, py_ast.Call):
            return self.compile_call(expr)
        if isinstance(expr, py_ast.BinOp):
            lhs = self.compile_expr(expr.left)
            rhs = self.compile_expr(expr.right)
            op_span = lhs.span.between(rhs.span)
            ty = type(expr.op)
            op = self.builtin_ops.get(ty)
            if op is None:
                self.error(
                    f"Binary operator {ty} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            return Call(expr_span, Op(op_span, op), [lhs, rhs])
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
            ty = type(expr.ops[0])
            # Desugar `a != b` to `not (a == b)`
            if ty == py_ast.NotEq:
                return Call(
                    expr_span,
                    Op(op_span, BuiltinOp.Not),
                    [Call(expr_span, Op(op_span, BuiltinOp.Eq), [lhs, rhs])],
                )
            op = self.builtin_ops.get(ty)
            if op is None:
                self.error(
                    f"Binary operator {ty} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            return Call(expr_span, Op(op_span, op), [lhs, rhs])
        if isinstance(expr, py_ast.UnaryOp):
            lhs = self.compile_expr(expr.operand)
            op_span = expr_span.subtract(lhs.span)
            ty = type(expr.op)
            op = self.builtin_ops.get(ty)
            if op is None:
                self.error(
                    f"Binary operator {ty} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            return Call(expr_span, Op(op_span, op), [lhs])
        if isinstance(expr, py_ast.BoolOp):
            lhs = self.compile_expr(expr.values[0])
            rhs = self.compile_expr(expr.values[1])
            op_span = lhs.span.between(rhs.span)
            ty = type(expr.op)
            op = self.builtin_ops.get(ty)
            if op is None:
                self.error(
                    f"Binary operator {ty} is not supported",
                    op_span,
                )
                op = BuiltinOp.Invalid
            call = Call(lhs.span.merge(rhs.span), Op(op_span, op), [lhs, rhs])
            for arg in expr.values[2:]:
                rhs = self.compile_expr(arg)
                call = Call(
                    call.span.merge(rhs.span),
                    Op(call.span.between(rhs.span), op),
                    [call, rhs],
                )
            return call

        self.error(f"Unexpected expression {type(expr)}", expr_span)
        return Expr(Span.invalid())

    def compile_call(self, call: py_ast.Call) -> Call:
        if len(call.keywords) > 0:
            self.error(
                "Keyword arguments are not allowed", self.span_from_asts(call.keywords)
            )

        func = self.compile_expr(call.func)
        if func is not None and not isinstance(func, Var):
            self.error(
                f"Expected function name, but got {type(func)}",
                self.span_from_ast(call.func),
            )
            func = Var.invalid()

        args = []
        for arg in call.args:
            args.append(self.compile_expr(arg))

        return Call(self.span_from_ast(call), func, args)

    def compile_class(self, cls: py_ast.ClassDef) -> Class:
        span = self.span_from_ast(cls)
        funcs: Dict[Name, Function] = {}
        for func in cls.body:
            func_span = self.span_from_ast(func)
            f = self.compile_def(func)
            if not isinstance(f, Function):
                self.error(
                    "Only functions definitions are allowed within a class", func_span
                )
            funcs[f.name] = f
        return Class(span, funcs)


def to_ast(
    program: Any,
    diagnostic_ctx: DiagnosticContext,
    transformer: Optional[Transformer] = None,
):
    source_name = inspect.getsourcefile(program)
    assert source_name, "source name must be valid"
    lines, start_line = inspect.getsourcelines(program)
    source = "".join(lines)
    diagnostic_ctx.add_source(source_name, source)
    program_ast = py_ast.parse(source)
    compiler = Compiler(start_line, transformer, diagnostic_ctx)
    assert isinstance(program_ast, py_ast.Module), "support module"
    prog = compiler.compile_module(program_ast)
    diagnostic_ctx.diag_ctx.render()
    return prog
