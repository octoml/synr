"""
The goal of synr is to provide a syntactic representation
for the core of the Python programming language for building
embedded domain specific langauges.

Synr provides
- A python version agnostic syntax tree.
- Line and column information for every node in the AST.
"""

__version__ = "0.1.0"

from .compiler import to_ast
from .diagnostic_context import PrinterDiagnosticContext, DiagnosticContext
from .transformer import Transformer
