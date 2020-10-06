"""
The goal of synr is to provide a syntactic representation
for the core of the Python programming language for building
embedded domain specific langauges.
"""

__version__ = "0.1.0"

from .compiler import to_ast
from .diagnostic_context import PrinterDiagnosticContext, DiagnosticContext
