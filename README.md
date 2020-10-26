# Synr

Synr is a library that provides a stable Abstract Syntax Tree for Python.

## Features

- The Synr AST does not change between Python versions.
- Every AST node contains line and column information.
- There is a single AST node for assignments (compared to three in Python's ast module).
- Support for returning multiple errors at once.
- Support for custom error reporting.

## Usage

```python
import synr

def test_program(x: int):
  return x + 2

# Parse a Python function into an AST
ast = synr.to_ast(test_program, synr.PrinterDiagnosticContext())
```

## Documentation

Please see [https://synr.readthedocs.io/en/latest/](https://synr.readthedocs.io/en/latest/) for documentation.
