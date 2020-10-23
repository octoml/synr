Synr: A stable AST for python
=============================

Converting from the Python AST
------------------------------

.. autofunction:: synr.to_ast

The Synr AST
------------

.. automodule:: synr.ast
   :members:
   :undoc-members:
   :show-inheritance:

Error Handling
--------------

Synr uses a :py:class:`DiagnosticContext` to accumulate errors that occurred
during parsing. Multiple errors can be accumulated and they are all return
after parsing. :py:class:`DiagnosticContext` can be subclassed to customize
error handling behavior. We also provide a
:py:class:`PrinterDiagnosticContext`, which prints out errors as they occur.

.. autoclass:: synr.DiagnosticContext
   :members:
   :undoc-members:

.. autoclass:: synr.PrinterDiagnosticContext
   :show-inheritance:

Transforming the Synr AST to a Different Representation
-------------------------------------------------------

:py:func:`to_ast` allows you to transform a Synr AST into your desired
representation while using the existing Synr error handling infrastructure.
First implement a class that inherits from :py:class:`Transformer`, then
pass an instance of this class to :py:func:`to_ast`. :py:func:`to_ast` will
either return your converted class or an errors that occurred.

.. autoclass:: synr.Transformer
   :members:
