``libgear``
===========================

.. autofunction:: libgear.print_environ

.. autoclass:: libgear.DataType

.. autoclass:: libgear.Range



``libgear.*Span``
------------------

The ``libgear.Span`` class serve as a intermediate data interface for sharing memory blocks with external libraries (combined with data type and size information). Construction and forwarding ``libgear.span`` in function calls will not lead to memory copy, nor any changes in ownership and memory region life span, should be used with consideration.

.. autoclass:: libgear.Uint8Span
    :members

.. autoclass:: libgear.BoolSpan

.. autoclass:: libgear.Int8Span

.. autoclass:: libgear.Int16Span

.. autoclass:: libgear.Int32Span

.. autoclass:: libgear.Int64Span

.. autoclass:: libgear.Float32Span

.. autoclass:: libgear.Float64Span
