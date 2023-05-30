from typing import Sequence

from .dtypes import DataType

class ColumnSpec:
    shape: Sequence[int]
    dtype: DataType

    def __init__(self, shape: Sequence[int], dtype: DataType) -> ColumnSpec: ...
    def size(self) -> int: ...

class TableSpec:
    rank: int
    worldsize: int
    trajectory_length: int
    capacity: int
    def __init__(
        self,
        rank: int,
        worldsize: int,
        trajectory_length: int,
        capacity: int,
        num_columns: int,
        column_specs: Sequence[ColumnSpec],
    ) -> TableSpec: ...
    @property
    def num_columns(self) -> int: ...
    @property
    def column_specs(self) -> Sequence[ColumnSpec]: ...
    def size(self) -> int: ...
