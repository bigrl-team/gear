from typing import Any, Dict, Sequence, Union

import torch
import libgear.storage as glibs
from gear.dtypes import CVT_DTYPES_TORCH_TO_GEAR, DataType


class ColumnSpec:
    @staticmethod
    def create(
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
        name: str = "",
        *args,
        **kwargs
    ):
        if isinstance(dtype, DataType):
            dtype = dtype
        else:
            dtype = CVT_DTYPES_TORCH_TO_GEAR[dtype]
        return glibs.ColumnSpec(shape, dtype, name)


class TableSpec:
    @staticmethod
    def create(
        rank: int = 0,
        worldsize: int = 1,
        trajectory_length: int = 100,
        capacity: int = 32,
        column_specs: Union[Sequence[Dict[str, Any]], glibs.ColumnSpec] = None,
        *args,
        **kwargs
    ):
        num_columns = 0
        cspecs = []
        if column_specs is not None:
            num_columns = len(column_specs)
            for column_spec in column_specs:
                if isinstance(column_spec, glibs.ColumnSpec):
                    cspecs.append(column_spec)
                else:
                    cspecs.append(ColumnSpec.create(**column_spec))
        return glibs.TableSpec(
            rank, worldsize, trajectory_length, capacity, num_columns, cspecs
        )
