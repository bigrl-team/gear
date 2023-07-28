"""
This file mainly define alias for the DataTypes that will be accepted by GEAR and their forward & backward conversions with PyTorch DataType.

"""

from typing import Dict

import libgear as glib
import torch

DataType = glib.DataType

"""
    Currently half types: int16 & float16 not supported
"""
CVT_DTYPES_TORCH_TO_GEAR: Dict[torch.dtype, glib.DataType] = {
    torch.bool: glib.bool,
    torch.uint8: glib.uint8,
    torch.int8: glib.int8,
    torch.int16: glib.int16,
    torch.int32: glib.int32,
    torch.int64: glib.int64,
    torch.float16: glib.float16,
    torch.float32: glib.float32,
    torch.float64: glib.float64,
}

CVT_DTYPES_GEAR_TO_TORCH: Dict[glib.DataType, torch.dtype] = {
    glib.bool: torch.bool,
    glib.uint8: torch.uint8,
    glib.int8: torch.int8,
    glib.short: torch.int16,
    glib.int16: torch.int16,
    glib.int: torch.int32,
    glib.int32: torch.int32,
    glib.long: torch.int64,
    glib.int64: torch.int64,
    glib.half: torch.float16,
    glib.float16: torch.float16,
    glib.float: torch.float32,
    glib.float32: torch.float32,
    glib.double: torch.float64,
    glib.float64: torch.float64,
}
