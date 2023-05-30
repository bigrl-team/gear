import libgear.core as glibc
import torch

DataType = glibc.DataType

"""
    Currently half types: int16 & float16 not supported
"""
CVT_DTYPES_TORCH_TO_GEAR = {
    torch.bool: glibc.bool,
    torch.uint8: glibc.uint8,
    torch.int8: glibc.int8,
    torch.int16: glibc.int16,
    torch.int32: glibc.int32,
    torch.int64: glibc.int64,
    torch.float16: glibc.float16,
    torch.float32: glibc.float32,
    torch.float64: glibc.float64,
}

CVT_DTYPES_GEAR_TO_TORCH = {
    glibc.bool: torch.bool,
    glibc.uint8: torch.uint8,
    glibc.int8: torch.int8,
    glibc.short: torch.int16,
    glibc.int16: torch.int16,
    glibc.int: torch.int32,
    glibc.int32: torch.int32,
    glibc.long: torch.int64,
    glibc.int64: torch.int64,
    glibc.half: torch.float16,
    glibc.float16: torch.float16,
    glibc.float: torch.float32,
    glibc.float32: torch.float32,
    glibc.double: torch.float64,
    glibc.float64: torch.float64,
}
