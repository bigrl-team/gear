import torch

torch_dtype_to_int = {
    torch.float: 0,
    torch.float32: 0,
    torch.float64: 1,
    torch.double: 1,
    torch.complex64: 2,
    torch.cfloat: 2,
    torch.complex128: 3,
    torch.cdouble: 3,
    torch.float16: 4,
    torch.half: 4,
    torch.bfloat16: 5,
    torch.uint8: 6,
    torch.int8: 7,
    torch.int16: 8,
    torch.short: 8,
    torch.int32: 9,
    torch.int: 9,
    torch.int64: 10,
    torch.long: 10,
    torch.bool: 11,
}

torch_int_to_dtype = {
    0: torch.float32,
    1: torch.float64,
    2: torch.complex64,
    3: torch.complex128,
    4: torch.float16,
    5: torch.bfloat16,
    6: torch.uint8,
    7: torch.int8,
    8: torch.int16,
    9: torch.int32,
    10: torch.int64,
    11: torch.bool,
}
