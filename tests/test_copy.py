import libgear.core as glibc
import torch


def test_copy():
    device = torch.device("cuda:0")
    column_shapes = (4, 3, 2)
    data = torch.ones(
        (
            32,
            100,
        )
        + column_shapes,
        dtype=torch.float32,
        device="cpu",
        pin_memory=True,
    )
    copied = torch.zeros((10,) + column_shapes, dtype=torch.float32, device=device)
    src_offset = torch.zeros(1, dtype=torch.long, device=device)
    dst_offset = torch.zeros(1, dtype=torch.long, device=device)
    stride = (
        copied.numel()
        * copied.element_size()
        * torch.ones(1, dtype=torch.long, device=device)
    )
    print(stride)
    glibc.copy_debug(data, copied, src_offset, dst_offset, stride, stride[0])
    assert torch.all(copied)
