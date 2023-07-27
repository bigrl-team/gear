import torch
import libgear as glib
import libgear.storage as glibs
from libgear.core import (ColumnSpec, SubscribePattern, TableSpec,
                          TrajectoryTable)


def test_sub():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cspecs = [ColumnSpec(glib.float32, [1, 2, 3, 4])]
    tspec = TableSpec(0, 1, 100, 32, 1, cspecs)
    # table used to subscribe test can be used without allocate
    table = TrajectoryTable(tspec, 7, True)
    handler = glibs.get_cpu_handler(table)
    patterns = [SubscribePattern(-1, 100, SubscribePattern.PadOption.head)]
    handler.set(patterns)

    indices = torch.arange(10, dtype=torch.long)
    timesteps = torch.arange(10, dtype=torch.long)
    lengths = torch.ones(10, dtype=torch.long) * 10
    column_id = 0

    src_ofsts = torch.zeros(10, dtype=torch.long)
    dst_ofsts = torch.zeros(10, dtype=torch.long)
    copy_lens = torch.zeros(10, dtype=torch.long)
    handler.sub(
        indices, timesteps, lengths, column_id, src_ofsts, dst_ofsts, copy_lens
    )
    mock_src = torch.ones(
        (
            32,
            100,
        )
        + tuple(cspecs[0].shape),
        dtype=torch.float32,
        device=device,
    )
    dst = torch.empty(
        (
            10,
            100,
        )
        + tuple(cspecs[0].shape),
        dtype=torch.float32,
        device=device,
    )
    glib.cuda.vcopy(
        glib.MemoryPtr(mock_src.data_ptr()),
        dst,
        src_ofsts.to(device),
        dst_ofsts.to(device),
        copy_lens.to(device),
        patterns[0].length * 96,
    )


if __name__ == "__main__":
    test_sub()
