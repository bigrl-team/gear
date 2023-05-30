import deepspeed
import gear
import libgear.core as glibc
import torch
import torch.distributed as dist
from deepspeed.runtime.pipe.topology import (PipeDataParallelTopology,
                                             PipelineParallelGrid)
from gear.loader import OfflineLoader
from libgear.core import (ColumnSpec, SubscribePattern, TableSpec,
                          TrajectoryTable)

if __name__ == "__main__":
    deepspeed.init_distributed()
    import os

    glibc.init(int(os.environ["LOCAL_RANK"]))

    # print(f"=====>>>>>>>>RANK{dist.get_rank()} enter main body")
    num_dp = 2

    cspecs = [ColumnSpec(glibc.float32, [1])]
    history_length = 1
    capacity = 32
    batch_size = 4
    tspec = TableSpec(0, 1, history_length, num_dp * capacity, 1, cspecs)
    table = glibc.TrajectoryTable(tspec, 7, True)
    table.connect()

    mpu = PipelineParallelGrid(
        topology=PipeDataParallelTopology(num_dp=num_dp, num_pp=1)
    )
    # print(f"=====>>>>>>>>RANK{dist.get_rank()} build topology")
    dp_rank = mpu.get_data_parallel_id()
    mpu.device = torch.device(f"cuda:{dp_rank}")

    isrv = glibc.SharedMemoryIndexServer(42 + dp_rank, 0, capacity)
    rc = isrv.connect()
    # print(f"=====>>>>>>>>RANK{dist.get_rank()} build index table")
    patterns = [
        SubscribePattern(
            -history_length, history_length, SubscribePattern.PadOption.head
        )
    ]

    loader = gear.loader.OfflineLoader(
        trajectory_table=table,
        index_server=isrv,
        batch_size=batch_size,
        mpu=mpu,
        sampling_method="Uniform",
        patterns=patterns,
    )
    # print(f"=====>>>>>>>>RANK{dist.get_rank()} build offline loader")
    if dp_rank == 0:
        for tidx in range(num_dp * capacity):
            for col_id, cspec in enumerate(cspecs):
                byte_view = loader._handler.view(tidx, col_id)
                float32_view = byte_view.float32()
                tensor = glibc.Float32Span.to_tensor(float32_view)
                tensor.fill_(tidx + dp_rank * capacity)
                print(f"{tidx}-th traj filled")
                # print(
                #     tensor
                #     == glibc.Float32Span.to_tensor(
                #         loader._handler.view(tidx, col_id).float32()
                #     )
                # )
    # print(f"=====>>>>>>>>RANK{dist.get_rank()} begin data collecting")
    for i in range(1):
        data_batch = next(loader)

        torch.cuda.synchronize()
        # print("sadasdasdas")
        print(torch.sum(data_batch[0]))
