import os 
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store
import deepspeed

import gear
import libgear.core as glibc
import libgear.comm as glibcm
from gear.dataset import SharedDataset
from gear.specs import TableSpec
from gear.mpu import ModelParallelismUnit

if __name__ == "__main__":
    deepspeed.init_distributed()
    gear.init()

    default_store = _get_default_store()
    if dist.get_rank() == 0:
        nccl_id = glibcm.create_nccl_id()
        default_store.set(
            "NCCL_COMM_ID",
            f"{str(nccl_id)}",
        )
        dist.barrier()
    else:
        dist.barrier()
        nccl_id = eval(default_store.get("NCCL_COMM_ID"))
    print(f"RANK={dist.get_rank()} nccl_id {nccl_id}")

    table_spec_params = {
        "rank": dist.get_rank(),  # node rank
        "worldsize": dist.get_world_size(),  # node worldsize
        "trajectory_length": 1000,  # max episode steps
        "capacity": 32,  # max number of trajectories
        "column_specs": list(
            [
                {
                    "shape": (1000, 3),
                    "dtype": torch.float32,
                    "name": "mock_obs",
                }
            ]
        ),
    }

    local_dataset = SharedDataset.create(
        key=7, spec=TableSpec.create(**table_spec_params), create=True
    )

    for i in range(table_spec_params["capacity"]):
        local_dataset[i] = (
            1.0,
            100,
            [
                (i + table_spec_params["rank"] * table_spec_params["capacity"])
                * torch.ones(
                    (table_spec_params["trajectory_length"],)
                    + table_spec_params["column_specs"][0]["shape"],
                    dtype=table_spec_params["column_specs"][0]["dtype"],
                )
            ],
        )

    offline_loader_params = {
        "data_path": local_dataset,
        "mpu": None,
        "batch_size": 32,
        "sampling_method": "Uniform",
        "patterns": [
            {
                "name": "mock_obs",
                "pad": "tail",  # Literal["head", "tail"]
                "offset": -1000,
                "length": 1000,  # fetch entire sequence for a trajectory no longer than 1000 steps
            },
        ],
    }

    mpu = ModelParallelismUnit(device=torch.device("cuda"))
    mpu.build_ds_mpu()
    offline_loader_params["mpu"] = mpu
    offline_loader_params["attach"] = int(os.environ["LOCAL_RANK"]) != 0
    print(local_dataset._iset.timesteps)
    print(mpu.get_data_parallel_world_size())
    loader = gear.loader.DistributedOfflineLoader.create(**offline_loader_params)

    print(next(loader))
    print("xxxxxxxx")
    torch.cuda.synchronize()