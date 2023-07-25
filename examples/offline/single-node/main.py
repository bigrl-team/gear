import argparse
import os
import gymnasium as gym
import deepspeed
import numpy as np
import time
from datetime import datetime
from typing import Union, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributions import Normal
from deepspeed.runtime.pipe.topology import (
    PipeDataParallelTopology,
    PipelineParallelGrid,
)
from torch.utils.tensorboard import SummaryWriter

import gear
from gear.mpu import ModelParallelismUnit

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shared_memory_seed",
    type=int,
    default=None,
    help="Seed to generate shared memory key SharedDataset, hash of local host name will be used if 'shared_memory_seed' is not set.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/tmp/gear/checkpoints/example_shared_dataset.pt",
    help="Seed to generate shared memory key SharedDataset, hash of local host name will be used if 'shared_memory_seed' is not set.",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank passed from distributed launcher",
)
parser.add_argument(
    "--enable-tensorboard",
    type=bool,
    default=True,
    help="Logging training losses to the tensorboard files",
)
parser.add_argument(
    "--tensorboard-logdir", type=str, default="./logs", help="Tensorboard logging path"
)
parser.add_argument(
    "--expr-name",
    type=str,
    default=datetime.now().strftime("%d-%m-%Y-%H:%M:%S"),
    help="Experiment name, used as logging file name",
)
parser = deepspeed.add_config_arguments(parser)


train_step: Callable = None
eval_step: Callable = None


def rank_0_evaluation(
    model,
    num_eval_trajectories,
    step_id,
    tensorboard_writer: Union[SummaryWriter, None],
):
    if dist.get_rank() != 0:
        dist.barrier()
    else:
        eval_ret = eval_step(model, num_eval_trajectories, step_id, tensorboard_writer)
        dist.barrier()
        return eval_ret


def rank_0_get_tensorboard_writer(args) -> Union[SummaryWriter, None]:
    if dist.get_rank() == 0 and args.enable_tensorboard:
        return SummaryWriter(
            log_dir=os.path.abspath(args.tensorboard_logdir) + f"/{args.expr_name}"
        )
    else:
        return None


def setup(args):
    global train_step
    global eval_step

    deepspeed.init_distributed(dist_init_required=True)
    gear.init()

    offline_loader_params = {
        "data_path": args.data_path,
        "mpu": None,
        "batch_size": 32,
        "sampling_method": "Uniform",
        "patterns": [
            {
                "name": "observations",
                "pad": "tail",  # Literal["head", "tail"]
                "offset": -1000,
                "length": 1000,  # fetch entire sequence for a trajectory no longer than 1000 steps
            },
            {
                "name": "actions",
                "pad": "tail",  # Literal["head", "tail"]
                "offset": -1000,
                "length": 1000,
            },
        ],
    }

    mpu = ModelParallelismUnit(device=torch.device("cuda"))
    mpu.build_ds_mpu()
    offline_loader_params["mpu"] = mpu
    offline_loader_params["attach"] = int(os.environ["LOCAL_RANK"]) != 0
    loader = gear.loader.OfflineLoader.create(**offline_loader_params)

    # setting up the MLP model
    table_spec = loader.table_spec
    in_dim = np.prod(
        table_spec.column_specs[table_spec.index("observations")].shape
    ).item()
    from models import mlp

    # print(f"Model configuration: input_dim {in_dim}")
    raw_model = mlp.model(in_dim=in_dim, hidden_dim=128, out_dim=3)
    train_step = mlp.train_step
    eval_step = mlp.eval_step

    model, optimizer, _, _ = deepspeed.initialize(
        args=args, model=raw_model, model_parameters=raw_model.parameters()
    )
    torch.cuda.synchronize()
    tensorboard_writer = rank_0_get_tensorboard_writer(args)

    return loader, model, optimizer, tensorboard_writer


def run(
    loader,
    model,
    optimizer,
    num_iter,
    tensorboard_writer: Union[SummaryWriter, None],
):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step_id in range(num_iter):
        train_step(loader, model, optimizer, step_id, tensorboard_writer)

        if step_id % 10 == 0:
            rank_0_evaluation(model, 100, step_id, tensorboard_writer)


if __name__ == "__main__":
    args = parser.parse_args()
    loader, model, optimizer, tbwriter = setup(args)
    run(
        loader=loader,
        model=model,
        optimizer=optimizer,
        num_iter=10000,
        tensorboard_writer=tbwriter,
    )

    if tbwriter:
        tbwriter.close()
