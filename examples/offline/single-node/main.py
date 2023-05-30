import argparse
import os
import gymnasium as gym
import deepspeed
import numpy as np
import time
from datetime import datetime
from typing import Union
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

from models import MLP

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


def rank_0_evaluation(env, model, num_eval_trajectories):
    if dist.get_rank() != 0:
        dist.barrier()
    else:
        model.eval()
        rsums = []
        for _ in range(num_eval_trajectories):
            obs, info = env.reset()
            rsum = 0
            while True:
                # print(obs)
                mean, sigma = model(torch.from_numpy(obs).cuda().float())
                distri = Normal(mean, sigma)

                act = distri.rsample()
                obs, reward, terminated, truncated, info = env.step(
                    act.detach().cpu().numpy()
                )
                rsum += reward
                if terminated or truncated:
                    rsums.append(rsum)
                    break
        model.train()
        dist.barrier()
        return np.mean(rsums)


def rank_0_get_tensorboard_writer(args) -> Union[SummaryWriter, None]:
    if dist.get_rank() == 0 and args.enable_tensorboard:
        return SummaryWriter(
            log_dir=os.path.abspath(args.tensorboard_logdir) + f"/{args.expr_name}"
        )
    else:
        return None


def setup(args):
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
    # print(f"Model configuration: input_dim {in_dim}")
    raw_model = MLP(in_dim=in_dim, hidden_dim=128, out_dim=3)
    model, optimizer, _, _ = deepspeed.initialize(
        args=args, model=raw_model, model_parameters=raw_model.parameters()
    )
    torch.cuda.synchronize()
    tensorboard_writer = rank_0_get_tensorboard_writer(args)

    return loader, model, optimizer, tensorboard_writer


def generate_padding_mask(max_length):
    return torch.cat(
        [
            torch.zeros(1, max_length, dtype=torch.int32),
            torch.tril(torch.ones(max_length, max_length, dtype=torch.int32)),
        ]
    )


def run(
    loader,
    model,
    optimizer,
    num_iter,
    tensorboard_writer: Union[SummaryWriter, None],
):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mask = generate_padding_mask(max_length=1000).cuda()
    env = gym.make("Hopper-v4") if dist.get_rank() == 0 else None
    for i in range(num_iter):
        model.zero_grad()

        timesteps, data_batch = next(loader)
        obs = data_batch[0]
        act = data_batch[1]
        # print(obs.shape, data_batch)
        mean, sigma = model(obs)
        distri = Normal(mean, sigma)
        loss = -distri.log_prob(act)
        loss = torch.sum(loss.mean(-1) * mask[timesteps]) / timesteps.sum()
        # print(loss, timesteps)

        # print(loss.detach().cpu().numpy())
        model.backward(loss)
        model.step()

        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                tag="training-losses", scalar_value=loss.item(), global_step=i
            )

        if i % 10 == 0:
            eval_reward = rank_0_evaluation(env, model, 100)
            if eval_reward:
                print(f"Iteration: {i} evalution reward {eval_reward.item()}")
                if tensorboard_writer:
                    tensorboard_writer.add_scalar(
                        tag="evaluation-rewards",
                        scalar_value=eval_reward.item(),
                        global_step=i,
                    )


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
