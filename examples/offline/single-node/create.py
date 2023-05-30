import argparse
import os

import gear
import h5py
import torch
import numpy as np
from gear.dataset import SharedDataset
from gear.specs import TableSpec

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hdf5_data_url",
    type=str,
    default="http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5",
    help="D4RL's hopper-expert-v0 offline dataset.",
)
parser.add_argument(
    "--hdf5_data_path",
    type=str,
    default="/tmp/gear/datasets/example.hdf5",
    help="Path of the hdf5 dataset(folloing D4RL).",
)
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


def convert_d4rl_dataset(args):
    """
    convert open-source D4RL dataset(https://github.com/Farama-Foundation/D4RL) to SharedDataset
    The D4RL dataset is under the Creative Commons Attribution 4.0 License (CC BY)(https://creativecommons.org/licenses/by/4.0/)

    Here we use hopper-expert dataset as a minimal example.
    """

    if not os.path.exists(args.hdf5_data_path):
        print()
        gear.utils.download_from_url(args.hdf5_data_url, args.hdf5_data_path)

    data = None
    with h5py.File(args.hdf5_data_path, "r") as hf:
        data = gear.utils.load_hdf5_dataset(hf)
    for column_key in data.keys():
        for trajectory_id in range(len(data[column_key])):
            data[column_key][trajectory_id] = torch.from_numpy(
                data[column_key][trajectory_id]
            )

    num_trajectory = len(data["observations"])
    table_spec_params = {
        "rank": 0,  # node rank
        "worldsize": 1,  # node worldsize
        "trajectory_length": 1000,  # max episode steps
        "capacity": num_trajectory,  # max number of trajectories
        "column_specs": list(
            [
                {
                    "shape": data[k][0].shape[1:],
                    "dtype": data[k][0].dtype,
                    "name": str(k),
                }
                for k in data.keys()
            ]
        ),
    }

    dataset = SharedDataset.create(
        key=args.shared_memory_seed,
        spec=TableSpec.create(**table_spec_params),
        create=True,
    )
    column_keys = list(data.keys())
    for trajectory_id in range(num_trajectory):
        columns = list(
            data[column_keys[cid]][trajectory_id] for cid in range(len(column_keys))
        )
        weight = 1.0
        timestep = columns[0].shape[0]
        dataset[trajectory_id] = weight, timestep, columns
        assert dataset[trajectory_id].weight == weight
        assert dataset[trajectory_id].timestep == timestep
    return dataset


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = convert_d4rl_dataset(args)

    dataset.checkpoint(args.data_path)
    import pickle as pkl

    with open("/tmp/gear/checkpoints/iset.pt", "wb") as f:
        pkl.dump(dataset._iset.get_state(), f)
