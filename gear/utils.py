import os
import pathlib
import socket
import pprint

import urllib
from typing import Union

from tqdm import tqdm, trange
import numpy as np


def ensure_path(path: Union[str, pathlib.Path]) -> None:
    path = pathlib.Path(path)
    if not path.parent.exists():
        os.makedirs(path.parent)


def get_local_node_hashed_int32() -> int:
    from .config import KEY_T_RANGE

    hash_key = hash(socket.gethostname()) % KEY_T_RANGE
    return hash_key


def download_from_url(url: str, path: str):
    class DownloadProgressBar(tqdm):
        block_count = 0

        def hook(self, num_block=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((num_block - self.block_count) * block_size)
            self.block_count = num_block
    
    ensure_path(path)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=f"Downloading dataset from {url}") as pbar:
        urllib.request.urlretrieve(url, path, pbar.hook)
    if not os.path.exists(path):
        raise IOError(f"Failed to download dataset from {url} to {path}")


def load_hdf5_dataset(hf):
    """
    reference: https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L80
    """
    must_have_keys = ["observations", "actions", "rewards", "terminals"]
    assert np.all([k in hf.keys() for k in must_have_keys]), "Key(s) missing."
    data_dict = {key: hf.get(key) for key in hf.keys()}

    total_steps = data_dict[must_have_keys[0]].shape[0]

    built_dict = {key: [] for key in hf.keys()}
    prev_terminal_step = 0
    for step_id in trange(0, total_steps, 1, desc="loading dataset ..."):
        if data_dict["terminals"][step_id] or data_dict["timeouts"][step_id]:
            for k in built_dict.keys():
                built_dict[k].append(data_dict[k][prev_terminal_step:step_id])
            prev_terminal_step = step_id

    num_trajectory = len(built_dict[must_have_keys[0]])
    max_traj_len = np.max(
        [built_dict[must_have_keys[0]][tid].shape[0] for tid in range(num_trajectory)]
    )
    expected_mem_consumption = (
        max_traj_len
        * num_trajectory
        * np.sum([np.prod(built_dict[ckey][0].shape[1:]) for ckey in built_dict.keys()])
    )
    dataset_info = {
        "num_trajectory": num_trajectory,
        "total_steps": total_steps,
        "avg_trajectory_length": total_steps / num_trajectory,
        "max_trajectory_length": max_traj_len,
        "expected_memory_consumption": expected_mem_consumption,
    }
    pprint.pprint(dataset_info)
    return built_dict
