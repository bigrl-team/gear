import os
from collections import namedtuple
from itertools import product
from typing import Sequence, Union
import logging

import torch
import torch.distributed as dist

# DeepSpeed reference: https://github.com/microsoft/DeepSpeed/blob/3f5e4931098bf533f8217afb6d986c90f81aed80/deepspeed/runtime/pipe/topology.py

DEFAULT_WORLD_SIZE = 1
DEFAULT_LOCAL_WORLD_SIZE = torch.cuda.device_count()
try:
    DEFAULT_WORLD_SIZE = dist.get_world_size()
except RuntimeError as e:
    logging.info("Parallism detect failed, world size set to 1")

axes = ["pipe", "data"]
ProcessCoord = namedtuple("ProcessCoord", axes)


class Grid:
    def __init__(self, num_pp: int, num_dp: int) -> None:
        import torch.distributed as dist
        from torch.distributed import ProcessGroup

        ws = num_pp * num_dp
        self._global_rank_to_parallel_coord: Sequence[ProcessCoord] = [None] * ws
        self._global_rank_to_data_parallel_group_ranks: Sequence[Sequence[int]] = [
            None
        ] * ws
        self._global_rank_to_data_parallel_group: Sequence[ProcessGroup] = [None] * ws
        self._global_rank_to_pipe_parallel_group_ranks: Sequence[Sequence[int]] = [
            None
        ] * ws
        self._global_rank_to_pipe_parallel_group: Sequence[ProcessGroup] = [None] * ws

        for global_rank, coord in enumerate(product(range(num_pp), range(num_dp))):
            self._global_rank_to_parallel_coord[global_rank] = ProcessCoord(
                {axis_name: axis_rank for axis_name, axis_rank in zip(axes, coord)}
            )

        data_parallel_group_ranks = [None] * num_pp
        for stage_id in range(num_dp):
            data_parallel_group_ranks[stage_id] = list(
                [
                    rank
                    for rank in range(ws)
                    if self._global_rank_to_parallel_coord[rank].pipe == stage_id
                ]
            )


def get_world_size(coord: ProcessCoord) -> int:
    return coord.data * coord.pipe


class ModelParallelismUnit:
    def __init__(
        self,
        proc_coord: Union[ProcessCoord, dict, Sequence[int]] = ProcessCoord(
            pipe=1, data=DEFAULT_WORLD_SIZE
        ),
        global_rank: Union[int, None] = None,
        local_rank: Union[int, None] = None,
        device: Union[torch.device, None] = None,
    ) -> None:
        if isinstance(proc_coord, Sequence):
            proc_coord = ProcessCoord(*proc_coord)
        elif isinstance(proc_coord, dict):
            proc_coord = ProcessCoord(**proc_coord)

        self.coord = proc_coord
        self.global_rank = global_rank if global_rank else dist.get_rank()
        self.local_rank = local_rank if local_rank else os.environ["LOCAL_RANK"]
        self.local_world = DEFAULT_LOCAL_WORLD_SIZE
        self.local_proc_ranks = list(
            range(
                self.global_rank - local_rank,
                self.global_rank - local_rank + self.local_world,
            )
        )
        self.local_proc_identifiers = [False] * get_world_size(proc_coord)
        for rank in self.local_proc_ranks:
            self.local_proc_identifiers[rank] = True

        self.device = device if device else torch.device(f"cuda:{self.local_rank}")

        self.ds_mpu = None

    def build_ds_mpu(self):
        import deepspeed
        from deepspeed.runtime.pipe.topology import (
            PipeDataParallelTopology,
            PipelineParallelGrid,
        )

        self.ds_mpu = PipelineParallelGrid(
            PipeDataParallelTopology(num_dp=self.coord.data, num_pp=self.coord.pipe)
        )
        return self.ds_mpu

    def get_global_rank(self):
        if self.ds_mpu:
            return self.ds_mpu.get_global_rank()
        else:
            return dist.get_rank()

    def get_model_parallel_rank(self) -> int:
        """
        Not support yet, always return 0
        """
        return 0

    def get_model_parallel_world_size(self) -> int:
        """
        Not support yet, always return 1
        """
        return 1

    def get_stage_id(self):
        return self.ds_mpu.get_stage_id()

    def get_pipe_parallel_rank(self):
        """The stage of the pipeline this rank resides in."""
        return self.ds_mpu.get_pipe_parallel_rank()

    def get_pipe_parallel_world_size(self):
        """The number of stages in the pipeline."""
        return self.ds_mpu.get_pipe_parallel_world_size()

    def get_pipe_parallel_group(self):
        """The group of ranks within the same pipeline."""
        return self.ds_mpu.get_pipe_parallel_group()

    def get_data_parallel_rank(self):
        """Which pipeline this rank resides in."""
        return self.ds_mpu.get_data_parallel_rank()

    def get_data_parallel_world_size(self):
        """The number of pipelines."""
        return self.ds_mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        """The group of ranks within the same stage of all pipelines."""
        return self.ds_mpu.get_data_parallel_group()

    def is_on_the_same_node(self, global_rank: int) -> bool:
        return self.local_proc_identifiers[global_rank]

    def get_local_group_ranks(self) -> Sequence[int]:
        return self.local_proc_ranks
