import os
from collections import namedtuple
from itertools import product
from typing import Sequence, Union
import logging

import torch
import torch.distributed as dist

# DeepSpeed reference: https://github.com/microsoft/DeepSpeed/blob/3f5e4931098bf533f8217afb6d986c90f81aed80/deepspeed/runtime/pipe/topology.py


axes = ["pipe", "data"]
ProcessCoord = namedtuple("ProcessCoord", axes)
ProcessCoord.__doc__ = """
    A namedtuple describing the 2-dimensional process topology.

    The implementation is copied from DeepSpeed.
    DeepSpeed Reference: https://github.com/microsoft/DeepSpeed/blob/3f5e4931098bf533f8217afb6d986c90f81aed80/deepspeed/runtime/pipe/topology.py
    
    ProcesseTopology: [pipe, data]

    :type data: int
    :param data:
        Data parallel group size.

    :type pipe: int
    :param pipe:
        pipe parallel group size.

    .. code-block:: python

        pcoord = ProcessCoord(3, 5)
        pcoord = ProcessCoord(pipe=3, data=5)

"""


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
    """
    A class describe the possible 2D parallelism topology of current launched task, similar to DeepSpeed's ``ParallelGrid`` class. ModelParallelismUnit(MPU) is a commonly used technical term. GEAR relies heavily on parallelism schemas(also known as the term ProcessTopology) to address trajectory data. DeepSpeed seems to have different launching strategies for single-processing and multi-processing tasks, which may cause inconsistent logics in parallelism detecting and address translating. Hence we implemente the class to serve as an abstraction layer.

    :type proc_coord: Union[ProcessCoord, dict, Sequence[int], None]
    :param proc_coord:
        Describe the parallelism schema, set to ``ProcessCoord(pipe=1,data=1)`` if not specified.
        
        .. seealso::
         
           :py:class:`gear.mpu.ProcessCoord`. 

    :type global_rank: Union[int, None]
    :param global_rank:
        Maunally set global rank of the current process or auto detect by :py:func:`torch.distributed.get_rank`. If no integer provided and the the default torch process group not initialized(run with single card), 0 is set.

    :type local_rank: Union[int, None]
    :param local_rank:
        Local rank refers to the rank of the current process within the process group that consists of all processes located on the current local node. Since existing Deep Learning frameworks(such as PyTorch and DeepSpeed) usually adopt a SPMD paradigm for parallelism, in which each training process has exclusive ownership of a GPU device, ``local_rank`` also used as the device rank. Fallback to 0 if auto-detect failed without specification.

    :type device: Union[torch.device, None]
    :param device:
        Set the device in the cuda context. Fallback to cuda:{local_rank} if not specified.

    .. code-block:: python

        # run on single GPU device
        mpu = ModelParallelismUnit()

        # run with a data parallel degree of 2
        mpu = ModelParallelismUnit([1, 2])
        mpu = ModelParallelismUnit({"pipe":1, "data":2})

    """

    def __init__(
        self,
        proc_coord: Union[ProcessCoord, dict, Sequence[int], None] = None,
        global_rank: Union[int, None] = None,
        local_rank: Union[int, None] = None,
        device: Union[torch.device, None] = None,
    ) -> None:
        DEFAULT_WORLD_SIZE = 1
        DEFAULT_LOCAL_WORLD_SIZE = torch.cuda.device_count()
        try:
            DEFAULT_WORLD_SIZE = dist.get_world_size()
        except RuntimeError as e:
            logging.info("Parallism detect failed, world size set to 1")

        DEFAULT_GLOBAL_RANK = 0
        try:
            DEFAULT_GLOBAL_RANK = dist.get_rank()
        except RuntimeError as e:
            logging.info("Parallism detect failed, global rank set to 0")

        if proc_coord is None:
            proc_coord = ProcessCoord(pipe=1, data=DEFAULT_WORLD_SIZE)
        elif isinstance(proc_coord, Sequence):
            proc_coord = ProcessCoord(*proc_coord)
        elif isinstance(proc_coord, dict):
            proc_coord = ProcessCoord(**proc_coord)

        self.coord = proc_coord
        self.global_rank = global_rank if global_rank else DEFAULT_GLOBAL_RANK

        self.local_rank = (
            local_rank
            if local_rank
            else (int(os.environ["LOCAL_RANK"]) if os.environ["LOCAL_RANK"] else 0)
        )
        print(self.global_rank, self.local_rank)
        self.local_world = DEFAULT_LOCAL_WORLD_SIZE
        self.local_proc_ranks = list(
            range(
                self.global_rank - self.local_rank,
                self.global_rank - self.local_rank + self.local_world,
            )
        )
        self.local_proc_identifiers = [False] * get_world_size(proc_coord)
        for rank in self.local_proc_ranks:
            self.local_proc_identifiers[rank] = True

        self.device = device if device else torch.device(f"cuda:{self.local_rank}")

        self.ds_mpu = None

    def build_ds_mpu(self) -> None:
        """
        Trying to import and generate a DeepSpeed MPU(ParallelGrid) and set it as ``mpu.ds_mpu`` member. An easy-to-use interface when being integreted with DeepSpeed.
        """
        import deepspeed
        from deepspeed.runtime.pipe.topology import (
            PipeDataParallelTopology,
            PipelineParallelGrid,
        )

        self.ds_mpu = PipelineParallelGrid(
            PipeDataParallelTopology(num_dp=self.coord.data, num_pp=self.coord.pipe)
        )
        return self.ds_mpu

    def get_global_rank(self) -> int:
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

    def get_stage_id(self) -> int:
        return self.ds_mpu.get_stage_id()

    def get_pipe_parallel_rank(self) -> int:
        """The stage of the pipeline this rank resides in."""
        return self.ds_mpu.get_pipe_parallel_rank()

    def get_pipe_parallel_world_size(self) -> int:
        """The number of stages in the pipeline."""
        return self.ds_mpu.get_pipe_parallel_world_size()

    def get_pipe_parallel_group(self) -> int:
        """The group of ranks within the same pipeline."""
        return self.ds_mpu.get_pipe_parallel_group()

    def get_data_parallel_rank(self) -> int:
        """Which pipeline this rank resides in."""
        return self.ds_mpu.get_data_parallel_rank()

    def get_data_parallel_world_size(self) -> int:
        """The number of pipelines."""
        return self.ds_mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self) -> int:
        """The group of ranks within the same stage of all pipelines."""
        return self.ds_mpu.get_data_parallel_group()

    def is_on_the_same_node(self, global_rank: int) -> bool:
        """Determine whether a process with certain global rank locates on the same node"""
        return self.local_proc_identifiers[global_rank]

    def get_local_group_ranks(self) -> Sequence[int]:
        """Get global ranks of all local processes"""
        return self.local_proc_ranks
