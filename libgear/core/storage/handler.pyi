from abc import abstractmethod
from enum import Enum
from typing import Sequence

import torch

from .table import TrajectoryTable

class SubscribePattern:
    class PadOption(Enum):
        inactive = ...
        head = ...
        tail = ...
    def __init__(self, offset: int, length: int) -> SubscribePattern: ...
    @property
    def offset(self) -> int: ...
    @property
    def length(self) -> int: ...

class TrajectoryHandler:
    @abstractmethod
    def set(self):
        raise NotImplementedError
    @abstractmethod
    def sub(self):
        raise NotImplementedError

class CpuTrajectoryStorageHandler(TrajectoryHandler):
    def set(self, patterns=Sequence[SubscribePattern]) -> None: ...
    def sub(
        self,
        indices: torch.LongTensor,
        timesteps: torch.LongTensor,
        lengths: torch.LongTensor,
        column: int,
        source_offsets: torch.LongTensor,
        dest_offsets: torch.LongTensor,
        nbytes_to_copy: torch.LongTensor,
    ) -> int: ...
    def sub_(
        sself,
        indices: torch.LongTensor,
        timesteps: torch.LongTensor,
        lengths: torch.LongTensor,
        column: int,
        source_offsets: torch.LongTensor,
        dest_offsets: torch.LongTensor,
        nbytes_to_copy: torch.LongTensor,
    ) -> int: ...
    def subcopy(
        self,
        idxs: torch.cuda.LongTensor,
        timesteps: torch.cuda.LongTensor,
        lengths: torch.cuda.LongTensor,
        column: int,
        dst: torch.cuda.Tensor,
    ) -> None: ...
    def view(self, index: int, column: int) -> torch.Tensor: ...
    def raw(self, offset: int, length: int) -> torch.Tensor: ...

def get_cpu_handler(t: TrajectoryTable, global_capacity: int) -> CpuTrajectoryStorageHandler: ...
