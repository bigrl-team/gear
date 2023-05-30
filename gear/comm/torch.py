import libgear.comm as lgcm
import torch
from gear.comm.base import Communicator


class TorchDistributedCommunicator(Communicator):
    import torch
    import torch.distributed as dist

    def __init__(self, context) -> None:
        super().__init__()
        self._context = context
        self._id = lgcm.create_nccl_id()
        self._comm = lgcm.NcclComm(
            rank=self._context.get_rank(),
            ws=self._context.get_world_size(),
            id=self._id,
        )

    def send(self, tensor: torch.Tensor, dst: int):
        self._comm.send(tensor, dst)

    def recv(self, tensor: torch.Tensor, src: int):
        self._comm.recv(tensor, src)
