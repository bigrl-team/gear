import socket
import time

import libgear.comm as lgcm
import torch
from gear.comm.base import Communicator


class NcclCommunicator(Communicator):
    def __init__(self, context, port=13500, address="127.0.0.1") -> None:
        super().__init__()
        self._context = context

        if self._context.get_rank() == 0:
            self._id = lgcm.create_nccl_id()
            sock = socket.socket(socket.AF_INET)
            sock.bind((address, port))
            sock.listen()
            for i in range(1, self._context.get_world_size()):
                conn, caddr = sock.accept()
                conn.sendall(self._id)
                conn.close()
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET)

            sleep_time = 1
            for i in range(10):
                try:
                    sock.connect((address, port))
                    break
                except ConnectionRefusedError as e:
                    time.sleep(sleep_time)
                    sleep_time *= 2

            buf = sock.recv(1024)
            self._id = buf
            sock.close()

        self._comm = lgcm.NcclComm(
            self._context.get_rank(),
            self._context.get_world_size(),
            self._id,
        )

    def send(self, tensor: torch.Tensor, dst: int):
        self._comm.send(tensor, dst)

    def recv(self, tensor: torch.Tensor, src: int):
        self._comm.recv(tensor, src)
