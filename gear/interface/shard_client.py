from typing import Union

import libgear.core as glibc
import torch
from gear.errors import ArgumentError


class TrajectoryWriterParamChecker:
    @staticmethod
    def __call__(
        shard_group_rank, shard_group_world, shard_rank, node_rank
    ) -> Union[None, ArgumentError]:
        if shard_group_world <= 0:
            raise ArgumentError(f"Invalid shard group world size: {shard_group_world}")

        if shard_group_rank == 0 or shard_group_rank >= shard_group_world:
            raise ArgumentError(
                f"Valid rank range for TrajectoryWriter would be [1, {shard_group_world}], got {shard_group_rank}"
            )

        if shard_rank < 0:
            raise ArgumentError(
                f"A positive number expected for the shard rank, got {shard_rank}"
            )

        if node_rank < 0:
            raise ArgumentError(
                f"A positive number expected for the node rank, got {node_rank}"
            )


class TrajectoryShardClient:
    """
    TrajectoryShardClient holds a local view of the TrajectoryTable shard,
    all the clients and the server that operating on the shard forms
    a process group named "Shard Group", where the server takes rank 0 of
    the group.
    """

    def __init__(
        self,
        shard_key: int,
        shard_group_rank: int,
        shard_group_world: int,
        shard_rank: int,
        node_key: int,
        node_rank: int,
    ) -> None:
        TrajectoryWriterParamChecker(
            shard_group_rank, shard_group_rank, shard_rank, node_rank
        )
        self.shard_key = shard_key
        self.shard_group_rank = shard_group_rank
        self.shard_group_world = shard_group_world
        self.node_key = node_key
        self.node_rank = node_rank
        self.create: bool = False

        self._index_client = glibc.SharedMemoryIndexClient(
            shard_key, shard_group_rank - 1, shard_group_world - 1
        )

    @property
    def index(self):
        return self._index_client.get_index()

    @property
    def timestep(self):
        return self._index_client.get_timestep()

    def status(self):
        return self._index_client.get_status_unsafe()

    def connect(self, force_wait: bool = True):
        num_active_clients = self._index_client.connect()
        while force_wait and num_active_clients != self.shard_group_world - 1:
            num_active_clients = self._index_client.connect()
        return num_active_clients

    def create(self):
        self._index_client.acquire()
        return self.index

    def close(self, create=False):
        self._index_client.writeback(False, create)
        pass
