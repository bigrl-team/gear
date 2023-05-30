from typing import List, Union

import libgear.core as glibc
import torch
from gear.errors import ArgumentError
from gear.interface.params import TrajectoryInterfaceParams as Params


class TrajectoryShardServerParamChecker:
    @staticmethod
    def __call__(params: Params) -> Union[None, ArgumentError]:
        if params.node_world <= 0:
            raise ArgumentError(
                f"A positive number expected for the node worldsize, got {params.node_rank}"
            )
        if params.node_rank < 0:
            raise ArgumentError(
                f"A non-negative number expected for the node rank, got {params.node_rank}"
            )
        if params.shard_world <= 0:
            raise ArgumentError(f"Invalid shard world: {params.shard_world}")
        if params.shard_rank < 0:
            raise ArgumentError(f"Invalid shard rank: {params.shard_rank}")
        if params.shard_capacity <= 0:
            raise ArgumentError(f"Invalid shard capacity: {params.shard_capacity}")


class InferenceLoader:
    def __init__(
        self,
        index_server: glibc.SharedMemoryIndexServer,
        trajectory_table: glibc.TrajectoryTable,
        cache: glibc.CachedRequest,
        batch_size: int,
        sub_patterns: List[glibc.SubscribePattern],
        strict: bool,
    ) -> None:
        self._batch_size = batch_size
        self._table = trajectory_table
        self._index_server = index_server
        self._patterns = sub_patterns
        self._cache = cache
        self._strict = strict

        self._handler = glibc.get_cpu_handler(self._table)
        self._handler.set(sub_patterns)

    def __next__(self):
        while self._strict and self._cache.iarr.count <= self._batch_size:
            self._index_server.scan(self._cache)
            self._handler.sub_infer_cache(self._cache)

    def callback(self):
        pass


class TraininingLoader:
    def __init__(
        self,
        index_server: glibc.SharedMemoryIndexServer,
        trajectory_table: glibc.TrajectoryTable,
        cache: glibc.CachedRequest,
        batch_size: int,
        sub_patterns: List[glibc.SubscribePattern],
        strict: bool,
    ) -> None:
        pass

    def __next__(self):
        pass


class TrajectoryShardServer:
    def __init__(
        self, params: Params, table_spec: glibc.TableSpec
    ) -> Union[None, ArgumentError]:
        TrajectoryShardServerParamChecker(params)

        self._params = params
        self._index_server = glibc.SharedMemoryIndexServer(
            params.shard_key, params.shard_group_world - 1, params.shard_capacity
        )
        self._table = glibc.TrajectoryTable(
            table_spec,
            params.node_key,
            # server of first local shard create the entire trajectory table
            params.shard_rank == 0,
        )
        self._cache = glibc.CachedRequest(params.shard_capacity)

    def connect(self, force_all=False):
        num_active_agents = self._index_server.connect()
        while force_all and num_active_agents != self._params.shard_group_world:
            num_active_agents = self._index_server.connect()

        return num_active_agents

    def get_inference_loader(
        self,
        batch_size: int,
        sub_patterns: List[glibc.SubscribePattern],
        strict: bool = True,
    ) -> InferenceLoader:
        return InferenceLoader(
            self._index_server,
            self._table,
            self._cache,
            batch_size,
            sub_patterns,
            strict,
        )
