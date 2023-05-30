import os
import pathlib
import pickle
import random
from collections import namedtuple
from typing import Sequence, Union

import libgear.core as glibc
import torch
from gear.config import KEY_T_RANGE
from gear.dtypes import DataType
from gear.specs import ColumnSpec, TableSpec
from gear.utils import get_local_node_hashed_int32, ensure_path
from torch.utils.data.dataset import Dataset

SharedDatasetMeta = namedtuple("SharedDatasetMeta", ["storage", "index"])
IndexMeta = namedtuple(
    "IndexMeta", ["global_capacity", "local_capacity", "index_offset", "shared", "key"]
)


class SharedDataset(Dataset):
    @staticmethod
    def create(
        spec: TableSpec = Union[TableSpec, None],
        create: bool = True,
        shard_rank: int = 0,
        shard_world: int = 1,
        proc_rank: int = 0,
        # max_shared_procs: int = 1,
        key: Union[int, None] = None,
    ):
        if key is None:
            key = get_local_node_hashed_int32()
        rng = random.Random(key)
        table_key = rng.randint(0, KEY_T_RANGE)
        iset_key = rng.randint(0, KEY_T_RANGE)

        if create:
            if spec is None:
                raise RuntimeError(
                    "SharedDataset creator process should provide storage specs!"
                )
            # assert max_shared_procs >= 1, "'max_shared_procs' should be larger than 1."
            table = glibc.TrajectoryTable(spec, table_key, True)
            iset = glibc.Indexset(
                int(spec.capacity * spec.worldsize),
                int(spec.capacity / shard_world),
                int(
                    spec.capacity * spec.rank + spec.capacity * shard_rank / shard_world
                ),
                shared=True,
                key=iset_key,
                create=True,
            )
        else:
            # assert (
            #     0 < proc_rank < max_shared_procs
            # ), "'proc_rank' should be in the range [1, max_shard_procs) for pure clients, proc_rank 0 is reserved for shard server."
            table = glibc.TrajectoryTable(spec, table_key, False)
            iset = glibc.Indexset(
                spec.capacity * spec.worldsize,
                spec.capacity / shard_world,
                index_offset=(
                    spec.capacity * spec.rank + spec.capacity * shard_rank / shard_world
                ),
                shared=True,
                key=iset_key,
                create=False,
            )

        trc = -1
        while trc == -1:
            trc = table.connect()
        return SharedDataset(
            table,
            iset,
        )

    def __init__(
        self,
        table: Union[glibc.TrajectoryTable, None] = None,
        index_set: Union[glibc.Indexset, None] = None,
    ) -> None:
        super().__init__()

        self._table = table
        self._iset = index_set
        if table and index_set:
            self._entry_cls = namedtuple(
                "SharedDatasetEntry",
                ["weight", "timestep"]
                + list(
                    [f"column{i}" for i in range(table.get_table_spec().num_columns)]
                ),
            )
            self._handler = glibc.get_cpu_handler(
                table,
                index_set.global_capacity,
                glibc.Range(0, 0),
                glibc.Range(
                    index_set.index_offset,
                    index_set.index_offset + index_set.local_capacity,
                ),
            )
        else:
            self._handler = None

    @property
    def weights(self):
        return self._iset.weights

    @property
    def timesteps(self):
        return self._iset.timesteps

    @property
    def num_col(self):
        return self._table.ncolumns

    def __getitem__(self, index):
        ret = (
            self.weights[index],
            self.timesteps[index],
        )
        ret += self.get_tensor_view(index)
        return self._entry_cls(*ret)

    def __len__(self):
        return self._iset.capacity

    def __setitem__(self, index, value):
        weight, timestep, columns = value
        self._iset.weights[index] = weight
        self._iset.timesteps[index] = timestep
        for cid in range(self.num_col):
            self._handler.view(index, cid).copy(columns[cid].view(-1))

    def get_column_spec(self, cid: int) -> ColumnSpec:
        return self._table.get_column_spec(cid)

    def get_column_dtype(self, cid: int) -> DataType:
        """
        Get column id specified datatype from the TableSpec
        """
        return self._table.get_column_spec(cid).dtype

    def get_column_shape(self, cid: int) -> DataType:
        """
        Get column id specified datatype from the TableSpec
        """
        return (self._table.get_table_spec().trajectory_length,) + tuple(
            self._table.get_column_spec(cid).shape
        )

    def get_raw_view(
        self, index: int, cids: Sequence[int] = None
    ) -> Union[glibc.Uint8Span, Sequence[glibc.Uint8Span]]:
        if cids is None:
            cids = range(self.num_col)

        return tuple([self._handler.view(index, cid) for cid in cids])

    def get_tensor_view(
        self, index: int, cids: Sequence[int] = None
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        if cids is None:
            cids = range(self.num_col)
        raw_views = self.get_raw_view(index, cids)
        return tuple(
            [
                raw_views[idx]
                .cast_tensor(self.get_column_dtype(cid))
                .view(self.get_column_shape(cid))
                for idx, cid in enumerate(cids)
            ]
        )

    def checkpoint(self, path: Union[pathlib.Path, str]):
        ensure_path(path)

        with open(path, "wb") as of:
            pickler = pickle.Pickler(of)
            pickler.dump(
                SharedDatasetMeta(
                    self._table.get_table_spec(),
                    IndexMeta(
                        self._iset.global_capacity,
                        self._iset.local_capacity,
                        self._iset.index_offset,
                        self._iset.is_sharing,
                        self._iset.shm_key,
                    ),
                )
            )
            print("meta dumped")
            pickler.dump(self._table)
            print("table dumped")
            pickler.dump(self._iset.get_state())

    def load(
        self,
        path: Union[pathlib.Path, str],
        key: Union[int, None] = None,
        create: bool = True,
        proc_rank: int = 0,
    ):
        self._release_resources()

        if key is None:
            key = get_local_node_hashed_int32()

        rng = random.Random(key)
        table_key = rng.randint(0, KEY_T_RANGE)
        iset_key = rng.randint(0, KEY_T_RANGE)

        path = pathlib.Path(path)

        with open(path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            meta: SharedDatasetMeta = unpickler.load()
            if create:
                self._table = unpickler.load()
                self._iset = glibc.Indexset.load_state(
                    unpickler.load(), meta.index.shared, iset_key, True
                )
            else:
                self._table = glibc.TrajectoryTable(meta.storage, table_key, False)
                self._iset = glibc.Indexset(
                    meta.index.global_capacity,
                    meta.index.local_capacity,
                    meta.index.index_offset,
                    meta.index.shared,
                    iset_key,
                    False,
                )
            self._entry_cls = namedtuple(
                "SharedDatasetEntry",
                ["weights", "timesteps"]
                + list(
                    [
                        f"column{i}"
                        for i in range(self._table.get_table_spec().num_columns)
                    ]
                ),
            )

        self._ensure_attached()
        self._handler = glibc.get_cpu_handler(
            self._table,
            self._iset.global_capacity,
            glibc.Range(0, 0),
            glibc.Range(
                self._iset.index_offset,
                self._iset.index_offset + self._iset.local_capacity,
            ),
        )
        return self

    def _release_resources(self):
        if self._table:
            del self._table
        if self._iset:
            del self._iset

    def _ensure_attached(self, blocking: bool = True):
        resources_list = [self._table]
        for res in resources_list:
            rc = -1
            while rc == -1:
                rc = res.connect()
                if not blocking:
                    break
